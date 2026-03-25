"""
IMAP email client — connects to the user's real inbox via IMAP.
Supports Gmail, Outlook, and any standard IMAP provider.
"""
from __future__ import annotations

import email
import imaplib
import re
from dataclasses import dataclass, field
from datetime import datetime
from email.header import decode_header
from typing import Optional


@dataclass
class EmailConfig:
    user: str
    password: str
    host: str
    port: int = 993
    use_ssl: bool = True


@dataclass
class EmailMetadata:
    uid: str
    message_id: str
    subject: str
    from_addr: str
    to_addr: str
    date: str
    is_seen: bool
    is_flagged: bool
    has_attachments: bool
    size: int
    mailbox: str = "INBOX"


@dataclass
class EmailFull(EmailMetadata):
    text_body: str = ""
    html_body: str = ""
    snippet: str = ""  # first 400 chars of text body


@dataclass
class SearchFilter:
    unread_only: bool = False
    flagged_only: bool = False
    from_addr: str = ""
    to_addr: str = ""
    subject: str = ""
    text: str = ""  # IMAP TEXT — searches headers + body
    since: Optional[datetime] = None
    before: Optional[datetime] = None
    limit: int = 30
    mailbox: str = "INBOX"


def _decode_header_value(value: str) -> str:
    """Decode encoded email header value to a plain string."""
    parts = decode_header(value or "")
    result = []
    for part, charset in parts:
        if isinstance(part, bytes):
            try:
                result.append(part.decode(charset or "utf-8", errors="replace"))
            except Exception:
                result.append(part.decode("utf-8", errors="replace"))
        else:
            result.append(str(part))
    return "".join(result).strip()


def _extract_body(msg: email.message.Message) -> tuple[str, str, bool]:
    """Extract (text_body, html_body, has_attachments) from a parsed email."""
    text_body = ""
    html_body = ""
    has_attachments = False

    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            disposition = str(part.get("Content-Disposition", ""))

            if "attachment" in disposition.lower():
                has_attachments = True
                continue

            if content_type == "text/plain" and not text_body:
                payload = part.get_payload(decode=True)
                if payload:
                    charset = part.get_content_charset() or "utf-8"
                    text_body = payload.decode(charset, errors="replace")
            elif content_type == "text/html" and not html_body:
                payload = part.get_payload(decode=True)
                if payload:
                    charset = part.get_content_charset() or "utf-8"
                    html_body = payload.decode(charset, errors="replace")
    else:
        payload = msg.get_payload(decode=True)
        if payload:
            charset = msg.get_content_charset() or "utf-8"
            content_type = msg.get_content_type()
            text = payload.decode(charset, errors="replace")
            if content_type == "text/html":
                html_body = text
            else:
                text_body = text

    return text_body, html_body, has_attachments


class EmailClient:
    def __init__(self, config: EmailConfig):
        self.config = config

    def _connect(self) -> imaplib.IMAP4_SSL | imaplib.IMAP4:
        if self.config.use_ssl:
            conn = imaplib.IMAP4_SSL(self.config.host, self.config.port)
        else:
            conn = imaplib.IMAP4(self.config.host, self.config.port)
        conn.login(self.config.user, self.config.password)
        return conn

    def list_mailboxes(self) -> list[str]:
        """Return all mailbox/folder names available."""
        conn = self._connect()
        try:
            _, data = conn.list()
            mailboxes = []
            for item in data:
                if item:
                    decoded = item.decode() if isinstance(item, bytes) else item
                    # Parse: (\HasNoChildren) "/" "INBOX"
                    match = re.search(r'"([^"]+)"$|(\S+)$', decoded)
                    if match:
                        name = (match.group(1) or match.group(2)).strip('"')
                        mailboxes.append(name)
            return mailboxes
        finally:
            conn.logout()

    def fetch_metadata(self, search_filter: SearchFilter | None = None) -> list[EmailMetadata]:
        """Fetch email metadata (headers only, fast) for the inbox."""
        f = search_filter or SearchFilter()
        conn = self._connect()
        try:
            conn.select(f.mailbox, readonly=True)

            # Build IMAP search criteria
            criteria: list[str] = []
            if f.unread_only:
                criteria.append("UNSEEN")
            if f.flagged_only:
                criteria.append("FLAGGED")
            if f.from_addr:
                criteria.append(f'FROM "{f.from_addr}"')
            if f.to_addr:
                criteria.append(f'TO "{f.to_addr}"')
            if f.subject:
                criteria.append(f'SUBJECT "{f.subject}"')
            if f.text:
                criteria.append(f'TEXT "{f.text}"')
            if f.since:
                criteria.append(f'SINCE "{f.since.strftime("%d-%b-%Y")}"')
            if f.before:
                criteria.append(f'BEFORE "{f.before.strftime("%d-%b-%Y")}"')

            search_str = " ".join(criteria) if criteria else "ALL"
            _, data = conn.search(None, search_str)

            uids = data[0].split() if data[0] else []
            if not uids:
                return []

            # Take the most recent N
            target_uids = uids[-f.limit:]

            results: list[EmailMetadata] = []
            uid_str = ",".join(u.decode() for u in target_uids)
            _, msg_data = conn.fetch(uid_str, "(UID FLAGS RFC822.SIZE BODY.PEEK[HEADER.FIELDS (FROM TO SUBJECT DATE MESSAGE-ID)])")

            i = 0
            while i < len(msg_data):
                item = msg_data[i]
                if not isinstance(item, tuple):
                    i += 1
                    continue

                meta_str = item[0].decode() if isinstance(item[0], bytes) else str(item[0])
                header_bytes = item[1]

                # Parse UID and FLAGS from meta string
                uid_match = re.search(r"UID (\d+)", meta_str)
                uid = uid_match.group(1) if uid_match else str(i)
                is_seen = "\\Seen" in meta_str
                is_flagged = "\\Flagged" in meta_str
                size_match = re.search(r"RFC822\.SIZE (\d+)", meta_str)
                size = int(size_match.group(1)) if size_match else 0

                # Parse headers
                header_text = header_bytes.decode("utf-8", errors="replace") if isinstance(header_bytes, bytes) else str(header_bytes)
                msg = email.message_from_string(header_text)

                subject = _decode_header_value(msg.get("Subject", "(no subject)"))
                from_addr = _decode_header_value(msg.get("From", ""))
                to_addr = _decode_header_value(msg.get("To", ""))
                date = msg.get("Date", "")
                message_id = msg.get("Message-ID", "")

                results.append(EmailMetadata(
                    uid=uid,
                    message_id=message_id,
                    subject=subject,
                    from_addr=from_addr,
                    to_addr=to_addr,
                    date=date,
                    is_seen=is_seen,
                    is_flagged=is_flagged,
                    has_attachments=False,  # Can't tell from headers alone
                    size=size,
                    mailbox=f.mailbox,
                ))
                i += 1

            return list(reversed(results))  # Newest first
        finally:
            conn.logout()

    def fetch_full_email(self, uid: str, mailbox: str = "INBOX") -> EmailFull | None:
        """Fetch full email content for a specific UID."""
        conn = self._connect()
        try:
            conn.select(mailbox, readonly=True)
            _, data = conn.uid("FETCH", uid, "(UID FLAGS RFC822.SIZE RFC822)")

            for item in data:
                if not isinstance(item, tuple):
                    continue

                meta_str = item[0].decode() if isinstance(item[0], bytes) else str(item[0])
                raw = item[1]

                is_seen = "\\Seen" in meta_str
                is_flagged = "\\Flagged" in meta_str
                size_match = re.search(r"RFC822\.SIZE (\d+)", meta_str)
                size = int(size_match.group(1)) if size_match else 0

                msg = email.message_from_bytes(raw if isinstance(raw, bytes) else raw.encode())

                subject = _decode_header_value(msg.get("Subject", "(no subject)"))
                from_addr = _decode_header_value(msg.get("From", ""))
                to_addr = _decode_header_value(msg.get("To", ""))
                date = msg.get("Date", "")
                message_id = msg.get("Message-ID", "")

                text_body, html_body, has_attachments = _extract_body(msg)
                snippet = re.sub(r"\s+", " ", text_body).strip()[:400]

                return EmailFull(
                    uid=uid,
                    message_id=message_id,
                    subject=subject,
                    from_addr=from_addr,
                    to_addr=to_addr,
                    date=date,
                    is_seen=is_seen,
                    is_flagged=is_flagged,
                    has_attachments=has_attachments,
                    size=size,
                    mailbox=mailbox,
                    text_body=text_body,
                    html_body=html_body,
                    snippet=snippet,
                )
            return None
        finally:
            conn.logout()

    def mark_seen(self, uid: str, mailbox: str = "INBOX") -> None:
        conn = self._connect()
        try:
            conn.select(mailbox, readonly=False)
            conn.uid("STORE", uid, "+FLAGS", "\\Seen")
        finally:
            conn.logout()

    def mark_flagged(self, uid: str, mailbox: str = "INBOX") -> None:
        conn = self._connect()
        try:
            conn.select(mailbox, readonly=False)
            conn.uid("STORE", uid, "+FLAGS", "\\Flagged")
        finally:
            conn.logout()

    def test_connection(self) -> tuple[bool, str]:
        """Test IMAP connection. Returns (success, message)."""
        try:
            conn = self._connect()
            conn.logout()
            return True, f"Connected to {self.config.host} as {self.config.user}"
        except Exception as e:
            return False, str(e)
