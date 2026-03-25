"""
Email Classifier MCP Server
============================
Gives Claude Desktop AI-powered email classification tools that work
directly on the user's real IMAP inbox (Gmail, Outlook, or any provider).

Tools exposed:
  - classify_inbox        — fetch + classify recent emails, return ranked list
  - classify_single_email — classify one specific email by UID
  - get_inbox_summary     — AI narrative summary of what needs attention
  - list_mailboxes        — list all folders in the mailbox
  - fetch_email_content   — read a full email body by UID
  - mark_email_seen       — mark an email as read
  - mark_email_flagged    — flag an email for follow-up
  - test_connection       — verify IMAP credentials are working
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict

from dotenv import load_dotenv
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from classifier import ClassificationResult, EmailClassifier, EmailInput
from email_client import EmailClient, EmailConfig, SearchFilter

load_dotenv()

# ---------------------------------------------------------------------------
# Provider IMAP settings
# ---------------------------------------------------------------------------

PROVIDERS = {
    "gmail":      {"host": "imap.gmail.com",          "port": 993, "ssl": True},
    "outlook":    {"host": "outlook.office365.com",    "port": 993, "ssl": True},
    "hotmail":    {"host": "outlook.office365.com",    "port": 993, "ssl": True},
    "yahoo":      {"host": "imap.mail.yahoo.com",     "port": 993, "ssl": True},
    "icloud":     {"host": "imap.mail.me.com",        "port": 993, "ssl": True},
    "aol":        {"host": "imap.aol.com",            "port": 993, "ssl": True},
    "zoho":       {"host": "imap.zoho.com",           "port": 993, "ssl": True},
    "protonmail": {"host": "127.0.0.1",               "port": 1143, "ssl": False},
    "fastmail":   {"host": "imap.fastmail.com",       "port": 993, "ssl": True},
    "gmx":        {"host": "imap.gmx.com",            "port": 993, "ssl": True},
    "mail.ru":    {"host": "imap.mail.ru",            "port": 993, "ssl": True},
    "yandex":     {"host": "imap.yandex.com",         "port": 993, "ssl": True},
}

# ---------------------------------------------------------------------------
# Multi-account config
# ---------------------------------------------------------------------------


def _get_accounts() -> list[dict]:
    """
    Load accounts from environment.

    Multi-account:  EMAIL_ACCOUNTS='[{"provider":"gmail","username":"...","password":"..."},...]'
    Single account: EMAIL_USER + EMAIL_PASSWORD  (+ optional EMAIL_IMAP_HOST, EMAIL_IMAP_PORT, EMAIL_USE_SSL)
    """
    accounts_json = os.environ.get("EMAIL_ACCOUNTS")
    if accounts_json:
        try:
            return json.loads(accounts_json)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"EMAIL_ACCOUNTS is not valid JSON: {e}")

    # Fallback: single account from legacy env vars
    username = os.environ.get("EMAIL_USER")
    password = os.environ.get("EMAIL_PASSWORD")
    if not username or not password:
        return []

    account: dict = {"username": username, "password": password}
    if os.environ.get("EMAIL_IMAP_HOST"):
        account["host"] = os.environ["EMAIL_IMAP_HOST"]
        account["port"] = int(os.environ.get("EMAIL_IMAP_PORT", "993"))
        account["ssl"] = os.environ.get("EMAIL_USE_SSL", "true").lower() == "true"
    else:
        account["provider"] = os.environ.get("EMAIL_PROVIDER", "gmail")
    return [account]


def _make_client(account: dict) -> EmailClient:
    """Build an EmailClient from an account dict."""
    provider = account.get("provider", "").lower()
    if provider in PROVIDERS:
        settings = PROVIDERS[provider]
        host = account.get("host", settings["host"])
        port = account.get("port", settings["port"])
        ssl = account.get("ssl", settings["ssl"])
    elif "host" in account:
        host = account["host"]
        port = account.get("port", 993)
        ssl = account.get("ssl", True)
    else:
        raise RuntimeError(
            f"Unknown provider '{provider}' and no 'host' specified. "
            f"Known providers: {', '.join(PROVIDERS.keys())}"
        )

    return EmailClient(EmailConfig(
        user=account["username"],
        password=account["password"],
        host=host,
        port=port,
        use_ssl=ssl,
    ))


def _get_client(account_id: int = 0) -> EmailClient:
    accounts = _get_accounts()
    if not accounts:
        raise RuntimeError(
            "No email accounts configured. "
            "Set EMAIL_USER/EMAIL_PASSWORD, or EMAIL_ACCOUNTS as JSON."
        )
    if account_id >= len(accounts):
        raise RuntimeError(f"Account index {account_id} out of range (have {len(accounts)} accounts)")
    return _make_client(accounts[account_id])


def get_classifier() -> EmailClassifier:
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    return EmailClassifier(api_key=api_key)


# ---------------------------------------------------------------------------
# Label formatting helpers
# ---------------------------------------------------------------------------

LABEL_EMOJI = {
    "URGENT":     "🔴",
    "IMPORTANT":  "🟠",
    "NORMAL":     "🟡",
    "LOW":        "🔵",
    "NEWSLETTER": "📰",
    "SPAM":       "🗑️",
}


def format_classification(
    meta_dict: dict,
    result: ClassificationResult,
) -> dict:
    return {
        "uid": meta_dict.get("uid"),
        "subject": meta_dict.get("subject"),
        "from": meta_dict.get("from_addr"),
        "date": meta_dict.get("date"),
        "label": result.label,
        "label_emoji": LABEL_EMOJI.get(result.label, ""),
        "confidence": result.confidence,
        "reasoning": result.reasoning,
        "suggested_action": result.suggested_action,
        "key_points": result.key_points,
        "is_seen": meta_dict.get("is_seen"),
    }


# ---------------------------------------------------------------------------
# MCP server setup
# ---------------------------------------------------------------------------

server = Server("email-classifier-mcp")


ACCOUNT_ID_PROP = {
    "type": "integer",
    "description": "Account index (default 0). Use list_accounts to see all accounts.",
    "default": 0,
}


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="list_accounts",
            description="List all configured email accounts. Returns index, provider, and username for each.",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="classify_inbox",
            description=(
                "Fetch emails from the user's inbox and classify each one by importance "
                "(URGENT, IMPORTANT, NORMAL, LOW, NEWSLETTER, SPAM). "
                "Returns a sorted list with label, confidence score, reasoning, and suggested action. "
                "Use this as the main entry point for inbox triage."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "account_id": ACCOUNT_ID_PROP,
                    "limit": {
                        "type": "integer",
                        "description": "Number of recent emails to classify (default: 20, max: 50)",
                        "default": 20,
                    },
                    "unread_only": {
                        "type": "boolean",
                        "description": "Only classify unread emails (default: false)",
                        "default": False,
                    },
                    "mailbox": {
                        "type": "string",
                        "description": "Mailbox/folder to read from (default: INBOX)",
                        "default": "INBOX",
                    },
                    "from_filter": {
                        "type": "string",
                        "description": "Filter emails by sender address or name (optional)",
                    },
                    "subject_filter": {
                        "type": "string",
                        "description": "Filter emails by subject keyword (optional)",
                    },
                },
            },
        ),
        Tool(
            name="classify_single_email",
            description=(
                "Classify a single specific email by its UID. "
                "Fetches the full email body for higher accuracy classification. "
                "Use when you need detailed analysis of one email."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "account_id": ACCOUNT_ID_PROP,
                    "uid": {
                        "type": "string",
                        "description": "The UID of the email to classify",
                    },
                    "mailbox": {
                        "type": "string",
                        "description": "Mailbox/folder containing the email (default: INBOX)",
                        "default": "INBOX",
                    },
                },
                "required": ["uid"],
            },
        ),
        Tool(
            name="get_inbox_summary",
            description=(
                "Analyze recent inbox emails and generate a concise natural-language summary "
                "of what needs attention, what can wait, and what to delete. "
                "Great for a daily inbox briefing."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "account_id": ACCOUNT_ID_PROP,
                    "limit": {
                        "type": "integer",
                        "description": "Number of recent emails to analyze (default: 30)",
                        "default": 30,
                    },
                    "unread_only": {
                        "type": "boolean",
                        "description": "Only analyze unread emails (default: false)",
                        "default": False,
                    },
                },
            },
        ),
        Tool(
            name="list_mailboxes",
            description="List all mailbox folders available in the email account.",
            inputSchema={
                "type": "object",
                "properties": {
                    "account_id": ACCOUNT_ID_PROP,
                },
            },
        ),
        Tool(
            name="fetch_email_content",
            description=(
                "Fetch the full content (subject, body, headers) of a specific email by UID. "
                "Returns both plain text and metadata. Use before classify_single_email "
                "if you want to read the email yourself."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "account_id": ACCOUNT_ID_PROP,
                    "uid": {
                        "type": "string",
                        "description": "The UID of the email to fetch",
                    },
                    "mailbox": {
                        "type": "string",
                        "description": "Mailbox/folder (default: INBOX)",
                        "default": "INBOX",
                    },
                },
                "required": ["uid"],
            },
        ),
        Tool(
            name="mark_email_seen",
            description="Mark an email as read/seen in the inbox.",
            inputSchema={
                "type": "object",
                "properties": {
                    "account_id": ACCOUNT_ID_PROP,
                    "uid": {"type": "string", "description": "Email UID to mark as seen"},
                    "mailbox": {"type": "string", "default": "INBOX"},
                },
                "required": ["uid"],
            },
        ),
        Tool(
            name="mark_email_flagged",
            description="Flag an email for follow-up (marks with the IMAP \\Flagged flag, shows as starred in Gmail).",
            inputSchema={
                "type": "object",
                "properties": {
                    "account_id": ACCOUNT_ID_PROP,
                    "uid": {"type": "string", "description": "Email UID to flag"},
                    "mailbox": {"type": "string", "default": "INBOX"},
                },
                "required": ["uid"],
            },
        ),
        Tool(
            name="test_connection",
            description="Test the IMAP connection to verify credentials and server are working.",
            inputSchema={
                "type": "object",
                "properties": {
                    "account_id": ACCOUNT_ID_PROP,
                },
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    try:
        result = await _dispatch(name, arguments)
        return [TextContent(type="text", text=json.dumps(result, indent=2, ensure_ascii=False))]
    except Exception as e:
        error_response = {"error": str(e), "tool": name}
        return [TextContent(type="text", text=json.dumps(error_response, indent=2))]


async def _dispatch(name: str, args: dict) -> dict | list:
    account_id = int(args.get("account_id", 0))

    # ------------------------------------------------------------------
    if name == "list_accounts":
        accounts = _get_accounts()
        return {
            "accounts": [
                {
                    "index": i,
                    "provider": a.get("provider", "custom"),
                    "username": a.get("username", ""),
                }
                for i, a in enumerate(accounts)
            ],
            "total": len(accounts),
        }

    client = _get_client(account_id)

    # ------------------------------------------------------------------
    if name == "test_connection":
        success, message = client.test_connection()
        return {"success": success, "message": message, "account_id": account_id}

    # ------------------------------------------------------------------
    elif name == "list_mailboxes":
        mailboxes = client.list_mailboxes()
        return {"mailboxes": mailboxes, "count": len(mailboxes), "account_id": account_id}

    # ------------------------------------------------------------------
    elif name == "fetch_email_content":
        uid = str(args["uid"])
        mailbox = args.get("mailbox", "INBOX")
        full = client.fetch_full_email(uid, mailbox)
        if not full:
            return {"error": f"Email UID {uid} not found in {mailbox}"}
        d = asdict(full)
        # Don't return huge HTML body by default
        d.pop("html_body", None)
        return d

    # ------------------------------------------------------------------
    elif name == "mark_email_seen":
        client.mark_seen(str(args["uid"]), args.get("mailbox", "INBOX"))
        return {"success": True, "uid": args["uid"]}

    # ------------------------------------------------------------------
    elif name == "mark_email_flagged":
        client.mark_flagged(str(args["uid"]), args.get("mailbox", "INBOX"))
        return {"success": True, "uid": args["uid"]}

    # ------------------------------------------------------------------
    elif name == "classify_single_email":
        uid = str(args["uid"])
        mailbox = args.get("mailbox", "INBOX")
        full = client.fetch_full_email(uid, mailbox)
        if not full:
            return {"error": f"Email UID {uid} not found"}

        classifier = get_classifier()
        email_input = EmailInput(
            subject=full.subject,
            from_addr=full.from_addr,
            to_addr=full.to_addr,
            date=full.date,
            snippet=full.snippet or full.text_body[:400],
        )
        result = classifier.classify(email_input)
        return format_classification(asdict(full), result)

    # ------------------------------------------------------------------
    elif name == "classify_inbox":
        limit = min(int(args.get("limit", 20)), 50)
        search_filter = SearchFilter(
            unread_only=bool(args.get("unread_only", False)),
            from_addr=str(args.get("from_filter", "")),
            subject=str(args.get("subject_filter", "")),
            mailbox=str(args.get("mailbox", "INBOX")),
            limit=limit,
        )

        emails_meta = client.fetch_metadata(search_filter)
        if not emails_meta:
            return {
                "classified": [],
                "message": "No emails found matching the criteria",
                "total": 0,
            }

        classifier = get_classifier()
        inputs = [
            EmailInput(
                subject=m.subject,
                from_addr=m.from_addr,
                to_addr=m.to_addr,
                date=m.date,
                snippet="",  # Metadata only for batch — fast
            )
            for m in emails_meta
        ]
        results = classifier.classify_batch(inputs)

        # Sort by priority: URGENT → IMPORTANT → NORMAL → LOW → NEWSLETTER → SPAM
        priority = {"URGENT": 0, "IMPORTANT": 1, "NORMAL": 2, "LOW": 3, "NEWSLETTER": 4, "SPAM": 5}
        classified_pairs = list(zip(emails_meta, results))
        classified_pairs.sort(key=lambda p: priority.get(p[1].label, 99))

        classified_output = [
            format_classification(asdict(meta), result)
            for meta, result in classified_pairs
        ]

        # Quick stats
        label_counts: dict[str, int] = {}
        for _, r in classified_pairs:
            label_counts[r.label] = label_counts.get(r.label, 0) + 1

        return {
            "classified": classified_output,
            "total": len(classified_output),
            "summary": label_counts,
        }

    # ------------------------------------------------------------------
    elif name == "get_inbox_summary":
        limit = min(int(args.get("limit", 30)), 50)
        search_filter = SearchFilter(
            unread_only=bool(args.get("unread_only", False)),
            mailbox="INBOX",
            limit=limit,
        )

        emails_meta = client.fetch_metadata(search_filter)
        if not emails_meta:
            return {"summary": "Your inbox appears to be empty or no emails match the criteria."}

        classifier = get_classifier()
        inputs = [
            EmailInput(
                subject=m.subject,
                from_addr=m.from_addr,
                to_addr=m.to_addr,
                date=m.date,
                snippet="",
            )
            for m in emails_meta
        ]
        results = classifier.classify_batch(inputs)
        pairs = list(zip(inputs, results))
        summary_text = classifier.generate_inbox_summary(pairs)

        label_counts: dict[str, int] = {}
        for _, r in results:
            label_counts[r.label] = label_counts.get(r.label, 0) + 1

        return {
            "summary": summary_text,
            "breakdown": label_counts,
            "total_analyzed": len(emails_meta),
        }

    else:
        return {"error": f"Unknown tool: {name}"}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
