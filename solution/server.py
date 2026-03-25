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
  - search_emails         — search inbox by keyword, sender, subject, date range
  - test_connection       — verify IMAP credentials are working
"""
from __future__ import annotations

import json
import os
import pathlib
import time
from dataclasses import asdict

from dotenv import load_dotenv
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from classifier import ClassificationResult, DraftResult, EmailClassifier, EmailInput, ResponseOptions
from email_client import EmailClient, EmailConfig, EmailMetadata, SearchFilter

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
# Classification cache
# ---------------------------------------------------------------------------

_CACHE_TTL = 3600.0  # seconds — re-classify after 1 hour
_CACHE_FILE = pathlib.Path(__file__).parent / "classification_cache.json"
_classification_cache: dict[str, tuple[ClassificationResult, float]] = {}


def _cache_key(meta: EmailMetadata) -> str:
    """Stable key: prefer message_id (globally unique), fall back to mailbox:uid."""
    mid = (meta.message_id or "").strip()
    return mid if mid else f"{meta.mailbox}:{meta.uid}"


def _get_cached(key: str) -> ClassificationResult | None:
    entry = _classification_cache.get(key)
    if entry is None:
        return None
    result, ts = entry
    if time.time() - ts > _CACHE_TTL:
        del _classification_cache[key]
        return None
    return result


def _set_cached(key: str, result: ClassificationResult) -> None:
    _classification_cache[key] = (result, time.time())
    _save_cache()


def _save_cache() -> None:
    data = {
        key: {**asdict(result), "timestamp": ts}
        for key, (result, ts) in _classification_cache.items()
    }
    _CACHE_FILE.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def _load_cache() -> None:
    if not _CACHE_FILE.exists():
        return
    try:
        data = json.loads(_CACHE_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return
    now = time.time()
    for key, entry in data.items():
        ts = float(entry.get("timestamp", 0))
        if now - ts > _CACHE_TTL:
            continue
        try:
            _classification_cache[key] = (
                ClassificationResult(
                    label=entry["label"],
                    confidence=entry["confidence"],
                    reasoning=entry["reasoning"],
                    suggested_action=entry["suggested_action"],
                    key_points=entry["key_points"],
                ),
                ts,
            )
        except (KeyError, TypeError):
            continue


_load_cache()


# ---------------------------------------------------------------------------
# Content-fallback helpers
# ---------------------------------------------------------------------------

_LOW_CONFIDENCE_THRESHOLD = 60


def _enrich_with_body(
    client: EmailClient,
    classifier: EmailClassifier,
    emails_meta: list[EmailMetadata],
    results: list[ClassificationResult],
) -> list[ClassificationResult]:
    """Re-classify emails whose initial confidence is below the threshold.

    Fetches the full body for each such email and runs a second classification
    pass so that content — not just headers — informs the label.
    """
    enriched = list(results)
    for i, (meta, result) in enumerate(zip(emails_meta, results)):
        if result.confidence >= _LOW_CONFIDENCE_THRESHOLD:
            continue
        full = client.fetch_full_email(meta.uid, meta.mailbox)
        if not full:
            continue
        email_input = EmailInput(
            subject=full.subject,
            from_addr=full.from_addr,
            to_addr=full.to_addr,
            date=full.date,
            snippet=full.snippet or full.text_body[:400],
        )
        try:
            enriched[i] = classifier.classify(email_input)
        except Exception as ex:
            # Keep the original result; don't fail the whole batch
            enriched[i] = ClassificationResult(
                label=result.label,
                confidence=result.confidence,
                reasoning=f"{result.reasoning} (body fetch failed: {ex})",
                suggested_action=result.suggested_action,
                key_points=result.key_points,
            )
    return enriched


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
                        "description": "Number of recent emails to classify (default: 30, max: 100)",
                        "default": 30,
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
            name="search_emails",
            description=(
                "Search emails by any combination of keyword (searches headers + body), "
                "sender, recipient, subject, and date range. Returns matching email metadata. "
                "Use this when the user asks to find, look up, or search for specific emails."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Keyword to search in headers and body (optional)",
                    },
                    "from_filter": {
                        "type": "string",
                        "description": "Filter by sender address or name (optional)",
                    },
                    "to_filter": {
                        "type": "string",
                        "description": "Filter by recipient address or name (optional)",
                    },
                    "subject_filter": {
                        "type": "string",
                        "description": "Filter by subject keyword (optional)",
                    },
                    "unread_only": {
                        "type": "boolean",
                        "description": "Only return unread emails (default: false)",
                        "default": False,
                    },
                    "flagged_only": {
                        "type": "boolean",
                        "description": "Only return flagged/starred emails (default: false)",
                        "default": False,
                    },
                    "since": {
                        "type": "string",
                        "description": "Return emails on or after this date (YYYY-MM-DD, optional)",
                    },
                    "before": {
                        "type": "string",
                        "description": "Return emails before this date (YYYY-MM-DD, optional)",
                    },
                    "mailbox": {
                        "type": "string",
                        "description": "Mailbox to search (default: INBOX)",
                        "default": "INBOX",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max results to return (default: 30, max: 100)",
                        "default": 30,
                    },
                },
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
        Tool(
            name="analyze_email_for_response",
            description=(
                "Read a full email and return contextually appropriate response options, "
                "the email's detected tone, a summary of what it's asking, and whether "
                "clarification from the user is needed before drafting a reply. "
                "Call this first before draft_email_response."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "uid": {
                        "type": "string",
                        "description": "UID of the email to analyze",
                    },
                    "mailbox": {
                        "type": "string",
                        "description": "Mailbox containing the email (default: INBOX)",
                        "default": "INBOX",
                    },
                },
                "required": ["uid"],
            },
        ),
        Tool(
            name="draft_email_response",
            description=(
                "Draft a reply to an email given the user's chosen core answer and optional "
                "clarification. Fetches the full email for context; tone of the reply is "
                "matched to the original. Use analyze_email_for_response first to get "
                "the list of response options to present to the user."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "uid": {
                        "type": "string",
                        "description": "UID of the email to reply to",
                    },
                    "mailbox": {
                        "type": "string",
                        "description": "Mailbox containing the email (default: INBOX)",
                        "default": "INBOX",
                    },
                    "core_answer": {
                        "type": "string",
                        "description": "The user's core answer, e.g. 'Yes, I can attend'",
                    },
                    "clarification": {
                        "type": "string",
                        "description": "Optional extra context or reason from the user",
                        "default": "",
                    },
                },
                "required": ["uid", "core_answer"],
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
    if name == "search_emails":
        from datetime import datetime as _dt
        sf = SearchFilter(
            unread_only=bool(args.get("unread_only", False)),
            flagged_only=bool(args.get("flagged_only", False)),
            from_addr=str(args.get("from_filter", "")),
            to_addr=str(args.get("to_filter", "")),
            subject=str(args.get("subject_filter", "")),
            text=str(args.get("query", "")),
            since=_dt.strptime(args["since"], "%Y-%m-%d") if args.get("since") else None,
            before=_dt.strptime(args["before"], "%Y-%m-%d") if args.get("before") else None,
            mailbox=str(args.get("mailbox", "INBOX")),
            limit=min(int(args.get("limit", 30)), 100),
        )
        results = client.fetch_metadata(sf)
        return {
            "results": [asdict(m) for m in results],
            "total": len(results),
            "query": {k: v for k, v in args.items() if v},
        }

    # ------------------------------------------------------------------
    elif name == "test_connection":
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

        key = (full.message_id or "").strip() or f"{mailbox}:{uid}"
        cached = _get_cached(key)
        if cached:
            return {**format_classification(asdict(full), cached), "from_cache": True}

        classifier = get_classifier()
        email_input = EmailInput(
            subject=full.subject,
            from_addr=full.from_addr,
            to_addr=full.to_addr,
            date=full.date,
            snippet=full.snippet or full.text_body[:400],
        )
        result = classifier.classify(email_input)
        _set_cached(key, result)
        return format_classification(asdict(full), result)

    # ------------------------------------------------------------------
    elif name == "classify_inbox":
        limit = min(int(args.get("limit", 30)), 100)
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

        # Serve cached results; only classify emails we haven't seen recently
        cache_keys = [_cache_key(m) for m in emails_meta]
        results: list[ClassificationResult | None] = [_get_cached(k) for k in cache_keys]

        uncached_idx = [i for i, r in enumerate(results) if r is None]
        if uncached_idx:
            classifier = get_classifier()
            uncached_inputs = [
                EmailInput(
                    subject=emails_meta[i].subject,
                    from_addr=emails_meta[i].from_addr,
                    to_addr=emails_meta[i].to_addr,
                    date=emails_meta[i].date,
                    snippet="",
                )
                for i in uncached_idx
            ]
            uncached_meta = [emails_meta[i] for i in uncached_idx]
            new_results = classifier.classify_batch(uncached_inputs)
            new_results = _enrich_with_body(client, classifier, uncached_meta, new_results)
            for i, result in zip(uncached_idx, new_results):
                results[i] = result
                _set_cached(cache_keys[i], result)

        final_results: list[ClassificationResult] = results  # type: ignore[assignment]

        # Sort by priority: URGENT → IMPORTANT → NORMAL → LOW → NEWSLETTER → SPAM
        priority = {"URGENT": 0, "IMPORTANT": 1, "NORMAL": 2, "LOW": 3, "NEWSLETTER": 4, "SPAM": 5}
        classified_pairs = list(zip(emails_meta, final_results))
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
        limit = min(int(args.get("limit", 30)), 100)
        search_filter = SearchFilter(
            unread_only=bool(args.get("unread_only", False)),
            mailbox="INBOX",
            limit=limit,
        )

        emails_meta = client.fetch_metadata(search_filter)
        if not emails_meta:
            return {"summary": "Your inbox appears to be empty or no emails match the criteria."}

        cache_keys = [_cache_key(m) for m in emails_meta]
        results: list[ClassificationResult | None] = [_get_cached(k) for k in cache_keys]

        uncached_idx = [i for i, r in enumerate(results) if r is None]
        if uncached_idx:
            classifier = get_classifier()
            uncached_inputs = [
                EmailInput(
                    subject=emails_meta[i].subject,
                    from_addr=emails_meta[i].from_addr,
                    to_addr=emails_meta[i].to_addr,
                    date=emails_meta[i].date,
                    snippet="",
                )
                for i in uncached_idx
            ]
            uncached_meta = [emails_meta[i] for i in uncached_idx]
            new_results = classifier.classify_batch(uncached_inputs)
            new_results = _enrich_with_body(client, classifier, uncached_meta, new_results)
            for i, result in zip(uncached_idx, new_results):
                results[i] = result
                _set_cached(cache_keys[i], result)
        else:
            classifier = get_classifier()

        final_results: list[ClassificationResult] = results  # type: ignore[assignment]
        summary_inputs = [
            EmailInput(
                subject=m.subject, from_addr=m.from_addr,
                to_addr=m.to_addr, date=m.date, snippet="",
            )
            for m in emails_meta
        ]
        pairs = list(zip(summary_inputs, final_results))
        summary_text = classifier.generate_inbox_summary(pairs)

        label_counts: dict[str, int] = {}
        for r in final_results:
            label_counts[r.label] = label_counts.get(r.label, 0) + 1

        return {
            "summary": summary_text,
            "breakdown": label_counts,
            "total_analyzed": len(emails_meta),
        }

    # ------------------------------------------------------------------
    elif name == "analyze_email_for_response":
        uid = str(args["uid"])
        mailbox = args.get("mailbox", "INBOX")
        full = client.fetch_full_email(uid, mailbox)
        if not full:
            return {"error": f"Email UID {uid} not found in {mailbox}"}

        classifier = get_classifier()
        options: ResponseOptions = classifier.analyze_for_response(
            subject=full.subject,
            from_addr=full.from_addr,
            body=full.text_body or full.snippet,
        )
        return {
            "uid": uid,
            "email_summary": options.email_summary,
            "detected_tone": options.detected_tone,
            "suggested_responses": options.suggested_responses,
            "needs_clarification": options.needs_clarification,
            "clarification_prompt": options.clarification_prompt,
        }

    # ------------------------------------------------------------------
    elif name == "draft_email_response":
        uid = str(args["uid"])
        mailbox = args.get("mailbox", "INBOX")
        core_answer = str(args["core_answer"])
        clarification = str(args.get("clarification", ""))

        full = client.fetch_full_email(uid, mailbox)
        if not full:
            return {"error": f"Email UID {uid} not found in {mailbox}"}

        classifier = get_classifier()
        draft: DraftResult = classifier.draft_response(
            subject=full.subject,
            from_addr=full.from_addr,
            to_addr=full.to_addr,
            body=full.text_body or full.snippet,
            core_answer=core_answer,
            clarification=clarification,
        )
        return {
            "uid": uid,
            "reply_to": full.from_addr,
            "subject": draft.subject,
            "body": draft.body,
            "tone_used": draft.tone_used,
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
