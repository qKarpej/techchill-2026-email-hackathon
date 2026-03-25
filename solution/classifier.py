"""
AI-powered email importance classifier using the Anthropic API.
Classifies emails into 6 tiers and generates inbox summaries.
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Literal

import anthropic

ImportanceLabel = Literal["URGENT", "IMPORTANT", "NORMAL", "LOW", "NEWSLETTER", "SPAM"]

LABEL_DESCRIPTIONS: dict[str, str] = {
    "URGENT":     "Requires immediate attention — deadlines, emergencies, time-sensitive requests from real people",
    "IMPORTANT":  "Should be read today — meaningful messages that need a response but aren't on fire",
    "NORMAL":     "Standard correspondence — no urgency, read within a day or two",
    "LOW":        "FYI / informational — no action required, read at leisure",
    "NEWSLETTER": "Marketing, newsletters, digests, promotional content, subscription emails",
    "SPAM":       "Unsolicited, suspicious, phishing, or irrelevant mass mail",
}

SYSTEM_PROMPT = """You are an email importance classifier. Analyze email metadata and content and assign one of 6 importance labels.

Labels:
- URGENT: Requires immediate attention — deadlines, emergencies, time-sensitive requests from real people
- IMPORTANT: Should be read today — meaningful messages that need a response
- NORMAL: Standard correspondence — no urgency, read within a couple of days
- LOW: FYI / informational — no action required
- NEWSLETTER: Marketing, newsletters, digests, promotions, subscriptions
- SPAM: Unsolicited, suspicious, phishing attempts, irrelevant mass mail

Classification rules:
- Prioritize sender signal (known person vs automated/bulk), subject urgency language, and content
- Emails with action words (deadline, urgent, ASAP, please respond, invoice due) from real humans → URGENT or IMPORTANT
- Automated notifications, receipts, shipping updates → NORMAL or LOW
- Any subscription/marketing/promotional content → NEWSLETTER
- Suspicious links, grammar errors, pressure tactics, too-good-to-be-true offers → SPAM
- When uncertain between URGENT/IMPORTANT, prefer IMPORTANT
- Never classify a clearly personal email from a real person as SPAM

Respond with valid JSON only — no markdown, no explanation outside the JSON."""


@dataclass
class ResponseOptions:
    suggested_responses: list[str]  # 4–6 contextual response options
    needs_clarification: bool        # Whether a reason/extra detail would help
    clarification_prompt: str        # Question to ask the user if clarification is needed
    detected_tone: str               # "formal" | "professional" | "casual" | "friendly"
    email_summary: str               # One-sentence summary of what the email wants


@dataclass
class DraftResult:
    subject: str    # "Re: <original subject>"
    body: str       # Full email body text
    tone_used: str  # Tone applied to the draft


@dataclass
class ClassificationResult:
    label: ImportanceLabel
    confidence: int          # 0–100
    reasoning: str           # One sentence
    suggested_action: str    # One short action phrase
    key_points: list[str]    # 1–3 bullet points


@dataclass
class EmailInput:
    subject: str
    from_addr: str
    to_addr: str
    date: str
    snippet: str  # first ~400 chars of body


class EmailClassifier:
    def __init__(self, api_key: str):
        self.client = anthropic.Anthropic(api_key=api_key)

    def classify(self, email_input: EmailInput) -> ClassificationResult:
        """Classify a single email."""
        user_prompt = f"""Classify this email:

Subject: {email_input.subject}
From: {email_input.from_addr}
To: {email_input.to_addr}
Date: {email_input.date}
Content preview:
{email_input.snippet or "(no content available)"}

Respond with JSON in exactly this format:
{{
  "label": "URGENT|IMPORTANT|NORMAL|LOW|NEWSLETTER|SPAM",
  "confidence": <integer 0-100>,
  "reasoning": "<one sentence explaining why>",
  "suggested_action": "<one short action: e.g. Reply today / Read later / Archive / Unsubscribe / Delete>",
  "key_points": ["<point1>", "<point2>"]
}}"""

        response = self.client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=512,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )

        text = "".join(
            block.text for block in response.content if block.type == "text"
        ).strip()

        try:
            data = json.loads(text)
            valid_labels = {"URGENT", "IMPORTANT", "NORMAL", "LOW", "NEWSLETTER", "SPAM"}
            label = data.get("label", "NORMAL")
            if label not in valid_labels:
                label = "NORMAL"

            return ClassificationResult(
                label=label,
                confidence=max(0, min(100, int(data.get("confidence", 50)))),
                reasoning=str(data.get("reasoning", "")),
                suggested_action=str(data.get("suggested_action", "Read manually")),
                key_points=list(data.get("key_points", [])),
            )
        except (json.JSONDecodeError, KeyError, ValueError):
            return ClassificationResult(
                label="NORMAL",
                confidence=0,
                reasoning="Classification response could not be parsed",
                suggested_action="Read manually",
                key_points=[],
            )

    def classify_batch(
        self, emails: list[EmailInput]
    ) -> list[ClassificationResult]:
        """Classify a list of emails in parallel, returning results in the same order."""
        if not emails:
            return []

        def _safe(e: EmailInput) -> ClassificationResult:
            try:
                return self.classify(e)
            except Exception as ex:
                return ClassificationResult(
                    label="NORMAL",
                    confidence=0,
                    reasoning=f"Error: {ex}",
                    suggested_action="Read manually",
                    key_points=[],
                )

        # Cap workers to avoid hitting Anthropic rate limits
        max_workers = min(len(emails), 8)
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            return list(pool.map(_safe, emails))

    def analyze_for_response(
        self, subject: str, from_addr: str, body: str
    ) -> ResponseOptions:
        """Read a full email and return contextual response options plus tone/summary."""
        prompt = f"""You are helping a user decide how to reply to an email.
Read the email below and:
1. Write 4-6 short, contextually appropriate response options (e.g. "Yes, I can attend",
   "I need more time", "I've already sent it"). Tailor these to what the email is actually asking.
2. Decide if providing a reason or extra detail would meaningfully improve the reply
   (needs_clarification = true/false).
3. If needs_clarification is true, write a short, specific question to ask the user
   (e.g. "What is the reason you cannot attend?").
4. Detect the tone/formality of the original email: one of "formal", "professional",
   "casual", or "friendly".
5. Write a one-sentence summary of what the email is requesting or asking.

Email:
Subject: {subject}
From: {from_addr}
Body:
{body or "(no body)"}

Respond with valid JSON only - no markdown:
{{
  "suggested_responses": ["<option1>", "<option2>", ...],
  "needs_clarification": <true|false>,
  "clarification_prompt": "<question or empty string>",
  "detected_tone": "<formal|professional|casual|friendly>",
  "email_summary": "<one sentence>"
}}"""

        response = self.client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(
            block.text for block in response.content if block.type == "text"
        ).strip()

        try:
            data = json.loads(text)
            return ResponseOptions(
                suggested_responses=list(data.get("suggested_responses", [])),
                needs_clarification=bool(data.get("needs_clarification", False)),
                clarification_prompt=str(data.get("clarification_prompt", "")),
                detected_tone=str(data.get("detected_tone", "professional")),
                email_summary=str(data.get("email_summary", "")),
            )
        except (json.JSONDecodeError, KeyError, ValueError):
            return ResponseOptions(
                suggested_responses=["Yes", "No", "I need more time", "Let me get back to you"],
                needs_clarification=False,
                clarification_prompt="",
                detected_tone="professional",
                email_summary="Could not parse email summary",
            )

    def draft_response(
        self,
        subject: str,
        from_addr: str,
        to_addr: str,
        body: str,
        core_answer: str,
        clarification: str = "",
    ) -> DraftResult:
        """Draft a reply email given the original email and the user's chosen answer."""
        prompt = f"""You are drafting an email reply on behalf of the user.

Original email:
Subject: {subject}
From: {from_addr}
To: {to_addr}
Body:
{body or "(no body)"}

User's core answer: {core_answer}
Additional context from user: {clarification or "(none provided)"}

Instructions:
- Match the tone and formality of the original email exactly.
- Be concise - do not pad with unnecessary pleasantries or filler sentences.
- Do NOT invent facts, commitments, or details the user did not provide.
- Start the body directly (no "Subject:" line in the body field).
- The subject should be "Re: {subject}" unless a different subject is clearly appropriate.

Respond with valid JSON only - no markdown:
{{
  "subject": "<reply subject>",
  "body": "<full reply body text>",
  "tone_used": "<formal|professional|casual|friendly>"
}}"""

        response = self.client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(
            block.text for block in response.content if block.type == "text"
        ).strip()

        try:
            data = json.loads(text)
            return DraftResult(
                subject=str(data.get("subject", f"Re: {subject}")),
                body=str(data.get("body", "")),
                tone_used=str(data.get("tone_used", "professional")),
            )
        except (json.JSONDecodeError, KeyError, ValueError):
            return DraftResult(
                subject=f"Re: {subject}",
                body="Could not generate draft - please try again.",
                tone_used="unknown",
            )

    def generate_inbox_summary(
        self,
        classified: list[tuple[EmailInput, ClassificationResult]],
    ) -> str:
        """Generate a concise natural-language inbox summary."""
        counts: dict[str, int] = {}
        for _, result in classified:
            counts[result.label] = counts.get(result.label, 0) + 1

        urgent_lines = [
            f'- "{e.subject}" from {e.from_addr}: {r.reasoning}'
            for e, r in classified
            if r.label == "URGENT"
        ]

        prompt = f"""Write a concise 3–4 sentence inbox summary for the user based on this analysis:

Breakdown:
{chr(10).join(f"  {label}: {count}" for label, count in sorted(counts.items()))}
Total analyzed: {len(classified)} emails

Urgent emails:
{chr(10).join(urgent_lines) if urgent_lines else "  None"}

Be friendly, specific about urgent items, and keep it under 100 words."""

        response = self.client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}],
        )

        return "".join(
            block.text for block in response.content if block.type == "text"
        ).strip()
