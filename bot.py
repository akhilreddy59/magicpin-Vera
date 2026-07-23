"""
magicpin AI Challenge — bot.py

Env vars needed (see .env.example):
    LLM_PROVIDER — "groq" | "openrouter" (default: groq). The OTHER one is
    used automatically as a fallback if the primary provider's API call fails.
    GROQ_API_KEY, OPENROUTER_API_KEY
"""

import os
import re
import time
import json
import asyncio
import difflib
import threading
from datetime import datetime, timezone
from typing import Any, Optional

import requests
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()
START_TIME = time.time()

# ---------------------------------------------------------------------------
# LLM provider config. Two providers only: Groq (primary, fast) and
# OpenRouter (automatic fallback on error — already proven working since
# it runs the judge in this setup).
# ---------------------------------------------------------------------------
PROVIDERS = {
    "groq": {
        "url": "https://api.groq.com/openai/v1/chat/completions",
        "key_env": "GROQ_API_KEY",
        "model": os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
    },
    "openrouter": {
        "url": "https://openrouter.ai/api/v1/chat/completions",
        "key_env": "OPENROUTER_API_KEY",
        "model": os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct"),
    },
}

_configured_provider = os.getenv("LLM_PROVIDER", "groq").lower()
if _configured_provider not in PROVIDERS:
    # A typo here (e.g. "gorq") would otherwise KeyError deep inside
    # PROVIDERS[LLM_PROVIDER] and crash /v1/metadata and every compose()
    # call. Fail safe instead: warn loudly, default to groq, keep serving.
    print(f"[CONFIG WARNING] LLM_PROVIDER={_configured_provider!r} is not a recognized provider "
          f"(expected 'groq' or 'openrouter') — defaulting to 'groq'. Check your .env for a typo.")
    _configured_provider = "groq"
LLM_PROVIDER = _configured_provider
FALLBACK_PROVIDER = "openrouter" if LLM_PROVIDER == "groq" else "groq"
PROVIDER_ORDER = [LLM_PROVIDER, FALLBACK_PROVIDER]

# Multiple merchants can now compose in parallel (asyncio.gather + to_thread
# in /v1/tick). Uncapped, a busy tick could fire many simultaneous outbound
# calls at once and trip Groq/OpenRouter's free-tier concurrency limits.
# This caps how many LLM HTTP calls run at the same time across all threads.
LLM_CONCURRENCY_LIMIT = int(os.getenv("LLM_CONCURRENCY_LIMIT", "5"))
_llm_semaphore = threading.Semaphore(LLM_CONCURRENCY_LIMIT)

# ---------------------------------------------------------------------------
# In-memory state.
# ---------------------------------------------------------------------------
contexts: dict[tuple[str, str], dict] = {}
conversations: dict[str, list] = {}
sent_suppression_keys: set[str] = set()
merchant_message_log: dict[str, list[str]] = {}
AUTO_REPLY_SIMILARITY_THRESHOLD = 0.85


@app.get("/v1/healthz")
@app.head("/v1/healthz")
async def healthz():
    counts = {"category": 0, "merchant": 0, "customer": 0, "trigger": 0}
    for (scope, _cid) in contexts.keys():
        counts[scope] = counts.get(scope, 0) + 1
    return {
        "status": "ok",
        "uptime_seconds": int(time.time() - START_TIME),
        "contexts_loaded": counts,
        "conversations_active": len(conversations),
    }


@app.get("/v1/metadata")
async def metadata():
    return {
        "team_name": "Akhil Kumar Reddy",
        "team_members": ["Akhil Kumar Reddy"],
        "model": f"{LLM_PROVIDER}:{PROVIDERS[LLM_PROVIDER]['model']} (fallback: {FALLBACK_PROVIDER})",
        "approach": (
            "Selector (urgency+kind+expiry scoring, one trigger/merchant, cross-tick suppression) "
            "+ composer (LLM-driven, trigger-kind-specific prompts, language-preference aware, "
            "temperature=0, automatic provider fallback) + output validator (stakes/CTA/taboo/"
            "URL/anti-repetition checks, one corrective retry) + reply handler (fuzzy auto-reply "
            "detection, deterministic hostile/auto-reply/intent-handoff, LLM-composed contextual "
            "reply otherwise using the original trigger context)."
        ),
        "contact_email": "akhilkumarreddy2006@gmail.com",
        "version": "0.3.0",
        "submitted_at": datetime.now(timezone.utc).isoformat(),
    }


class ContextBody(BaseModel):
    scope: str
    context_id: str
    version: int
    payload: dict[str, Any]
    delivered_at: str


@app.post("/v1/context")
async def push_context(body: ContextBody):
    key = (body.scope, body.context_id)
    current = contexts.get(key)
    # Re-pushing the SAME version is idempotent and accepted — safe to
    # re-run the judge against a long-lived server without restarting it.
    if current and current["version"] > body.version:
        return {"accepted": False, "reason": "stale_version", "current_version": current["version"]}
    contexts[key] = {"version": body.version, "payload": body.payload}
    return {"accepted": True, "ack_id": f"ack_{body.context_id}_v{body.version}", "stored_at": datetime.now(timezone.utc).isoformat()}


@app.post("/v1/teardown")
async def teardown():
    """Wipe all in-memory state. Useful for local testing — lets you reset
    between judge_simulator.py runs without restarting uvicorn (which is
    also fine, this is just a faster option for local iteration)."""
    contexts.clear()
    conversations.clear()
    sent_suppression_keys.clear()
    merchant_message_log.clear()
    return {"status": "ok", "message": "all state cleared"}


def get_payload(scope: str, context_id: str) -> Optional[dict]:
    entry = contexts.get((scope, context_id))
    return entry["payload"] if entry else None


# ---------------------------------------------------------------------------
# Composer.
# ---------------------------------------------------------------------------
BASE_SYSTEM_PROMPT = """You write short WhatsApp messages for Vera, an AI growth \
assistant for local merchants in India.

AUDIENCE — read this first, it changes everything below:
- If CUSTOMER CONTEXT is "none": you are writing TO THE MERCHANT. Use \
send_as="vera". Clinical/peer/professional tone per category.
- If CUSTOMER CONTEXT is present: you are writing TO THE MERCHANT'S \
CUSTOMER, on the merchant's behalf. Use send_as="merchant_on_behalf". \
The message must read like it's from the clinic/shop itself — naturally \
include the merchant/clinic name early (e.g. "Hi Priya, this is Dr. \
Meera's Dental Clinic —"), plain customer-friendly language, and the CTA \
should be a booking/confirmation action.

Universal rules:
- Anchor the message on ONE concrete, verifiable fact from the context \
given (a number, date, headline, or peer stat). Never invent data not \
present in the context.
- Match the category voice exactly when writing to a merchant.
- Exactly ONE call-to-action, specific and tied to the fact you cited — \
never "let me know" or "checking in." Name the concrete next step. Pick \
whichever CTA form converts best (binary, scheduling offer, or specific \
low-friction ask) — don't default to binary just because it's binary.
- STATE THE STAKES, not just the fact — one sentence of "why this matters \
now": a deadline, a cost of inaction, or a peer comparison, grounded in \
what's actually in the context. Never fabricate a stake.
- No preamble. Get to the point in the first sentence. Never include URLs.
- Never repeat a message you've already sent in this conversation.
- Return ONLY valid JSON, no markdown fences, no commentary. Shape:
{"body": "...", "cta": "open_ended|binary|none", \
"send_as": "vera OR merchant_on_behalf", \
"suppression_key": "...", "rationale": "one sentence: why this message, why now"}
"""

# Trigger-kind-specific guidance — sharpens Category Fit and Decision
# Quality by telling the model HOW to frame each kind of signal, not just
# handing it raw JSON and hoping. Falls back to the base prompt alone for
# any kind not listed here (e.g. reply-mode's generic stub trigger).
TRIGGER_KIND_PROMPTS = {
    "research_digest": (
        "RESEARCH DIGEST: a new clinical paper or publication just landed. Anchor on the "
        "specific finding (title, source, effect size). Cite the source explicitly. "
        "CTA: open-ended, curiosity-driven (e.g. 'Want me to pull the abstract?')."
    ),
    "recall_due": (
        "RECALL DUE: a patient is due for a follow-up. This is customer-facing "
        "(send_as='merchant_on_behalf'). Mention the specific service, how long since their "
        "last visit, and offer a specific slot if available. Warm tone, no jargon."
    ),
    "perf_dip": (
        "PERFORMANCE DIP: metrics dropped. Don't panic the merchant — if category context "
        "suggests this is seasonal/normal, reassure and reframe. If genuinely concerning, "
        "offer one concrete fix using the merchant's actual numbers."
    ),
    "perf_spike": (
        "PERFORMANCE SPIKE: good news, no urgent action needed. Keep it brief and positive. "
        "Only suggest a follow-up if there's a natural one (e.g. 'keep the momentum with a post?')."
    ),
    "active_planning_intent": (
        "ACTIVE PLANNING INTENT: the merchant already showed buying/action intent. Treat this "
        "as continuing an active discussion — move toward action, not re-qualification."
    ),
    "competitor_opened": (
        "COMPETITOR OPENED: state the fact neutrally (distance, type), no fear-mongering. "
        "Offer one specific defensive action (profile update, offer refresh, post)."
    ),
    "review_theme_emerged": (
        "REVIEW THEME: a pattern across recent reviews. If negative, offer to help respond or "
        "fix the issue. If positive, suggest amplifying it. Use actual review data if present."
    ),
    "milestone_reached": (
        "MILESTONE: the merchant crossed a threshold. Celebrate briefly and genuinely. "
        "Suggest one follow-up only if natural (post about it, thank customers)."
    ),
    "dormant_with_vera": (
        "RE-ENGAGEMENT: merchant hasn't interacted in a while. Lead with something useful or "
        "interesting — don't guilt-trip about the gap. Low-friction CTA."
    ),
    "supply_alert": (
        "SUPPLY ALERT: urgent compliance/safety issue. Lead with specifics (batch, product, "
        "risk level). Precise, trustworthy, calm-but-urgent tone."
    ),
    "renewal_due": (
        "RENEWAL DUE: subscription expiring. State the expiry date and days remaining clearly. "
        "Reinforce value delivered so far. Frictionless renewal CTA."
    ),
    "regulation_change": (
        "REGULATION CHANGE: cite the specific regulation and effective date. Explain what the "
        "merchant needs to do in plain language. Offer to help draft required updates."
    ),
}


def call_llm(system_prompt: str, user_prompt: str) -> dict:
    last_error = None
    for provider in PROVIDER_ORDER:
        cfg = PROVIDERS[provider]
        api_key = os.getenv(cfg["key_env"], "")
        if not api_key:
            last_error = f"{cfg['key_env']} not set"
            continue
        try:
            # Cap concurrent outbound calls across all threads — protects
            # against tripping free-tier rate/concurrency limits when
            # several merchants compose at once.
            with _llm_semaphore:
                resp = requests.post(
                    cfg["url"],
                    headers={"Authorization": f"Bearer {api_key}"},
                    json={
                        "model": cfg["model"],
                        "temperature": 0,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                    },
                    timeout=10,  # fail over to the other provider quickly rather than eating the whole budget on one slow call
                )
            resp.raise_for_status()
            raw = resp.json()["choices"][0]["message"]["content"]
            if not raw or not raw.strip():
                raise ValueError("provider returned an empty completion (common under free-tier load)")
            return _extract_json(raw)
        except Exception as e:
            print(f"[COMPOSER ERROR] provider={provider} -> {e}")
            last_error = e
    return {
        "body": "Vera here — checking in on your listing.",
        "cta": "open_ended",
        "send_as": "vera",
        "suppression_key": "fallback",
        "rationale": f"all LLM providers failed: {last_error}",
    }


def _extract_json(raw: str) -> dict:
    """Tolerate markdown fences or stray commentary around the JSON —
    some models don't follow 'ONLY valid JSON' perfectly even when told to."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise


STAKES_KEYWORDS = [
    "before", "deadline", "expir", "risk", "non-complian", "miss out",
    "last chance", "limited", "urgent", "today", "this week",
]
HARD_FAILURES = {"empty_body", "multiple_ctas", "taboo_vocab"}  # never send, even after retry


def validate_output(result: dict, category: Optional[dict], conversation_bodies: list[str]) -> tuple[bool, str]:
    """Returns (ok, reason). Hard failures never get sent even after a
    retry; the caller decides whether a soft failure (no_stakes) still
    ships after one corrective attempt."""
    body = (result.get("body") or "").strip()
    if not body:
        return False, "empty_body"

    if re.search(r"https?://\S+", body):
        body = re.sub(r"https?://\S+", "", body).strip()
        result["body"] = body
        if not body:
            return False, "empty_body"

    if body.count("?") > 1:
        return False, "multiple_ctas"

    taboo = []
    if category:
        taboo = category.get("voice", {}).get("taboo_words", []) or category.get("voice", {}).get("vocab_forbidden", [])
    if taboo and any(word.lower() in body.lower() for word in taboo):
        return False, "taboo_vocab"

    for prev in conversation_bodies:
        if prev and difflib.SequenceMatcher(None, body, prev).ratio() > AUTO_REPLY_SIMILARITY_THRESHOLD:
            return False, "repeated_message"

    has_digit = any(ch.isdigit() for ch in body)
    has_stakes_word = any(kw in body.lower() for kw in STAKES_KEYWORDS)
    if not has_digit and not has_stakes_word:
        return False, "no_stakes"

    if result.get("cta") not in ("open_ended", "binary", "none"):
        result["cta"] = "open_ended"
    if result.get("send_as") not in ("vera", "merchant_on_behalf"):
        result["send_as"] = "vera"

    return True, "ok"


def get_conversation_bodies(conversation_id: str) -> list[str]:
    turns = conversations.get(conversation_id, [])
    return [t.get("body", "") for t in turns if t.get("from") == "bot" and t.get("body")]


def compose(
    category: dict,
    merchant: dict,
    trigger: dict,
    customer: Optional[dict],
    conversation_id: Optional[str] = None,
    reply_mode: bool = False,
    merchant_reply: Optional[str] = None,
) -> dict:
    kind_suffix = TRIGGER_KIND_PROMPTS.get(trigger.get("kind", ""), "")

    lang_pref = None
    if customer:
        lang_pref = customer.get("identity", {}).get("language_pref")
    elif merchant:
        langs = merchant.get("identity", {}).get("languages", [])
        lang_pref = "hi-en mix (Hindi-English code-mix preferred)" if "hi" in langs else "en (English)"

    system = BASE_SYSTEM_PROMPT
    if kind_suffix:
        system += f"\n\nTRIGGER-SPECIFIC GUIDANCE:\n{kind_suffix}"
    if lang_pref:
        system += f"\n\nLANGUAGE: recipient's preference is '{lang_pref}'. Honor this strictly."
    if reply_mode:
        system += (
            "\n\nREPLY MODE: you're responding within an ongoing conversation. If the merchant "
            "shows intent to proceed, move to action — do not re-qualify. If they ask a "
            "question, answer it. Always advance the conversation."
        )

    parts = [
        f"CATEGORY CONTEXT:\n{json.dumps(category, ensure_ascii=False)}",
        f"MERCHANT CONTEXT:\n{json.dumps(merchant, ensure_ascii=False)}",
        f"TRIGGER CONTEXT:\n{json.dumps(trigger, ensure_ascii=False)}",
        f"CUSTOMER CONTEXT:\n{json.dumps(customer, ensure_ascii=False) if customer else 'none'}",
    ]
    if reply_mode and merchant_reply:
        parts.append(f"MERCHANT'S LATEST MESSAGE:\n{merchant_reply}")
    parts.append("Compose the next message now, per the rules above.")
    user_prompt = "\n\n".join(parts)

    result = call_llm(system, user_prompt)
    conv_bodies = get_conversation_bodies(conversation_id) if conversation_id else []
    ok, reason = validate_output(result, category, conv_bodies)

    if not ok:
        print(f"[VALIDATOR] first attempt failed ({reason}) — retrying once with correction")
        corrected_prompt = user_prompt + f"\n\nYOUR PREVIOUS ATTEMPT FAILED VALIDATION: {reason}. Fix specifically that issue and resend the full JSON."
        retry_result = call_llm(system, corrected_prompt)
        retry_ok, retry_reason = validate_output(retry_result, category, conv_bodies)
        if retry_ok or reason not in HARD_FAILURES:
            result = retry_result
        else:
            print(f"[VALIDATOR] retry still hard-failed ({retry_reason}) — skipping this message")
            return {"body": "", "cta": "none", "send_as": "vera", "suppression_key": "validation_failed", "rationale": f"skipped after retry: {retry_reason}"}

    return result


# ---------------------------------------------------------------------------
# Selector — Phase 2.
# ---------------------------------------------------------------------------
class TickBody(BaseModel):
    now: str
    available_triggers: list[str] = []


TRIGGER_KIND_BOOST = {
    "active_planning_intent": 25, "supply_alert": 20, "customer_lapsed_hard": 10,
    "competitor_opened": 8, "perf_dip": 8, "review_theme_emerged": 8,
    "gbp_unverified": 6, "renewal_due": 6, "chronic_refill_due": 6,
    "recall_due": 5, "winback_eligible": 4, "trial_followup": 4,
    "regulation_change": 4, "wedding_package_followup": 3, "ipl_match_today": 3,
    "dormant_with_vera": 3, "category_seasonal": 2, "seasonal_perf_dip": 1,
    "milestone_reached": 1, "cde_opportunity": 1, "curious_ask_due": 0,
    "festival_upcoming": 0, "research_digest": 0, "perf_spike": -5,
}


def score_trigger(trigger: dict, now: datetime) -> Optional[float]:
    expires_at = trigger.get("expires_at")
    hours_left = None
    if expires_at:
        try:
            exp = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
            hours_left = (exp - now).total_seconds() / 3600
            if hours_left <= 0:
                return None
        except ValueError:
            pass
    score = trigger.get("urgency", 0) * 10 + TRIGGER_KIND_BOOST.get(trigger.get("kind", ""), 0)
    if hours_left is not None:
        if hours_left <= 24:
            score += 15
        elif hours_left <= 72:
            score += 8
        elif hours_left <= 24 * 7:
            score += 3
    return score


def select_best_trigger_per_merchant(available_trigger_ids: list[str], now: datetime) -> dict[str, str]:
    candidates_by_merchant: dict[str, list[tuple[float, str]]] = {}
    for trg_id in available_trigger_ids:
        trigger = get_payload("trigger", trg_id)
        if not trigger:
            print(f"[SELECTOR] trigger {trg_id!r} not in contexts — was it pushed via /v1/context?")
            continue
        if trigger.get("suppression_key") in sent_suppression_keys:
            continue
        score = score_trigger(trigger, now)
        if score is None:
            continue
        merchant_id = trigger.get("merchant_id") or trigger.get("payload", {}).get("merchant_id")
        if not merchant_id:
            continue
        candidates_by_merchant.setdefault(merchant_id, []).append((score, trg_id))

    selected = {}
    for mid, candidates in candidates_by_merchant.items():
        candidates.sort(key=lambda pair: -pair[0])
        selected[mid] = candidates[0][1]
    return selected


@app.post("/v1/tick")
async def tick(body: TickBody):
    try:
        tick_now = datetime.fromisoformat(body.now.replace("Z", "+00:00"))
    except ValueError:
        tick_now = datetime.now(timezone.utc)
    selected = select_best_trigger_per_merchant(body.available_triggers, tick_now)

    # Gather everything needed per merchant first — this part is pure
    # in-memory lookup, no I/O, safe to do synchronously.
    jobs = []  # (merchant_id, trg_id, trigger, merchant, category, customer, customer_id, conversation_id)
    for merchant_id, trg_id in selected.items():
        trigger = get_payload("trigger", trg_id)
        merchant = get_payload("merchant", merchant_id)
        if not merchant:
            continue
        category_slug = merchant.get("identity", {}).get("category_slug") or merchant.get("category_slug")
        category = get_payload("category", category_slug) if category_slug else None
        if not category:
            continue
        customer_id = trigger.get("customer_id") or trigger.get("payload", {}).get("customer_id")
        customer = get_payload("customer", customer_id) if customer_id else None
        conversation_id = f"conv_{merchant_id}_{trg_id}"
        jobs.append((merchant_id, trg_id, trigger, merchant, category, customer, customer_id, conversation_id))

    # compose() makes blocking network calls (requests library). Running it
    # directly here would freeze the whole event loop — including any other
    # request arriving concurrently (a context push, a healthz check) — for
    # the full duration of every LLM call, sequentially, per merchant. That
    # was the actual cause of the "[FAIL] context push" + "tick timed out"
    # pattern seen when multiple merchants needed composing in one tick.
    # asyncio.to_thread runs each compose() call in a background thread, and
    # gather runs all of them in PARALLEL, not one after another.
    results = await asyncio.gather(*[
        asyncio.to_thread(compose, category, merchant, trigger, customer, conversation_id)
        for (_mid, _tid, trigger, merchant, category, customer, _cid, conversation_id) in jobs
    ])

    actions = []
    for (merchant_id, trg_id, trigger, merchant, category, customer, customer_id, conversation_id), result in zip(jobs, results):
        if not result.get("body"):
            continue

        # Store trigger_id on the turn — this is what lets /v1/reply later
        # reconstruct the real trigger context instead of falling back to
        # a generic stub.
        conversations.setdefault(conversation_id, []).append(
            {"from": "bot", "body": result["body"], "trigger_id": trg_id}
        )
        suppression_key = result.get("suppression_key") or trigger.get("suppression_key", "")
        sent_suppression_keys.add(suppression_key)

        trigger_kind = trigger.get("kind", "generic")
        actions.append({
            "conversation_id": conversation_id,
            "merchant_id": merchant_id,
            "customer_id": customer_id,
            "send_as": result.get("send_as", "vera"),
            "trigger_id": trg_id,
            "template_name": f"vera_{trigger_kind}_v1",
            "template_params": [merchant.get("identity", {}).get("name", "")],
            "body": result["body"],
            "cta": result.get("cta", "open_ended"),
            "suppression_key": suppression_key,
            "rationale": result.get("rationale", ""),
        })
    return {"actions": actions}


# ---------------------------------------------------------------------------
# Reply classifier — Phase 4. Hostile / auto-reply / intent-handoff are
# deterministic and PROVEN (all 4 judge_simulator.py scenarios pass) —
# left untouched. Only the "normal" fallthrough now optionally composes a
# real contextual reply via the LLM, using the actual original trigger
# (properly threaded via trigger_id now, not the broken lookup from the
# candidate version) — falls back to the safe generic line if that fails.
# ---------------------------------------------------------------------------
class ReplyBody(BaseModel):
    conversation_id: str
    merchant_id: Optional[str] = None
    customer_id: Optional[str] = None
    from_role: str
    message: str
    received_at: str
    turn_number: int


HOSTILE_KEYWORDS = [
    "stop messaging", "stop texting", "unsubscribe", "spam", "harassment",
    "leave me alone", "not interested", "annoying", "block", "report",
    "who gave you this number", "how did you get my number",
    "do not contact", "don't text", "stop bothering",
]
INTENT_KEYWORDS = [
    "yes", "lets do it", "let's do it", "sure", "ok let", "okay let",
    "sounds good", "go ahead", "whats next", "what's next", "im in",
    "i'm in", "do it", "sign me up", "i want to", "count me in",
    "tell me more", "i'm interested",
]


def classify_reply(conversation_id: str, merchant_id: Optional[str], message: str) -> str:
    lower = message.lower().strip()
    if any(kw in lower for kw in HOSTILE_KEYWORDS):
        return "hostile"

    tracking_key = merchant_id or conversation_id
    log = merchant_message_log.setdefault(tracking_key, [])
    log.append(lower)
    # Fuzzy match — catches near-duplicate auto-replies, not just exact text.
    repeat_count = sum(1 for m in log if difflib.SequenceMatcher(None, lower, m).ratio() >= AUTO_REPLY_SIMILARITY_THRESHOLD)
    if repeat_count >= 3:
        return "auto_reply_confirmed"
    if repeat_count == 2:
        return "auto_reply_suspected"

    if any(kw in lower for kw in INTENT_KEYWORDS):
        return "intent_accept"
    return "normal"


@app.post("/v1/reply")
async def reply(body: ReplyBody):
    conversations.setdefault(body.conversation_id, []).append({"from": body.from_role, "msg": body.message})
    label = classify_reply(body.conversation_id, body.merchant_id, body.message)

    # --- Proven deterministic paths — unchanged from the tested version ---
    if label == "hostile":
        return {"action": "end", "body": "Understood — I won't message you about this again. Take care!", "cta": "none",
                "rationale": "Detected an opt-out / hostile signal; ending immediately rather than risking further fatigue."}
    if label == "auto_reply_confirmed":
        return {"action": "end", "body": "", "cta": "none",
                "rationale": "Same message received 3+ times — treating as an unattended auto-reply and ending to avoid wasting turns."}
    if label == "auto_reply_suspected":
        return {"action": "send", "body": "Just making sure this reaches the right person — are you able to review this, or should I check back another time?", "cta": "binary",
                "rationale": "Second identical reply suggests a possible auto-reply; one differentiated probe before giving up."}
    if label == "intent_accept":
        return {"action": "send", "body": "Perfect — I'll set this up now and confirm here once it's live. No further steps needed on your end.", "cta": "none",
                "rationale": "Merchant expressed clear acceptance; skipping re-qualification, moving straight to action."}

    # --- "normal" only: try a real contextual LLM reply, using the actual
    # original trigger (now correctly found via trigger_id) — fall back to
    # the safe generic line if anything goes wrong. ---
    try:
        conv_turns = conversations.get(body.conversation_id, [])
        trigger_id = next((t.get("trigger_id") for t in reversed(conv_turns) if t.get("from") == "bot" and t.get("trigger_id")), None)
        trigger = get_payload("trigger", trigger_id) if trigger_id else None
        merchant = get_payload("merchant", body.merchant_id) if body.merchant_id else None
        customer = get_payload("customer", body.customer_id) if body.customer_id else None
        category = None
        if merchant:
            cat_slug = merchant.get("identity", {}).get("category_slug") or merchant.get("category_slug")
            category = get_payload("category", cat_slug) if cat_slug else None

        if trigger and merchant and category:
            result = await asyncio.to_thread(
                compose, category, merchant, trigger, customer,
                conversation_id=body.conversation_id, reply_mode=True, merchant_reply=body.message,
            )
            if result.get("body"):
                conversations[body.conversation_id].append({"from": "bot", "body": result["body"]})
                return {"action": "send", "body": result["body"], "cta": result.get("cta", "open_ended"), "rationale": result.get("rationale", "")}
    except Exception as e:
        print(f"[REPLY ERROR] falling back to generic — {e}")

    return {"action": "send", "body": "Got it — noted. I'll follow up with the next step shortly.", "cta": "open_ended",
            "rationale": "No strong signal, or insufficient context for a tailored reply — acknowledging and continuing normally."}