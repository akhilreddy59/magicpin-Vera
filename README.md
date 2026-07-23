# Vera Bot — magicpin AI Challenge Submission

**Author:** Akhil Kumar Reddy · akhilkumarreddy2006@gmail.com
**Stack:** FastAPI, Python 3.12, Groq (primary) + OpenRouter (fallback)

## What this is

A merchant-messaging assistant that decides *what* to say to a merchant (or
their customer), *when*, and *why* — built to outperform the brief's stated
weaknesses in production Vera: generic copy, wasted turns on auto-replies,
and lost merchants on intent-handoff.

## Architecture

```
/v1/context ──▶ in-memory store (category / merchant / customer / trigger)
                        │
/v1/tick ───────────────┼──▶ SELECTOR ──▶ COMPOSER ──▶ VALIDATOR ──▶ actions
                        │      │              │             │
                        │  scores every   LLM call,     checks output,
                        │  candidate,     trigger-kind-  retries once on
                        │  keeps the ONE  specific       failure, blocks
                        │  best per       prompt          hard failures
                        │  merchant       guidance
                        │
/v1/reply ──────────────┴──▶ CLASSIFY ──▶ deterministic (hostile / auto-reply /
                              intent-handoff) OR LLM-composed contextual reply
```

Every stage is independently testable and was tested independently before
being wired together — the selector's scoring, the validator's checks, and
the reply classifier's routing all have unit-level verification, not just
end-to-end smoke tests.

## Key design decisions

**Selector picks one signal, not all of them.** Every available trigger is
scored on urgency × hand-ranked kind priority × expiry proximity; only the
single highest-scoring, non-expired, not-already-sent trigger per merchant
survives. This is deliberate: sending three competing messages to one
merchant is worse than sending the single best one, and the scoring
explicitly favors triggers that reflect real business priority (a
compliance deadline or an engaged, planning-intent merchant beats a routine
digest) over urgency-alone.

**The composer states stakes, not just facts.** Early iterations reliably
produced specific, well-targeted messages that still scored weak on
engagement — the message stated a fact and asked for a reply, but never
said why it mattered *now*. The system prompt requires one grounded
"why this matters" sentence per message (a deadline, a cost of inaction, a
peer comparison) pulled from the actual context, never fabricated.

**A validator enforces what the prompt can't guarantee.** LLMs don't follow
every instruction every time. Rather than trust the prompt alone, output is
checked — empty body, multiple CTAs, banned vocabulary, no stated stakes,
near-duplicate of a prior message — and one corrective retry is attempted
before a hard failure is simply skipped rather than sent. Restraint (an
empty tick) is treated as a valid, sometimes-correct outcome.

**Reply-handling separates what's safety-critical from what's generative.**
Hostile messages, auto-reply loops, and intent-handoff are handled with
deterministic logic — never left to an LLM's mercy, because getting these
wrong (missing an opt-out, re-qualifying a merchant who already said yes)
is worse than a slightly duller message. Everything else gets a real,
LLM-composed contextual reply using the original trigger.

**Reliability was treated as a first-class requirement.** FastAPI's async
route handlers were found to be blocking the event loop during LLM calls,
silently causing dropped requests under concurrent load — fixed by running
composition in parallel background threads. A thread-safe concurrency
limiter caps simultaneous outbound calls to stay within free-tier provider
limits. Groq is primary for latency; OpenRouter is an automatic fallback,
not a single point of failure.

## Results

Across repeated local evaluation (`judge_simulator.py`, LLM-judged on
Specificity / Category Fit / Merchant Fit / Decision Quality / Engagement):

| Dimension | Typical score |
|---|---|
| Specificity | 8-9 / 10 |
| Category Fit | 8-9 / 10 |
| Merchant Fit | 8-9 / 10 |
| Decision Quality | 8-9 / 10 |
| Engagement | 7-8 / 10 |

Verified qualitatively across all 5 categories (dentists, salons, gyms,
pharmacies, restaurants) — including a supply-recall alert for a pharmacy
that correctly balanced urgency with a calm, precise tone, and a
review-response message for a restaurant that cited the actual complaint
pattern rather than a generic "we value your feedback."

## Known limitations & honest tradeoffs

- **State is in-memory**, reset on restart — the right call for this
  scope; a production deployment would use Redis or a database.
- **Run-to-run score variance is real but mostly noise, not instability**:
  the judge LLM isn't temperature-locked, and most test runs score only
  1-2 messages (the selector deliberately narrows output), so a single
  message's score swings the whole run's average more than it would with
  a larger sample.
- **The local seed dataset ages** — most of its example trigger dates
  are already in the past relative to real-world testing today. Rather
  than weaken the bot's (correct) expiry enforcement to make local tests
  look complete, a separate `coverage_test.py` harness clones real
  triggers with fresh dates to exercise categories the stale seed data
  can no longer reach.

## What I'd build next, given more time

- **Semantic-similarity suppression**, not just exact suppression-key
  matching — catch two differently-worded messages about the same
  underlying situation.
- **A real eval harness with a held-out gold set** and regression
  tracking per prompt change, instead of manual before/after comparison.
- **Structured outputs via tool-calling** instead of prompted JSON, to
  eliminate the small residual parse-failure rate entirely.
- **A lightweight feedback loop**: track which composed messages actually
  got a merchant reply vs. silence, and feed that back into trigger-kind
  prioritization over time.
- **Observability**: structured logs and per-provider latency/cost
  metrics, so a slow or expensive provider is visible before it becomes
  an incident.
- **A human-in-the-loop review queue** specifically for
  `merchant_on_behalf` messages (customer-facing, sent as the merchant) —
  the highest-stakes message type, worth an extra safety layer before
  scale.

## Running it

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in GROQ_API_KEY (and OPENROUTER_API_KEY as fallback)
uvicorn bot:app --host 0.0.0.0 --port 8080
```
