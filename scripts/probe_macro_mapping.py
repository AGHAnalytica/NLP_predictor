"""
Probe: can the configured LLM (Config["LLM_MODEL"], currently gpt-5-nano) map an
aggregate market CLIMATE to each instrument's effect (tailwind / headwind / neutral)?

This de-risks step 3b before we build it. It tests the real production call shape — one
climate, all instruments, structured output — against canonical ground truth, in two
prompt modes:
  - MINIMAL : no guidance, pure reasoning.
  - GUIDED  : with the risk-asset/safe-haven framework + "default neutral unless clear".

It reports overall accuracy, and separately the accuracy on the HARD subset (safe havens
/ FX / rates), which is where small models and the regime-dependent sign tend to break.
If GUIDED clears the bar, nano is good enough with scaffolding; if not, bump the model
for this one (shared, once-per-tick) call.

Run:  .venv/Scripts/python.exe scripts/probe_macro_mapping.py
"""

import os

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

from nlp_predictor.config import Config

load_dotenv(override=True)

# Instruments flagged as the "hard" regime-dependent subset (havens / FX / rates).
HARD = {"GLD", "US10Y", "DXY", "EURUSD"}


class MacroEffect(BaseModel):
    ticker: str = Field(description="Exact ticker from the provided list.")
    effect: str = Field(description="One of: 'tailwind', 'headwind', 'neutral'.")
    strength: float = Field(description="Conviction of the effect, 0.0-1.0.")
    rationale: str = Field(description="One short sentence.")


class MacroMapping(BaseModel):
    effects: list[MacroEffect] = Field(description="One entry per instrument.")


# Each scenario: a climate + (ticker, full_name, acceptable_effects). Acceptable sets
# allow 'neutral' on the genuinely indirect cases, but demand the correct sign on the
# unambiguous ones (especially the havens — that is the real test).
SCENARIOS = [
    {
        "name": "Risk-on (de-escalation + global growth)",
        "climate": (
            "Broad geopolitical de-escalation and strong global growth. Equity markets "
            "are rallying, volatility is falling, and investors are rotating into risk "
            "assets. Clear risk-on environment."
        ),
        "instruments": [
            ("SPY", "SPDR S&P 500 ETF", {"tailwind"}),
            ("AAPL", "Apple Inc.", {"tailwind"}),
            ("GOOGL", "Alphabet Inc.", {"tailwind"}),
            ("BTC", "Bitcoin", {"tailwind"}),
            ("GLD", "Gold (spot)", {"headwind"}),
            ("US10Y", "US 10-Year Treasury (bond price)", {"headwind"}),
        ],
    },
    {
        "name": "Risk-off (banking crisis + recession fear)",
        "climate": (
            "A banking crisis is spreading in Europe and US recession fears rise. "
            "Investors are fleeing to safety and selling equities broadly. Clear "
            "risk-off, flight-to-safety environment."
        ),
        "instruments": [
            ("SPY", "SPDR S&P 500 ETF", {"headwind"}),
            ("AAPL", "Apple Inc.", {"headwind"}),
            ("GOOGL", "Alphabet Inc.", {"headwind"}),
            ("BTC", "Bitcoin", {"headwind"}),
            ("GLD", "Gold (spot)", {"tailwind"}),
            ("US10Y", "US 10-Year Treasury (bond price)", {"tailwind"}),
            ("DXY", "US Dollar Index", {"tailwind"}),
        ],
    },
    {
        "name": "Oil supply shock",
        "climate": (
            "OPEC+ production cuts, Strait of Hormuz tensions and tightened sanctions "
            "are sharply reducing global crude supply and raising a geopolitical risk "
            "premium."
        ),
        "instruments": [
            ("OIL", "Crude Oil", {"tailwind"}),
            ("SPY", "SPDR S&P 500 ETF", {"headwind", "neutral"}),
            ("AAPL", "Apple Inc.", {"headwind", "neutral"}),
        ],
    },
    {
        "name": "Hawkish Fed (rising yields, strong USD)",
        "climate": (
            "Fed officials signal further rate hikes amid sticky inflation. Yields "
            "are rising and the US dollar is strengthening broadly."
        ),
        "instruments": [
            ("EURUSD", "Euro / US Dollar", {"headwind"}),
            ("DXY", "US Dollar Index", {"tailwind"}),
            ("GLD", "Gold (spot)", {"headwind"}),
            ("SPY", "SPDR S&P 500 ETF", {"headwind", "neutral"}),
        ],
    },
]

_BASE = """You are a macro strategist. Below is the current market CLIMATE, then a list
of instruments. For EACH instrument, decide how this climate affects its price over the
next ~12-16 hours:
- "tailwind" = the climate pushes its price UP
- "headwind" = the climate pushes its price DOWN
- "neutral"  = no clear, direct effect
Return an effect, a strength (0.0-1.0) and a one-line rationale for every instrument.
{guidance}
CLIMATE:
{climate}

INSTRUMENTS:
{instruments}
"""

_GUIDANCE = (
    "Guidance: risk assets (equities, oil, crypto) tend to RISE in optimistic/risk-on "
    "climates and FALL in fearful/risk-off ones. Safe havens (gold, government bonds, "
    "and often the US dollar) tend to move the OPPOSITE way, rising when fearful. "
    "Weigh what the climate is specifically about, not just whether it is "
    "positive or negative. Only choose tailwind/headwind when the effect is clear and "
    "direct; otherwise neutral.\n"
)

_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def _call(prompt: str) -> MacroMapping:
    resp = _client.beta.chat.completions.parse(
        model=str(Config["LLM_MODEL"]),
        messages=[{"role": "user", "content": prompt}],
        response_format=MacroMapping,
    )
    return resp.choices[0].message.parsed  # type: ignore


def _run_mode(guided: bool) -> None:
    label = "GUIDED" if guided else "MINIMAL"
    print(f"\n{'=' * 70}\n  MODE: {label}  (model: {Config['LLM_MODEL']})\n{'=' * 70}")

    total = hits = hard_total = hard_hits = 0
    misses: list[str] = []

    for scn in SCENARIOS:
        instruments_block = "\n".join(
            f"- {t} — {name}" for t, name, _ in scn["instruments"]
        )
        prompt = _BASE.format(
            guidance=_GUIDANCE if guided else "",
            climate=scn["climate"],
            instruments=instruments_block,
        )
        try:
            parsed = _call(prompt)
        except Exception as e:  # noqa: BLE001
            print(f"[ERROR] {scn['name']}: {e}")
            continue

        pred = {
            e.ticker.upper(): (e.effect.strip().lower(), e.strength, e.rationale)
            for e in parsed.effects
        }

        print(f"\n-- {scn['name']} --")
        for ticker, _name, acceptable in scn["instruments"]:
            got = pred.get(ticker.upper())
            effect = got[0] if got else "<missing>"
            ok = effect in acceptable
            total += 1
            hits += int(ok)
            if ticker in HARD:
                hard_total += 1
                hard_hits += int(ok)
            mark = "OK " if ok else "XX "
            exp = "/".join(sorted(acceptable))
            print(f"  {mark}{ticker:7s} got={effect:9s} exp={exp}")
            if not ok:
                why = got[2] if got else "(no answer returned)"
                misses.append(
                    f"    [{label}] {scn['name']} / {ticker}: "
                    f"got '{effect}', expected {exp} — {why}"
                )

    acc = hits / total * 100 if total else 0
    hard_acc = hard_hits / hard_total * 100 if hard_total else 0
    print(
        f"\n  {label} overall: {hits}/{total} = {acc:.0f}%   "
        f"| HARD (havens/FX/rates): {hard_hits}/{hard_total} = {hard_acc:.0f}%"
    )
    if misses:
        print("  Misses:")
        for m in misses:
            print(m)


if __name__ == "__main__":
    print("====== Macro -> instrument mapping probe ======")
    _run_mode(guided=False)
    _run_mode(guided=True)
    print(
        "\nRule of thumb: if GUIDED overall >= ~85% and HARD >= ~70%, nano is adequate "
        "for this (shared, once-per-tick) call. Otherwise upgrade the model for it."
    )
