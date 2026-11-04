# AGENTS.md — Architectural & Engineering Guidelines for AI Agents

This file defines mandatory rules for AI coding assistants (Antigravity, Cursor, Claude Code, GitHub Copilot) operating in the `NLP_predictor` repository.

---

## 1. Project Architecture Overview

`NLP_predictor` is a dual-path financial NLP and quantitative signal generation pipeline:

- **`nlp_predictor.config`**: Pydantic-validated YAML configuration (`configs/config.yaml`, `configs/hyperparams/`, `configs/paths/`, `configs/severity/`, `configs/backtesting/`). Access all settings exclusively via typed attributes on `from nlp_predictor.config import config` (e.g., `config.hyperparams.upper_path.dcim.half_life_hours`). **Never** add dictionary-style `config["..."]` accessors or metaclass shims.
- **`nlp_predictor.core`**: Shared base classes and pure mathematical models:
  - `base_analyzer.py`: `BaseAnalyzer` (OpenAI Structured Outputs with Tenacity retries, thread-pooled `_parallel_map`, PyArrow Parquet persistence) and `load_instrument_universe()`.
  - `severity_models.py`: Pure, stateless functions for instrument severity (`SEVERITY_REGISTRY`: `baseline`, `dcim`, `abnormal_volume`) and macro cluster intensity (`MACRO_SEVERITY_REGISTRY`: `count`, `decayed_weighted`).
  - `logging_config.py`: `structlog` + stdlib logging setup.
- **`nlp_predictor.scraping`**: Extensible content ingestion built on `BaseScraper`:
  - HTML/RSS scrapers use `process_article()` and `_run_rss_pipeline()`.
  - Text/API/Social scrapers (e.g., Twitter/X, wire feeds) use `process_text_item()` and custom `min_content_length`.
- **`nlp_predictor.preprocessing`**:
  - `clean_data.py`: HTML extraction (`trafilatura`), Unicode/whitespace normalization, UTC date parsing, SHA-256 deduplication hashing.
  - `topic_extractor.py`: LLM extraction of market-moving events (`ExtractedTopics`).
  - `feature_store.py` & `feature_ingestor.py`: Rolling Parquet cache (`FeatureStore`) storing per-topic embeddings and instrument impacts keyed by `doc_hash` so each article is processed by the LLM/embedding API **only once**.
- **`nlp_predictor.upper_path`**: Instrument-specific sentiment mapping (`UpperPathAnalyzer.map_topics`) and multi-variant severity scoring (`score_variants`, `generate_variant_trends`). **Must never import from `lower_path`.**
- **`nlp_predictor.lower_path`**: Unsupervised semantic clustering (`HDBSCAN` in `LowerPathAnalyzer`) and per-instrument macro tailwind/headwind classification (`MacroOverlayAnalyzer`). **Must never import from `upper_path`.**
- **`nlp_predictor.signal`**: Deterministic fusion of upper-path severity and lower-path macro context (`SignalGenerator`) and append-only Parquet evaluation logging (`SignalLog`).
- **`nlp_predictor.tick`**: Single-tick A/B/n orchestrator (`TickRunner`) evaluating all configured severity variants over a shared cached window and shared macro climate.
- **`nlp_predictor.backtesting`**: Historical news ingestion (`alpaca_news.py`), cleaning (`clean_news.py`), and synchronized market price fetching (`price_fetcher.py`).

---

## 2. Strict Anti-AI-Slop & Code Quality Rules

1. **No Cryptic Names or Dead Shims**:
   - Never create single-letter functions or variables like `def _p(k):` or `x1, x2`. Use descriptive, domain-accurate identifiers.
   - Do not leave dead code, commented-out blocks, or backward-compatibility shims unless explicitly requested.
2. **Concise, High-Signal Google Docstrings**:
   - Every module, class, and function must have a Google-style docstring.
   - **Do NOT write bloated, repetitive essays.** State the responsibility in 1–2 sentences, document input/output schemas (`Args`, `Returns`, DataFrame columns), and include mathematical formulas where applicable.
   - Wrap all lines (including docstrings) at **88 characters** maximum.
3. **Type Hints & Pydantic Everywhere**:
   - Annotate all function arguments and return types using Python 3.11+ syntax (`list[str]`, `dict[str, Any]`, `str | None`).
   - Use `Pydantic` (`BaseModel`) for structured LLM outputs and configuration schemas.
4. **OOP, Composition & Dependency Injection**:
   - Inject collaborators (`FeatureStore`, analyzers, loggers) via `__init__` and provide a `@classmethod def build_default(cls)` factory for production wiring.
   - Never instantiate external API clients or read files at module import time.
5. **Logging, Never `print()`**:
   - Use `logger = logging.getLogger(__name__)` across all modules and scripts. Never use `print()`.
6. **Pandas & I/O Performance**:
   - **Never** use `df.iterrows()`, `df.apply(pd.Series)`, or `pd.concat` inside a loop.
   - Use vectorized operations, `df.to_dict(orient="records")`, `pd.DataFrame.from_records()`, and `O(1)` set lookups for deduplication.
   - Use `pyarrow` for Parquet I/O (do not reintroduce `fastparquet`).

---

## 3. Mandatory Verification Commands

Before finishing any task or opening a Pull Request, run all checks via `uv`:

```bash
uv run ruff format src tests scripts
uv run ruff check src tests scripts
uv run pytest --no-cov
uv run sphinx-build -b html -E docs/source docs/build/html
```

All unit tests in `tests/` must be deterministic and run 100% offline using fakes/mocks instead of live OpenAI or network calls.

