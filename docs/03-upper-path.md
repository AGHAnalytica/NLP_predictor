# Module: Upper Path (Instrument Trends)

## 1. Architectural Overview
The Upper Path is a dedicated analytical pipeline answering the specific question: *"What exactly is moving on the market?"*. Operating in parallel with the macro-level Lower Path, this module processes raw, isolated news topics and translates them into granular, per-instrument trading signals (e.g., assigning a "US-Iran peace deal" topic directly to crude oil `OIL` with a downward price pressure).

The module adheres to a strict data contract, ensuring decoupling from the scraping layer (Group 1) and topic extraction layer (Group 2).

## 2. Pipeline Execution Flow

### Phase 1: Topic-to-Instrument Mapping
*   **Action:** Iterates through the incoming topics DataFrame.
*   **Mechanism:** Utilizes an LLM (e.g., Gemini Flash) constrained by Pydantic schemas (Structured Outputs) to map a topic to one or more valid instrument tickers from the static reference database (`instruments.csv`). 
*   **Filtering:** Topics that do not map to any predefined trading asset are classified as noise in this path and explicitly dropped.

### Phase 2: Data Aggregation
*   **Action:** Groups the validated topics by their respective instrument tickers.
*   **Time Windowing:** Operates on batches of topics collected within a specific timeframe (e.g., last 15 minutes or 1 day) to prevent historical data contamination.

### Phase 3: Severity Calculation
*   **Action:** Quantifies the strength of the incoming signal.
*   **Metric:** In the MVP phase, this is represented by a numerical score based on the raw count of topics associated with the instrument in the current time window.

### Phase 4: Trend Generation
*   **Action:** Consolidates the grouped topics into a final trading context.
*   **Mechanism:** Prompts the LLM with the aggregated context for a specific instrument to determine the overarching price direction (`up`, `down`, `neutral`) and generates a human-readable title and description for the final signal.

## 3. Data Contracts

### Input Contract
Expects a Pandas DataFrame from the Topic Extraction module (Group 2) containing:
*   `topic_id` (string/UUID)
*   `doc_hash` (string)
*   `topic_name` (string)
*   `topic_description` (string)
*   `date` (datetime)

### Output Contract (`INSTRUMENT_TRENDS`)
Yields a structured Pandas DataFrame ready for the final signal generation layer:

| Column | Type | Description |
| :--- | :--- | :--- |
| `trend_id` | string (UUID) | Unique identifier for the generated trend. |
| `date` | datetime | Timestamp of trend generation. |
| `instrument` | string | The specific asset ticker (e.g., `OIL`, `GOOGL`). |
| `title` | string | Short summary of the trend. |
| `description` | string | 1-2 sentence explanation of the market impact. |
| `severity` | float | Signal strength indicator. |
| `direction` | enum | Market direction: `up`, `down`, or `neutral`. |
| `topic_ids` | list[string] | Array of source topic IDs justifying the trend. |

## 4. Module Methods Overview
TODO