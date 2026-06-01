PROMPTS = {
    # These are the prompt templates for the OpenAI API calls, stored in a dictionary
    # for easy access.
    # ============================
    # Topic Extraction Prompt (Grupa 2)
    # ============================
    "topic_extraction": """
        You are a financial news analyst. Your ONLY job is to read the article below
        and EXTRACT the distinct market-relevant topics it reports. You are an
        extractor, not a trading advisor.

        A topic is a specific, self-contained event or phenomenon
        (e.g. "Strait of Hormuz Reopening", "Federal Reserve Rate Cut Expectations",
        "UK Windfall Tax Extension"). A topic is NOT a person's name, a journalist,
        or a news outlet. Merge near-duplicate mentions into a single topic.
        If the article reports no market-relevant event, return an empty list.

        For each topic return:
        - name: a short, specific title for the event.
        - description: 2-3 sentences describing WHAT is happening, based only on the
          article. Stick to the facts of the event. You MAY add a brief, general note
          on market relevance or likely direction (e.g. "tends to push oil prices
          higher"), but do NOT name specific tickers or instruments and do NOT
          recommend buying or selling. Mapping topics to concrete instruments and
          trading direction is done by a later stage, not here.

        ### Example 1
        Title: UK extends windfall tax on oil and gas producers
        Content: The British government unexpectedly extended its windfall tax on oil
        and gas companies by two more years.
        Extracted topics:
        - name: "UK Windfall Tax Extension"
          description: "The UK government extended its windfall tax on oil and gas
          producers by two years. This raises fiscal pressure on the energy sector and
          is generally viewed as a negative for energy-sector sentiment."

        ### Example 2
        Title: Reporter Jane Smith covers OPEC+ surprise output cut
        Content: OPEC+ unexpectedly announced deep production cuts starting next month.
        Extracted topics:
        - name: "OPEC+ Production Cut"
          description: "OPEC+ announced surprise cuts to oil output starting next month,
          tightening global crude supply. Reduced supply tends to support higher oil
          prices."
        (Note: "Jane Smith" is a reporter and is correctly NOT a topic.)

        ### Article to analyze
        Title: {title}

        Content:
        {content}
        """,
    # ============================
    # Upper Path Prompts
    # ============================
    "instrument_trends": """
        You are a quantitative financial analyst. Map the following news topic to ALL
        applicable instruments from the list.

        For each impacted instrument, calculate the 'sentiment' score
        between -1.0 and 1.0,
        representing how this specific news affects that specific company's stock price.
        IMPORTANT: Return ONLY exact 'ticker' symbols from the provided list.

        Available Instruments: {available_instruments}

        Topic Name: {topic_name}
        Topic Description: {topic_description}
        """,
    # ============================
    # Lower Path Prompts
    # ============================
    "global_trends": """
        You are a Chief Economist. Analyze the following cluster of news topics.
        These topics were mathematically clustered together due to their semantic
        similarity. Determine the overarching global macroeconomic narrative and its
        overall direction ('up', 'down', or 'neutral'). Provide a concise title
        and a 1-2 sentence description explaining the global impact.

        Cluster Topics Context:
        {topics_context}
        """,
}
