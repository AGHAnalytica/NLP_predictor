PROMPTS = {
    # These are the prompt templates for the OpenAI API calls, stored in a dictionary
    # for easy access.
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
