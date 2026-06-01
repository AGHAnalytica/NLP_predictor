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

        A topic is a specific, self-contained MARKET-MOVING EVENT
        (e.g. "Strait of Hormuz Reopening", "Federal Reserve Rate Cut Expectations",
        "UK Windfall Tax Extension"). A topic is NOT a person's name, a journalist,
        or a news outlet.

        Granularity rules:
        - Extract distinct EVENTS, not their sub-facts. Do NOT split one event into its
          individual details, consequences, or the market's reaction to it — fold those
          into the single parent topic. For example: an earnings report is ONE topic,
          not separate topics for revenue, margins, dividend and demand; a supply-chain
          disruption is ONE topic, not separate topics for port congestion, freight
          rates and input costs.
        - Merge near-duplicate mentions into a single topic.
        - For a typical single-story article return 1-3 topics; only return more if the
          article genuinely covers several independent events.
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
        You are a quantitative financial analyst. Map the news topic below ONLY to
        instruments it affects SPECIFICALLY. Be conservative: prefer FEWER, directly
        linked instruments over many weak ones.

        First decide what KIND of topic this is:
        - SPECIFIC (about one company, sector, commodity or currency): map only the
          directly affected instrument(s). E.g. an Apple lawsuit -> Apple; an OPEC
          supply cut -> oil and only names whose costs/revenue depend directly on oil.
        - BROAD MACRO (interest rates, inflation, jobs, recession, overall risk
          sentiment, the general level of the dollar): this moves the WHOLE market,
          not one company. Map it to the broad market index, plus the currency pair if
          it is a rates/FX story. Do NOT attach broad macro to individual stocks just
          because "everything moves with the macro" — that market-wide effect is
          captured elsewhere, so spreading it over every ticker only double-counts it.

        Map an individual stock to a macro topic ONLY if it is uniquely and directly
        exposed (more than the average company), and even then be cautious. If a topic
        has no specific link to any listed instrument, return an empty list.

        Score each impacted instrument's 'sentiment' on this FIXED 5-point scale
        (pick the single closest level — do not invent in-between values):
          +1.0 = decisive positive, a major boost to that instrument's value
          +0.5 = clear positive
           0.0 = irrelevant, or positive and negative effects roughly cancel
          -0.5 = clear negative
          -1.0 = decisive negative, a major threat to that instrument's value
        Judge each instrument independently: the SAME topic can be +0.5 for one and
        -0.5 for another.

        ### Example 1 (broad macro)
        Topic: "Federal Reserve signals more interest-rate hikes"
        -> SPY = -0.5 (higher-for-longer rates weigh on the broad equity market)
        -> EURUSD = -0.5 (tighter Fed policy lifts the dollar, pressuring EUR/USD)
        Do NOT map this to individual technology or consumer stocks.

        ### Example 2 (specific commodity)
        Topic: "OPEC+ announces a surprise oil production cut"
        -> OIL = +1.0 (tighter crude supply pushes oil prices up)

        ### Example 3 (company-specific)
        Topic: "Apple hit with an antitrust lawsuit over the App Store"
        -> AAPL = -0.5 (regulatory overhang on a key profit center)

        Return ONLY exact 'ticker' symbols from the provided list.
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
