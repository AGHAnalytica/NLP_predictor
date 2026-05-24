import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from .prompts import SYSTEM_PROMPT


def load_api_key(dotenv_path: Path | str | None = None) -> str:
    load_dotenv(dotenv_path)
    API_KEY = os.getenv("API_KEY")
    if API_KEY is None:
        raise ValueError("Missing API_KEY in .env file")
    return API_KEY


@dataclass(slots=True, eq=False)
class TopicExtractor:
    OPENAI_API_KEY: str
    prompts: dict[str, str] = {"SYSTEM_PROMPT": SYSTEM_PROMPT}
    _client: OpenAI = field(init=False)

    def __post_init__(self):
        self._client = OpenAI(api_key=self.OPENAI_API_KEY)

    def extract_content(self, article: str) -> dict:
        """
        Calls GPT-5 Nano in order to extract financial topics from given article.
        """
        try:
            response = self._client.chat.completions.create(
                model="gpt-5-nano",
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": f"Artykuł do analizy:\n\n{article}",
                    },
                ],
            )
        except Exception as e:
            print("OpenAI API error:", e)
            raise

        raw_response = response.choices[0].message.content
        return json.loads(raw_response)


if __name__ == "__main__":
    topic_extractor = TopicExtractor(load_api_key())

    # API and connection test
    dummy_article = (
        "Reuters: W poniedziałek rano brytyjski rząd ogłosił niespodziewane wydłużenie "
        "podatku od ponadnormatywnych zysków (Windfall Tax) dla spółek naftowych i gazowych o kolejne dwa lata. "
        "Decyzja ta natychmiast wywołała spadki notowań gigantów energetycznych na giełdzie w Londynie, "
        "ponieważ inwestorzy obawiają się mniejszych dywidend."
    )

    print("Sending request to GPT-5 Nano...")
    result = topic_extractor.extract_content(dummy_article)

    print("\nRecieved JSON response:")
    print(json.dumps(result, indent=4, ensure_ascii=False))
