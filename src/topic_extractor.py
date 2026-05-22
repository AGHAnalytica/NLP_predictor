import os
import json
from openai import OpenAI

# Lekki loader .env (bez zewnętrznych zależności). Odczytuje proste linie KEY=VALUE.
def _load_env_file(path: str = ".env") -> None:
  if not os.path.exists(path):
    return
  try:
    with open(path, encoding="utf-8") as f:
      for line in f:
        line = line.strip()
        if not line or line.startswith("#"):
          continue
        if "=" not in line:
          continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        os.environ.setdefault(key, val)
  except Exception:
    # Ignoruj błędy podczas czytania .env; polegaj na zmiennych środowiskowych
    pass


# Spróbuj wczytać .env, jeśli istnieje (opcjonalne)
_load_env_file()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
  raise RuntimeError("OPENAI_API_KEY not set. Set the env var or create a .env file.")

client = OpenAI(api_key=OPENAI_API_KEY)

SYSTEM_PROMPT = """
Jesteś analitykiem finansowym. Z artykułu wyodrębnij listę tematów, które mogą wpłynąć na rynki finansowe.

Topic to konkretne wydarzenie/zjawisko (NIE nazwiska, NIE nazwy serwisów).
Dla każdego topica zwróć: nazwę, opis (2-3 zdania: co się dzieje, dlaczego ważne dla rynku, jaki mechanizm wpływu na ceny).

# OUTPUT FORMAT (strict JSON)
{
  "topics": [
    {
      "name": "Strait of Hormuz Reopening",
      "description": "US-Iran negotiations are progressing toward a memorandum of understanding..."
    }
  ]
}
"""


def extract_topics_from_article(article_content: str) -> dict:
    """
    Wywołuje model GPT-5 Nano, aby wyodrębnić tematy finansowe z treści podanego artykułu.
    """
    try:
      response = client.chat.completions.create(
        model="gpt-5-nano",
        response_format={"type": "json_object"},
        messages=[
          {"role": "system", "content": SYSTEM_PROMPT},
          {"role": "user", "content": f"Artykuł do analizy:\n\n{article_content}"}
        ],
      )
    except Exception as e:
      print("Błąd API OpenAI:", e)
      raise

    raw_response = response.choices[0].message.content
    return json.loads(raw_response)


if __name__ == "__main__":
    # Test naszego promptu i połączenia z API
    dummy_article = (
        "Reuters: W poniedziałek rano brytyjski rząd ogłosił niespodziewane wydłużenie "
        "podatku od ponadnormatywnych zysków (Windfall Tax) dla spółek naftowych i gazowych o kolejne dwa lata. "
        "Decyzja ta natychmiast wywołała spadki notowań gigantów energetycznych na giełdzie w Londynie, "
        "ponieważ inwestorzy obawiają się mniejszych dywidend."
    )

    print("Sending request to GPT-5 Nano...")
    result = extract_topics_from_article(dummy_article)

    print("\nOtrzymany wynik JSON:")
    print(json.dumps(result, indent=2, ensure_ascii=False))