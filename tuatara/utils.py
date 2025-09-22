import json
from importlib import resources


def parse_json_pairs(response: str) -> list[tuple[str, str]]:
    response = response.replace("```json", "").replace("```", "").strip()

    try:
        data = json.loads(response)
        return [(item["question"], item["answer"]) for item in data]
    except Exception:
        return []


def load_prompt_template(name: str) -> str:
    return resources.read_text("tuatara.prompts", f"{name}.txt")
