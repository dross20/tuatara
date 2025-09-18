import json

def parse_json_pairs(response: str) -> list[tuple[str, str]]:
    # Remove markdown code blocks
    response = response.replace('```json', '').replace('```', '').strip()
    
    try:
        data = json.loads(response)
        return [(item['question'], item['answer']) for item in data]
    except:
        return []