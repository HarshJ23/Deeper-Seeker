import re
import json

# This is for custom models that do not directly support the response_format for chat completion
def extract_json_from_response(content: str):
    # Try to find JSON between tags if the model adds them
    json_match = re.search(r'<json>(.*?)</json>', content, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1).strip())
        except json.JSONDecodeError:
            pass
    
    # Try to find anything that looks like a JSON object
    json_match = re.search(r'\{.*\}', content, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass
    
    raise ValueError(f"Failed to parse valid json response from inputs: {content}")