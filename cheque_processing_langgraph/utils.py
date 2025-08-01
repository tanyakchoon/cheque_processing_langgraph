import json
import re
import base64
from PIL import Image
import io

def parse_json_from_response(content: str) -> dict | None:
    """
    Robustly parses a JSON object from an LLM's string response.
    Handles plain JSON and JSON wrapped in markdown backticks.
    """
    if not isinstance(content, str):
        print(f"ERROR: Expected a string response, but got {type(content)}")
        return None
        
    try:
        # Case 1: The response is already a perfect JSON string.
        return json.loads(content)
    except json.JSONDecodeError:
        # Case 2: The JSON is wrapped in markdown backticks.
        match = re.search(r"```(?:json\s*)?({.*})\s*```", content, re.DOTALL)
        if match:
            json_str = match.group(1)
            try:
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                print(f"ERROR: Found JSON in markdown, but failed to parse: {json_str}\nError: {e}")
                return None
        else:
            # Case 3: JSON is not wrapped, but might have prefixes/suffixes.
            try:
                start = content.find('{')
                end = content.rfind('}') + 1
                if start != -1 and end != 0:
                    potential_json = content[start:end]
                    return json.loads(potential_json)
                else:
                    print(f"ERROR: Could not find a JSON object in the response: {content}")
                    return None
            except json.JSONDecodeError as e:
                print(f"ERROR: Failed to parse extracted JSON substring. Error: {e}\nContent: {content}")
                return None

# --- NEW HELPER FUNCTION ---
def pil_to_base64_uri(pil_image: Image.Image) -> str:
    """Converts a PIL Image object to a Base64 encoded Data URI."""
    buffered = io.BytesIO()
    pil_image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{img_str}"