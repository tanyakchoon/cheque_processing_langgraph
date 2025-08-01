import cv2
import numpy as np
import json
import base64
import io
from PIL import Image
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

# === HELPER FUNCTIONS ADDED HERE ===
def convert_to_pil_image(image_array: np.ndarray) -> Image.Image:
    """Converts a BGR numpy array from OpenCV to an RGB PIL Image."""
    return Image.fromarray(cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB))

def encode_pil_to_base64_data_uri(pil_image: Image.Image) -> str:
    """Encodes a PIL image to a Base64 data URI."""
    buffered = io.BytesIO()
    pil_image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{img_str}"
# ==================================

def llm_detect_tampering(image: np.ndarray, llm: ChatGoogleGenerativeAI) -> (bool, str):
    print("INFO: Analyzing for tampering using Gemini...")
    pil_image = convert_to_pil_image(image)
    data_uri = encode_pil_to_base64_data_uri(pil_image)

    prompt = HumanMessage(
        content=[
            {
                "type": "text",
                "text": """You are a forensic document examiner. Analyze the attached cheque image for any signs of tampering.
                Look for:
                1. Mismatched fonts in the payee name or amount.
                2. Smudges, discoloration, or alignment issues in the amount fields.
                3. Any visual anomalies that suggest the cheque has been altered.
                
                Respond with a JSON object containing two keys:
                - "is_tampered": boolean (true if you suspect tampering, false otherwise)
                - "reason": string (a brief explanation of your findings)
                """,
            },
            # --- Use the correct image_url format ---
            {
                "type": "image_url",
                "image_url": data_uri
            },
            # ----------------------------------------
        ]
    )
    
    try:
        response = llm.invoke([prompt])
        result = json.loads(response.content)
        return result.get("is_tampered", False), result.get("reason", "No reason provided.")
    except Exception as e:
        print(f"ERROR: Failed to detect tampering with Gemini: {e}")
        return True, "Analysis failed, flagging for review."