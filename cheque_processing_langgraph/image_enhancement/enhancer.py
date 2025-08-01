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

def correct_skew(image: np.ndarray) -> np.ndarray:
    print("INFO: Correcting image skew...")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thresh > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

def enhance_brightness_contrast(image: np.ndarray, alpha=1.5, beta=10) -> np.ndarray:
    print("INFO: Enhancing brightness and contrast...")
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

def llm_check_readability(image: np.ndarray, llm: ChatGoogleGenerativeAI) -> (bool, str):
    print("INFO: Checking image readability using Gemini...")
    pil_image = convert_to_pil_image(image)
    data_uri = encode_pil_to_base64_data_uri(pil_image)
    
    prompt = HumanMessage(
        content=[
            {
                "type": "text",
                "text": """
                You are an image quality inspector for a cheque processing system.
                Assess the quality of the attached image. Is it clear, well-lit, and complete?
                Or is it blurry, skewed, dark, or cut off?
                
                Respond with a JSON object with two keys:
                - "is_readable": boolean (true if the quality is acceptable)
                - "feedback": string (a very brief, user-facing comment, e.g., "Image is too dark" or "Quality is good")
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
        return result.get("is_readable", False), result.get("feedback", "No feedback provided.")
    except Exception as e:
        print(f"ERROR: Failed to check readability with Gemini: {e}")
        return False, "Failed to analyze image quality."