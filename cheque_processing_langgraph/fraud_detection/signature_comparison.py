import cv2
import numpy as np
import json
import base64
import io
from PIL import Image
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

def convert_to_pil_image(image_array: np.ndarray) -> Image.Image:
    """Converts a BGR numpy array from OpenCV to an RGB PIL Image."""
    return Image.fromarray(cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB))

def encode_pil_to_base64_data_uri(pil_image: Image.Image) -> str:
    """Encodes a PIL image to a Base64 data URI."""
    buffered = io.BytesIO()
    pil_image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{img_str}"

def llm_compare_signatures(
    cheque_signature: np.ndarray,
    reference_signature: np.ndarray,
    llm: ChatGoogleGenerativeAI
) -> (bool, str):
    """
    Uses Gemini Vision with a forensic analysis prompt to compare two signature images.
    This is a much more robust method than SSIM.
    """
    print("INFO: Comparing signatures using Gemini forensic analysis agent...")
    
    if cheque_signature is None or reference_signature is None:
        return False, "One of the signature images is missing."

    cheque_sig_pil = convert_to_pil_image(cheque_signature)
    ref_sig_pil = convert_to_pil_image(reference_signature)
    
    cheque_sig_uri = encode_pil_to_base64_data_uri(cheque_sig_pil)
    ref_sig_uri = encode_pil_to_base64_data_uri(ref_sig_pil)
    
    prompt = HumanMessage(
        content=[
            {
                "type": "text",
                "text": """
                You are a forensic document examiner AI with expertise in signature verification. I will provide two images:
                - **Image 1**: The signature provided on a cheque.
                - **Image 2**: The reference signature on file for the account holder.

                Your task is to perform a detailed comparison. Analyze and compare the following forensic features:
                1.  **Stroke Style**: Are the lines smooth and flowing, or sharp and angular? Compare the thickness and pressure.
                2.  **Angle/Slant**: Is the overall writing slanted to the left, right, or is it upright? Are they consistent?
                3.  **Loops and Circles**: Examine the circular formations (like in 'a', 'o', 'l'). Are they open or closed? Are they similar in shape and size?
                4.  **Baseline Alignment**: Do the signatures follow a straight line (real or imaginary), or do they slant up or down?
                5.  **Overall Flow and Spacing**: Is the spacing between letters and the overall rhythm of the signatures similar?

                After your analysis, make a final conclusion. Do these signatures appear to be from the same person?

                **Output:**
                Return a single JSON object with two keys:
                - `signatures_match`: boolean (true if you conclude they match, false otherwise).
                - `reason`: string (A brief, expert justification for your decision based on the features you analyzed).
                """
            },
            {"type": "image_url", "image_url": cheque_sig_uri},
            {"type": "image_url", "image_url": ref_sig_uri},
        ]
    )

    try:
        response = llm.invoke([prompt])
        result = json.loads(response.content)
        match = result.get("signatures_match", False)
        reason = result.get("reason", "No reason provided.")
        print(f"INFO: Gemini signature comparison result: Match = {match}, Reason = {reason}")
        # Return only the boolean match and the reason
        return match, reason
    except Exception as e:
        print(f"ERROR: Failed to compare signatures with Gemini: {e}")
        return False, "Signature comparison analysis failed due to an error."