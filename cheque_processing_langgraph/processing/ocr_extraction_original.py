import cv2
import numpy as np
from PIL import Image
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from ..utils import parse_json_from_response, pil_to_base64_uri

def convert_to_pil_image(image_array: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB))

def llm_extract_cheque_data(image: np.ndarray, llm: ChatGoogleGenerativeAI) -> dict:
    """
    Extracts data using a highly specific prompt that guides the model on how to
    visually parse complex fields like boxed dates.
    """
    print("INFO: Extracting data using Gemini Vision with hyper-specific date prompt...")
    pil_image = convert_to_pil_image(image)
    image_uri = pil_to_base64_uri(pil_image)

    # --- FINAL, MOST ROBUST PROMPT ---
    messages = [
        SystemMessage(
            content="You are an expert financial document OCR system. Your response MUST be a single, perfectly formatted JSON object. Pay extremely close attention to the visual layout of the document."
        ),
        HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": """From the attached cheque image, you must meticulously extract the details.

                    **Instructions for Extraction:**

                    1.  **Date Field**: This is the most complex field.
                        - First, locate the area labeled "Date".
                        - You will see six individual boxes for Day (D D), Month (M M), and Year (Y Y).
                        - Perform OCR on each box to get the digit inside.
                        - Combine the digits for day, month, and year.
                        - **CRITICAL**: A two-digit year like '18' must be interpreted as '2018'. A year like '24' is '2024'.
                        - Assemble the final date string in "DD/MM/YYYY" format.

                    2.  **Amount Field**: Find the numerical amount. A comma (,) might be the decimal separator. Convert it to a standard float.

                    3.  **Payee Field**: Identify the recipient's name written on the "Pay" line.

                    4.  **Account Number**: Extract the account number from the MICR line at the bottom of the cheque. It is the longest set of digits at the end.

                    **Example JSON Output:**
                    {"payee": "Apple Tan", "amount": 1628.99, "date": "18/10/2018", "account_number": "123456678"}

                    You must return only the JSON object.
                    """,
                },
                { "type": "image_url", "image_url": {"url": image_uri} },
            ]
        ),
    ]

    try:
        response = llm.invoke(messages)
        data = parse_json_from_response(response.content)

        # We will keep the defensive check for required keys
        if data and all(k in data for k in ["payee", "amount", "date", "account_number"]):
             # Add a final check to ensure the date format itself is valid before returning
            if isinstance(data.get("date"), str) and len(data["date"]) == 10:
                print(f"INFO: Successfully extracted data: {data}")
                return data
            else:
                print(f"ERROR: Model returned an invalid date format. Value: {data.get('date')}")
                data['error'] = f"Model returned an invalid date format: {data.get('date')}"
                return data
        else:
            print(f"ERROR: Model returned incomplete JSON from data extraction. Raw response: {response.content}")
            return {"error": "Failed to extract data due to incomplete model response."}

    except Exception as e:
        print(f"ERROR: API call failed during data extraction: {e}")
        return {"error": f"Failed to extract data due to API error: {e}"}