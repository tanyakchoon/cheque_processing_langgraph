import cv2
import numpy as np
import json
import base64
import io
from PIL import Image
from datetime import datetime, timedelta # Import timedelta for stale date checks
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

def standardize_keys(data: dict) -> dict:
    """Converts all keys in a dictionary to lowercase and replaces spaces with underscores."""
    return {k.lower().replace(' ', '_'): v for k, v in data.items()}

def parse_bounding_box(bbox_data, image_width, image_height):
    """Parses bounding box data robustly."""
    try:
        if isinstance(bbox_data, list) and len(bbox_data) == 4:
            x_min, y_min, x_max, y_max = bbox_data
            return int(x_min * image_width), int(y_min * image_height), int(x_max * image_width), int(y_max * image_height)
    except Exception as e:
        print(f"WARNING: Could not parse bounding box data '{bbox_data}'. Error: {e}")
    return None

def validate_cheque_date(date_str: str) -> (bool, str):
    """
    Validates a date string, expecting a DDMMYYYY format.
    """
    if not isinstance(date_str, str):
        return False, "Date is not a valid string"

    if len(date_str) == 6:
        year_suffix = int(date_str[4:])
        current_year_suffix = int(datetime.now().strftime("%y"))
        century = "20" if year_suffix <= current_year_suffix else "19"
        date_str = date_str[:4] + century + date_str[4:]
        
    if len(date_str) != 8:
        return False, f"Invalid format (Expected DDMMYYYY, got {date_str})"

    today = datetime.now().date()
    stale_limit = today - timedelta(days=180)

    try:
        cheque_date = datetime.strptime(date_str, "%d%m%Y").date()
    except ValueError:
        return False, f"Invalid calendar date (e.g., Feb 30th)"

    if cheque_date > today:
        return False, f"Post-dated cheque (Date: {cheque_date.strftime('%Y-%m-%d')})"
    
    if cheque_date < stale_limit:
        return False, f"Stale-dated cheque (Date is older than 180 days)"

    return True, "Date is valid"


def llm_extract_and_validate_cheque_data(image: np.ndarray, llm: ChatGoogleGenerativeAI) -> dict:
    """
    Extracts, validates, and standardizes cheque data using a robust multi-prompt strategy.
    """
    print("INFO: Starting multi-step extraction process...")
    pil_image = convert_to_pil_image(image)
    data_uri = encode_pil_to_base64_data_uri(pil_image)

    try:
        # Step 1 & 2: Text and Signature Extraction (Unchanged)
        print("INFO: Step 1: Extracting text fields...")
        text_extraction_prompt = HumanMessage(content=[
            {"type": "text", "text": "You are an OCR AI. Extract the following from the image as a JSON object: Payee, Date (as a raw string of digits), Amount, Amount in Words, and the full MICR Line."},
            {"type": "image_url", "image_url": data_uri},
        ])
        text_response = llm.invoke([text_extraction_prompt])
        text_data = standardize_keys(json.loads(text_response.content))
        print(f"INFO: Text data extracted: {text_data}")

        print("INFO: Step 2: Locating signature bounding box...")
        signature_location_prompt = HumanMessage(content=[
            {"type": "text", "text": 'You are a visual analysis AI. Identify the signature bounding box. Return ONLY a JSON object with one key, "signature_bbox", containing a list of four relative coordinates: [x_min, y_min, x_max, y_max].'},
            {"type": "image_url", "image_url": data_uri},
        ])
        signature_response = llm.invoke([signature_location_prompt])
        signature_data = standardize_keys(json.loads(signature_response.content))
        print(f"INFO: Signature location found: {signature_data}")

        raw_data = {**text_data, **signature_data}

        # Step 3: LLM-based Validation (Amounts and MICR only)
        print("INFO: Step 3: Validating data with LLM...")
        
        # ===== REFINED AND MORE ROBUST VALIDATION PROMPT =====
        validation_prompt_text = f"""
        You are a meticulous bank compliance officer AI. Your task is to validate the extracted data from a cheque.

        **Primary Task: Amount Consistency Check**
        Analyze the `amount` and `amount_in_words` from the JSON data below. You must determine if they are financially equivalent. Pay close attention to variations in how words can represent numbers.

        **Crucial Examples of Equivalence (This is your guide):**
        - `amount: 150.25`, `amount_in_words: "ONE HUNDRED FIFTY & 25/100"` -> `is_amount_consistent: true`
        - `amount: 200.80`, `amount_in_words: "TWO HUNDRED DOLLARS AND EIGHTY CENTS"` -> `is_amount_consistent: true`
        - `amount: 50.00`, `amount_in_words: "FIFTY DOLLARS ONLY"` -> `is_amount_consistent: true`
        - `amount: 100.00`, `amount_in_words: "One Hundred Dollars"` -> `is_amount_consistent: true`
        - `amount: 150.25`, `amount_in_words: "ONE HUNDRED AND TWENTY-FIVE CENTS"` -> `is_amount_consistent: false` (This is 1.25, not 150.25)
        - `amount: 100.00`, `amount_in_words: "Ten Dollars"` -> `is_amount_consistent: false`

        **Secondary Task: Account Number Parsing**
        - From the `micr_line`, extract the Payer's Account Number. It is typically the longest set of digits in the middle.

        **Output requirements:**
        Return a single JSON object with three keys:
        1. `is_amount_consistent`: boolean (true if the amounts match, false otherwise).
        2. `validation_reason`: string (Briefly explain your reasoning for the amount consistency check. E.g., "Amounts 150.25 and 'ONE HUNDRED FIFTY & 25/100' are consistent.").
        3. `payer_account_number`: string (The parsed account number).

        **Data to Validate:**
        ```json
        {json.dumps(raw_data)}
        ```
        Produce the validation JSON object now.
        """
        # ==========================================================

        validation_response = llm.invoke([HumanMessage(content=validation_prompt_text)])
        validation_results = standardize_keys(json.loads(validation_response.content))
        print(f"INFO: Validation results from LLM: {validation_results}")

        # STEP 4: RIGOROUS PROGRAMMATIC DATE VALIDATION (Unchanged)
        raw_date_str = raw_data.get("date")
        is_valid, reason = validate_cheque_date(raw_date_str)
        validation_results['is_date_valid'] = is_valid
        validation_results['date_validation_reason'] = reason
        print(f"INFO: Programmatic date validation result: {is_valid}, Reason: {reason}")
        
        # Step 5: Final Merge and Processing (Unchanged)
        final_data = {**raw_data, **validation_results}
        
        if 'date' in final_data:
            final_data['formatted_date'] = final_data.pop('date')

        if 'payer_account_number' in final_data:
            final_data['account_number'] = final_data['payer_account_number'].replace('"', '').strip()
        
        signature_image = None
        bbox_raw = final_data.get("signature_bbox")
        if bbox_raw:
            h, w, _ = image.shape
            parsed_coords = parse_bounding_box(bbox_raw, w, h)
            if parsed_coords:
                x1_abs, y1_abs, x2_abs, y2_abs = parsed_coords
                x_padding, y_padding = int((x2_abs - x1_abs) * 0.1), int((y2_abs - y1_abs) * 0.15)
                x1, y1 = max(0, x1_abs - x_padding), max(0, y1_abs - y_padding)
                x2, y2 = min(w, x2_abs + x_padding), min(h, y2_abs + y_padding)
                signature_image = image[y1:y2, x1:x2]
        
        final_data["signature_image"] = signature_image

        if 'amount' in final_data and isinstance(final_data['amount'], str):
            final_data['amount'] = float(final_data['amount'].replace(',', '.'))

        return final_data
        
    except Exception as e:
        import traceback
        print(f"ERROR: An exception occurred during the multi-step extraction process: {e}")
        traceback.print_exc()
        return {"error": str(e)}