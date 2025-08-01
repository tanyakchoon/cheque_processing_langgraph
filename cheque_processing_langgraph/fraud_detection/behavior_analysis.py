import pandas as pd
import json
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

def llm_analyze_historical_behavior(
    cheque_data: dict,
    payer_database: dict, # Changed from historical_data for clarity
    llm: ChatGoogleGenerativeAI
) -> (bool, str):
    """
    Uses a "Chain of Thought" prompt to analyze a transaction for behavioral anomalies.
    """
    print("INFO: Analyzing historical behavior using Gemini...")
    account_number = cheque_data.get("account_number")
    amount = cheque_data.get("amount")

    payer_record = payer_database.get(account_number.strip()) if account_number else None
    if not payer_record:
        return True, f"Account number '{account_number}' not found in payer database."

    # For this demonstration, we'll create a simple history summary.
    # In a real system, you'd query a transaction database.
    history_summary = {
        "avg_amount": 500.00,
        "max_amount": 4000.00,
        "typical_payees": ["Utility Company", "Rentals Inc", "Some Company"],
        "account_holder": payer_record.get("payer_name")
    }

    prompt_text = f"""
    You are a senior fraud analyst AI. Analyze the following new cheque transaction based on the provided historical summary for the account.

    **Historical Behavior Summary:**
    ```json
    {json.dumps(history_summary, indent=2)}
    ```

    **New Transaction to Analyze:**
    ```json
    {json.dumps(cheque_data, indent=2, default=str)}
    ```

    **Your Task (Reason Step-by-Step):**
    1.  **Amount Check**: Is the new transaction amount (`{amount}`) unusually high compared to the historical average and maximums?
    2.  **Payee Check**: Is the payee (`{cheque_data.get('payee')}`) one of the typical payees? If not, is the payee the same as the account holder (self-payment can be unusual)?
    3.  **Conclusion**: Based on your analysis, is this transaction behaviorally anomalous?

    **Output:**
    Return a single JSON object with two keys:
    - `is_anomalous`: boolean (true if you suspect an anomaly, false otherwise).
    - `reason`: string (A brief, one-sentence justification for your conclusion).
    """

    try:
        response = llm.invoke([HumanMessage(content=prompt_text)])
        result = json.loads(response.content)
        is_anomalous = result.get("is_anomalous", False)
        reason = result.get("reason", "No reason provided.")
        print(f"INFO: Behavior analysis result: Anomalous = {is_anomalous}, Reason = {reason}")
        return is_anomalous, reason
    except Exception as e:
        print(f"ERROR: Failed to analyze behavior with Gemini: {e}")
        return True, "Behavioral analysis failed."