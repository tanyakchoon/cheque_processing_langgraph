import os
import uuid
import json
from typing import TypedDict, List, Any
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI

from .image_enhancement.enhancer import llm_check_readability
from .processing.ocr_extraction import llm_extract_and_validate_cheque_data
from .processing.validation import validate_account_details
from .fraud_detection.tampering_detection import llm_detect_tampering
from .fraud_detection.behavior_analysis import llm_analyze_historical_behavior
# === IMPORT THE NEW LLM-BASED SIGNATURE COMPARISON MODULE ===
from .fraud_detection.signature_comparison import llm_compare_signatures

# This 'database' maps account numbers to payer info, including the signature file.
PAYER_DATABASE = {
    "12345678": { 
        "payer_name": "Apple Tan",
        "payer_signature_path": "reference_signature.png"
    },
    "12345678901": {
        "payer_name": "Elton Lim",
        "payer_signature_path": "elton_lim_signature.png"
    },
    "55556666": {
        "payer_name": "Susan Wong",
        "payer_signature_path": "susan_wong_signature.png"
    }
}

class ChequeState(TypedDict):
    project_root: str
    image: np.ndarray
    cheque_data: dict
    signature_image: np.ndarray | None
    amount_in_words: str | None
    audit_trail: Any
    is_readable: bool
    fraud_detected: bool
    final_decision: str
    feedback: List[str]

def build_graph():
    """Builds and returns the LangGraph compiled workflow."""
    load_dotenv()
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0)
    json_llm = llm.bind(response_mime_type="application/json")

    def start_processing(state: ChequeState) -> ChequeState:
        from .audit.trail import AuditTrail
        cheque_id = f"cheque-{uuid.uuid4().hex[:8]}"
        audit_trail = AuditTrail(cheque_id)
        audit_trail.log_step("Start", "Success", "Image data received.")
        return {**state, "audit_trail": audit_trail, "feedback": []}

    def check_image_quality(state: ChequeState) -> ChequeState:
        is_readable, msg = llm_check_readability(state["image"], json_llm)
        if not is_readable:
            state["audit_trail"].highlight_anomaly("Image Quality", msg)
            return {**state, "is_readable": False}
        state["audit_trail"].log_step("Image Quality Check", "Success", "Gemini approved image quality.")
        return {**state, "is_readable": True}

    def extract_data(state: ChequeState) -> ChequeState:
        data = llm_extract_and_validate_cheque_data(state["image"], json_llm)
        if "error" in data or not all(k in data for k in ["amount", "payee", "payer_account_number", "is_date_valid"]):
            err_msg = data.get("error", "Gemini Vision failed to extract/validate all key fields.")
            state["audit_trail"].log_step("Extraction & Validation", "Failed", err_msg)
            return {**state, "final_decision": "MANUAL_REVIEW"}
        state["audit_trail"].log_step("Extraction & Validation", "Success", f"Data extracted and validated.")
        return {**state, "cheque_data": data, "signature_image": data.get("signature_image"), "amount_in_words": data.get("amount_in_words")}

    def run_fraud_detection(state: ChequeState) -> ChequeState:
        fraud_found = False
        audit_trail = state["audit_trail"]
        
        if not state["cheque_data"].get("is_date_valid", True):
            reason = state["cheque_data"].get('date_validation_reason', 'The extracted date is invalid.')
            audit_trail.highlight_anomaly("Date Validation", reason)
            fraud_found = True
            
        if not state["cheque_data"].get("is_amount_consistent", True):
            reason = state["cheque_data"].get('validation_reason', 'Amounts do not match.')
            audit_trail.highlight_anomaly("Amount Verification", reason)
            fraud_found = True

        is_tampered, msg = llm_detect_tampering(state["image"], json_llm)
        if is_tampered:
            audit_trail.highlight_anomaly("Tampering Detection", msg)
            fraud_found = True
            
        is_anomalous, msg = llm_analyze_historical_behavior(state["cheque_data"], PAYER_DATABASE, json_llm)
        if is_anomalous:
            audit_trail.highlight_anomaly("Behavior Analysis", msg)
            fraud_found = True

        # === UPDATE: The signature comparison logic now calls the Gemini agent ===
        cheque_signature = state.get("signature_image")
        payer_account_number = state["cheque_data"].get("payer_account_number")

        if cheque_signature is not None and payer_account_number:
            payer_record = PAYER_DATABASE.get(payer_account_number.strip())
            if not payer_record:
                audit_trail.highlight_anomaly("Signature Verification", f"Payer account '{payer_account_number}' not found in database.")
                fraud_found = True
            else:
                try:
                    ref_sig_path = Path(state["project_root"]) / payer_record["payer_signature_path"]
                    reference_signature = cv2.imread(str(ref_sig_path))
                    if reference_signature is None:
                        raise FileNotFoundError(f"Signature file not found at {ref_sig_path}")
                    
                    # Call the new Gemini-based comparison function
                    match, reason = llm_compare_signatures(cheque_signature, reference_signature, json_llm)
                    
                    if not match:
                        audit_trail.highlight_anomaly("Signature Verification", reason)
                        fraud_found = True
                    else:
                        audit_trail.log_step("Signature Verification", "Success", reason)
                except Exception as e:
                    audit_trail.highlight_anomaly("Signature Verification", f"Error during comparison: {e}")
        
        audit_trail.log_step("Fraud Detection", "Completed", f"Fraud found: {fraud_found}")
        return {**state, "fraud_detected": fraud_found}

    def validate_and_process(state: ChequeState) -> ChequeState:
        data = state["cheque_data"]
        is_valid, msg = validate_account_details(data.get("payer_account_number", ""))
        if not is_valid:
            state["audit_trail"].highlight_anomaly("Account Validation", msg)
            return {**state, "final_decision": "REJECT"}
        state["audit_trail"].log_step("Account Validation", "Success", "Account is valid.")
        state["feedback"].append("Cheque processed successfully.")
        return {**state, "final_decision": "APPROVE"}
        
    def route_after_start(state: ChequeState): return "check_image_quality"
    def route_after_quality_check(state: ChequeState): return END if not state.get("is_readable") else "extract_data"
    def route_after_extraction(state: ChequeState): return END if state.get("final_decision") == "MANUAL_REVIEW" else "run_fraud_detection"
    def route_after_fraud_check(state: ChequeState): return "manual_review" if state.get("fraud_detected") else "validate_and_process"

    workflow = StateGraph(ChequeState)
    workflow.add_node("start", start_processing); workflow.add_node("check_image_quality", check_image_quality); workflow.add_node("extract_data", extract_data); workflow.add_node("run_fraud_detection", run_fraud_detection); workflow.add_node("validate_and_process", validate_and_process); workflow.add_node("manual_review", lambda state: {**state, "final_decision": "MANUAL_REVIEW"})
    workflow.set_entry_point("start")
    workflow.add_conditional_edges("start", route_after_start); workflow.add_conditional_edges("check_image_quality", route_after_quality_check); workflow.add_conditional_edges("extract_data", route_after_extraction); workflow.add_conditional_edges("run_fraud_detection", route_after_fraud_check)
    workflow.add_edge("validate_and_process", END); workflow.add_edge("manual_review", END)
    return workflow.compile(), llm

def main():
    """Main function for command-line testing."""
    print("Initializing Cheque Processing System for command-line test...")
    try:
        project_root = str(Path(__file__).parent.parent.resolve())
    except NameError:
        project_root = os.getcwd()
    dbs_cheque_path = Path(project_root) / "dbs_cheque.png"
    if not dbs_cheque_path.exists():
        print(f"FATAL: Ensure 'dbs_cheque.png' exists in {project_root}")
        return
    cheque_image_data = cv2.imread(str(dbs_cheque_path))
    app, text_llm = build_graph()
    initial_state = {"image": cheque_image_data, "project_root": project_root}
    print(f"\nStarting Cheque Processing Workflow...")
    final_state = app.invoke(initial_state)
    print("\n\n" + "=" * 50)
    print("           FINAL CHEQUE PROCESSING OUTCOME")
    print("=" * 50)
    print(f"Final Decision: {final_state.get('final_decision', 'N/A')}")
    print(f"Feedback: {final_state.get('feedback')}")
    if final_state.get("audit_trail"):
        summary = final_state["audit_trail"].generate_llm_summary_report(text_llm)
        print("\n--- Gemini-Generated Audit Summary ---")
        print(summary)
    print("=" * 50)

if __name__ == "__main__":
    main()