import gradio as gr
import numpy as np
import os
import cv2
import uuid
from pathlib import Path

from cheque_processing_langgraph.__main__ import build_graph

# --- Startup: Define root and build graph ---
try:
    project_root = str(Path(__file__).parent.resolve())
except NameError:
    project_root = os.getcwd()
print(f"Project root for Gradio app: {project_root}")

print("Building the LangGraph application for the Gradio UI...")
app, text_llm = build_graph()
print("LangGraph application built successfully.")


def get_signature_check_result(final_state: dict) -> str:
    """
    Parses the audit trail to find the result of the signature verification step.
    """
    audit_trail = final_state.get("audit_trail")
    if not audit_trail or not hasattr(audit_trail, 'logs'):
        return "Not Performed"

    # Search the audit logs for the signature verification step
    for log in reversed(audit_trail.logs): # Search backwards for the most recent entry
        if "Signature Verification" in log:
            if "Success" in log:
                # Extract the reason, which contains the score
                reason_part = log.split("Reason: ")[-1]
                return f"✅ Match ({reason_part})"
            if "ANOMALY" in log: # Check for anomalies as well
                reason_part = log.split("Reason: ")[-1]
                return f"❌ Mismatch ({reason_part})"
    
    # Check anomalies specifically if not found in logs
    if hasattr(audit_trail, 'anomalies'):
        for anomaly in reversed(audit_trail.anomalies):
            if "Signature Verification" in anomaly:
                reason_part = anomaly.split("Reason: ")[-1]
                return f"❌ Mismatch ({reason_part})"

    return "Not Performed"


def process_cheque_with_ui(cheque_image_np: np.ndarray):
    """ Main interface for the Gradio UI. """
    if cheque_image_np is None:
        return None, "## Error\n\nPlease upload a cheque image first."

    cheque_image_bgr = cv2.cvtColor(cheque_image_np, cv2.COLOR_RGB2BGR)
    
    initial_state = {
        "image": cheque_image_bgr,
        "project_root": project_root
    }
    
    print(f"Invoking graph with image data and project root...")
    final_state = app.invoke(initial_state)
    
    # --- Generate the Report ---
    final_decision = final_state.get('final_decision', 'Error')
    feedback = "\n".join(final_state.get('feedback', ['An unknown error occurred.']))
    
    report = f"## Cheque Processing Report\n\n"
    report += f"**Final Decision:** `{final_decision}`\n\n"
    
    cheque_data = final_state.get("cheque_data", {})
    if cheque_data:
        report += "### Extracted & Validated Details\n\n"
        report += "| Field                 | Extracted Value |\n"
        report += "| --------------------- | --------------- |\n"
        report += f"| Payee                 | {cheque_data.get('payee', 'N/A')} |\n"
        report += f"| Amount (Numeric)      | {cheque_data.get('amount', 'N/A')} |\n"
        report += f"| Amount (in Words)     | {cheque_data.get('amount_in_words', 'N/A')} |\n"
        report += f"| Date                  | {cheque_data.get('formatted_date', 'N/A')} |\n"
        report += f"| Payer Account No.     | {cheque_data.get('payer_account_number', 'N/A')} |\n"
        
        # Date Validation Row
        is_date_valid = cheque_data.get('is_date_valid', False)
        if is_date_valid:
            date_valid_text = "✅ Yes"
        else:
            date_reason = cheque_data.get('date_validation_reason', 'Validation failed')
            date_valid_text = f"❌ No ({date_reason})"
        report += f"| Is Date Valid?        | {date_valid_text} |\n"
        
        # Amount Consistency Row
        is_consistent = cheque_data.get('is_amount_consistent', False)
        if is_consistent:
            consistency_text = "✅ Yes"
        else:
            consistency_reason = cheque_data.get('validation_reason', 'Mismatch')
            consistency_text = f"❌ No ({consistency_reason})"
        report += f"| Amounts Consistent?   | {consistency_text} |\n"
        
        # === NEW: Signature Comparison Result Row ===
        signature_result_text = get_signature_check_result(final_state)
        report += f"| Signature Match?      | {signature_result_text} |\n"
        
        report += "\n"

    report += f"**Processing Feedback:**\n```\n{feedback}\n```\n\n"
    
    if final_state.get("audit_trail"):
        summary = final_state["audit_trail"].generate_llm_summary_report(text_llm)
        report += "### AI-Generated Audit Summary\n\n"
        report += summary
    else:
        report += "### Audit trail could not be generated."

    return cheque_image_np, report

# --- Gradio Interface Definition ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Multi-Agent Cheque Processing System")
    gr.Markdown("Upload a cheque image to begin the automated extraction and fraud detection process.")
    
    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="numpy", label="Upload Cheque")
            submit_button = gr.Button("Process Cheque", variant="primary")
        with gr.Column(scale=1):
            image_output = gr.Image(type="numpy", label="Cheque Preview")
            report_output = gr.Markdown(label="Processing Report")
            
    submit_button.click(
        fn=process_cheque_with_ui,
        inputs=[image_input],
        outputs=[image_output, report_output]
    )
    
    gr.Examples(
        examples=[os.path.join(project_root, "dbs_cheque.png")],
        inputs=[image_input],
    )

if __name__ == "__main__":
    demo.launch()