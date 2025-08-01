import gradio as gr
import numpy as np
import os
import cv2
import uuid
from pathlib import Path

from cheque_processing_langgraph.__main__ import build_graph

# --- Startup: Define root, load assets, build graph ---
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
    for log in reversed(audit_trail.logs):
        if "Signature Verification" in log:
            if "Success" in log: return f"✅ Match ({log.split('Reason: ')[-1]})"
    if hasattr(audit_trail, 'anomalies'):
        for anomaly in reversed(audit_trail.anomalies):
            if "Signature Verification" in anomaly: return f"❌ Mismatch ({anomaly.split('Reason: ')[-1]})"
    return "Not Performed"


def process_cheque_with_ui(cheque_image_np: np.ndarray):
    """ Main interface for the Gradio UI. """
    if cheque_image_np is None:
        return None, "<h2>Error</h2><p>Please upload a cheque image first.</p>"

    cheque_image_bgr = cv2.cvtColor(cheque_image_np, cv2.COLOR_RGB2BGR)
    initial_state = {"image": cheque_image_bgr, "project_root": project_root}
    print(f"Invoking graph with image data and project root...")
    final_state = app.invoke(initial_state)
    
    # --- Generate the HTML Report ---
    final_decision = final_state.get('final_decision', 'Error')
    feedback = "\n".join(final_state.get('feedback', ['An unknown error occurred.']))
    
    html_report = f"<h2>Cheque Processing Report</h2>"
    html_report += f"<p><strong>Final Decision:</strong> <code>{final_decision}</code></p>"
    
    cheque_data = final_state.get("cheque_data", {})
    if cheque_data:
        html_report += "<h3>Extracted & Validated Details</h3>"
        
        # === FIX: Generate a pure HTML table for the gr.HTML() component ===
        table_style = "width: 100%; border-collapse: collapse; text-align: left;"
        th_style = "border: 1px solid #ddd; padding: 8px; background-color: #f2f2f2;"
        td_style = "border: 1px solid #ddd; padding: 8px;"
        field_col_style = "min-width: 170px; font-weight: bold;"
        
        html_report += f"<table style='{table_style}'><thead><tr><th style='{th_style}{field_col_style}'>Field</th><th style='{th_style}'>Extracted Value</th></tr></thead><tbody>"
        
        # Populate table rows
        html_report += f"<tr><td style='{td_style}{field_col_style}'>Payee</td><td style='{td_style}'>{cheque_data.get('payee', 'N/A')}</td></tr>"
        html_report += f"<tr><td style='{td_style}{field_col_style}'>Amount (Numeric)</td><td style='{td_style}'>{cheque_data.get('amount', 'N/A')}</td></tr>"
        html_report += f"<tr><td style='{td_style}{field_col_style}'>Amount (in Words)</td><td style='{td_style}'>{cheque_data.get('amount_in_words', 'N/A')}</td></tr>"
        html_report += f"<tr><td style='{td_style}{field_col_style}'>Date</td><td style='{td_style}'>{cheque_data.get('formatted_date', 'N/A')}</td></tr>"
        html_report += f"<tr><td style='{td_style}{field_col_style}'>Payer Account No.</td><td style='{td_style}'>{cheque_data.get('payer_account_number', 'N/A')}</td></tr>"
        
        is_date_valid = cheque_data.get('is_date_valid', False)
        date_reason = cheque_data.get('date_validation_reason', 'Validation failed')
        date_valid_text = "✅ Yes" if is_date_valid else f"❌ No ({date_reason})"
        html_report += f"<tr><td style='{td_style}{field_col_style}'>Is Date Valid?</td><td style='{td_style}'>{date_valid_text}</td></tr>"
        
        is_consistent = cheque_data.get('is_amount_consistent', False)
        consistency_reason = cheque_data.get('validation_reason', 'Mismatch')
        consistency_text = "✅ Yes" if is_consistent else f"❌ No ({consistency_reason})"
        html_report += f"<tr><td style='{td_style}{field_col_style}'>Amounts Consistent?</td><td style='{td_style}'>{consistency_text}</td></tr>"
        
        signature_result_text = get_signature_check_result(final_state)
        html_report += f"<tr><td style='{td_style}{field_col_style}'>Signature Match?</td><td style='{td_style}'>{signature_result_text}</td></tr>"
        
        html_report += "</tbody></table>"

    html_report += f"<h3>Processing Feedback:</h3><pre><code>{feedback}</code></pre>"
    
    if final_state.get("audit_trail"):
        summary = final_state["audit_trail"].generate_llm_summary_report(text_llm)
        html_report += f"<h3>AI-Generated Audit Summary</h3><p>{summary.replace('/n', '<br>')}</p>"
    else:
        html_report += "<h3>Audit trail could not be generated.</h3>"

    return cheque_image_np, html_report

def clear_outputs():
    """Returns empty values to clear the output components."""
    return None, ""

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
            # === FIX: Use gr.HTML() instead of gr.Markdown() for the report ===
            report_output = gr.HTML(label="Processing Report")
            
    submit_button.click(
        fn=process_cheque_with_ui,
        inputs=[image_input],
        outputs=[image_output, report_output]
    )
    
    image_input.upload(
        fn=clear_outputs,
        inputs=[],
        outputs=[image_output, report_output]
    )
    
    gr.Examples(
        examples=[os.path.join(project_root, "dbs_cheque.png")],
        inputs=[image_input],
    )

if __name__ == "__main__":
    demo.launch()