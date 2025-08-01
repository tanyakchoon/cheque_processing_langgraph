import logging
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AuditTrail:
    def __init__(self, cheque_id: str):
        self.cheque_id = cheque_id
        self.logs = []
        self.anomalies = []
        print(f"INFO: Started audit trail for Cheque ID: {self.cheque_id}")

    def log_step(self, step_name: str, status: str, summary: str):
        log_entry = f"Step: {step_name}, Status: {status}, Summary: {summary}"
        self.logs.append(log_entry)
        logging.info(f"[{self.cheque_id}] {log_entry}")

    def highlight_anomaly(self, anomaly_source: str, details: str):
        anomaly_entry = f"Source: {anomaly_source}, Details: {details}"
        self.anomalies.append(anomaly_entry)
        logging.warning(f"[{self.cheque_id}] ANOMALY DETECTED: {anomaly_entry}")

    def generate_llm_summary_report(self, llm: ChatGoogleGenerativeAI) -> str:
        print("INFO: Generating final audit summary with Gemini...")
        if not self.logs:
            return "No processing steps were logged."

        full_log = "\n".join(self.logs)
        anomaly_log = "\n".join(self.anomalies) if self.anomalies else "None"

        prompt_template = ChatPromptTemplate.from_template(
            """You are an AI audit assistant.
            The following logs detail the automated processing of a cheque (ID: {cheque_id}).
            Please generate a concise, human-readable summary report.

            The report should include:
            1. A brief overview of the final outcome (e.g., processed successfully, sent for manual review).
            2. A summary of any anomalies detected.
            3. A conclusion.

            Here are the detailed processing logs:
            {full_log}

            Here are the specific anomalies flagged:
            {anomaly_log}

            Generate the summary report now.
            """
        )
        chain = prompt_template | llm
        response = chain.invoke({
            "cheque_id": self.cheque_id,
            "full_log": full_log,
            "anomaly_log": anomaly_log,
        })
        return response.content