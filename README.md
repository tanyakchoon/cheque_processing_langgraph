# Multi-Agent Cheque Processing System with LangGraph and Gemini

This project demonstrates a multi-agent system for end-to-end cheque processing, built using the LangGraph framework. It leverages Google's Gemini 2.5 Pro model for complex tasks like fraud detection, multimodal data extraction, and audit summarization.

## Features

- **Gemini-Powered Image Quality Check**: Uses Gemini Pro Vision to assess image readability before processing.
- **Advanced Data Extraction**: Employs Gemini to extract structured data (payee, amount, date) directly from the cheque image using its multimodal and JSON output capabilities.
- **Multi-faceted Fraud Detection**:
  - **Tampering**: Gemini "looks" for visual signs of alteration in the cheque image.
  - **Behavioral**: Gemini analyzes the transaction against historical data for anomalies.
- **Predictive Lien Management**: A risk-analyst persona powered by Gemini predicts the necessity of placing a lien on an account.
- **AI-Generated Audit Trail**: At the end of the process, Gemini generates a human-readable summary of the cheque's entire journey.

## Setup and Installation

### 1. Prerequisites
- Python 3.9+
- A Google API Key

### 2. Get Your Google API Key
- Go to [Google AI Studio](https://aistudio.google.com/).
- Click on "**Get API key**" and create one.

### 3. Configure Your Environment
Create a file named `.env` in the root of the project directory. Add your Google API key to this file: