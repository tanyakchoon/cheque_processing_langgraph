from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from ..utils import parse_json_from_response

def llm_predict_lien_necessity(cheque_data: dict, llm: ChatGoogleGenerativeAI) -> (bool, str):
    # ... (prompt setup is the same) ...
    chain = prompt | llm
    try:
        response = chain.invoke(cheque_data)
        result = parse_json_from_response(response.content)

        if result:
            standardized_result = {k.lower(): v for k, v in result.items()}
            if "predict_lien" in standardized_result and "reason" in standardized_result:
                return standardized_result["predict_lien"], standardized_result["reason"]

        print(f"ERROR: Model returned incomplete/unparseable JSON from lien prediction. Raw: {response.content}")
        return False, "Analysis failed due to model response error."

    except Exception as e:
        print(f"ERROR: API call failed during lien prediction: {e}")
        return False, "Analysis failed due to API error."