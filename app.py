from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import google.generativeai as genai
import os
import requests
from dotenv import load_dotenv


load_dotenv();

gemini_key = os.getenv('API_KEY')
genai.configure(api_key=gemini_key)
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

# Load the model and tokenizer from Hugging Face
model_name = "hamzab/roberta-fake-news-classification"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Set up the FastAPI app
app = FastAPI()

rhetoric_prompt = """
You are an expert in rhetoric and argumentation. Analyze the following text for logical flaws, persuasive techniques, and rhetoric. Is this text likely fake or legitimate? Respond with 'Likely Fake' or 'Likely Legitimate' and explain your reasoning briefly.
Text: {text}
"""

sociologist_prompt = """
You are a sociologist analyzing societal norms, trends, and human behavior. Does the following text align with typical societal patterns, or does it seem fabricated? Respond with 'Likely Fake' or 'Likely Legitimate' and explain your reasoning briefly.
Text: {text}
"""

skeptic_prompt = """
You are a skeptic. Question the validity of the following text rigorously. Look for inconsistencies or implausible claims. Is this text likely fake or legitimate? Respond with 'Likely Fake' or 'Likely Legitimate' and explain your reasoning briefly.
Text: {text}
"""

neutral_prompt = """
You are a neutral observer with no biases. Analyze the following text objectively. Is this text likely fake or legitimate? Respond with 'Likely Fake' or 'Likely Legitimate' and explain your reasoning briefly.
Text: {text}
"""

normal_person_prompt = """
You are an average person encountering this information. Based on your intuition and general knowledge, is this text likely fake or legitimate? Respond with 'Likely Fake' or 'Likely Legitimate' and explain your reasoning briefly.
Text: {text}
"""
fact_checker_prompt = """
You are a fact-checker tasked with validating claims against existing knowledge and verifiable data. Analyze the following text for accuracy. If it includes implausible or unverifiable information, mark it as 'Likely Fake.' Otherwise, mark it as 'Likely Legitimate' and briefly explain your reasoning.
Text: {text}
"""


# Define input data structure
class TextInput(BaseModel):
    text: str

def detect_satire(text: str) -> str:
    """
    Use Gemini to determine if the text is satire.
    """
    try:
        if not text.strip():
            return "Error: No text provided"
        
        prompt = (
            "You are an expert at understanding humor and absurdity. "
            f"Analyze the following text and determine if it is satire. "
            f"Respond with 'Likely Satire' or 'Likely Not Satire': {text}"
        )
        
        response = gemini_model.generate_content(prompt)
        return response.text.strip() if response and response.text else "Error: No response"
    except Exception as e:
        return f"Error: {str(e)}" 
    
def analyze_with_role(role_prompt: str, text: str) -> str:
    """
    Analyze the text with a specific role using Gemini API.
    """
    try:
        prompt = role_prompt.format(text=text)
        print("Prompt sent to Gemini:", prompt)  # Debugging
        response = gemini_model.generate_content(prompt)
        return response.text.strip() if response and response.text else "Error: No response"
    except Exception as e:
        return f"Error: {str(e)}"


def fact_check_with_newsapi(claim: str) -> str:
    """
    Use NewsAPI to search for credible articles related to the claim.
    """
    API_KEY = os.getenv("NEWSAPI_KEY")  # Store API key in an environment variable
    API_URL = "https://newsapi.org/v2/everything"

    params = {
        "q": claim,  # Search query
        "language": "en",  # Search only English articles
        "sortBy": "relevancy",  # Rank articles by relevance to the claim
        "apiKey": API_KEY
    }

    try:
        response = requests.get(API_URL, params=params)
        if response.status_code == 200:
            articles = response.json().get("articles", [])
            
            # No articles found
            if not articles:
                return "No credible articles found. The claim might be fabricated or obscure."

            # Process articles
            sources = [article["source"]["name"] for article in articles]
            unique_sources = set(sources)

            return (
                f"Found {len(articles)} articles from credible sources: {', '.join(unique_sources)}. "
                "The claim appears to be supported by reliable reporting."
            )
        else:
            return f"NewsAPI Error: {response.status_code} - {response.text}"
    except Exception as e:
        return f"Error: {str(e)}"


def multi_role_analysis(text: str):
    """
    Analyze the text with multiple roles and determine a majority verdict.
    """
    roles = {
        "Rhetoric Expert": rhetoric_prompt,
        "Sociologist": sociologist_prompt,
        "Skeptic": skeptic_prompt,
        "Neutral Observer": neutral_prompt,
        "Normal Person": normal_person_prompt,
        "Fact-Checker":  fact_check_with_newsapi
    }
    
    results = {}
    for role_name, role_prompt in roles.items():
        if callable(role_prompt):  # If the role is a function (like Fact-Checker)
            result = role_prompt(text)
        else:  # If the role is a prompt string
            result = analyze_with_role(role_prompt, text)
        results[role_name] = result
        print(f"{role_name} Result: {result}")
    
    # Majority rule to determine verdict
    result_values = list(results.values())
    majority = max(set(result_values), key=result_values.count)
    return {
        "summary_verdict": {
            "verdict": majority.split('.')[0],  # Only keep 'Likely Fake' or 'Likely Legitimate'
            "explanation": majority  # Full explanation
        },
        "role_insights": results
    }


@app.post("/check")
async def check_fake_news(input_data: TextInput):
    max_tokens = 1000
    truncated_text = input_data.text[:max_tokens]
    print("Received text:", input_data.text)
    
    # Prepare the input text
    input_str = "<content>" + input_data.text + "<end>"
    print("Formatted input string:", input_str)  # Debugging

    input_ids = tokenizer.encode_plus(
        input_str,
        max_length=512,
        padding="max_length",
        truncation=True, 
        return_tensors="pt"
    )

    # Perform inference
    with torch.no_grad():
        output = model(input_ids["input_ids"],attention_mask=input_ids["attention_mask"])
        probabilities = torch.nn.functional.softmax(output.logits, dim=1)[0]
        fake_prob = probabilities[0].item()
        real_prob = probabilities[1].item()
    
    # Satire Detection with Gemini
    print("Truncated text for satire detection:", truncated_text)

    satire_result = detect_satire(truncated_text)

    return {
        "classification": {
            "fake_news_result": "Fake" if fake_prob > real_prob else "Real",
            "probabilities": {"Fake": fake_prob, "Real": real_prob},
        },
        "satire_result": satire_result
    }

@app.post("/analyze")
async def analyze_text(input_data: TextInput):
    """
    Endpoint to analyze text with multiple roles.
    """
    analysis_result = multi_role_analysis(input_data.text)
    return {
        "summary_verdict": analysis_result["summary_verdict"],
        "role_insights": analysis_result["role_insights"]
    }



@app.get("/")
def read_root():
    return {"message": "Hello, world!"}
