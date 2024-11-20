from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import google.generativeai as genai
import os
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

# Define input data structure
class TextInput(BaseModel):
    text: str

def detect_satire(text: str) -> str:
    """
    Use Gemini to determine if the text is satire.
    """
    try:
        prompt = (
            "You are an expert at understanding humor and absurdity. "
            "Analyze the following text and determine if it is satire. "
            "Respond with 'Likely Satire' or 'Likely Not Satire': {text}"
        )
        
        response = gemini_model.generate_content(prompt)
        return response.text.strip() if response and response.text else "Error: No response"
    except Exception as e:
        return f"Error: {str(e)}"   

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
    satire_result = detect_satire(truncated_text)

    return {
        "classification": {
            "fake_news_result": "Fake" if fake_prob > real_prob else "Real",
            "probabilities": {"Fake": fake_prob, "Real": real_prob},
        },
        "satire_result": satire_result
    }




@app.get("/")
def read_root():
    return {"message": "Hello, world!"}
