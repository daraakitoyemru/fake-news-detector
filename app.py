from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load the model and tokenizer from Hugging Face
model_name = "hamzab/roberta-fake-news-classification"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Set up the FastAPI app
app = FastAPI()

# Define input data structure
class TextInput(BaseModel):
    text: str

@app.post("/check")
async def check_fake_news(input_data: TextInput):
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

    # Return the results
    return {
        "result": "Fake" if fake_prob > real_prob else "Real",
        "probabilities": {"Fake": fake_prob, "Real": real_prob}
    }

@app.get("/")
def read_root():
    return {"message": "Hello, world!"}
