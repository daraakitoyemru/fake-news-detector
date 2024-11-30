import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv();
gemini_key = os.getenv('API_KEY')


genai.configure(api_key=gemini_key)
model = genai.GenerativeModel("gemini-1.5-flash")

test_text = "NASA confirmed a giant claw is lifting Earth."
prompt = (
    "You are an expert in identifying satire in written text. "
    f"Analyze the following text and respond with 'Satire' or 'Not Satire':\n{test_text}"
)

response = model.generate_content(prompt)
print("Prompt:", prompt)
print("Response:", response.text)
