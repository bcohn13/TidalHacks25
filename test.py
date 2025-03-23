import os 
import google.generativeai as genai
genai.configure(api_key=os.environ.get("api_key"))

model = genai.GenerativeModel("models/gemini-pro")
response = model.generate_content("Hello!")
print(response.text)
