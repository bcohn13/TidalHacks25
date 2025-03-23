import os
import google.generativeai as genai

genai.configure(api_key=os.environ.get("api_key"))
model = genai.GenerativeModel("gemini-2.0-flash")

response = model.generate_content("Hello Gemini! Just say hi back.")
print(response.text)
