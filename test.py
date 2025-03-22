import google.generativeai as genai
genai.configure(api_key="AIzaSyAIelnAg8MwqCDQ1LwYwvcW_aJon29vE2g")

model = genai.GenerativeModel("models/gemini-pro")
response = model.generate_content("Hello!")
print(response.text)
