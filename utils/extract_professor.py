import os
import json
import google.generativeai as genai

# 🔐 Configure Gemini (set your API key here or in environment)
genai.configure(api_key=os.environ.get("api_key"))  # replace with your real key
print("Gemini API key loaded:", os.environ.get("api_key") is not None)


model = genai.GenerativeModel("gemini-2.0-flash")

def load_professor_names(json_path='data/csce_professors.json'):
    with open(json_path, 'r') as file:
        return json.load(file)

def extract_professor_name(prompt, professor_list):
    names = "\n".join(professor_list)
    full_prompt = f"""
You are helping a student identify which CSCE professor at Texas A&M they are referring to in a message.

Here is the query:
"{prompt}"

From this list of professors:
{names}

Respond with only the full name of the most relevant professor from the list. If you can't determine who it is, respond with "Unknown".
"""

    response = model.generate_content(full_prompt)
    return response.text.strip()
