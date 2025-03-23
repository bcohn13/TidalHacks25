import os
import json
import google.generativeai as genai

# Setup Gemini
genai.configure(api_key=os.environ.get("api_key"))
model = genai.GenerativeModel("gemini-2.0-flash")

def load_course_names(json_path='data/courses.json'):
    with open(json_path, 'r') as f:
        raw_data = json.load(f)
        # Flatten to list of course titles
        return raw_data.get("CSCE", [])

def extract_course_name(prompt, course_list):
    full_prompt = f"""
You are helping a TAMU student figure out which CSCE course their query is about.

Query: "{prompt}"

Here is a list of courses:
{chr(10).join(course_list)}

Respond with only the full course name (e.g., CSCE 411 - Artificial Intelligence), or say "Unknown".
"""

    try:
        response = model.generate_content(full_prompt)
        return response.text.strip()
    except Exception as e:
        return "Unknown"
