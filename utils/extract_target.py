import os
import json
import google.generativeai as genai

# 🔐 Setup Gemini
genai.configure(api_key=os.environ.get("api_key"))
model = genai.GenerativeModel("models/gemini-pro")

def load_data():
    with open("data/csce_professors.json") as f:
        professors = json.load(f)
    with open("data/courses.json") as f:
        courses = json.load(f)
    course_list = courses.get("CSCE", [])
    return professors, course_list

def extract_target(prompt, professors, courses):
    professor_list = "\n".join(professors)
    course_list = "\n".join([f"{c['code']} - {c['title']}" for c in courses])

    full_prompt = f"""
You are helping a TAMU CS student figure out whether their query is about a professor or a course.

Student Query:
"{prompt}"

Here's a list of professors:
{professor_list}

Here's a list of CSCE courses:
{course_list}

If the query is about a professor, respond with:
"PROFESSOR: <full professor name>"

If it's about a course, respond with:
"COURSE: <course code>"

If you can't tell, respond with:
"UNKNOWN"
"""

    response = model.generate_content(full_prompt)
    return response.text.strip()
