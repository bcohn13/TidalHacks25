from flask import Flask, render_template, request, redirect, url_for
import json
import os
import google.generativeai as genai
from utils.extract_professor import extract_professor_name, load_professor_names
from utils.extract_course import extract_course_name, load_course_names

# Setup Gemini
genai.configure(api_key=os.environ.get("api_key"))
model = genai.GenerativeModel("gemini-2.0-flash")

app = Flask(__name__, template_folder="frontend/templates", static_folder="frontend/static")

# Load professors & courses
professor_names = load_professor_names("data/csce_professors.json")
course_list = load_course_names("data/courses.json")

@app.route("/")
def home():
    return render_template("index.html", courses=course_list)

@app.route("/search", methods=["POST"])
def search():
    query = request.form["query"]

    matched_prof = extract_professor_name(query, professor_names)
    matched_course = extract_course_name(query, course_list)

    if matched_prof and matched_prof != "Unknown":
        return redirect(url_for("summary_page", matched_type="professor", match=matched_prof, query=query))

    elif matched_course and matched_course != "Unknown":
        return redirect(url_for("summary_page", matched_type="course", match=matched_course, query=query))

    else:
        return redirect(url_for("summary_page", matched_type="none", match="Unknown", query=query))

@app.route("/summary")
def summary_page():
    query = request.args.get("query", "")
    matched_type = request.args.get("matched_type", "none")
    match = request.args.get("match", "Unknown")

    if matched_type == "professor":
        prompt = f"""
The user asked: "{query}"

Generate a helpful and friendly overview about TAMU CSCE professor: {match}

Include teaching style, what classes they’re known for, and any unique insight students might care about.
"""

    elif matched_type == "course":
        prompt = f"""
The user asked: "{query}"

Generate a TAMU CSCE course summary for: {match}

Explain what the course is about, what kind of student would enjoy it, and the key skills taught.
"""

    else:
        return render_template("summary.html", query=query, summary="❌ No course or professor could be matched.", courses=course_list)

    try:
        response = model.generate_content(prompt)
        summary = response.text.strip()
    except Exception as e:
        summary = f"⚠️ Gemini API error: {e}"

    return render_template("summary.html", query=query, summary=summary, courses=course_list)


@app.route("/course/<course_code>")
def course_page(course_code):
    course = next((c for c in course_list if c["code"] == course_code), None)
    if not course:
        return f"Course {course_code} not found", 404
    return render_template("course.html", course=course, courses=course_list)

if __name__ == "__main__":
    app.run(debug=True)
