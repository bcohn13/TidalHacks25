from flask import Flask, render_template, request, redirect, url_for
import json
from utils.extract_professor import extract_professor_name, load_professor_names

app = Flask(__name__, template_folder="frontend/templates", static_folder="frontend/static")

# Load subject list for Gemini matching
with open("data/subjects.json") as f:
    subjects_list = json.load(f)

# Load all courses
with open("data/courses.json") as f:
    all_courses = json.load(f)
csce_courses = all_courses.get("CSCE", [])

# Load CSCE professor names for Gemini extraction
professor_names = load_professor_names("data/csce_professors.json")

@app.route("/")
def home():
    return render_template("index.html", courses=csce_courses)

@app.route("/search", methods=["POST"])
def search():
    query = request.form["query"]

    # Basic fallback subject (hardcoded since we're CSCE-only now)
    subject = "CSCE"

    # Gemini extraction
    matched_professor = extract_professor_name(query, professor_names)

    return redirect(url_for("summary_page", subject_code=subject, query=query, professor=matched_professor))

@app.route("/summary/<subject_code>")
def summary_page(subject_code):
    query = request.args.get("query", "")
    professor = request.args.get("professor", "Unknown")

    # Placeholder Gemini summary
    fake_summary = f"""
Based on your query: '{query}', here's what we found for {subject_code}:

👨‍🏫 **Recommended Professor**: {professor}  
(This is a demo summary — AI-driven customization goes here.)
"""

    return render_template("summary.html", subject=subject_code, query=query, summary=fake_summary, courses=csce_courses)

# OPTIONAL: Set up course-specific view
@app.route("/course/<course_code>")
def course_page(course_code):
    course = next((c for c in csce_courses if c["code"] == course_code), None)
    if not course:
        return f"Course {course_code} not found", 404
    return render_template("course.html", course=course, courses=csce_courses)

if __name__ == "__main__":
    app.run(debug=True)
