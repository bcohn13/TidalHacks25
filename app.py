from flask import Flask, render_template, request, redirect, url_for, jsonify, send_file
import json
import os
import io
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for Flask
import matplotlib.pyplot as plt
from utils.extract_professor import extract_professor_name, load_professor_names
from redditAnalysisScraper import analyze_professor_sentiment
from pdf_comparison_analysis import compare_pdfs
from visualiation import visualize_reddit_sentiments, ai_viewer_test_difficulty

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

@app.route("/analyze_professor", methods=["POST"])
def analyze_professor():
    data = request.get_json()
    professor_name = data.get("professor_name")
    
    if not professor_name:
        return jsonify({"error": "No professor name provided"}), 400
    
    # Analyze professor sentiment
    results = analyze_professor_sentiment(professor_name)
    
    # Generate visualizations and save them to memory
    if results['all_comments']:
        # Use BytesIO to store image data
        sentiment_breakdown = io.BytesIO()
        sentiment_distribution = io.BytesIO()
        
        # Create visualizations
        visualize_reddit_sentiments(professor_name, results, sentiment_breakdown, sentiment_distribution)
        
        # Get sentiment statistics for the summary
        sentiment_stats = {
            'comment_count': results['overall_sentiment']['comment_count'],
            'interpretation': results['overall_sentiment']['interpretation'],
            'average_score': results['overall_sentiment'].get('average_score', 0),
        }
        
        # Get AI analysis
        ai_analysis = results['overall_sentiment'].get('ai_analysis', 'No AI analysis available')
        
        return jsonify({
            'success': True,
            'has_data': True,
            'sentiment_stats': sentiment_stats,
            'ai_analysis': ai_analysis
        })
    else:
        return jsonify({
            'success': True,
            'has_data': False,
            'message': f"No comments found for {professor_name}"
        })

@app.route("/compare_pdfs", methods=["POST"])
def analyze_pdfs():
    files = request.files.getlist("pdfs")
    
    if len(files) < 2:
        return jsonify({"error": "At least two PDF files are required for comparison"}), 400
    
    # Save uploaded files temporarily
    temp_paths = []
    for file in files:
        temp_path = os.path.join(os.environ.get('TEMP_DIR', '/tmp'), file.filename)
        file.save(temp_path)
        temp_paths.append(temp_path)
    
    # Compare PDFs
    comparison_result = compare_pdfs(temp_paths)
    
    # Clean up temporary files
    for path in temp_paths:
        try:
            os.remove(path)
        except:
            pass
    
    return jsonify({
        'success': True,
        'comparison_result': comparison_result
    })

@app.route("/get_visualization/<image_type>")
def get_visualization(image_type):
    professor_name = request.args.get("professor")
    
    if not professor_name:
        return "Professor name is required", 400
    
    # Generate visualizations on-the-fly
    results = analyze_professor_sentiment(professor_name)
    
    if not results['all_comments']:
        return "No data available for this professor", 404
    
    img_data = io.BytesIO()
    
    if image_type == "breakdown":
        visualize_reddit_sentiments(professor_name, results, img_data, None)
    elif image_type == "distribution":
        visualize_reddit_sentiments(professor_name, results, None, img_data)
    else:
        return "Invalid visualization type", 400
    
    img_data.seek(0)
    return send_file(img_data, mimetype='image/png')

@app.route("/comprehensive_summary", methods=["POST"])
def comprehensive_summary():
    data = request.get_json()
    professor_name = data.get("professor_name")
    pdf_comparison = data.get("pdf_comparison", None)
    
    if not professor_name:
        return jsonify({"error": "Professor name is required"}), 400
    
    # Get professor sentiment analysis
    results = analyze_professor_sentiment(professor_name)
    professor_data = {
        'has_data': len(results['all_comments']) > 0,
        'sentiment_stats': {},
        'ai_analysis': ""
    }
    
    if professor_data['has_data']:
        professor_data['sentiment_stats'] = {
            'comment_count': results['overall_sentiment']['comment_count'],
            'interpretation': results['overall_sentiment']['interpretation'],
            'average_score': results['overall_sentiment'].get('average_score', 0)
        }
        professor_data['ai_analysis'] = results['overall_sentiment'].get('ai_analysis', '')
    
    # Get negative comments for test difficulty comparison
    negative_comments = []
    if professor_data['has_data']:
        negative_comments = [c['text'] for c in results.get('all_comments', []) if c['sentiment'] < 0]
    
    # Generate test difficulty vs reality analysis if we have both PDFs and negative comments
    test_reality_analysis = ""
    if pdf_comparison and negative_comments:
        test_reality_analysis = ai_viewer_test_difficulty(professor_name, negative_comments)
    
    return jsonify({
        'success': True,
        'professor_data': professor_data,
        'pdf_comparison': pdf_comparison,
        'test_reality_analysis': test_reality_analysis,
        'has_complete_data': professor_data['has_data'] and bool(pdf_comparison)
    })

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
