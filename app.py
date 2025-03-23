from flask import Flask, render_template, request, redirect, url_for, jsonify, send_file
import json
import os
import io
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for Flask
import google.generativeai as genai
from utils.extract_professor import extract_professor_name, load_professor_names
from utils.extract_course import extract_course_name, load_course_names
from redditAnalysisScraper import analyze_professor_sentiment
from visualiation import visualize_reddit_sentiments, ai_viewer_test_difficulty
from pdf_comparison_analysis import compare_pdfs

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

@app.route("/professor_visualization/<professor_name>")
def professor_visualization(professor_name):
    """Generate and show visualization for a specific professor"""
    professor_data = {}
    
    # Get sentiment analysis results
    results = analyze_professor_sentiment(professor_name)
    
    # Check if we have data in either comments or submissions
    has_comments = bool(results.get('all_comments'))
    has_submissions = bool(results.get('all_submissions'))
    
    if has_comments or has_submissions:
        # Generate visualization images
        sentiment_breakdown = io.BytesIO()
        sentiment_distribution = io.BytesIO()
        
        viz_stats = visualize_reddit_sentiments(
            professor_name, 
            results, 
            sentiment_breakdown,
            sentiment_distribution
        )
        
        sentiment_breakdown.seek(0)
        sentiment_distribution.seek(0)
        
        # Extract negative comments from BOTH comments and submissions
        negative_comments = [c['text'] for c in results.get('all_comments', []) if c['sentiment'] < 0]
        negative_submissions = [s['text'] for s in results.get('all_submissions', []) if s['sentiment'] < 0]
        all_negative_content = negative_comments + negative_submissions
        
        ai_analysis = results['overall_sentiment'].get('ai_analysis', '')
        
        professor_data = {
            'has_data': True,
            'stats': viz_stats,
            'sentiment_breakdown_url': url_for('get_visualization_image', professor_name=professor_name, image_type='breakdown'),
            'sentiment_distribution_url': url_for('get_visualization_image', professor_name=professor_name, image_type='distribution'),
            'ai_analysis': ai_analysis,
            'overall_sentiment': results['overall_sentiment'].get('interpretation', 'Neutral'),
            'comment_count': len(results.get('all_comments', [])),
            'submission_count': len(results.get('all_submissions', [])),
            'total_count': len(results.get('all_comments', [])) + len(results.get('all_submissions', []))
        }
        
        # Get PDF comparison analysis if available
        try:
            pdf_analysis = analyze_user_pdf()
            professor_data['pdf_analysis'] = pdf_analysis
            
            # Generate combined AI analysis if we have negative content
            if all_negative_content:
                test_difficulty_analysis = ai_viewer_test_difficulty(professor_name, all_negative_content)
                professor_data['test_difficulty_analysis'] = test_difficulty_analysis
        except Exception as e:
            professor_data['pdf_error'] = str(e)
    else:
        professor_data = {
            'has_data': False,
            'message': f"No Reddit content found for Professor {professor_name}"
        }
    
    return render_template('professor_visualization.html', 
                           professor_name=professor_name, 
                           data=professor_data,
                           courses=course_list)

@app.route("/get_visualization_image/<professor_name>/<image_type>")
def get_visualization_image(professor_name, image_type):
    """Serve visualization images"""
    results = analyze_professor_sentiment(professor_name)
    
    # Check for content in either comments or submissions
    has_data = bool(results.get('all_comments')) or bool(results.get('all_submissions'))
    
    if not has_data:
        return "No data available for this professor", 404
    
    img_data = io.BytesIO()
    
    if image_type == 'breakdown':
        visualize_reddit_sentiments(professor_name, results, img_data, None)
    elif image_type == 'distribution':
        visualize_reddit_sentiments(professor_name, results, None, img_data)
    else:
        return "Invalid image type", 400
    
    img_data.seek(0)
    return send_file(img_data, mimetype='image/png')

@app.route("/summary")
def summary_page():
    query = request.args.get("query", "")
    matched_type = request.args.get("matched_type", "none")
    match = request.args.get("match", "Unknown")

    if matched_type == "professor":
        # Check if we have Reddit sentiment data
        results = analyze_professor_sentiment(match)
        has_reddit_data = bool(results.get('all_comments')) or bool(results.get('all_submissions'))
        
        prompt = f"""
The user asked: "{query}"

Generate a helpful and friendly overview about TAMU CSCE professor: {match}

Include teaching style, what classes they're known for, and any unique insight students might care about.
"""
        try:
            response = model.generate_content(prompt)
            summary = response.text.strip()
            
            # Add visualization link if data exists
            if has_reddit_data:
                summary += f"\n\n[View Reddit Sentiment Analysis](/professor_visualization/{match})"
        except Exception as e:
            summary = f"⚠️ Gemini API error: {e}"
            
            # If Gemini fails but we have Reddit data, redirect to visualization
            if has_reddit_data:
                return redirect(url_for('professor_visualization', professor_name=match))

    elif matched_type == "course":
        prompt = f"""
The user asked: "{query}"

Generate a TAMU CSCE course summary for: {match}

Explain what the course is about, what kind of student would enjoy it, and the key skills taught.
"""
        try:
            response = model.generate_content(prompt)
            summary = response.text.strip()
        except Exception as e:
            summary = f"⚠️ Gemini API error: {e}"

    else:
        return render_template("summary.html", query=query, summary="❌ No course or professor could be matched.", courses=course_list)

    return render_template("summary.html", query=query, summary=summary, courses=course_list)

@app.route("/course/<course_code>")
def course_page(course_code):
    course = next((c for c in course_list if c["code"] == course_code), None)
    if not course:
        return f"Course {course_code} not found", 404
    return render_template("course.html", course=course, courses=course_list)

if __name__ == "__main__":
    app.run(debug=True)
