import matplotlib.pyplot as plt
import textwrap
import os
from google import genai
from google.genai import types
from redditAnalysisScraper import analyze_professor_sentiment
from pdf_comparison_analysis import compare_pdfs

# Initialize Gemini API client if API key is available
api_key = os.environ.get('api_key')
api_key = os.environ.get('api_key')
if api_key:
    try:
        client = genai.Client(api_key=api_key)
        gemini_available = True
    except Exception as e:
        print(f"Error initializing Gemini client: {e}")
        gemini_available = False
else:
    gemini_available = False
    print("Gemini API key not found. AI-powered analysis will not be available.")

def visualize_reddit_sentiments(professor_name):
    # Get sentiment analysis results from redditAnalysisScraper
    results = analyze_professor_sentiment(professor_name)
    comments = results.get('all_comments', [])
    
    # Count positive and negative comments
    positive = sum(1 for c in comments if c['sentiment']['compound'] >= 0.05)
    negative = sum(1 for c in comments if c['sentiment']['compound'] <= -0.05)
    neutral = len(comments) - positive - negative

    labels = ['Positive', 'Negative', 'Neutral']
    counts = [positive, negative, neutral]

    plt.figure(figsize=(6,4))
    bars = plt.bar(labels, counts, color=['green','red','gray'])
    plt.title(f"Sentiment Breakdown for {professor_name}")
    plt.ylabel("Number of Comments")
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x()+bar.get_width()/2.0, yval + 0.5, int(yval), ha='center', va='bottom')
    plt.tight_layout()
    plt.show()

def ai_viewer_test_difficulty(professor_name, negative_comments):
    """
    Use Gemini API to generate a viewer comparing test difficulty (from the test document)
    and the negative sentiments from Reddit, alongside the difference in expectations (syllabus)
    versus reality (test content).
    
    negative_comments is a list of comment texts from reddit with negative sentiment.
    """
    if not gemini_available:
        return "Gemini API not available."
    
    try:
        neg_text = "\n".join(f"- {c}" for c in negative_comments)
        prompt = f"""
        You are an academic analyst.
        
        Please generate a summary comparing:
        1. The expected course difficulty as described in the syllabus
        2. The actual test difficulty as inferred from the test document
        3. The negative sentiment evident in the Reddit comments
        
        Highlight any differences between expectation and reality, and how these differences may relate to student concerns.
        Format your answer with clear sections and bullet points.
        """
        testVsSyllabus = compare_pdfs()
        response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[
            testVsSyllabus,
            neg_text,         # Unpack the list of Parts directly into contents
            prompt      # Add the text prompt
        ])
        print(compare_pdfs())
        return response.text
    except Exception as e:
        return f"Error generating AI viewer: {e}"

def main():
    professor_name = input("Enter professor name for visualization: ")
    # Visualize Reddit sentiment breakdown
    visualize_reddit_sentiments(professor_name)
        
    results = analyze_professor_sentiment(professor_name)
    comments = results.get('all_comments', [])
    negative_comments = [c['text'] for c in comments if c['sentiment']['compound'] <= -0.05]
    
    # Generate AI viewer comparing expectations vs reality
    ai_view = ai_viewer_test_difficulty(professor_name, negative_comments)
    print("\n" + "="*80)
    print("AI GENERATED VIEWER: Test Difficulty vs Negative Comments & Syllabus-Expectation Gap")
    print("="*80)
    print(textwrap.fill(ai_view, width=80))
    
    comparison_result = compare_pdfs()
    comparison_result = compare_pdfs()
    print("\n" + "="*80)
    print("PDF COMPARISON ANALYSIS")
    print("="*80)
    print(textwrap.fill(comparison_result, width=80))

if __name__ == "__main__":
    main()