import matplotlib.pyplot as plt
import textwrap
import os
from google import genai
from google.genai import types
from redditAnalysisScraper import analyze_professor_sentiment
from pdf_comparison_analysis import compare_pdfs

# Initialize Gemini API client if API key is available
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
    
    if not comments:
        print(f"No comments found for {professor_name}")
        return
    
    # Count positive and negative comments using integer scores
    positive = sum(1 for c in comments if c['sentiment'] > 0)
    negative = sum(1 for c in comments if c['sentiment'] < 0)
    neutral = sum(1 for c in comments if c['sentiment'] == 0)

    labels = ['Positive', 'Negative', 'Neutral']
    counts = [positive, negative, neutral]

    plt.figure(figsize=(10,6))
    bars = plt.bar(labels, counts, color=['green','red','gray'])
    plt.title(f"Sentiment Breakdown for {professor_name}")
    plt.ylabel("Number of Comments")
    
    # Add count labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{int(height)}', ha='center', va='bottom')
    
    # Add percentage labels inside or above bars
    total = sum(counts)
    if total > 0:  # Avoid division by zero
        for i, bar in enumerate(bars):
            height = bar.get_height()
            percentage = (counts[i] / total) * 100
            if height > 0:  # Only add text if bar has height
                y_pos = height / 2 if height > 3 else height + 0.5
                plt.text(bar.get_x() + bar.get_width()/2., y_pos,
                        f'{percentage:.1f}%', ha='center', va='center')
    
    plt.tight_layout()
    plt.savefig('sentiment_breakdown.png')
    plt.show()
    
    # Add a score distribution histogram
    sentiment_values = [c['sentiment'] for c in comments]
    plt.figure(figsize=(10,6))
    plt.hist(sentiment_values, bins=range(min(sentiment_values)-1, max(sentiment_values)+2), 
             color='skyblue', edgecolor='black')
    plt.title(f"Sentiment Score Distribution for {professor_name}")
    plt.xlabel("Sentiment Score")
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.3)
    plt.savefig('sentiment_distribution.png')
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
        neg_text = "\n".join(negative_comments)
        prompt = f"""
        You are an academic analyst.
        
        Please generate a summary comparing:
        1. The expected course difficulty as described in the syllabus
        2. The actual test difficulty as inferred from the test document
        3. The negative sentiment evident in the Reddit comments
        
        Highlight any differences between expectation and reality, and how these differences may relate to student concerns.
        Format your answer with clear sections and don't add any recommendations on how to fix. 
        Assume this is for a student attempting to sign up for a class and first learning about the professor.
        """
        testVsSyllabus = compare_pdfs()
        response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[
            testVsSyllabus,
            neg_text,         # Unpack the list of Parts directly into contents
            prompt      # Add the text prompt
        ])
        return response.text
    except Exception as e:
        return f"Error generating AI viewer: {e}"

def main():
    professor_name = input("Enter professor name for visualization: ")
    
    # Visualize Reddit sentiment breakdown
    results = analyze_professor_sentiment(professor_name)
    visualize_reddit_sentiments(professor_name)
    
    comments = results.get('all_comments', [])
    
    # Get negative comments using integer sentiment scores
    negative_comments = [c['text'] for c in comments if c['sentiment'] < 0]
    
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