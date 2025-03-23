import matplotlib.pyplot as plt
import textwrap
import os
from google import genai
from google.genai import types
from redditAnalysisScraper import analyze_professor_sentiment
from pdf_comparison_analysis import analyze_user_pdf

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

def visualize_reddit_sentiments(professor_name, results=None, sentiment_breakdown_file= None, sentiment_distribution_file=None):
    """
    Generate visualizations for professor sentiment analysis using combined results from
    all_comments and all_submissions (treated as equivalent sources), and save to provided file objects.
    If results are not provided, they will be fetched using analyze_professor_sentiment.
    """
    # Get sentiment analysis results if not provided
    if results is None:
        results = analyze_professor_sentiment(professor_name)
    
    # Combine comments and submissions into a single list
    all_content = results.get('all_comments', []) + results.get('all_submissions', [])
    
    if not all_content:
        print(f"No content found for {professor_name}")
        return {
            'positive': 0,
            'negative': 0,
            'neutral': 0,
            'total': 0
        }
    
    # Count positive and negative comments using integer scores
    positive = sum(1 for c in all_content if c['sentiment'] > 0)
    negative = sum(1 for c in all_content if c['sentiment'] < 0)
    neutral = sum(1 for c in all_content if c['sentiment'] == 0)

    labels = ['Positive', 'Negative', 'Neutral']
    counts = [positive, negative, neutral]

    # Create the sentiment breakdown visualization if requested
    if sentiment_breakdown_file is not None:
        # Create a larger figure for better spacing
        plt.figure(figsize=(12, 8))
        
        # Increase spacing between bars
        ax = plt.subplot(111)
        bars = ax.bar(labels, counts, color=['green', 'red', 'gray'], width=0.6)
        
        # Customize title and labels
        plt.title(f"Sentiment Breakdown for {professor_name}", fontsize=16, pad=20)
        plt.ylabel("Number of Comments/Posts", fontsize=12)
        
        # Add count labels ABOVE the bars with more padding
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + 0.3, 
                    f'{int(height)}', ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        # Adjust y-axis to leave room for labels
        max_count = max(counts) if counts else 1
        plt.ylim(0, max_count * 1.2)
        
        # Add grid for better readability
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Ensure everything fits
        plt.tight_layout()
        
        # Save to file object instead of disk file
        plt.savefig(sentiment_breakdown_file, format='png')
        plt.close()
    
    # Create the sentiment distribution visualization if requested
    if sentiment_distribution_file is not None:
        # Improve the histogram visualization as well
        sentiment_values = [c['sentiment'] for c in all_content]
        
        if sentiment_values:
            min_val = min(sentiment_values)
            max_val = max(sentiment_values)
            
            plt.figure(figsize=(12, 8))
            
            # Determine appropriate bin width based on the range of values
            range_vals = max_val - min_val
            bin_count = min(range_vals + 2, 15)  # Cap at 15 bins for readability
            
            plt.hist(sentiment_values, bins=int(bin_count) if bin_count > 0 else 3, 
                     color='skyblue', edgecolor='black', alpha=0.8)
                     
            plt.title(f"Sentiment Score Distribution for {professor_name}", fontsize=16, pad=20)
            plt.xlabel("Sentiment Score", fontsize=12)
            plt.ylabel("Frequency", fontsize=12)
            plt.grid(True, alpha=0.3)
            
            # Add mean line
            mean_val = sum(sentiment_values) / len(sentiment_values)
            plt.axvline(mean_val, color='red', linestyle='dashed', linewidth=2, 
                        label=f'Mean: {mean_val:.2f}')
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(sentiment_distribution_file, format='png')
            plt.close()
    
    return {
        'positive': positive,
        'negative': negative,
        'neutral': neutral,
        'total': len(all_content)
    }

def ai_viewer_test_difficulty(professor_name, negative_comments):
    """
    Use Gemini API to generate a viewer comparing test difficulty (from the test document)
    and the negative sentiments from Reddit, alongside the difference in expectations (syllabus)
    versus reality (test content).
    
    negative_comments is a list of comment texts from reddit with negative sentiment.
    """
    if not gemini_available:
        return "Gemini API not available."
    if not negative_comments:
        return "No comments"
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
        # Get PDF analysis
        try:
            testVsSyllabus = analyze_user_pdf()
        except Exception as e:
            print(f"Error getting PDF analysis: {e}")
            testVsSyllabus = "PDF comparison not available."
            
        response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[
            testVsSyllabus,
            neg_text,
            prompt
        ])
        return response.text
    except Exception as e:
        return f"Error generating AI viewer: {e}"

def main():
    professor_name = input("Enter professor name for visualization: ")
    
    # Visualize Reddit sentiment breakdown
    results = analyze_professor_sentiment(professor_name)
    visualize_reddit_sentiments(professor_name)
    
    # Combine comments and submissions for analysis
    comments = results.get('all_comments', [])
    submissions = results.get('all_submissions', [])
    all_content = comments + submissions
    
    # Get negative comments using integer sentiment scores (fixed the filter condition)
    negative_content = [c['text'] for c in all_content if c['sentiment'] < 0]
    
    # Generate AI viewer comparing expectations vs reality
    ai_view = ai_viewer_test_difficulty(professor_name, negative_content)
    print("\n" + "="*80)
    print("AI GENERATED VIEWER: Test Difficulty vs Negative Comments & Syllabus-Expectation Gap")
    print("="*80)
    print(textwrap.fill(ai_view, width=80))

if __name__ == "__main__":
    main()