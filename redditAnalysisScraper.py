import praw
import re
from collections import Counter
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import os
from google import genai
from google.genai import types
import textwrap

client = genai.Client(api_key = os.environ.get('api_key'))

# Download necessary NLTK data
try:
    nltk.download('vader_lexicon', quiet=True)
except:
    pass

reddit = praw.Reddit(
    client_id="pxDKTjqMU6pQXNfkhXWGkQ",
    client_secret="KX--8WzjC-y2bU9xc6y-vonLAwLUhQ",  # Fixed client secret (removed trailing 'T')
    redirect_uri = "http://localhost:1337",
    user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:137.0) Gecko/20100101 Firefox/137.0",
    check_for_async=False  # Disable async check for synchronous code
)

# Verify we're in read-only mode which doesn't require username/password
def analyze_with_gemini(comments, submissions, professor_name):
    """Use Google's Gemini API to analyze and summarize professor comments"""
    try:
        # Prepare the comments for analysis
        comments_text = "\n\n".join([f"Comment {i+1}: {c['text']}" for i, c in enumerate(comments)])
        submissions_text = "\n\n".join([f"Comment {i+1}: {c['text']}" for i, c in enumerate(submissions)])
        
        prompt = f"""
        You are an academic sentiment analyst. Review these Reddit comments about Professor {professor_name} and provide:
        
        1. A comprehensive summary of student opinions
        2. Key strengths mentioned by students
        3. Areas for improvement or concerns raised
        4. Teaching style characteristics described
        5. Course difficulty and workload assessment
        6. Overall sentiment across the comments
        
        
        Format your analysis in clear sections with bullet points where appropriate.
        Keep your analysis factual and balanced, based only on the comments provided.
        If there are contradictory views, highlight these differences.
        """
        # Configure the model
        response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=[
                    comments_text,
                    submissions_text,
                    prompt      # Add the text prompt
                ])
        
        # Generate content
        print("Analyzing comments with Gemini AI...")        
        # Return the formatted response
        return response.text
        
    except Exception as e:
        print(f"Error using Gemini API: {e}")
        return "AI-powered analysis unavailable due to an error. Falling back to standard analysis."

# Add this new function for Gemini-based sentiment
def gemini_sentiment_score(comment_text):
    """
    Use Gemini API to determine sentiment score(s) as an integer.
    
    Comment_text is provided, returns the sentiment score for the comment.
    
    Assumes that a forced mention in a professor's section is positive.
    """
    try:
        # Analyze the comment text
        prompt_comment = (
            f"Determine the sentiment of the following comment as an integer score. Return only the integer value.\n"
            f"Comment: {comment_text}"
        )
        response_comment = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[prompt_comment]
        )
        match_comment = re.search(r'(-?\d+)', response_comment.text)
        score_comment = int(match_comment.group(1)) if match_comment else 0

        return score_comment
    except Exception as e:
        print(f"Gemini sentiment error: {e}")
        return 0

def analyze_professor_sentiment(professor_name, subreddits=None, post_limit=25, comment_limit=100):
    """Analyze sentiment about a professor across Reddit posts and comments."""
    # Initialize sentiment analyzer (kept for fallback if needed)
    
    # Default to education-related subreddits if none specified
    if subreddits is None:
        subreddits = [ 
            'aggies'
        ]
    
    # Define functions to check for professor mentions and analyze submissions
    all_comments = []
    useful_posts = []
    subreddit_data = {}
    post_data = []
    
    # Search each subreddit for mentions of the professor
    for subreddit_name in subreddits:
        try:
            subreddit = reddit.subreddit(subreddit_name)
            search_results = list(subreddit.search(professor_name, limit=post_limit))
            if search_results:
                subreddit_data[subreddit_name] = {'posts_found': len(search_results), 'comments': []}
                
                # Process each post
                for post in search_results:
                    post_sentiment = gemini_sentiment_score(post.title + " " + post.selftext)
                    post_info = {
                        'title': post.title,
                        'text': post.title + " " + post.selftext,
                        'url': post.url,
                        'sentiment': post_sentiment,
                        'comments': []
                    }
                    useful_posts.append(post_info)
                    
                    # Get comments for this post
                    post.comments.replace_more(limit=0)
                    post_comments = list(post.comments.list())[:comment_limit]
                    
                    for comment in post_comments:
                        # Skip if comment has no body or is deleted
                        if not hasattr(comment, 'body') or comment.body == '[deleted]':
                            continue
                            
                        # Check if comment mentions professor name
                        if re.search(professor_name, comment.body, re.IGNORECASE):
                            # Use Gemini to determine sentiment as an integer
                            gemini_score = gemini_sentiment_score(comment.body)
                            comment_info = {
                                'text': comment.body,
                                'author': str(comment.author),
                                'score': comment.score,
                                'sentiment': gemini_score
                            }
                            
                            post_info['comments'].append(comment_info)
                            subreddit_data[subreddit_name]['comments'].append(comment_info)
                            all_comments.append(comment_info)
                    
                    # Only add posts that have relevant comments
                    if post_info['comments']:
                        post_data.append(post_info)
                
        except Exception as e:
            print(f"Error processing subreddit {subreddit_name}: {str(e)}")
    
    # Calculate overall sentiment - adjusted for integer scores
    if len(all_comments) | len(useful_posts):
        # Since sentiment is now an integer, we can directly average them
        sentiment_scores = [c['sentiment'] for c in all_comments]
        submission_sentiment_scores = [c['sentiment'] for c in useful_posts]
        
        overall_sentiment = {
            'average_score': (sum(sentiment_scores) + sum(submission_sentiment_scores)) / (len(sentiment_scores) + len(submission_sentiment_scores)),
            'comment_count': len(sentiment_scores) + len(submission_sentiment_scores),
        }
        
        # Interpret the sentiment
        if overall_sentiment['average_score'] > 0:
            overall_sentiment['interpretation'] = 'Positive'
        elif overall_sentiment['average_score'] < 0:
            overall_sentiment['interpretation'] = 'Negative'
        else:
            overall_sentiment['interpretation'] = 'Neutral'
        
        # Get AI-powered analysis if we have enough comments
        if len(all_comments) + len(useful_posts) >= 3:
            ai_analysis = analyze_with_gemini(all_comments, useful_posts, professor_name)
            overall_sentiment['ai_analysis'] = ai_analysis
        else:
            overall_sentiment['ai_analysis'] = "Not enough comments for AI-powered analysis."
    else:
        overall_sentiment = {
            'interpretation': 'No data found',
            'comment_count': 0
        }
    
    return {
        'professor_name': professor_name,
        'overall_sentiment': overall_sentiment,
        'posts': post_data,
        'subreddits': subreddit_data,
        'all_comments': all_comments,
        'all_submissions' : useful_posts
    }

def print_sentiment_results(results):
    """Print a summary of the sentiment analysis results without individual comments."""
    professor = results['professor_name']
    sentiment = results['overall_sentiment']
    
    print(f"\n{'='*80}")
    print(f"SENTIMENT ANALYSIS FOR: {professor}")
    print(f"{'='*80}")
    
    if sentiment['comment_count'] == 0:
        print(f"No comments found mentioning {professor}.")
        return
    
    # Display only the statistical summary
    print(f"Comments analyzed: {sentiment['comment_count']}")
    print(f"Overall sentiment: {sentiment['interpretation']}")
    print(f"Average Score: {sentiment.get('average_score', 0):.2f}")
    
    # Calculate additional statistics
    if results['all_comments']:
        scores = [c['sentiment'] for c in results['all_comments']]
        print(f"Highest sentiment score: {max(scores)}")
        print(f"Lowest sentiment score: {min(scores)}")
        print(f"Sentiment score range: {max(scores) - min(scores)}")
    
    # Print AI analysis if available (keeping this as it provides a summary without showing actual comments)
    if 'ai_analysis' in sentiment and sentiment['ai_analysis']:
        print(f"\n{'='*80}")
        print(f"AI-POWERED ANALYSIS")
        print(f"{'='*80}")
        print(textwrap.fill(sentiment['ai_analysis'], width=80))

def analyze_professor_from_input():
    """Get professor name from user and run analysis"""
    professor_name = input("Enter professor name to analyze: ")
    print(f"Searching Reddit for mentions of {professor_name}...")
    results = analyze_professor_sentiment(professor_name)
    print_sentiment_results(results)
    return results

# Only run this when the script is executed directly
if __name__ == "__main__":
    print("Choose an option:")
    print("Analyze professor sentiment across general Reddit")   
    # Default to the original analysis
    results = analyze_professor_from_input()