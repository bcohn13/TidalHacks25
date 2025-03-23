import praw
import re
from collections import Counter
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import os
from google import genai
from google.genai import types
import textwrap

# Download necessary NLTK data
try:
    nltk.download('vader_lexicon', quiet=True)
except:
    pass

# Set up Reddit API credentials
# You need to create a Reddit app at https://www.reddit.com/prefs/apps
# and fill in these details
reddit = praw.Reddit(
    client_id="pxDKTjqMU6pQXNfkhXWGkQ",
    client_secret="KX--8WzjC-y2bU9xc6y-vonLAwLUhQ",  # Fixed client secret (removed trailing 'T')
    redirect_uri = "http://localhost:1337",
    user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:137.0) Gecko/20100101 Firefox/137.0",
    check_for_async=False  # Disable async check for synchronous code
)

# Verify we're in read-only mode which doesn't require username/password
print(f"Running in read-only mode: {reddit.read_only}")

def analyze_with_gemini(comments, professor_name):
    """Use Google's Gemini API to analyze and summarize professor comments"""
    if not comments:
        return "No comments available for analysis."
        
    try:
        # Prepare the comments for analysis
        comments_text = "\n\n".join([f"Comment {i+1}: {c['text']}" for i, c in enumerate(comments)])
        
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
        client = genai.Client(api_key = os.environ.get('api_key'))
        # Configure the model
        response = response = client.models.generate_content(
                model="gemini-1.5-flash",
                contents=[
                    comments_text,
                    prompt      # Add the text prompt
                ])
        
        # Generate content
        print("Analyzing comments with Gemini AI...")        
        # Return the formatted response
        return response.text
        
    except Exception as e:
        print(f"Error using Gemini API: {e}")
        return "AI-powered analysis unavailable due to an error. Falling back to standard analysis."

def analyze_professor_sentiment(professor_name, subreddits=None, post_limit=25, comment_limit=100):
    """Analyze sentiment about a professor across Reddit posts and comments."""
    # Initialize sentiment analyzer
    analyzer = SentimentIntensityAnalyzer()
    
    # Default to education-related subreddits if none specified
    if subreddits is None:
        subreddits = [
            'college', 'academia', 'professors', 'AskAcademia', 
            'GradSchool', 'highereducation', 'university', 'aggies'
        ]
    
    all_comments = []
    subreddit_data = {}
    post_data = []
    
    # Search each subreddit for mentions of the professor
    for subreddit_name in subreddits:
        try:
            print(f"Searching r/{subreddit_name} for mentions of '{professor_name}'...")
            subreddit = reddit.subreddit(subreddit_name)
            search_results = list(subreddit.search(professor_name, limit=post_limit))
            
            if search_results:
                subreddit_data[subreddit_name] = {'posts_found': len(search_results), 'comments': []}
                
                # Process each post
                for post in search_results:
                    post_sentiment = analyzer.polarity_scores(post.title + " " + post.selftext)
                    post_info = {
                        'title': post.title,
                        'url': post.url,
                        'score': post.score,
                        'sentiment': post_sentiment,
                        'comments': []
                    }
                    
                    # Get comments for this post
                    post.comments.replace_more(limit=0)
                    post_comments = list(post.comments.list())[:comment_limit]
                    
                    for comment in post_comments:
                        # Skip if comment has no body or is deleted
                        if not hasattr(comment, 'body') or comment.body == '[deleted]':
                            continue
                            
                        # Check if comment mentions professor name
                        if re.search(professor_name, comment.body, re.IGNORECASE):
                            comment_sentiment = analyzer.polarity_scores(comment.body)
                            comment_info = {
                                'text': comment.body,
                                'author': str(comment.author),
                                'score': comment.score,
                                'sentiment': comment_sentiment
                            }
                            
                            post_info['comments'].append(comment_info)
                            subreddit_data[subreddit_name]['comments'].append(comment_info)
                            all_comments.append(comment_info)
                    
                    # Only add posts that have relevant comments
                    if post_info['comments']:
                        post_data.append(post_info)
                
        except Exception as e:
            print(f"Error processing subreddit {subreddit_name}: {str(e)}")
    
    # Calculate overall sentiment
    if all_comments:
        compound_scores = [c['sentiment']['compound'] for c in all_comments]
        positive_scores = [c['sentiment']['pos'] for c in all_comments]
        negative_scores = [c['sentiment']['neg'] for c in all_comments]
        
        overall_sentiment = {
            'compound_avg': sum(compound_scores) / len(compound_scores),
            'positive_avg': sum(positive_scores) / len(positive_scores),
            'negative_avg': sum(negative_scores) / len(negative_scores),
            'comment_count': len(all_comments),
        }
        
        # Interpret the sentiment
        if overall_sentiment['compound_avg'] >= 0.05:
            overall_sentiment['interpretation'] = 'Positive'
        elif overall_sentiment['compound_avg'] <= -0.05:
            overall_sentiment['interpretation'] = 'Negative'
        else:
            overall_sentiment['interpretation'] = 'Neutral'
        
        # Get AI-powered analysis if we have enough comments
        if len(all_comments) >= 3:
            ai_analysis = analyze_with_gemini(all_comments, professor_name)
            overall_sentiment['ai_analysis'] = ai_analysis
        else:
            overall_sentiment['ai_analysis'] = "Not enough comments for AI-powered analysis or Gemini API not available."
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
        'all_comments': all_comments
    }

def print_sentiment_results(results):
    """Print a summary of the sentiment analysis results."""
    professor = results['professor_name']
    sentiment = results['overall_sentiment']
    
    print(f"\n{'='*80}")
    print(f"SENTIMENT ANALYSIS FOR: {professor}")
    print(f"{'='*80}")
    
    if sentiment['comment_count'] == 0:
        print(f"No comments found mentioning {professor}.")
        return
    
    # Basic sentiment stats
    print(f"Comments analyzed: {sentiment['comment_count']}")
    print(f"Overall sentiment: {sentiment['interpretation']}")
    print(f"Compound Score: {sentiment['compound_avg']:.3f}")
    print(f"Positive Score: {sentiment['positive_avg']:.3f}")
    print(f"Negative Score: {sentiment['negative_avg']:.3f}")
    
    # Print AI analysis if available
    if 'ai_analysis' in sentiment and sentiment['ai_analysis']:
        print(f"\n{'='*80}")
        print(f"AI-POWERED ANALYSIS")
        print(f"{'='*80}")
        print(textwrap.fill(sentiment['ai_analysis'], width=80))
    
    # Print sample comments
    print(f"\n{'='*80}")
    print(f"SAMPLE COMMENTS")
    print(f"{'='*80}")
    
    if results['all_comments']:
        sorted_comments = sorted(results['all_comments'], 
                               key=lambda x: x['sentiment']['compound'], 
                               reverse=True)
        
        print("\nMost positive comments:")
        for comment in sorted_comments[:3]:
            print(f"Score: {comment['sentiment']['compound']:.3f}")
            print(f"Comment: {comment['text'][:100]}...")
            print("-" * 40)
        
        print("\nMost negative comments:")
        for comment in sorted_comments[-3:]:
            print(f"Score: {comment['sentiment']['compound']:.3f}")
            print(f"Comment: {comment['text'][:100]}...")
            print("-" * 40)

def analyze_professor_from_input():
    """Get professor name from user and run analysis"""
    professor_name = input("Enter professor name to analyze: ")
    print(f"Searching Reddit for mentions of {professor_name}...")
    results = analyze_professor_sentiment(professor_name)
    print_sentiment_results(results)
    return results


    # Run the analysis on the list of university subreddits
    print(f"Searching {len(university_subreddits)} university subreddits for mentions of {professor_name}...")
    
    # Initialize counters for progress reporting
    total_subreddits = len(university_subreddits)
    processed = 0
    found_mentions = 0
    all_comments = []
    
    for subreddit_name in university_subreddits:
        processed += 1
        print(f"[{processed}/{total_subreddits}] Searching r/{subreddit_name}...", end="", flush=True)
        
        try:
            # Search the subreddit
            subreddit = reddit.subreddit(subreddit_name)
            
            # Try two search approaches: professor name alone and with university name
            search_terms = [professor_name]
            if university_name:
                # Try both full name and last name with university
                last_name = professor_name.split()[-1]
                if last_name != professor_name:
                    search_terms.append(f"{last_name} {university_name}")
            
            # Search with each term
            found_posts = False
            for term in search_terms:
                search_results = list(subreddit.search(term, limit=post_limit))
                if search_results:
                    found_posts = True
                    found_mentions += 1
                    print(f" Found {len(search_results)} posts mentioning '{term}'")
                    
                    # Process these posts
                    for post in search_results:
                        # Get comments
                        post.comments.replace_more(limit=0)
                        
                        for comment in list(post.comments.list())[:comment_limit]:
                            # Skip if comment has no body or is deleted
                            if not hasattr(comment, 'body') or comment.body == '[deleted]':
                                continue
                                
                            # Check if comment mentions professor
                            if re.search(professor_name, comment.body, re.IGNORECASE):
                                all_comments.append({
                                    'text': comment.body,
                                    'author': str(comment.author),
                                    'score': comment.score,
                                    'subreddit': subreddit_name,
                                    'sentiment': SentimentIntensityAnalyzer().polarity_scores(comment.body)
                                })
            
            if not found_posts:
                print(" No relevant posts found")
                
        except Exception as e:
            print(f" Error: {str(e)}")
    
    print(f"\nSearch complete. Found mentions in {found_mentions} out of {total_subreddits} university subreddits.")
    print(f"Total relevant comments found: {len(all_comments)}")
    
    # Analyze comments with Gemini if we have enough
    if len(all_comments) >= 3:
        ai_analysis = analyze_with_gemini(all_comments, professor_name)
        print("\n" + "="*80)
        print("AI ANALYSIS OF COMMENTS FROM UNIVERSITY SUBREDDITS")
        print("="*80)
        print(textwrap.fill(ai_analysis, width=80))
    
    # Return a summary
    return {
        "professor_name": professor_name,
        "university_name": university_name,
        "subreddits_searched": total_subreddits,
        "subreddits_with_mentions": found_mentions,
        "comments": all_comments
    }

def university_subreddit_search_from_input():
    """Interactive function to search university subreddits for professor mentions"""
    professor_name = input("Enter professor name to search for: ")
    university_name = input("Enter university name (optional, press Enter to skip): ").strip()
    
    if not university_name:
        university_name = None
        
    # Run the search
    results = search_university_subreddits(professor_name, university_name)
    
    # Display a summary
    print("\nSEARCH SUMMARY:")
    print(f"Professor: {results['professor_name']}")
    if results['university_name']:
        print(f"University: {results['university_name']}")
    print(f"Subreddits searched: {results['subreddits_searched']}")
    print(f"Subreddits with mentions: {results['subreddits_with_mentions']}")
    print(f"Total comments found: {len(results['comments'])}")

# Only run this when the script is executed directly
if __name__ == "__main__":
    print("Choose an option:")
    print("Analyze professor sentiment across general Reddit")   
    # Default to the original analysis
    results = analyze_professor_from_input()