from flask import Flask, render_template, request, jsonify
from pymongo import MongoClient
import requests
import os
from dotenv import load_dotenv
from datetime import datetime
from textblob import TextBlob

load_dotenv()

app = Flask(__name__)

# MongoDB connection
client = MongoClient(os.getenv('MONGODB_URI'))
db = client.sentiment_db
tweets_collection = db.tweets

def analyze_sentiment_hf(text):
    """Analyze sentiment using Hugging Face API"""
    try:
        url = "https://api-inference.huggingface.co/models/distilbert-base-uncased-finetuned-sst-2-english"
        headers = {"Authorization": f"Bearer {os.getenv('HUGGINGFACE_API_KEY')}"}
        response = requests.post(url, headers=headers, json={"inputs": text})
        
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                sentiment_data = result[0]
                best_sentiment = max(sentiment_data, key=lambda x: x['score'])
                label = 'POSITIVE' if best_sentiment['label'] == 'POSITIVE' else 'NEGATIVE'
                return {'label': label, 'score': best_sentiment['score'], 'method': 'HuggingFace'}
    except Exception as e:
        print(f"HF API error: {e}")
    return None

def analyze_sentiment_textblob(text):
    """Analyze sentiment using TextBlob"""
    try:
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        
        if polarity > 0.1:
            return {'label': 'POSITIVE', 'score': abs(polarity), 'method': 'TextBlob'}
        elif polarity < -0.1:
            return {'label': 'NEGATIVE', 'score': abs(polarity), 'method': 'TextBlob'}
        else:
            return {'label': 'NEUTRAL', 'score': 0.5, 'method': 'TextBlob'}
    except Exception as e:
        print(f"TextBlob error: {e}")
        return {'label': 'NEUTRAL', 'score': 0.5, 'method': 'TextBlob'}

def analyze_sentiment(text):
    """Analyze sentiment using both methods"""
    # Try Hugging Face first, fallback to TextBlob
    hf_result = analyze_sentiment_hf(text)
    if hf_result:
        return hf_result
    return analyze_sentiment_textblob(text)

@app.route('/')
def dashboard():
    """Main dashboard"""
    return render_template('dashboard.html')

@app.route('/add_tweet', methods=['POST'])
def add_tweet():
    """Add a new tweet and analyze sentiment"""
    data = request.json
    text = data.get('text', '')
    
    if not text:
        return jsonify({'error': 'Tweet text is required'}), 400
    
    # Analyze sentiment
    sentiment = analyze_sentiment(text)
    
    # Store in MongoDB
    tweet_doc = {
        'text': text,
        'sentiment_label': sentiment['label'],
        'sentiment_score': sentiment['score'],
        'analysis_method': sentiment.get('method', 'Unknown'),
        'timestamp': datetime.utcnow(),
        'brand': data.get('brand', 'default')
    }
    
    result = tweets_collection.insert_one(tweet_doc)
    tweet_doc['_id'] = str(result.inserted_id)
    
    return jsonify(tweet_doc)

@app.route('/api/tweets')
def get_tweets():
    """Get all tweets with pagination"""
    page = int(request.args.get('page', 1))
    limit = int(request.args.get('limit', 10))
    skip = (page - 1) * limit
    
    tweets = list(tweets_collection.find().sort('timestamp', -1).skip(skip).limit(limit))
    
    # Convert ObjectId to string
    for tweet in tweets:
        tweet['_id'] = str(tweet['_id'])
    
    return jsonify(tweets)

@app.route('/api/sentiment_stats')
def sentiment_stats():
    """Get sentiment statistics"""
    pipeline = [
        {
            '$group': {
                '_id': '$sentiment_label',
                'count': {'$sum': 1},
                'avg_score': {'$avg': '$sentiment_score'}
            }
        }
    ]
    
    stats = list(tweets_collection.aggregate(pipeline))
    return jsonify(stats)

@app.route('/api/brand_stats')
def brand_stats():
    """Get brand-specific sentiment statistics"""
    pipeline = [
        {
            '$group': {
                '_id': {
                    'brand': '$brand',
                    'sentiment': '$sentiment_label'
                },
                'count': {'$sum': 1},
                'avg_score': {'$avg': '$sentiment_score'}
            }
        },
        {
            '$group': {
                '_id': '$_id.brand',
                'sentiments': {
                    '$push': {
                        'sentiment': '$_id.sentiment',
                        'count': '$count',
                        'avg_score': '$avg_score'
                    }
                },
                'total': {'$sum': '$count'}
            }
        }
    ]
    
    stats = list(tweets_collection.aggregate(pipeline))
    return jsonify(stats)

@app.route('/api/sentiment_timeline')
def sentiment_timeline():
    """Feature 1: Get sentiment trends over time"""
    pipeline = [
        {
            '$group': {
                '_id': {
                    'date': {'$dateToString': {'format': '%Y-%m-%d', 'date': '$timestamp'}},
                    'sentiment': '$sentiment_label'
                },
                'count': {'$sum': 1}
            }
        },
        {'$sort': {'_id.date': 1}}
    ]
    
    timeline = list(tweets_collection.aggregate(pipeline))
    return jsonify(timeline)

@app.route('/api/emotion_analysis')
def emotion_analysis():
    """Feature 2: Advanced emotion detection"""
    from textblob import TextBlob
    
    tweets = list(tweets_collection.find().limit(50))
    emotions = []
    
    for tweet in tweets:
        blob = TextBlob(tweet['text'])
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        if polarity > 0.5 and subjectivity > 0.5:
            emotion = 'Joy'
        elif polarity < -0.5 and subjectivity > 0.5:
            emotion = 'Anger'
        elif polarity < -0.3 and subjectivity < 0.5:
            emotion = 'Sadness'
        elif polarity > 0.3 and subjectivity < 0.3:
            emotion = 'Trust'
        elif subjectivity > 0.7:
            emotion = 'Surprise'
        else:
            emotion = 'Neutral'
            
        emotions.append({
            'text': tweet['text'][:100] + '...',
            'emotion': emotion,
            'polarity': polarity,
            'subjectivity': subjectivity,
            'timestamp': tweet['timestamp']
        })
    
    return jsonify(emotions)

@app.route('/api/keyword_analysis')
def keyword_analysis():
    """Feature 3: Extract and analyze keywords"""
    from collections import Counter
    import re
    
    tweets = list(tweets_collection.find())
    positive_words = []
    negative_words = []
    
    for tweet in tweets:
        words = re.findall(r'\b\w+\b', tweet['text'].lower())
        words = [w for w in words if len(w) > 3]
        
        if tweet['sentiment_label'] == 'POSITIVE':
            positive_words.extend(words)
        elif tweet['sentiment_label'] == 'NEGATIVE':
            negative_words.extend(words)
    
    pos_counter = Counter(positive_words).most_common(10)
    neg_counter = Counter(negative_words).most_common(10)
    
    return jsonify({
        'positive_keywords': pos_counter,
        'negative_keywords': neg_counter
    })

@app.route('/api/sentiment_heatmap')
def sentiment_heatmap():
    """Feature 4: Hourly sentiment heatmap data"""
    pipeline = [
        {
            '$group': {
                '_id': {
                    'hour': {'$hour': '$timestamp'},
                    'day': {'$dayOfWeek': '$timestamp'},
                    'sentiment': '$sentiment_label'
                },
                'count': {'$sum': 1}
            }
        }
    ]
    
    heatmap_data = list(tweets_collection.aggregate(pipeline))
    return jsonify(heatmap_data)

@app.route('/api/comparative_analysis', methods=['POST'])
def comparative_analysis():
    """Feature 5: Compare sentiment between brands/topics"""
    data = request.json
    brands = data.get('brands', [])
    
    if not brands:
        return jsonify({'error': 'No brands provided'}), 400
    
    pipeline = [
        {'$match': {'brand': {'$in': brands}}},
        {
            '$group': {
                '_id': {
                    'brand': '$brand',
                    'sentiment': '$sentiment_label'
                },
                'count': {'$sum': 1},
                'avg_score': {'$avg': '$sentiment_score'}
            }
        }
    ]
    
    comparison = list(tweets_collection.aggregate(pipeline))
    return jsonify(comparison)

if __name__ == '__main__':
    app.run(debug=True)
