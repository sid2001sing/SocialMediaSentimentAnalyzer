# Social Media Sentiment Analyzer

A Flask-based dashboard for real-time social media sentiment analysis using MongoDB and Hugging Face AI models.


## Features

- Real-time sentiment analysis using Cardiff NLP Twitter RoBERTa model
- MongoDB storage for tweets and sentiment data
- Interactive dashboard with sentiment distribution charts
- Brand-specific sentiment tracking
- RESTful API endpoints

## API Endpoints

- `POST /add_tweet` - Add and analyze new tweet
- `GET /api/tweets` - Get paginated tweets
- `GET /api/sentiment_stats` - Get sentiment statistics

