import os
import requests
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

def test_mongodb():
    try:
        client = MongoClient(os.getenv('MONGODB_URI'))
        client.admin.command('ping')
        print("MongoDB connection successful")
        return True
    except Exception as e:
        print(f"MongoDB connection failed: {e}")
        return False

def test_textblob():
    try:
        from textblob import TextBlob
        blob = TextBlob("I love this product!")
        sentiment = blob.sentiment.polarity
        print(f"TextBlob sentiment analysis working: {sentiment}")
        return True
    except Exception as e:
        print(f"TextBlob error: {e}")
        return False

if __name__ == "__main__":
    print("Testing credentials...")
    mongo_ok = test_mongodb()
    textblob_ok = test_textblob()
    
    if mongo_ok and textblob_ok:
        print("\nAll services are working!")
    else:
        print("\nSome services need fixing")
