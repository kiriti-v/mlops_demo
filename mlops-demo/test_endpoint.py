import requests
import json
import sys

def test_sentiment_analysis(url):
    """Test the sentiment analysis endpoint."""
    
    # Test cases
    test_texts = [
        "I love this product, it's amazing!",
        "This is terrible, I hate it.",
        "The weather is nice today.",
    ]
    
    print("\nTesting Sentiment Analysis API at:", url)
    print("-" * 50)
    
    # Test health endpoint
    try:
        health_response = requests.get(f"{url}/health")
        print(f"\nHealth Check Status: {health_response.status_code}")
        print(f"Response: {health_response.json()}")
    except Exception as e:
        print(f"Error checking health endpoint: {str(e)}")
        return
    
    # Test prediction endpoint
    for text in test_texts:
        try:
            print(f"\nTesting text: '{text}'")
            response = requests.post(
                f"{url}/predict",
                json={"text": text}
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"Status: {response.status_code}")
                print(f"Sentiment: {result['sentiment']}")
                print(f"Confidence: {result['confidence']:.4f}")
            else:
                print(f"Error: Status code {response.status_code}")
                print(f"Response: {response.text}")
        
        except Exception as e:
            print(f"Error making prediction: {str(e)}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test_endpoint.py <service_url>")
        print("Example: python test_endpoint.py https://sentiment-analysis-api-xxx.run.app")
        sys.exit(1)
    
    service_url = sys.argv[1].rstrip('/')
    test_sentiment_analysis(service_url) 