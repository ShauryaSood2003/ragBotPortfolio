import requests
import json

BASE_URL = "http://localhost:8000"

def test_health():
    response = requests.get(f"{BASE_URL}/health")
    print("Health Check:", response.json())
    return response.status_code == 200

def test_upload_sample():
    sample_text = """
    This is a sample document for testing the RAG bot.
    It contains information about artificial intelligence and machine learning.
    AI is transforming various industries including healthcare, finance, and education.
    Machine learning algorithms can learn patterns from data to make predictions.
    """
    
    with open("sample_doc.txt", "w") as f:
        f.write(sample_text)
    
    with open("sample_doc.txt", "rb") as f:
        response = requests.post(
            f"{BASE_URL}/upload",
            files={"file": ("sample_doc.txt", f, "text/plain")}
        )
    
    print("Upload Response:", response.json())
    return response.status_code == 200

def test_question():
    question_data = {
        "question": "What industries is AI transforming?",
        "session_id": "test_session"
    }
    
    response = requests.post(
        f"{BASE_URL}/question",
        headers={"Content-Type": "application/json"},
        data=json.dumps(question_data)
    )
    
    print("Question Response:", response.json())
    return response.status_code == 200

def test_sessions():
    response = requests.get(f"{BASE_URL}/sessions")
    print("Active Sessions:", response.json())
    return response.status_code == 200

if __name__ == "__main__":
    print("Testing RAG Bot API...")
    
    if test_health():
        print("✓ Health check passed")
    else:
        print("✗ Health check failed")
        exit(1)
    
    if test_upload_sample():
        print("✓ Document upload passed")
    else:
        print("✗ Document upload failed")
        exit(1)
    
    if test_question():
        print("✓ Question answering passed")
    else:
        print("✗ Question answering failed")
        exit(1)
    
    if test_sessions():
        print("✓ Session management passed")
    else:
        print("✗ Session management failed")
        exit(1)
    
    print("\nAll tests passed! 🎉")