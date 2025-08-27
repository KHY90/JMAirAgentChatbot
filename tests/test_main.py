import os
from fastapi.testclient import TestClient
from app.main import app

# Use a test client for the app
client = TestClient(app)


def setup_module(module):
    """Create a dummy document for testing."""
    test_doc_path = "documents/infomation.md"
    os.makedirs(os.path.dirname(test_doc_path), exist_ok=True)
    with open(test_doc_path, "w", encoding="utf-8") as f:
        f.write("벽걸이형 에어컨 기본 설치 비용은 30,000원입니다.")

def teardown_module(module):
    """Remove the dummy document after tests."""
    test_doc_path = "documents/infomation.md"
    if os.path.exists(test_doc_path):
        os.remove(test_doc_path)

def test_health_check():
    """Test the /health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "message": "정상!"}

def test_ask_question_integration():
    """An integration test to ensure the /ask endpoint and RAG chain work.
    
    This test can be slow as it may trigger model downloads on the first run.
    It checks if the endpoint returns a successful response with a non-empty answer.
    """
    # The API key verification is currently commented out in main.py
    # If it's re-enabled, a header will be needed here.
    response = client.post("/ask", json={"q": "벽걸이 에어컨 설치비 얼마야?"})
    
    # Check for a successful response
    assert response.status_code == 200
    
    # Check the response body
    json_response = response.json()
    assert "answer" in json_response
    assert isinstance(json_response["answer"], str)
    assert len(json_response["answer"]) > 0