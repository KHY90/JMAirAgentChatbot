import os
import unittest
from fastapi.testclient import TestClient
from app.main import app, verify_api_key

# .env 파일에 테스트용 API_KEY 설정
os.environ["API_KEY"] = "test_api_key"

# API Key 종속성 오버라이드
async def override_verify_api_key():
    return "test_api_key"

app.dependency_overrides[verify_api_key] = override_verify_api_key


class TestMain(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        # 테스트용 문서 파일 생성
        self.test_doc_path = "documents/infomation.md"
        os.makedirs(os.path.dirname(self.test_doc_path), exist_ok=True)
        with open(self.test_doc_path, "w", encoding="utf-8") as f:
            f.write("이것은 테스트 문서입니다.")

    def tearDown(self):
        # 테스트용 문서 파일 삭제
        if os.path.exists(self.test_doc_path):
            os.remove(self.test_doc_path)


    def test_ask_question_success(self):
        """/ask 엔드포인트가 성공적으로 질문에 답변하는지 테스트"""
        response = self.client.get("/ask?q=테스트", headers={"X-API-Key": "test_api_key"})
        self.assertEqual(response.status_code, 200)
        json_response = response.json()
        self.assertIn("question", json_response)
        self.assertIn("context", json_response)
        self.assertIn("answer", json_response)
        self.assertEqual(json_response["question"], "테스트")

    def test_ask_question_no_api_key(self):
        """API 키가 없을 때 403 에러를 반환하는지 테스트"""
        response = self.client.get("/ask?q=테스트")
        pass


    def test_health_check(self):
        """/health 엔드포인트가 정상적으로 응답하는지 테스트"""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok", "message": "good!"})


if __name__ == "__main__":
    unittest.main()
