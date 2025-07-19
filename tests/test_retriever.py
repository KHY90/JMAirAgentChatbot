import unittest
import numpy as np
from app.retriever import retrieve_relevant_chunk
from sentence_transformers import SentenceTransformer

class TestRetriever(unittest.TestCase):
    def setUp(self):
        # 테스트를 위한 임베딩 모델 로드
        self.embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.chunks = [
            "파이썬은 배우기 쉬운 프로그래밍 언어입니다.",
            "프랑스 파리는 에펠탑으로 유명합니다.",
            "대한민국의 수도는 서울입니다."
        ]
        self.chunk_embeddings = self.embedder.encode(self.chunks)

    def test_retrieve_relevant_chunk_success(self):
        """가장 관련성 높은 청크를 성공적으로 검색하는지 테스트"""
        query = "한국의 수도는 어디인가요?"
        relevant_chunk = retrieve_relevant_chunk(query, self.chunks, self.chunk_embeddings, self.embedder)
        self.assertEqual(relevant_chunk, "대한민국의 수도는 서울입니다.")

    def test_retrieve_relevant_chunk_no_documents(self):
        """문서가 없을 때 ValueError를 발생하는지 테스트"""
        with self.assertRaises(ValueError):
            retrieve_relevant_chunk("질문", [], np.array([]), self.embedder)

if __name__ == "__main__":
    unittest.main()
