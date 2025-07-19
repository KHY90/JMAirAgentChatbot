import unittest
import numpy as np
from app.embedder import embed_chunks

class TestEmbedder(unittest.TestCase):
    def test_embed_chunks_success(self):
        """청크 목록을 성공적으로 임베딩하는지 테스트"""
        chunks = ["이것은 첫 번째 문장입니다.", "이것은 두 번째 문장입니다."]
        embeddings = embed_chunks(chunks)
        self.assertIsInstance(embeddings, np.ndarray)
        self.assertEqual(embeddings.shape[0], len(chunks))
        # 임베딩 차원 수 확인 (all-MiniLM-L6-v2는 384차원)
        self.assertEqual(embeddings.shape[1], 384)

    def test_embed_chunks_empty_list(self):
        """빈 청크 목록에 대해 ValueError를 발생하는지 테스트"""
        with self.assertRaises(ValueError):
            embed_chunks([])

if __name__ == "__main__":
    unittest.main()
