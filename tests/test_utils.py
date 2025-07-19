import os
import unittest
from app.utils import load_md_file

class TestUtils(unittest.TestCase):
    def setUp(self):
        # 테스트용 Markdown 파일 생성
        self.test_md_path = "test_document.md"
        with open(self.test_md_path, "w", encoding="utf-8") as f:
            f.write("# 제목\n\n이것은 테스트 문서입니다. 문단이 올바르게 분리되는지 확인합니다.")

    def tearDown(self):
        # 테스트용 파일 삭제
        if os.path.exists(self.test_md_path):
            os.remove(self.test_md_path)

    def test_load_md_file_success(self):
        """Markdown 파일을 성공적으로 로드하고 청크로 분할하는지 테스트"""
        chunks = load_md_file(self.test_md_path, chunk_size=5, overlap=1)
        self.assertIsInstance(chunks, list)
        self.assertTrue(len(chunks) > 0)
        # 청크 내용 검증 (예상되는 청크 결과)
        expected_chunks = [
            '제목 이것은 테스트 문서입니다. 문단이',
            '문단이 올바르게 분리되는지 확인합니다.'
        ]
        self.assertEqual(chunks[0].split()[0], "제목")


    def test_load_md_file_not_found(self):
        """파일이 없을 때 FileNotFoundError를 발생하는지 테스트"""
        with self.assertRaises(FileNotFoundError):
            load_md_file("non_existent_file.md")

if __name__ == "__main__":
    unittest.main()
