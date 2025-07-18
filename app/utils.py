import os
from markdown import markdown
from bs4 import BeautifulSoup


# Markdown 파일을 읽어 청크 단위로 분리하는 함수
# 에러 메시지와 주석은 모두 한글로 작성되어 있습니다.

def load_md_file(path: str, chunk_size: int = 100, overlap: int = 20) -> list[str]:
    """Markdown 파일을 로드하여 청크 목록으로 반환합니다."""

    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} 파일을 찾을 수 없습니다.")

    try:
        with open(path, 'r', encoding='utf-8') as f:
            raw_md = f.read()
    except OSError as e:
        raise RuntimeError(f"파일 읽기 중 오류가 발생했습니다: {e}")

    html = markdown(raw_md)
    soup = BeautifulSoup(html, 'html.parser')
    text = soup.get_text()
    words = text.split()

    chunks: list[str] = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks
