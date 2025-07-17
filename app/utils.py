import os
from markdown import markdown
from bs4 import BeautifulSoup

def load_md_file(path: str, chunk_size: int = 100, overlap: int = 20) -> list[str]:
    with open(path, 'r', encoding='utf-8') as f:
        raw_md = f.read()
    html = markdown(raw_md)
    soup = BeautifulSoup(html, 'html.parser')
    text = soup.get_text()
    words = text.split()
    
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks
