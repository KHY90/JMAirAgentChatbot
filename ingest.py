import os
import json
import pickle
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
from app.supabase_client import supabase
from rank_bm25 import BM25Okapi
from kiwipiepy import Kiwi
from dotenv import load_dotenv

load_dotenv()

# 한국어 토크나이저 초기화
kiwi = Kiwi()

def tokenize_korean(text: str) -> list:
    """한국어 형태소 분석 기반 토큰화"""
    tokens = kiwi.tokenize(text)
    # 명사, 동사, 형용사 등 의미있는 형태소만 추출
    meaningful_pos = {'NNG', 'NNP', 'VV', 'VA', 'MAG', 'SL', 'SN'}
    return [token.form for token in tokens if token.tag in meaningful_pos]


def main():
    # 1. 문서 로드
    loader = UnstructuredMarkdownLoader("documents/information.md")
    documents = loader.load()

    # 2. 청크 분할 (사이즈 증가)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,        # 200 -> 500 (한국어 문맥 보존)
        chunk_overlap=100,     # 40 -> 100 (연속성 강화)
        separators=["\n---\n", "\n###", "\n\n", "\n", " "]
    )
    docs = text_splitter.split_documents(documents)

    # 3. 임베딩 생성
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # 4. Vector Store 저장 (Supabase)
    vector_store = SupabaseVectorStore.from_documents(
        docs,
        embeddings,
        client=supabase,
        table_name="documents",
        query_name="match_documents",
    )

    # 5. BM25 인덱스 생성
    corpus_texts = [doc.page_content for doc in docs]
    tokenized_corpus = [tokenize_korean(text) for text in corpus_texts]
    bm25_index = BM25Okapi(tokenized_corpus)

    # 6. BM25 데이터 저장 (로컬 파일)
    os.makedirs("data", exist_ok=True)

    bm25_data = {
        "corpus_texts": corpus_texts,
        "tokenized_corpus": tokenized_corpus,
    }

    with open("data/bm25_corpus.json", "w", encoding="utf-8") as f:
        json.dump(bm25_data, f, ensure_ascii=False, indent=2)

    with open("data/bm25_index.pkl", "wb") as f:
        pickle.dump(bm25_index, f)

    print(f"데이터 주입 완료: {len(docs)}개 청크")
    print("- Vector Store: Supabase 저장 완료")
    print("- BM25 Index: data/bm25_index.pkl 저장 완료")


if __name__ == "__main__":
    main()
