#!/bin/bash

DOC_FILE="documents/information.md"
HASH_FILE="data/doc_hash.txt"
INDEX_FILE="data/bm25_index.pkl"

# 현재 문서의 해시 계산
CURRENT_HASH=$(md5sum "$DOC_FILE" | cut -d' ' -f1)

# 저장된 해시 읽기 (없으면 빈 문자열)
if [ -f "$HASH_FILE" ]; then
    SAVED_HASH=$(cat "$HASH_FILE")
else
    SAVED_HASH=""
fi

# 인덱스가 없거나 문서가 변경되었으면 ingest.py 실행
if [ ! -f "$INDEX_FILE" ] || [ "$CURRENT_HASH" != "$SAVED_HASH" ]; then
    echo "Document changed or index not found. Running ingest.py..."
    python ingest.py
    # 새 해시 저장
    echo "$CURRENT_HASH" > "$HASH_FILE"
    echo "Ingest completed. Hash saved."
fi

# 서버 시작
exec uvicorn app.main:app --host 0.0.0.0 --port 8000
