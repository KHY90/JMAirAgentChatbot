# JMAirAgentChatbot (Hybrid RAG Version)

JMAirAgentChatbot은 에어컨 설치 정보를 제공하는 FastAPI 기반 챗봇입니다. BM25 키워드 검색과 Vector 의미 검색을 결합한 하이브리드 RAG(Retrieval Augmented Generation) 파이프라인을 구현하여, 사용자의 질문에 정확하고 자연스러운 한국어 답변을 생성합니다.

## 특징
- FastAPI 기반의 웹 서버 제공
- **하이브리드 검색**: BM25 키워드 검색 + Vector 의미 검색 (RRF 결합)
- **한국어 형태소 분석**: `kiwipiepy`를 활용한 정확한 한국어 토큰화
- `HuggingFaceEmbeddings` (`all-MiniLM-L6-v2`)를 사용한 텍스트 임베딩
- Supabase Vector DB를 활용한 효율적인 문서 검색
- `google/gemma-2-2b-it` 모델을 이용한 자연스러운 한국어 답변 생성
- LangChain을 활용한 RAG 파이프라인 구현
- Docker 및 docker-compose를 통한 손쉬운 배포 지원

## 아키텍처

```
사용자 질문
      ↓
┌─────────────────────┐
│  Hybrid Retriever   │
│  ├─ BM25 (키워드)    │
│  └─ Vector (의미)    │
│         ↓           │
│  RRF Fusion (k=3)   │
└─────────────────────┘
      ↓
프롬프트 템플릿
      ↓
Gemma-2-2b-it LLM
      ↓
답변
```

## 설치 방법
1. 이 저장소를 클론합니다.
   ```bash
   git clone https://github.com/your-repo/JMAirAgentChatbot.git
   cd JMAirAgentChatbot
   ```
2. 프로젝트 루트에 `.env` 파일을 생성하여 다음 값을 설정합니다.
   ```
   SUPABASE_URL=your_supabase_url
   SUPABASE_ANON_KEY=your_supabase_anon_key
   API_KEY=your_api_key
   ALLOWED_ORIGINS=http://localhost:3000
   ```
3. Python 환경에서 의존성을 설치합니다.
   ```bash
   pip install -r requirements.txt
   ```

## 실행 방법
### 1. 데이터 주입 (Ingestion)
`documents/information.md` 파일의 내용을 Supabase Vector DB와 BM25 인덱스에 저장합니다.
```bash
python ingest.py
```
이 과정은 한 번만 실행하면 됩니다. 문서 내용이 변경될 경우 다시 실행해주세요.

### 2. 로컬 실행
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### 3. Docker
```bash
docker-compose up --build
```

## API 엔드포인트
- **`GET /health`**: 서버의 상태를 확인합니다.
- **`POST /ask`**: 사용자 질문을 받아 답변을 반환합니다. (`X-API-Key` 헤더 필요)

**요청 예시:**
```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your_api_key" \
  -d '{"q": "배관 비용은 어떻게 되나요?"}'
```

## 폴더 구조
```
JMAirAgentChatbot/
├── app/
│   ├── main.py             # FastAPI 앱 및 API 엔드포인트 정의
│   ├── rag_chain.py        # 하이브리드 RAG 체인 (BM25 + Vector + Gemma LLM)
│   └── supabase_client.py  # Supabase 클라이언트 초기화
├── data/                   # BM25 인덱스 저장 (gitignore)
│   ├── bm25_index.pkl      # BM25 인덱스
│   └── bm25_corpus.json    # 코퍼스 텍스트
├── documents/
│   └── information.md      # RAG에 사용될 정보가 담긴 문서
├── tests/
│   └── test_main.py        # API 테스트
├── .env.example            # 환경 변수 예시 파일
├── .gitignore
├── docker-compose.yml      # Docker Compose 설정
├── Dockerfile              # Docker 이미지 빌드 설정
├── ingest.py               # 문서를 Supabase + BM25에 주입하는 스크립트
├── README.md               # 프로젝트 설명 파일
└── requirements.txt        # Python 의존성 목록
```

## 시스템 요구사항
- Python 3.9+
- RAM: 최소 8GB (Gemma-2-2b-it 모델 로드용)
- 첫 실행 시 Gemma 모델 다운로드 (~5GB)

## 성능 참고
- CPU 환경: 응답 시간 10-30초
- GPU 환경 (권장): 응답 시간 1-3초
