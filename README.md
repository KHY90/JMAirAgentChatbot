# JMAirAgentChatbot (Supabase RAG Version)

JMAirAgentChatbot은 에어컨 설치 정보를 제공하는 간단한 FastAPI 기반 챗봇 예제입니다. Supabase의 Vector DB를 사용하여 RAG(Retrieval Augmented Generation) 파이프라인을 구현하여, 사용자의 질문에 가장 연관성이 높은 내용을 찾아 답변을 생성합니다.

## 특징
- FastAPI 기반의 웹 서버 제공
- `HuggingFaceEmbeddings` (`all-MiniLM-L6-v2`)를 사용한 텍스트 임베딩
- Supabase Vector DB를 활용한 효율적인 문서 검색
- `google/flan-t5-small` 모델을 이용한 답변 생성
- LangChain을 활용한 RAG 파이프라인 구현
- Docker 및 docker-compose를 통한 손쉬운 배포 지원
- 상세한 설치 및 실행 방법 안내

## 설치 방법
1. 이 저장소를 클론합니다.
   ```bash
   git clone https://github.com/your-repo/JMAirAgentChatbot.git
   cd JMAirAgentChatbot
   ```
2. 프로젝트 루트에 `.env` 파일을 생성하여 다음 값을 설정합니다. (`.env.example` 파일 참조)
   ```
   SUPABASE_URL=your_supabase_url
   SUPABASE_ANON_KEY=your_supabase_anon_key
   ```
3. Python 환경에서 의존성을 설치합니다.
   ```bash
   pip install -r requirements.txt
   ```

## 실행 방법
### 1. 데이터 주입 (Ingestion)
먼저, `documents/information.md` 파일의 내용을 Supabase Vector DB에 저장해야 합니다. 다음 스크립트를 실행하세요.
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
- **`POST /ask`**: 사용자 질문을 받아 답변을 반환합니다.

**요청 예시:**
```bash
curl -X POST "http://localhost:8000/ask" -H "Content-Type: application/json" -d '{"q": "배관 비용은 어떻게 되나요?"}'
```

## 폴더 구조
```
JMAirAgentChatbot/
├── app/
│   ├── main.py             # FastAPI 앱 및 API 엔드포인트 정의
│   ├── rag_chain.py        # LangChain을 이용한 RAG 체인 구현
│   └── supabase_client.py  # Supabase 클라이언트 초기화
├── documents/
│   └── information.md      # RAG에 사용될 정보가 담긴 문서
├── .env.example            # 환경 변수 예시 파일
├── .gitignore
├── docker-compose.yml      # Docker Compose 설정
├── Dockerfile              # Docker 이미지 빌드 설정
├── ingest.py               # 문서를 Supabase에 주입하는 스크립트
├── README.md               # 프로젝트 설명 파일
└── requirements.txt        # Python 의존성 목록
```
