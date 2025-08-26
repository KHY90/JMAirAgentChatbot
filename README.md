# JMAirAgentChatbot (Supabase RAG Version)

JMAirAgentChatbot은 에어컨 설치 정보를 제공하는 간단한 FastAPI 기반 챗봇 예제입니다. Supabase의 Vector DB를 사용하여 RAG(Retrieval Augmented Generation) 파이프라인을 구현하여, 사용자의 질문에 가장 연관성이 높은 내용을 찾아 LLM을 통해 답변을 생성합니다.

## 특징
- FastAPI 웹 서버 제공
- Supabase Vector DB를 활용한 문서 검색
- LangChain 및 OpenAI를 이용한 RAG 파이프라인 구현
- Docker 및 docker-compose 지원
- 기본적인 오류 처리 로직 포함

## 설치 방법
1. 이 저장소를 클론합니다.
2. 프로젝트 루트에 `.env` 파일을 생성하여 다음 값을 설정합니다. (`.env.example` 파일 참조)
   ```
   OPENAI_API_KEY=your_openai_api_key
   SUPABASE_URL=your_supabase_url
   SUPABASE_ANON_KEY=your_supabase_anon_key
   ```
3. Python 환경에서 의존성을 설치합니다.
   ```bash
   pip install -r requirements.txt
   ```
   또는 Docker 환경을 사용할 수 있습니다.

## 실행 방법
### 1. 데이터 주입 (Ingestion)
먼저, `documents/infomation.md` 파일의 내용을 Supabase Vector DB에 저장해야 합니다. 다음 스크립트를 실행하세요.
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

## 사용 예시
- `/health` : 서버 상태 확인
- `/ask` : POST 요청으로 질문을 보내 답변을 얻습니다.

예)
```bash
curl -X POST "http://localhost:8000/ask" -H "Content-Type: application/json" -d '{"q": "배관 비용은?"}'
```

## 폴더 구조
```
app/
  supabase_client.py  Supabase 클라이언트 초기화
  rag_chain.py        RAG 체인 구현
  main.py             API 엔드포인트 정의

documents/            답변에 사용되는 정보가 담긴 Markdown 파일

ingest.py             문서 내용을 Supabase에 주입하는 스크립트
```