# JMAirAgentChatbot

JMAirAgentChatbot은 에어컨 설치 정보를 제공하는 간단한 FastAPI 기반 챗봇 예제입니다. 문서(`documents/infomation.md`)를 임베딩하여 사용자의 질문에 가장 연관성이 높은 내용을 찾아 응답합니다.

## 특징
- FastAPI 웹 서버 제공
- SentenceTransformer를 이용한 임베딩
- scikit-learn을 활용한 코사인 유사도 기반 검색
- 간단한 LLM 추론 예제(`llm_infer.py`)
- Docker 및 docker-compose 지원

## 설치 방법
1. 이 저장소를 클론합니다.
2. 프로젝트 루트에 `.env` 파일을 생성하여 `API_KEY` 값을 설정합니다.
   ```
   API_KEY=your_api_key
   ```
3. Python 환경에서 의존성을 설치합니다.
   ```bash
   pip install -r requirements.txt
   ```
   또는 Docker 환경을 사용할 수 있습니다.

## 실행 방법
### 로컬 실행
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Docker
```bash
docker-compose up --build
```

## 사용 예시
- `/health` : 서버 상태 확인
- `/ask?q=QUESTION` : 질문에 대한 답변을 얻습니다. 이때 `X-API-Key` 헤더로 위에서 설정한 `API_KEY` 값을 전달해야 합니다.

예)
```bash
curl -H "X-API-Key: your_api_key" "http://localhost:8000/ask?q=배관 비용은?"
```

## 폴더 구조
```
app/            FastAPI 애플리케이션 코드
  embedder.py   문서 임베딩 모듈
  llm_infer.py  LLM 추론 예제
  main.py       API 엔드포인트 정의
  retriever.py  문서 검색 로직
  utils.py      Markdown 로딩 및 청크화

documents/      답변에 사용되는 정보가 담긴 Markdown 파일
```
