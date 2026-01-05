FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Windows CRLF -> Linux LF 변환 및 실행 권한 부여
RUN sed -i 's/\r$//' entrypoint.sh && chmod +x entrypoint.sh

CMD ["./entrypoint.sh"]
