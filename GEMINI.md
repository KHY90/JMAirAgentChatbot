# Project Overview

This project is a simple FastAPI-based chatbot called JMAirAgentChatbot. It provides information about air conditioner installation by using a Retrieval Augmented Generation (RAG) approach. The chatbot embeds a Markdown document (`documents/infomation.md`) and uses cosine similarity to find the most relevant information to answer a user's question.

## Key Technologies

*   **Backend:** FastAPI
*   **Embedding:** Sentence-Transformers (`all-MiniLM-L6-v2`)
*   **Search:** scikit-learn (cosine similarity)
*   **LLM:** The current implementation uses a placeholder for the language model, returning the retrieved context directly. The presence of the `langchain` library suggests that a more sophisticated LLM integration is planned.
*   **Containerization:** Docker and Docker Compose

## Architecture

The application is structured as follows:

*   `app/main.py`: Defines the main FastAPI application, including the `/ask` and `/health` endpoints. It handles API key verification and CORS middleware.
*   `app/embedder.py`: Loads the Sentence-Transformer model and embeds the text chunks from the source document.
*   `app/retriever.py`: Retrieves the most relevant text chunk based on cosine similarity between the user's query and the embedded document chunks.
*   `app/llm_infer.py`: A placeholder module that is intended to generate a final answer using a large language model. Currently, it returns the retrieved context.
*   `app/utils.py`: Contains utility functions for loading and chunking Markdown files.
*   `documents/infomation.md`: The source document containing the information about air conditioner installation.
*   `Dockerfile` and `docker-compose.yml`: Used to build and run the application in a containerized environment.

# Building and Running

## Local Development

1.  **Create a `.env` file:**
    ```
    API_KEY=your_api_key
    ALLOWED_ORIGINS=http://localhost:3000/
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run the application:**
    ```bash
    uvicorn app.main:app --host 0.0.0.0 --port 8000
    ```

## Docker

1.  **Create a `.env` file** (as described above).
2.  **Build and run the container:**
    ```bash
    docker-compose up --build
    ```

# Development Conventions

*   **API Key Authentication:** All requests to the `/ask` endpoint must include a valid `X-API-Key` header.
*   **CORS:** The application is configured to allow cross-origin requests from the origins specified in the `ALLOWED_ORIGINS` environment variable.
*   **Modular Structure:** The code is organized into separate modules for embedding, retrieval, and LLM inference, promoting separation of concerns.
*   **Error Handling:** The application includes basic error handling for API key verification and question processing.
*   **TODO:** The `app/llm_infer.py` module is a placeholder and needs to be implemented with a proper language model to provide more sophisticated answers.
