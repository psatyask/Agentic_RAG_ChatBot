# 🤖 Agentic RAG Chatbot

### An AI-Powered Conversational Assistant with LangGraph, Google Gemini, ChromaDB, RAG, and Persistent Memory

An intelligent, tool-enabled AI chatbot built using **LangGraph, LangChain, Google Gemini, ChromaDB, Google Embeddings, FastAPI, SQLite, and Bootstrap**.

The application combines large language models (LLMs) with Retrieval-Augmented Generation (RAG), real-time web search, mathematical computation, and long-term memory to deliver context-aware conversational experiences.

Unlike a traditional LLM chatbot, this system can interact with external tools, retrieve information from uploaded documents, perform calculations, and maintain persistent conversation history across sessions.

The application also supports real-time streaming responses, document ingestion, Markdown rendering, syntax-highlighted code blocks, and a responsive web interface.

---

## ✨ Key Features

- **Agentic AI with LangGraph:** Tool-enabled AI agent capable of interacting with external resources.
- **Google Gemini LLM:** Natural language understanding, response generation, and tool selection.
- **Retrieval-Augmented Generation:** Document-grounded question answering using semantic retrieval.
- **ChromaDB Vector Database:** Vector storage and similarity-based document retrieval.
- **Google Embeddings:** Semantic vector representations of documents and user queries.
- **Tavily Web Search:** Retrieval of external web information.
- **Calculator Tool:** Mathematical expression evaluation.
- **Long-Term Memory:** Database-backed storage and recall of user information.
- **Conversation History:** Persistent chat messages organized through unique thread identifiers.
- **Real-Time Streaming:** Token streaming using FastAPI and Server-Sent Events (SSE).
- **Multi-Document Support:** PDF, DOCX, TXT, MD, PY, and CSV uploads.
- **Interactive Web Interface:** Bootstrap-based responsive interface with Markdown and syntax highlighting.
- **Voice Input:** Browser-supported speech-to-text interaction.
- **Performance Monitoring:** Backend logging for agent initialization, first-token latency, and execution duration.

---

## 🏗️ System Architecture

The application follows a modular architecture consisting of a frontend, API server, AI agent, retrieval system, external tools, and persistent storage.

```text
                      USER
                       |
                       v
          +--------------------------+
          |       FRONTEND           |
          |    HTML / CSS / JS       |
          |    Bootstrap 5           |
          |                          |
          | - Chat Interface         |
          | - Model Selection        |
          | - Document Upload        |
          | - Conversation History   |
          | - Markdown Rendering     |
          +--------------------------+
                       |
                       | HTTP / SSE
                       v
          +--------------------------+
          |      FASTAPI SERVER      |
          |         app.py           |
          |                          |
          | - API Endpoints          |
          | - Request Validation     |
          | - Document Ingestion     |
          | - Streaming Responses    |
          | - Chat Persistence       |
          +--------------------------+
                       |
                       v
          +--------------------------+
          |     LANGGRAPH AGENT      |
          |         agent.py         |
          |                          |
          |    Google Gemini LLM     |
          |    Tool Calling          |
          |    Agent Execution       |
          +--------------------------+
                       |
          +------------+-------------+
          |            |             |
          v            v             v
   +------------+ +------------+ +------------+
   | RAG SYSTEM | |  EXTERNAL  | |   MEMORY   |
   |   rag.py   | |   TOOLS    | |   SYSTEM   |
   +------------+ +------------+ +------------+
          |            |             |
          v            v             v
   +------------+ +------------+ +------------+
   | Google     | | Tavily     | | SQLite     |
   | Embeddings | | Web Search | | Database   |
   +------------+ +------------+ +------------+
          |            |
          v            v
   +------------+ +------------+
   | ChromaDB   | | Calculator |
   | Vector DB  | | Tool       |
   +------------+ +------------+
          |
          v
   +--------------------+
   | Relevant Document  |
   | Context Retrieval  |
   +--------------------+
          |
          v
   +--------------------+
   | Gemini Generates   |
   | Contextual Answer  |
   +--------------------+
          |
          v
   +--------------------+
   | FastAPI SSE Stream |
   +--------------------+
          |
          v
   +--------------------+
   | Formatted Chat UI  |
   +--------------------+
```

### Architecture Overview

**Frontend Layer**

The frontend uses HTML, CSS, JavaScript, and Bootstrap to provide an interactive chat interface. It manages user input, model selection, file uploads, chat navigation, and streaming response rendering.

**API Layer — FastAPI**

FastAPI handles HTTP requests, validates inputs, manages document uploads, retrieves conversation history, and delivers streaming model responses to the browser.

**Agent Layer — LangGraph**

LangGraph provides the agent execution framework. Google Gemini processes user requests and can invoke registered tools to access external capabilities.

**Retrieval Layer — ChromaDB and Google Embeddings**

The RAG system processes uploaded files, creates semantic embeddings, and uses vector similarity search to retrieve relevant document passages.

**Persistence Layer — SQLite**

SQLite stores conversation metadata, chat messages, and long-term memory records.

---

## 🛠️ Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Agent Orchestration | LangGraph |
| LLM Framework | LangChain |
| Large Language Model | Google Gemini |
| Embedding Model | Google Embedding Model |
| Vector Database | ChromaDB |
| Retrieval Architecture | Retrieval-Augmented Generation (RAG) |
| Web Search | Tavily Search API |
| Backend API | FastAPI |
| ASGI Server | Uvicorn |
| Database | SQLite |
| Frontend | HTML5, CSS3, JavaScript |
| UI Framework | Bootstrap 5 |
| Markdown Rendering | Marked.js |
| Syntax Highlighting | Highlight.js |
| HTML Sanitization | DOMPurify |
| Streaming | Server-Sent Events (SSE) |
| Configuration | python-dotenv |
| SSL Certificates | Certifi |
| Deployment | Render |

---

## 🧠 LangGraph Agent Architecture

LangGraph is used to build the chatbot's tool-enabled conversational workflow.

The agent combines Google Gemini with registered tools, allowing it to support requests requiring information retrieval, calculations, or persistent memory operations.

### Agent Execution Flow

```text
                 User Query
                     |
                     v
              LangGraph Agent
                     |
                     v
               Google Gemini
                     |
                     v
             Tool Required?
                /       \
              YES        NO
               |          |
               v          |
          Select Tool     |
               |          |
               v          |
          Execute Tool    |
               |          |
               v          |
          Tool Response   |
               |          |
               v          |
           Gemini LLM <---+
               |
               v
         Generate Response
               |
               v
         Stream to Frontend
```

The agent can execute multiple reasoning and tool-interaction steps before producing its final response.

The FastAPI backend receives the compiled agent through:

```python
from agent import get_agent
```

### Agent Caching

The application uses Python's `lru_cache` to reuse initialized agent instances for the same model.

```python
@lru_cache(maxsize=5)
def get_cached_agent(model_name: str):
    agent = get_agent(model_name)
    return agent
```

This reduces repeated agent initialization overhead, assuming the compiled agent is designed for safe reuse.

### Conversation Threading

Each conversation uses a unique thread identifier:

```python
config = {
    "configurable": {
        "thread_id": thread_id
    }
}
```

The identifier is passed into the agent execution configuration to associate requests with a conversation.

---

## 🔧 Integrated AI Tools

The application registers five tools that extend the capabilities of Google Gemini.

These tools are implemented in `tools.py`.

```python
tools = [
    calculator,
    search_uploaded_documents,
    remember_this,
    recall_memory,
    web_search_tool
]
```

### 1. Tavily Web Search

**Technology:** Tavily Search API and LangChain

The Tavily integration allows the agent to retrieve information from external web sources.

```python
web_search_tool = TavilySearch(
    max_results=5,
    topic="general",
    search_depth="advanced"
)
```

**Configuration:**

- Maximum results: 5
- Search topic: General
- Search depth: Advanced

**Example query:**

> What are the latest developments in Agentic AI?

The agent can invoke Tavily, retrieve relevant search results, and use the returned information to formulate an answer.

This tool is particularly useful for questions requiring current information rather than knowledge stored in the model's training data.

### 2. Document Retrieval Tool — RAG

**Technologies:** LangChain, ChromaDB, Google Embeddings

The document retrieval tool enables semantic search across uploaded files.

```python
@tool
def search_uploaded_documents(query: str) -> str:
    """
    Search uploaded documents for relevant information.
    """

    return retrieve_from_rag(
        query=query,
        thread_id=CURRENT_THREAD_ID
    )
```

**Example query:**

> Summarize the main findings in the PDF I uploaded.

The tool forwards the query and current thread identifier to the RAG retrieval function.

Relevant document content is retrieved and supplied to the agent for answer generation.

### 3. Calculator Tool

**Technology:** Python Math Library

The calculator tool supports mathematical operations.

```python
@tool
def calculator(expression: str) -> str:
    """
    Useful for simple math calculations.
    """

    try:
        allowed = {
            "math": math,
            "abs": abs,
            "round": round,
            "min": min,
            "max": max,
            "sum": sum
        }

        result = eval(
            expression,
            {"__builtins__": {}},
            allowed
        )

        return str(result)

    except Exception as e:
        return f"Calculation error: {str(e)}"
```

**Example queries:**

- Calculate 125 × 48 ÷ 6.
- What is the square root of 256?
- Calculate the average of several numbers.

The tool evaluates the requested expression and returns the result to the agent.

**Security note:** The current implementation uses a restricted `eval()` environment. It should be replaced with an AST-based expression evaluator before exposing the application to untrusted public users.

### 4. Long-Term Memory Storage

**Technologies:** SQLite and LangChain Tools

The `remember_this` tool stores user-provided information in the application's persistent memory database.

```python
@tool
def remember_this(memory: str) -> str:

    return save_memory(
        thread_id=CURRENT_THREAD_ID,
        memory=memory
    )
```

**Example query:**

> Remember that I prefer Python for backend development.

The memory tool passes the information to the database layer for storage.

### 5. Long-Term Memory Retrieval

The `recall_memory` tool retrieves previously stored information relevant to a query.

```python
@tool
def recall_memory(query: str) -> str:

    return search_memory(
        thread_id=CURRENT_THREAD_ID,
        query=query
    )
```

**Example query:**

> What programming language do I prefer?

The tool queries the memory database and returns matching information to the agent.

Together, the memory tools provide a way to store and retrieve information across user interactions.

---

## 📚 Retrieval-Augmented Generation (RAG)

Retrieval-Augmented Generation is an important component of this application.

RAG enhances language-model responses by retrieving relevant information from external documents before generating an answer.

Instead of relying entirely on pretrained knowledge, the agent can access information provided by the user through uploaded documents.

### Document Ingestion Pipeline

```text
             Document Upload
                    |
                    v
             Text Extraction
                    |
                    v
               Text Chunking
                    |
                    v
          Google Embedding Model
                    |
                    v
          Dense Vector Embeddings
                    |
                    v
               ChromaDB
                    |
                    v
           Vectorized Documents
```

### Document Retrieval Pipeline

```text
               User Question
                     |
                     v
            RAG Retrieval Tool
                     |
                     v
             Query Embedding
                     |
                     v
           ChromaDB Similarity
                  Search
                     |
                     v
          Relevant Text Chunks
                     |
                     v
           Retrieved Context
                     |
                     v
              Gemini LLM
                     |
                     v
          Context-Aware Answer
```

### Document Processing

When a document is uploaded, the FastAPI backend invokes the ingestion function:

```python
result = add_document_to_rag(
    file_path=file_path,
    thread_id=thread_id
)
```

The RAG pipeline is organized around the following operations:

**Text Extraction:** Extract readable information from uploaded documents.

**Chunking:** Divide larger text into smaller passages for semantic retrieval.

**Embedding Generation:** Convert document passages into numerical vectors using a Google embedding model.

**Vector Storage:** Store embeddings in ChromaDB.

**Semantic Retrieval:** Retrieve relevant passages using similarity between the query and stored document vectors.

**Answer Generation:** Provide retrieved context to Gemini to support the final response.

### Why ChromaDB?

ChromaDB is an open-source vector database designed for embedding storage and semantic similarity search.

Its advantages include:

- Integration with LangChain
- Vector similarity search
- Document metadata support
- Local development capabilities
- Simple integration into Python applications
- Suitability for RAG prototypes

### Why Google Embeddings?

Google embedding models convert text into numerical representations capturing semantic relationships.

Traditional keyword search relies primarily on matching specific words.

Semantic retrieval instead uses vector similarity, which can identify related content even when the user's question uses different wording.

For example:

**Document:** "The application reduces response latency through agent caching."

**User question:** "How does the chatbot improve performance?"

An embedding-based retrieval system can identify the relationship between these statements, even without exact keyword overlap.

The embedding model handles semantic retrieval, while Gemini handles conversational response generation.

---

## 💾 Persistent Memory and Conversation History

The application uses SQLite for structured persistent storage.

The database layer provides functions for:

- Initializing database tables
- Creating and updating conversation records
- Saving user and assistant messages
- Retrieving chat history
- Listing existing conversations
- Saving long-term memories
- Searching stored memories

### Conversation Lifecycle

```text
               New Conversation
                       |
                       v
               Generate Thread ID
                       |
                       v
              Save User Message
                       |
                       v
               LangGraph Agent
                       |
                       v
               Gemini Response
                       |
                       v
            Save Assistant Message
                       |
                       v
             SQLite Conversation DB
                       |
                       v
            Reload Conversation Later
```

### Conversation Storage

The application records both user and assistant messages using:

```python
save_chat_message(
    thread_id,
    "user",
    user_message
)
```

```python
save_chat_message(
    thread_id,
    "assistant",
    final_answer
)
```

This enables users to revisit earlier conversations through the frontend sidebar.

### Long-Term Memory vs. Chat History

| Feature | Chat History | Long-Term Memory |
|---|---|---|
| Purpose | Preserve conversation messages | Store selected important information |
| Storage | SQLite | SQLite |
| Organization | Thread ID | Thread ID |
| Retrieval | Conversation history endpoint | Memory retrieval tool |
| Typical Usage | Reopen previous conversations | Recall saved facts or preferences |

---

## ⚡ Real-Time Response Streaming

The chatbot implements Server-Sent Events (SSE) to deliver responses incrementally.

Instead of waiting for Gemini to generate the entire answer, the backend forwards text chunks as they become available.

### Streaming Architecture

```text
              User Message
                    |
                    v
              FastAPI API
                    |
                    v
             LangGraph Agent
                    |
                    v
              Gemini LLM
                    |
                    v
            Generated Text Chunks
                    |
                    v
         Server-Sent Events (SSE)
                    |
                    v
             Browser Fetch API
                    |
                    v
          Incremental Chat Rendering
```

The backend streams using:

```python
for chunk, metadata in agent.stream(
    inputs,
    config=config,
    stream_mode="messages"
):
    ...
```

Responses are returned through FastAPI's `StreamingResponse`.

```python
return StreamingResponse(
    event_generator(),
    media_type="text/event-stream",
    headers={
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no"
    }
)
```

### Streaming Output Filtering

The backend filters out raw tool messages, tool-call payloads, and tool-node output before forwarding normal AI text to the frontend.

This helps keep the displayed response focused on human-readable assistant content rather than intermediate tool payloads.

### Markdown Rendering

The frontend uses:

- Marked.js for Markdown parsing
- DOMPurify for HTML sanitization
- Highlight.js for syntax highlighting

Supported presentation features include formatted headings, lists, tables, inline code, code blocks, and copy-code buttons.

---

## 🌐 FastAPI Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Serves the chatbot frontend |
| GET | `/conversations` | Lists existing conversations |
| GET | `/history/{thread_id}` | Retrieves messages from a conversation |
| POST | `/upload` | Uploads and indexes a document |
| POST | `/chat/stream` | Streams agent responses using SSE |

### Chat Request Example

```json
{
  "message": "Summarize my uploaded document",
  "thread_id": "example-thread-123",
  "model": "your-configured-gemini-model"
}
```

### Streaming Response Example

```text
data: {"status": "thinking"}

data: {"token": "Here"}

data: {"token": " is"}

data: {"token": " the summary."}

data: {"done": true}
```

### Document Upload

The upload endpoint accepts multipart form data containing:

- `file`: Uploaded document
- `thread_id`: Conversation identifier

Supported file extensions:

```text
.pdf
.docx
.txt
.md
.py
.csv
```

Document indexing is executed through FastAPI's threadpool to avoid blocking the main asynchronous request loop during synchronous processing.

---

## 📁 Project Structure

The application is organized into modular Python components.

```text
Agentic_RAG_Chatbot/
│
├── app.py
│   └── FastAPI application and SSE streaming
│
├── agent.py
│   └── LangGraph agent configuration
│
├── tools.py
│   └── Registered agent tools
│
├── rag.py
│   └── Document ingestion and retrieval
│
├── database.py
│   └── SQLite persistence and memory
│
├── templates/
│   └── index.html
│       └── Bootstrap chatbot frontend
│
├── uploads/
│   └── Uploaded documents
│
├── data/
│   └── Application data and database files
│
├── .env
│   └── Local environment variables
│
├── .gitignore
│
├── requirements.txt
│
└── README.md
```

The structure above describes the application modules and supporting directories; exact generated database and vector-store filenames depend on their respective implementations.

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or newer with compatible dependencies
- Google Gemini API credentials
- Tavily API credentials
- Internet connection for hosted model and web-search services

### 1. Clone the Repository

```bash
git clone https://github.com/psatyask/Agentic_RAG_ChatBot.git

cd Agentic_RAG_ChatBot
```

### 2. Create a Virtual Environment

Using Conda:

```bash
conda create -n agent_rag python=3.11 -y

conda activate agent_rag
```

Alternatively:

```bash
python -m venv .venv
```

Activate on Windows:

```bash
.venv\Scripts\activate
```

Activate on Linux or macOS:

```bash
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

The application requires dependencies for FastAPI, LangGraph, LangChain, Gemini, Tavily, document processing, ChromaDB, and SQLite integration.

The exact package versions should follow the project's `requirements.txt`.

### 4. Configure Environment Variables

Create a `.env` file in the project root.

```env
GOOGLE_API_KEY=your_google_api_key

TAVILY_API_KEY=your_tavily_api_key
```

Obtain Google API credentials from:

https://aistudio.google.com/app/apikey

Obtain Tavily API credentials from:

https://tavily.com/

The application loads environment variables using `python-dotenv`.

**Never commit `.env` files or API credentials to GitHub.**

### 5. Start the Application

```bash
python app.py
```

The development server runs on:

```text
http://localhost:8080
```

Open the address in your browser to access the chatbot.

---

## 🧪 Example Use Cases

### 1. General AI Conversation

**User:**

> Explain the difference between synchronous and asynchronous programming in Python.

**Expected behavior:**

Gemini generates an explanation, and the frontend renders formatted text and syntax-highlighted Python examples.

### 2. Document-Based Question Answering

**User:**

> Summarize the research paper I uploaded.

**Expected behavior:**

The agent invokes the document retrieval tool, retrieves relevant text from the indexed document, and uses the retrieved context to generate an answer.

### 3. Real-Time Information Retrieval

**User:**

> What are the latest developments in generative AI?

**Expected behavior:**

The agent can use Tavily Search to obtain information from external sources before generating a response.

### 4. Mathematical Computation

**User:**

> Calculate 125 × 48 ÷ 6.

**Expected behavior:**

The calculator tool evaluates the expression and returns the mathematical result.

### 5. Long-Term Memory

**User:**

> Remember that I prefer Python for my AI projects.

**Expected behavior:**

The agent invokes the memory storage tool to persist the information.

**Follow-up question:**

> What language do I prefer for AI development?

**Expected behavior:**

The agent can retrieve the stored preference through the memory recall tool.

---

## ⚡ Performance Optimizations

The application incorporates several optimizations to improve responsiveness.

### Agent Caching

Python's `lru_cache` caches initialized agents by model name to reduce repeated setup work.

### Non-Blocking Document Processing

Document ingestion runs through FastAPI's threadpool to reduce blocking of the asynchronous event loop.

### Streaming Responses

Server-Sent Events allow text to appear incrementally rather than waiting for full response completion.

### Frontend Rendering Optimization

The frontend batches token updates and avoids unnecessary scrolling animations.

### Backend Performance Logging

The application logs:

- Agent initialization time
- Database operation time
- Time to first visible token
- Total agent execution time
- Number of streamed chunks
- Response persistence time

These measurements help identify delays related to model generation, tool execution, and application overhead.

---

## ☁️ Deployment

The application can be deployed on Render as a Python web service.

### Build Command

```bash
pip install -r requirements.txt
```

### Start Command

```bash
uvicorn app:app --host 0.0.0.0 --port $PORT
```

For deployment, configure the required API credentials as environment variables in the hosting platform.

### Deployment Considerations

- Do not commit API keys or secret files.
- Configure the appropriate Google Gemini and Tavily API credentials.
- Use supported Gemini model identifiers.
- Configure persistent storage for SQLite and ChromaDB if application data must survive service restarts.
- Avoid relying on temporary local storage for production document persistence.
- Add appropriate request limits, authentication, and error handling before public production use.

**Note:** The current implementation uses a process-global `CURRENT_THREAD_ID` variable for memory and document tools. This should be replaced with request-scoped tool context to isolate concurrent conversations safely.

---

## 🔒 Security Considerations

The application uses environment variables to separate API credentials from source code.

It also uses HTML sanitization to reduce the risks of displaying model-generated content in the browser.

Additional production hardening should include:

- Authentication and authorization for conversations and uploaded documents
- Request-scoped tool context
- Secure mathematical expression parsing instead of `eval()`
- Document size and content validation
- API rate limiting
- Persistent database storage and backups
- Secret rotation and protected environment variables
- Access controls for uploaded documents and memory
- Protection against prompt injection in retrieved documents and external search results

---

## 🔮 Future Enhancements

Potential improvements include:

- Hybrid retrieval combining vector similarity and keyword search
- Reranking retrieved document chunks
- Source citations for document-grounded responses
- RAG evaluation using retrieval and answer-quality metrics
- Multi-user authentication and session management
- Persistent managed vector database deployment
- Model fallback strategies for temporary API failures
- Structured tool-execution progress events
- Observability and tracing with LangSmith
- Automated testing for tools and retrieval pipelines
- Docker-based deployment and CI/CD integration

---

## 🎯 Project Objective

The primary objective of this project is to explore how **agentic workflows, external tools, semantic retrieval, and persistent memory** can be integrated into a practical conversational AI system.

The project demonstrates several important AI engineering concepts:

**Agent Orchestration:** Connecting an LLM to external tools using LangGraph.

**Retrieval-Augmented Generation:** Enriching language-model responses with information retrieved from user documents.

**Vector Search:** Using Google embeddings and ChromaDB for semantic retrieval.

**Tool Calling:** Allowing an AI agent to interact with web search, calculator, and memory functions.

**Stateful Conversations:** Maintaining chat history and database-backed information across sessions.

**AI Application Development:** Integrating a Python backend, streaming inference, document processing, and a responsive frontend.

The resulting application serves as an extensible foundation for building tool-enabled AI assistants and document-aware conversational systems.

---

## 👨‍💻 Author

**Pardhu SS Krishna Alapati**

AI / Machine Learning Engineer

GitHub: https://github.com/psatyask

---

**Built with Python, LangGraph, LangChain, Google Gemini, ChromaDB, Google Embeddings, Tavily, FastAPI, SQLite, and Bootstrap.**