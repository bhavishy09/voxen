# Voxen — Autonomous Voice AI Sales Platform

> **An intelligent, low-latency voice conversational agent for automated sales outreach and lead qualification.**  
> Built by **Bhavishya**

---

## ⚡ Overview

**Voxen** is a full-stack, voice-native conversational AI sales system designed to conduct interactive, human-like sales discovery and qualification calls. It unifies high-speed speech recognition, retrieval-augmented generation (RAG) over structured knowledge bases, and dynamic objection handling into a seamless, real-time pipeline.

### Why Voxen?
Traditional sales chatbots often feel rigid or introduce awkward delays. Voxen solves this by pairing **Groq-accelerated inference** with **in-memory vector search (FAISS)** and real-time audio synthesis, delivering natural conversational flow and sub-2-second voice response latency.

---

## 💎 Key Capabilities

### 🎙️ Bidirectional Voice Engine
* **Acoustic Noise Calibration**: Automatic ambient noise adjustment to ensure clear speech capture even in noisy room environments.
* **Speech-to-Text (STT)**: Direct microphone streaming via Google Speech Recognition with configurable timeout and phrase limits.
* **Speech Synthesis (TTS)**: High-fidelity natural voice audio synthesis with automatic audio playback management.

### 🧠 Dual Intelligence Architecture
* **Semantic RAG Pipeline**: Powered by LangChain and FAISS with `all-MiniLM-L6-v2` embeddings. Pulls targeted course/product information dynamically to answer specific prospect queries.
* **Groq LPU Acceleration**: Utilizes high-throughput `llama3-8b-8192` for rapid, context-rich reasoning and concise responses (under 4 sentences) tuned for conversational cadence.
* **Tactical Fallback Rules**: Deterministic rule-based objection handler for high-frequency objections (pricing concerns, schedule limitations, skepticism, and graceful call terminations).

### 🖥️ Modern Command Dashboard
* **Sleek Dark Interface**: High-contrast obsidian & light blue theme designed for sales representatives and supervisors.
* **Live Call Transcript**: Real-time message streaming distinguishing between prospect statements and agent responses.
* **Dynamic Mode Toggling**: Seamless switching between Semantic RAG mode and Deterministic Rule-based mode on active sessions.

---

## 🏗️ Architecture Pipeline

```
  ┌─────────────────────────────────────────────────────────────┐
  │                         User Audio                          │
  └──────────────────────────────┬──────────────────────────────┘
                                 │ Microphone Stream
                                 ▼
  ┌─────────────────────────────────────────────────────────────┐
  │         Voice Ingestion & STT (Google Speech Engine)        │
  └──────────────────────────────┬──────────────────────────────┘
                                 │ Transcribed Query Text
                                 ▼
  ┌─────────────────────────────────────────────────────────────┐
  │               FastAPI Orchestration Layer                   │
  └──────────────┬──────────────────────────────┬───────────────┘
                 │ (RAG Mode)                   │ (Rule Mode)
                 ▼                              ▼
  ┌──────────────────────────────┐ ┌────────────────────────────┐
  │   FAISS Vector Retrieval     │ │ Objection Resolution Logic │
  │   + Groq Llama-3 Inference   │ │  (Guaranteed Response Set) │
  └──────────────┬───────────────┘ └────────────┬───────────────┘
                 │ Generated Sales Script       │
                 └───────────────┬──────────────┘
                                 │
                                 ▼
  ┌─────────────────────────────────────────────────────────────┐
  │         Text-to-Speech Engine (Audio Playback Buffer)       │
  └──────────────────────────────┬──────────────────────────────┘
                                 │ Speaker Stream
                                 ▼
  ┌─────────────────────────────────────────────────────────────┐
  │                     Live Audio Output                       │
  └─────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Technology Stack

| Layer | Technologies |
|---|---|
| **API & Server** | FastAPI, Uvicorn, Pydantic |
| **LLM Inference** | Groq (`llama3-8b-8192`), LangChain |
| **Vector Search & RAG** | FAISS (CPU), Sentence-Transformers (`all-MiniLM-L6-v2`) |
| **Audio & Speech** | SpeechRecognition, gTTS, PyAudio, Pygame Mixer |
| **Frontend Console** | Vanilla HTML5, Modern CSS3 (Dark / Sky Blue), JavaScript (ES6+) |

---

## 🚀 Quick Setup Guide

### 1. Prerequisites
* **Python**: Version 3.8 to 3.11 installed on your system.
* **Groq API Key**: Obtain a free API key from [Groq Console](https://console.groq.com/).
* **Audio Hardware**: Working microphone and speakers / headphones.

### 2. Prepare Environment
Open your terminal in the `voxen` directory:

```bash
# Create a virtual environment
python -m venv venv

# Activate virtual environment
# On macOS / Linux:
source venv/bin/activate
# On Windows (cmd / PowerShell):
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

> **Note for macOS / Linux users**: If you encounter issues installing `pyaudio`, ensure development tools and `portaudio` are available (`brew install portaudio` on macOS, or `sudo apt install portaudio19-dev` on Ubuntu/Debian).

### 4. Configure Credentials
Create a `.env` file in the project root:

```bash
GROQ_API_KEY=your_actual_groq_api_key_here
```

### 5. Launch the Server
You can launch Voxen directly with Python:

```bash
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

*(On Windows, you can also double-click or run `start.bat`)*

### 6. Access the Dashboard
Open your browser and navigate to:
```
http://localhost:8000
```

---

## 📡 API Reference

Voxen provides clean RESTful endpoints for integration into existing CRM and dialer infrastructure:

### `POST /start-call`
Initiates a new voice session and generates a welcome greeting.
* **Request Body:**
  ```json
  {
    "customer_name": "Sarah Connor",
    "phone_number": "+14155552671"
  }
  ```
* **Response:**
  ```json
  {
    "call_id": "9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d",
    "message": "Calling Sarah Connor...",
    "first_message": "Hi Sarah Connor, this is your AI assistant..."
  }
  ```

### `POST /respond-rag/{call_id}`
Sends customer input through the knowledge-base RAG chain powered by Groq Llama-3.
* **Request Body:**
  ```json
  {
    "message": "What topics are covered in the machine learning modules?"
  }
  ```
* **Response:**
  ```json
  {
    "reply": "Our curriculum covers AI/ML foundations, deep learning, computer vision, and LLMs...",
    "should_end_call": false
  }
  ```

### `POST /respond/{call_id}`
Processes customer response via tactical rule-based objection engine.
* **Request Body:**
  ```json
  {
    "message": "The price is too high for me right now."
  }
  ```
* **Response:**
  ```json
  {
    "reply": "I understand the cost concern! Let me share some great news...",
    "should_end_call": false
  }
  ```

### `GET /simulate-call/{call_id}`
Triggers server-side microphone listening for real-time speech input and returns recognized text with agent reply.

### `GET /conversation/{call_id}`
Retrieves the full chronological interaction transcript for a given call ID.

### `GET /rag-status`
Returns diagnostic health and indexing status for the RAG vector store.

---

## 📂 Project Structure

```
voxen/
├── main.py                 # FastAPI application routing & session management
├── llm_service.py          # RAG vector index, Groq integration & objection engine
├── voice_service.py        # Speech recognition & audio synthesis service
├── models.py               # Pydantic schemas for calls & messaging
├── index.html              # Modern dark-mode web console (HTML5/CSS3/JS)
├── requirements.txt        # Python package specifications
├── start.bat               # Windows quick-launch automation script
├── .env                    # Environment secrets (GROQ_API_KEY)
├── test_api.py             # Integration test suite for REST endpoints
├── test_simulate.py        # Voice simulation verification script
└── check_server.py         # Fast server connectivity checker
```

---

## 🧪 Verification & Diagnostics

Test the system components independently with the included scripts:

```bash
# 1. Verify server health
python check_server.py

# 2. Test complete API lifecycle (Call Start -> Respond -> History)
python test_api.py

# 3. Test interactive microphone capture
python test_simulate.py
```

---

## 👨‍💻 Creator & Maintainer

Designed and implemented from the ground up by **Bhavishya**.
Built for high-performance voice AI sales operations.
