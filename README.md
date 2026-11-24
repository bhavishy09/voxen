# Voxen — Autonomous Voice AI Sales Platform

> **An intelligent, low-latency voice conversational agent for automated sales outreach and lead qualification.**  
> Built by **Bhavishya**

[![Watch Demo Video](https://img.shields.io/badge/▶_Watch_Demo_Video-Google_Drive-4285F4?style=for-the-badge&logo=googledrive&logoColor=white)](https://drive.google.com/file/d/16U1vL70WIzRiPWmgfcFyDr60SLQDGjj8/view?usp=sharing)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Groq LPU](https://img.shields.io/badge/Groq-Qwen_3.8_27B-F55036?style=for-the-badge&logo=fastly&logoColor=white)](https://groq.com/)

---

## 🎬 Live Product Demo

> 📺 **Click below to watch the full live walkthrough of Voxen handling real-time customer voice conversations, vehicle qualification, and test ride bookings:**
>
> 👉 **[Watch Voxen Demo Video (Google Drive)](https://drive.google.com/file/d/16U1vL70WIzRiPWmgfcFyDr60SLQDGjj8/view?usp=sharing)**

---

## ⚡ Overview

**Voxen** is a full-stack, voice-native conversational AI sales system designed to conduct interactive, human-like sales discovery and qualification calls. It unifies high-speed speech recognition, retrieval-augmented generation (RAG) over structured knowledge bases, and dynamic objection handling into a seamless, real-time pipeline.

### Why Voxen?
Traditional sales chatbots often feel rigid or introduce awkward delays. Voxen solves this by pairing **Groq-accelerated inference** with **in-memory vector search (FAISS)** and real-time audio synthesis, delivering natural conversational flow and sub-2-second voice response latency.

---

## 💎 Key Capabilities

### 🎙️ Bidirectional Voice Engine
* **Hybrid Dual-Layer STT**: Browser-native **Web Speech API (`en-IN` locale)** for client-side, zero-latency Roman-script Hinglish transcription, backed by Python **SpeechRecognition (Google Engine)** for headless CLI simulations.
* **Acoustic Noise Calibration**: Automatic ambient noise adjustment to ensure clear speech capture even in noisy room environments.
* **Speech Synthesis (TTS)**: High-fidelity natural voice audio synthesis with automatic audio playback management.

### 🧠 Dual Intelligence Architecture
* **Semantic RAG Pipeline**: Powered by LangChain and FAISS with `all-MiniLM-L6-v2` embeddings. Pulls verified TVS Motors specs (`tvs_bike_info.txt`) dynamically to answer prospect queries without hallucinations.
* **Groq LPU Acceleration**: Utilizes high-throughput **`qwen/qwen3.8-27b`** for ultra-fast, context-rich reasoning and concise voice responses (strictly 1–2 sentences, under 22 words) tuned for conversational phone cadence.
* **Tactical Fallback Rules**: Deterministic rule-based objection handler for high-frequency objections (pricing concerns, schedule limitations, skepticism, and graceful call terminations).

### 🖥️ Modern Command Dashboard
* **Sleek Dark Interface**: High-contrast obsidian & light blue theme designed for sales representatives and supervisors.
* **Live Call Transcript**: Real-time message streaming distinguishing between prospect statements and agent responses.
* **Dynamic Mode Toggling**: Seamless switching between Semantic RAG mode and Deterministic Rule-based mode on active sessions.

---

## 📊 Benchmark Evaluation Suite (`v1` vs `v2`)

Voxen features an automated 61-scenario evaluation harness ([`eval_voxen.py`](eval_voxen.py)) that stress-tests voice brevity, anti-hallucination grounding, objection handling, language mirroring, and multi-turn state retention.

In **Commit [`476a5db3`](https://github.com/bhavishy09/voxen/commit/476a5db3)**, targeted prompt engineering and conversation memory wiring increased the benchmark pass rate from **80% to 90%**:

| Metric | Baseline (`results_v1.csv`) | Optimized (`results_v2.csv`) | Impact |
|---|:---:|:---:|:---:|
| **Overall Pass Rate** | **80%** (49/61) | **90%** (55/61) | **+10% Net Improvement** |
| **Spoken Word Economy** | 16.6 words | 17.8 words | Strictly under 22-word limit |
| **Voice Cleanliness** | 100% | 100% | Zero emojis, asterisks, or markdown |

### Category Highlights
* 🟢 **Test-Ride Booking**: **100% (7/7)** — Slot capture, showroom timings (10 AM–7 PM), doorstep rides.
* 🟢 **Objection Resolution**: **100% (8/8)** — Price hesitation, timing delays, competitor comparisons.
* 🟢 **Grounded Deferrals**: **100% (5/5)** — Zero invented figures for on-road prices, discounts, or stock.
* 🟢 **Model Comparisons**: **100% (4/4)** — Accurate cross-model analysis (Jupiter, Raider, Apache, iQube).
* 🟢 **Warranty Coverage**: **100% (3/3)** — Verified TVS engine and EV battery terms.
* 🟢 **Callback Handling**: **100% (2/2)** — Structured `schedule_callback` action triggers.

```bash
# Compare benchmark runs directly
python eval_voxen.py --compare results_v1.csv results_v2.csv
```

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
  │   + Groq Qwen-3.8 Inference  │ │  (Guaranteed Response Set) │
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
| **LLM Inference** | Groq (`qwen/qwen3.8-27b`), LangChain |
| **Vector Search & RAG** | FAISS (CPU), Sentence-Transformers (`all-MiniLM-L6-v2`) |
| **Audio & Speech** | Web Speech API (`en-IN`), SpeechRecognition, gTTS, Pygame Mixer |
| **Frontend Console** | Vanilla HTML5, Modern CSS3 (Dark / Sky Blue), JavaScript (ES6+) |

---

## 🚀 Quick Setup Guide (Run Locally)

Follow these simple steps to clone and run Voxen on your local machine:

### 1. Clone the Repository
```bash
git clone https://github.com/bhavishy09/voxen.git
cd voxen
```

### 2. Prepare Virtual Environment
```bash
# Create virtual environment
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
Create a `.env` file in the project root with your free Groq API key:

```bash
GROQ_API_KEY=your_actual_groq_api_key_here
```
*(Get a free API key at [console.groq.com](https://console.groq.com/))*

### 5. Launch the Server

Choose your preferred execution mode:

* **Normal Voice Mode (With bot voice speaking aloud):**
  ```bash
  python -m uvicorn main:app --host 127.0.0.1 --port 8000
  ```
* **Silent Mode (Muted speaker output, best if listening to music or testing):**
  ```bash
  DISABLE_TTS=true python -m uvicorn main:app --host 127.0.0.1 --port 8000
  ```

*(On Windows, you can also double-click `start.bat`)*

### 6. Open the Web Dashboard
1. Open **Google Chrome** (recommended for native Web Speech API microphone support).
2. Navigate to:
   ```
   http://localhost:8000
   ```
3. Enter any sample lead info (e.g., Name: `Rahul Sharma`, Phone: `+91 9876543210`) and click **Start Call**.
4. Click **Allow** for microphone access.
5. Talk to Voxen or type your query:
   * *"Jupiter 125 ka mileage aur price kya hai?"*
   * *"Mujhe Raider ki test ride chahiye kal subah."*
   * *"Showroom timings kya hain?"*

---

### 🧪 Running Tests & Benchmarks

With the server running on port 8000, open a second terminal window to run:

```bash
# 1. View benchmark comparison (v1 vs v2)
python eval_voxen.py --compare results_v1.csv results_v2.csv

# 2. Test API endpoints
python test_api.py

# 3. Run a fresh 61-test evaluation suite
DISABLE_TTS=true python eval_voxen.py --label my_run
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
Sends customer input through the knowledge-base RAG chain powered by Groq Qwen-3.8.
* **Request Body:**
  ```json
  {
    "message": "Raider 125 ka mileage aur price range kya hai?"
  }
  ```
* **Response:**
  ```json
  {
    "reply": "Raider 125 ka mileage lagbhag 56.7 km per litre hai aur price 90,000 se start hota hai. Kya aap test ride lena chahenge?",
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
├── llm_service.py          # RAG vector index, Groq integration & structured output
├── voice_service.py        # Speech recognition & audio synthesis service
├── models.py               # Pydantic schemas for calls & messaging
├── tvs_bike_info.txt       # Grounded knowledge base for TVS two-wheelers
├── eval_voxen.py           # 61-scenario automated evaluation harness
├── results_v1.csv          # Baseline evaluation benchmark (80% pass rate)
├── results_v2.csv          # Optimized evaluation benchmark (90% pass rate)
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
