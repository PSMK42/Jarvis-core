import os
import re
import requests
import psutil
import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel
from typing import Optional
from ddgs import DDGS
import edge_tts
import chromadb
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- Device Configuration ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Running engine on device: {device}")

# --- Load Pre-trained SLM ---
MODEL_NAME = "HuggingFaceTB/SmolLM-135M-Instruct"

print("Loading model tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Loading pre-trained language model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if device == 'cuda' else torch.float32,
    device_map="auto" if device == 'cuda' else None
)
if device == 'cpu':
    model.to(device)
model.eval()
print("Language model loaded successfully.")
if device == 'cpu':
    model.to(device)
model.eval()
print("Language model loaded successfully.")

# --- Persistent Vector Database (ChromaDB) ---
chroma_client = chromadb.PersistentClient(path="./jarvis_memory")
memory_collection = chroma_client.get_or_create_collection(name="user_knowledge")

def store_memory(text: str) -> str:
    current_count = memory_collection.count()
    memory_id = f"mem_{current_count + 1}"
    memory_collection.add(
        documents=[text],
        ids=[memory_id]
    )
    return f"Logged into persistent vector memory: '{text}'."

def retrieve_context(query: str, n_results: int = 2) -> str:
    if memory_collection.count() == 0:
        return ""
    results = memory_collection.query(
        query_texts=[query],
        n_results=min(n_results, memory_collection.count())
    )
    docs = results.get("documents", [[]])[0]
    return " ".join(docs) if docs else ""

chat_history = []

# --- Output Sanitizer ---
def sanitize_response(reply: str, user_prompt: str) -> str:
    identity_queries = ["who are you", "your name", "what are you", "about yourself", "who is jarvis", "what is jarvis"]
    if any(k in user_prompt.lower() for k in identity_queries):
        return reply

    cleaned = re.sub(r'^\s*jarvis\s*:\s*', '', reply, flags=re.IGNORECASE)
    cleaned = re.sub(r'\bjarvis\b', 'I', cleaned, flags=re.IGNORECASE)
    return cleaned.strip()

# --- Knowledge Search Engine ---
def cross_reference_search(query: str) -> str:
    try:
        raw_snippets = []
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
            for r in results:
                body = r.get("body", "")
                if body:
                    raw_snippets.append(body)

        if not raw_snippets:
            return f"I cross-referenced live networks, but found no verified records for '{query}'."

        extracted_sentences = []
        for snippet in raw_snippets:
            cleaned = re.sub(r'https?://\S+|www\.\S+', '', snippet)
            cleaned = re.sub(r'\s+', ' ', cleaned)
            sentences = re.split(r'(?<=[.!?])\s+', cleaned)
            for s in sentences:
                s_clean = s.strip()
                if len(s_clean) > 25 and not any(junk in s_clean.lower() for junk in ["click here", "cookie", "privacy policy", "subscribe"]):
                    extracted_sentences.append(s_clean)

        unique_facts = []
        seen_words = set()

        for s in extracted_sentences:
            words = set(re.findall(r'\w+', s.lower())) - {"the", "a", "an", "and", "or", "is", "of", "to", "in", "for", "on", "with"}
            overlap = words.intersection(seen_words)
            if not words or (len(overlap) / len(words)) < 0.6:
                unique_facts.append(s)
                seen_words.update(words)

            if len(unique_facts) >= 3:
                break

        return " ".join(unique_facts) if unique_facts else f"Data retrieved for '{query}', but results were inconclusive."

    except Exception as e:
        return f"Knowledge retrieval system error: {e}"

# --- Speech Generation ---
async def generate_speech(text: str) -> bytes:
    communicate = edge_tts.Communicate(text, "en-GB-RyanNeural")
    audio_data = bytearray()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_data.extend(chunk["data"])
    return bytes(audio_data)

# --- FastAPI Setup ---
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryModel(BaseModel):
    text: Optional[str] = None
    prompt: Optional[str] = None
    message: Optional[str] = None

# --- Pre-trained SLM Generation Function ---
def generate_llm_response(prompt: str) -> str:
    try:
        context = retrieve_context(prompt)
        system_instructions = (
            "You are JARVIS, an intelligent and polite voice assistant. "
            "Keep your responses brief, direct, and conversational (1-3 sentences max)."
        )
        if context:
            system_instructions += f" Relevant Memory Context: {context}"

        messages = [
            {"role": "system", "content": system_instructions},
            {"role": "user", "content": prompt}
        ]

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        model_inputs = tokenizer([text], return_tensors="pt").to(device)

        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=80,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )

        generated_ids = [
            output_ids[len(input_ids):] 
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        final_response = response.strip()

        chat_history.append((prompt, final_response))
        return final_response

    except Exception as e:
        return f"System error processing query: {e}"

@app.get("/tts")
async def get_tts(text: str):
    audio_bytes = await generate_speech(text)
    return Response(content=audio_bytes, media_type="audio/mpeg")

@app.get("/", response_class=HTMLResponse)
def get_ui():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>JARVIS Voice Interface</title>
  <style>
    body {
      background-color: #0d1117;
      color: #58a6ff;
      font-family: 'Courier New', Courier, monospace;
      margin: 0;
      padding: 20px;
      display: flex;
      justify-content: center;
      align-items: center;
      min-height: 90vh;
    }

    .jarvis-container {
      width: 100%;
      max-width: 650px;
      background: #161b22;
      border: 1px solid #30363d;
      border-radius: 8px;
      padding: 20px;
      box-shadow: 0 0 15px rgba(56, 139, 253, 0.2);
    }

    .hud-header {
      text-align: center;
      font-weight: bold;
      letter-spacing: 2px;
      color: #79c0ff;
      margin-bottom: 15px;
      border-bottom: 1px solid #30363d;
      padding-bottom: 10px;
    }

    .chat-log {
      height: 350px;
      overflow-y: auto;
      background: #0d1117;
      border: 1px solid #30363d;
      padding: 10px;
      border-radius: 4px;
      margin-bottom: 15px;
      display: flex;
      flex-direction: column;
      gap: 10px;
    }

    .msg {
      padding: 8px 12px;
      border-radius: 4px;
      max-width: 85%;
      line-height: 1.4;
      word-wrap: break-word;
    }

    .msg.user {
      align-self: flex-end;
      background: #1f6feb;
      color: #ffffff;
    }

    .msg.jarvis {
      align-self: flex-start;
      background: #21262d;
      color: #79c0ff;
      border-left: 3px solid #58a6ff;
    }

    .input-area {
      display: flex;
      gap: 10px;
    }

    input {
      flex: 1;
      background: #0d1117;
      border: 1px solid #30363d;
      color: #c9d1d9;
      padding: 10px;
      border-radius: 4px;
      outline: none;
    }

    input:focus {
      border-color: #58a6ff;
    }

    button {
      background: #238636;
      color: #ffffff;
      border: none;
      padding: 10px 16px;
      border-radius: 4px;
      cursor: pointer;
      font-weight: bold;
    }

    button:hover {
      background: #2ea043;
    }

    #mic-btn {
      background: #388bfd;
    }

    #mic-btn:hover {
      background: #1f6feb;
    }
    
    #mic-btn.active {
      background: #da3633;
    }
  </style>
</head>
<body>

  <div class="jarvis-container">
    <div class="hud-header">JARVIS PROTOCOL INTERFACE</div>
    <div id="chat-log" class="chat-log"></div>
    <div class="input-area">
      <input type="text" id="user-input" placeholder="Type or activate wake-word mode..." onkeydown="if(event.key==='Enter') sendCommand()">
      <button id="mic-btn" onclick="toggleWakeWordListener()">🎙️ Listen</button>
      <button id="send-btn" onclick="sendCommand()">Send</button>
    </div>
  </div>

  <script>
    const BACKEND_URL = "/jarvis";

    let recognition = null;
    let isWakeWordActive = false;

    function toggleWakeWordListener() {
      if (isWakeWordActive) {
        stopWakeWordListener();
      } else {
        startWakeWordListener();
      }
    }

    function startWakeWordListener() {
      const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
      if (!SpeechRecognition) {
        alert("Speech recognition is not supported in this browser.");
        return;
      }

      if (recognition) {
        try { recognition.stop(); } catch(e) {}
      }

      recognition = new SpeechRecognition();
      recognition.continuous = true;
      recognition.interimResults = false;
      recognition.lang = 'en-US';

      recognition.onresult = (event) => {
        const lastIndex = event.results.length - 1;
        const transcript = event.results[lastIndex][0].transcript.trim();
        const lower = transcript.toLowerCase();

        if (lower.includes("jarvis")) {
          const jarvisIdx = lower.indexOf("jarvis");
          let command = transcript.substring(jarvisIdx + 6).replace(/^[,\s.]+/, "").trim();

          if (!command) {
            command = "jarvis";
          }

          document.getElementById("user-input").value = transcript;
          sendCommand(command);
        }
      };

      recognition.onerror = (e) => {
        console.error("Wake word error:", e.error);
      };

      recognition.onend = () => {
        if (isWakeWordActive) {
          try { recognition.start(); } catch(e) {}
        }
      };

      isWakeWordActive = true;
      updateMicStatus(true);
      try { recognition.start(); } catch(e) {}
    }

    function stopWakeWordListener() {
      isWakeWordActive = false;
      updateMicStatus(false);
      if (recognition) {
        try { recognition.stop(); } catch(e) {}
      }
    }

    async function sendCommand(customText = null) {
      const inputEl = document.getElementById("user-input");
      const query = customText || inputEl.value.trim();

      if (!query) return;

      appendMessage(query, "user");
      inputEl.value = "";

      const loadingDiv = appendMessage("Processing sequence...", "jarvis");

      try {
        const response = await fetch(BACKEND_URL, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text: query })
        });

        if (!response.ok) throw new Error(`HTTP ${response.status}`);

        const data = await response.json();
        const replyText = data.reply || data.response || "No output returned.";

        loadingDiv.textContent = replyText;
        speakReply(replyText);

      } catch (error) {
        console.error(error);
        loadingDiv.textContent = "Error communicating with brain.";
      }
    }

    function speakReply(text) {
      if (recognition) {
        try { recognition.stop(); } catch(e) {}
      }

      const audio = new Audio(`/tts?text=${encodeURIComponent(text)}`);
      
      audio.onended = () => {
        if (isWakeWordActive) {
          setTimeout(() => {
            if (isWakeWordActive && recognition) {
              try { recognition.start(); } catch(e) {}
            }
          }, 500);
        }
      };

      audio.play().catch(err => console.error("Audio playback error:", err));
    }

    function updateMicStatus(active) {
      const micBtn = document.getElementById("mic-btn");
      if (micBtn) {
        micBtn.textContent = active ? "🔴 Wake Word Active" : "🎙️ Listen";
        if (active) {
          micBtn.classList.add("active");
        } else {
          micBtn.classList.remove("active");
        }
      }
    }

    function appendMessage(text, sender) {
      const chatLog = document.getElementById("chat-log");
      const msgDiv = document.createElement("div");
      msgDiv.classList.add("msg", sender);
      msgDiv.textContent = text;
      chatLog.appendChild(msgDiv);
      chatLog.scrollTop = chatLog.scrollHeight;
      return msgDiv;
    }
  </script>
</body>
</html>
    """

@app.post("/jarvis")
def handle_jarvis_request(data: QueryModel):
    raw_text = data.text or data.prompt or data.message or ""
    text = raw_text.lower().strip()
    
    if not text:
        return {"response": "No command received.", "reply": "No command received."}

    # Greeting
    if text in ["jarvis", "hello jarvis", "hey jarvis", "hi jarvis"]:
        raw_reply = "Greetings, sir. How may I assist you today?"

    # Weather
    elif "weather in" in text:
        city = text.split("weather in")[-1].strip()
        try:
            res = requests.get(f"https://wttr.in/{city}?format=3")
            raw_reply = res.text.strip() if res.status_code == 200 else f"Could not retrieve weather for {city}."
        except Exception:
            raw_reply = f"Error fetching weather data for {city}."

    # System Status
    elif "system status" in text or "status report" in text:
        cpu = psutil.cpu_percent()
        ram = psutil.virtual_memory().percent
        raw_reply = f"System operational. CPU load at {cpu} percent. Memory utilization at {ram} percent."

    # Vector RAG Memory Storage
    elif "remember that" in text:
        fact = text.replace("remember that", "").strip()
        raw_reply = store_memory(fact)

    elif "what do you remember" in text or "recall memory" in text:
        context = retrieve_context("user facts and memory", n_results=5)
        raw_reply = f"Stored memory context: {context}" if context else "Vector memory storage is currently empty."

    # Web Search
    elif any(k in text for k in ["search for", "look up", "latest on"]):
        query = (
            text.replace("search for", "")
            .replace("look up", "")
            .replace("latest on", "")
            .strip()
        )
        raw_reply = cross_reference_search(query)

    # Clear History
    elif any(k in text for k in ["clear history", "reset conversation", "forget history"]):
        chat_history.clear()
        raw_reply = "Conversation history cleared. Ready for new input."

    # Pre-trained SLM Generation (Handles all general knowledge & reasoning)
    else:
        raw_reply = generate_llm_response(raw_text)

    final_reply = sanitize_response(raw_reply, raw_text)
    return {"response": final_reply, "reply": final_reply}