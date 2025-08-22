# app.py
"""
Single-file offline assistant with:
- Text + voice input (wav2vec2)
- Offline NLU (sentence-transformers)
- Offline TTS (pyttsx3)
- Features: time, date, jokes, calculator, memory, play YouTube, web search fallback
- New: open/launch services/apps (google, chrome, chatgpt, facebook, instagram,
       twitter, playstore, gmail, maps, perplexity, gemenai, github, calculator, calendar)
- Added features (Option A):
    - News reader (RSS)
    - Weather (wttr.in JSON fallback)
    - Dictionary/definitions (dictionaryapi.dev)
    - Note-taking (save/recall/list/delete)
    - Reminders/alarms (background Thread + persistence)
    - System info (psutil optional)
    - Story / fun facts generator
    - Stopwatch / Timer
    - Math quiz (interactive)
    - File search and open local files
"""

import os
import json
import sounddevice as sd
import soundfile as sf
import pyttsx3
import numpy as np
from datetime import datetime, timedelta
import random
import webbrowser
import platform
import subprocess
import shutil
import threading
import time
import re
import requests
import xml.etree.ElementTree as ET

from transformers import pipeline
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
from duckduckgo_search import DDGS
import pywhatkit

# Optional psutil for system info
try:
    import psutil
except Exception:
    psutil = None

# -------------------------
# Configuration & Files
# -------------------------
NOTES_FILE = "assistant_notes.json"
REMINDERS_FILE = "assistant_reminders.json"
MEMORY_FILE = "assistant_memory.json"

# Ensure persistence files exist
for f, default in [(NOTES_FILE, {}), (REMINDERS_FILE, []), (MEMORY_FILE, {})]:
    if not os.path.exists(f):
        with open(f, "w", encoding="utf-8") as fh:
            json.dump(default, fh)

# -------------------------
# AUDIO RECORDING (MIC)
# -------------------------
def record(duration=4, filename="input.wav", samplerate=16000):
    print(f"[INFO] Recording {duration}s... Speak now!")
    data = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1)
    sd.wait()
    sf.write(filename, data, samplerate)
    return filename

# -------------------------
# SPEECH TO TEXT (ASR)
# -------------------------
print("[INFO] Loading ASR model (wav2vec2-base-960h)...")
asr = pipeline("automatic-speech-recognition", model="facebook/wav2vec2-base-960h")

def transcribe(audio_path):
    try:
        res = asr(audio_path)
        return res.get("text", "").strip()
    except Exception as e:
        print("[WARN] ASR failed:", e)
        return ""

# -------------------------
# TEXT TO SPEECH (TTS)
# -------------------------
engine = pyttsx3.init()

def speak(text):
    if not text:
        return
    print("Assistant:", text)
    try:
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print("[WARN] TTS failed:", e)

# -------------------------
# INTENTS DATA (includes open/launch examples)
# -------------------------
intents_data = {
    "intents": [
        {"tag": "greeting", "examples": ["hello", "hi", "hey"], "responses": ["Hello! How can I help you today?"]},
        {"tag": "weather", "examples": ["weather today", "is it raining", "what's the weather", "weather in delhi"], "responses": [""]},
        {"tag": "thanks", "examples": ["thanks", "thank you"], "responses": ["You're welcome!"]},
        {"tag": "time", "examples": ["what time is it", "tell me the time"], "responses": [""]},
        {"tag": "date", "examples": ["what's the date", "today's date"], "responses": [""]},
        {"tag": "joke", "examples": ["tell me a joke", "make me laugh"], "responses": [""]},
        {"tag": "calculator", "examples": ["calculate 5 plus 7", "what is 12*3", "compute 12 / 4"], "responses": [""]},
        {"tag": "remember", "examples": ["my name is", "remember that my name is", "remember my name"], "responses": [""]},
        {"tag": "recall", "examples": ["what is my name", "do you remember my name"], "responses": [""]},
        {"tag": "play_youtube", "examples": ["play despacito on youtube", "play python tutorial", "play music on youtube"], "responses": [""]},
        {"tag": "open_service", "examples": ["open google", "launch chrome", "open chatgpt", "open gmail", "open maps", "open github", "open facebook", "open instagram", "open twitter", "open playstore", "open perplexity", "open gemenai", "open calculator", "open calendar"], "responses": [""]},
        {"tag": "exit", "examples": ["bye", "goodbye", "exit", "quit"], "responses": ["Goodbye! Have a nice day."]},

        # New intents
        {"tag": "news", "examples": ["read me the news", "what's the news", "top headlines"], "responses": [""]},
        {"tag": "define", "examples": ["define serendipity", "what does loquacious mean", "meaning of ubiquitous"], "responses": [""]},
        {"tag": "note_save", "examples": ["take a note", "remember this note", "save note"], "responses": [""]},
        {"tag": "note_list", "examples": ["list my notes", "show my notes", "what notes do i have"], "responses": [""]},
        {"tag": "note_delete", "examples": ["delete note", "remove note", "delete my note"], "responses": [""]},
        {"tag": "reminder_set", "examples": ["remind me to call mom at 6pm", "set a reminder tomorrow 9am"], "responses": [""]},
        {"tag": "reminder_list", "examples": ["list reminders", "what are my reminders"], "responses": [""]},
        {"tag": "system_info", "examples": ["system info", "cpu usage", "ram usage", "what's my cpu"], "responses": [""]},
        {"tag": "story", "examples": ["tell me a story", "tell me a fun fact", "fun fact"], "responses": [""]},
        {"tag": "stopwatch_start", "examples": ["start stopwatch", "start timer"], "responses": [""]},
        {"tag": "stopwatch_stop", "examples": ["stop stopwatch", "stop timer", "elapsed time"], "responses": [""]},
        {"tag": "timer_set", "examples": ["set timer 10 seconds", "timer for 5 minutes"], "responses": [""]},
        {"tag": "math_quiz", "examples": ["quiz me", "math quiz", "ask me a math question"], "responses": [""]},
        {"tag": "file_search", "examples": ["find file report", "search for resume", "open my presentation"], "responses": [""]},
    ]
}

# -------------------------
# NLU (INTENT DETECTION)
# -------------------------
print("[INFO] Loading sentence transformer for NLU...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

examples, labels = [], []
for intent in intents_data["intents"]:
    for ex in intent["examples"]:
        examples.append(ex)
        labels.append(intent["tag"])

embeddings = embedder.encode(examples, convert_to_numpy=True)
nn = NearestNeighbors(n_neighbors=1, metric="cosine").fit(embeddings)

def get_intent(text, threshold=0.55):
    emb = embedder.encode([text], convert_to_numpy=True)
    dist, idx = nn.kneighbors(emb, return_distance=True)
    sim = 1 - dist[0][0]
    label = labels[idx[0][0]]
    if sim < threshold:
        return None
    return next((it for it in intents_data["intents"] if it["tag"] == label), None)

# -------------------------
# Persistence helpers
# -------------------------
def load_notes():
    try:
        with open(NOTES_FILE, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return {}

def save_notes(notes):
    with open(NOTES_FILE, "w", encoding="utf-8") as fh:
        json.dump(notes, fh, indent=2)

def load_reminders():
    try:
        with open(REMINDERS_FILE, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return []

def save_reminders(reminders):
    with open(REMINDERS_FILE, "w", encoding="utf-8") as fh:
        json.dump(reminders, fh, indent=2)

def load_memory():
    try:
        with open(MEMORY_FILE, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return {}

def save_memory(mem):
    with open(MEMORY_FILE, "w", encoding="utf-8") as fh:
        json.dump(mem, fh, indent=2)

memory = load_memory()

# -------------------------
# Helper: Open URLs, apps, calculator (existing)
# -------------------------
def open_url(url):
    try:
        webbrowser.open(url, new=2)  # new tab
        return f"Opened {url}"
    except Exception as e:
        return f"Failed to open {url}: {e}"

def try_open_chrome():
    system = platform.system().lower()
    # try common executable names/locations
    chrome_execs = []
    if system == "windows":
        chrome_execs = [
            shutil.which("chrome"), shutil.which("chrome.exe"),
            os.path.join(os.environ.get("PROGRAMFILES", ""), "Google", "Chrome", "Application", "chrome.exe"),
            os.path.join(os.environ.get("PROGRAMFILES(X86)", ""), "Google", "Chrome", "Application", "chrome.exe")
        ]
    elif system == "darwin":
        # macOS
        chrome_execs = ["/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", shutil.which("google-chrome"), shutil.which("chrome")]
    else:
        # linux
        chrome_execs = [shutil.which("google-chrome"), shutil.which("chrome"), shutil.which("chromium"), shutil.which("chromium-browser")]

    for path in chrome_execs:
        if path and os.path.exists(path):
            try:
                subprocess.Popen([path])
                return True
            except:
                continue
    return False

def open_calculator():
    system = platform.system().lower()
    try:
        if system == "windows":
            subprocess.Popen(["calc.exe"])
            return "Opened Calculator."
        elif system == "darwin":
            subprocess.Popen(["open", "-a", "Calculator"])
            return "Opened Calculator."
        else:
            # try common linux calculators
            for cmd in (shutil.which("gnome-calculator"), shutil.which("kcalc"), shutil.which("galculator")):
                if cmd:
                    subprocess.Popen([cmd])
                    return "Opened Calculator."
            # fallback to web
            webbrowser.open("https://www.google.com/search?q=calculator", new=2)
            return "Opened web calculator."
    except Exception as e:
        return f"Failed to open calculator: {e}"

def open_service_from_text(text):
    t = text.lower()
    # mapping of service keywords to urls or special handlers
    services = {
        "google": "https://www.google.com",
        "gmail": "https://mail.google.com",
        "maps": "https://www.google.com/maps",
        "chatgpt": "https://chat.openai.com",
        "facebook": "https://www.facebook.com",
        "instagram": "https://www.instagram.com",
        "twitter": "https://twitter.com",
        "playstore": "https://play.google.com",
        "play store": "https://play.google.com",
        "perplexity": "https://www.perplexity.ai",
        "gemenai": "https://gemini.google.com",
        "gemeni": "https://gemini.google.com",
        "gemini": "https://gemini.google.com",
        "github": "https://github.com",
        "youtube": "https://www.youtube.com",
        "chrome": "chrome",  # special: try to open chrome app
        "calculator": "calculator",  # special
        "calendar": "https://calendar.google.com"
    }

    # detect which service user asked for (first match)
    for key, val in services.items():
        if key in t:
            if val == "chrome":
                ok = try_open_chrome()
                if ok:
                    return "Opened Chrome browser."
                else:
                    # fallback: open web google
                    return open_url("https://www.google.com")
            if val == "calculator":
                return open_calculator()
            # default: open url
            return open_url(val)

    # fallback: if text contains "open" then open google search for the phrase
    if t.startswith("open ") or t.startswith("launch "):
        # try to open whatever follows
        target = t.replace("open ", "").replace("launch ", "").strip()
        if target:
            # construct a google search
            return open_url(f"https://www.google.com/search?q={target.replace(' ', '+')}")
    return "I couldn't find that service to open."

# -------------------------
# WEB SEARCH (Fallback) - existing (uses DDGS)
# -------------------------
def web_search(query):
    print(f"[INFO] Searching online for: {query}")
    try:
        results = DDGS().text(query, max_results=3)  # top 3 results
        results = list(results)
    except Exception as e:
        return f"Web search failed: {e}"

    if not results:
        return "I couldn’t find anything online."

    # Print results
    for i, r in enumerate(results, start=1):
        print(f"\nResult {i}: {r.get('title','')}\n{r.get('body','')}\n{r.get('href','')}\n")

    # Speak only the top result summary
    top = results[0]
    title = top.get("title", "")
    body = top.get("body", "")
    return f"Here’s what I found: {title} - {body}"

# -------------------------
# NEWS (RSS) Reader
# -------------------------
def fetch_news(top_n=5):
    """
    Fetch top headlines from Google News RSS feed (no API key required).
    Returns list of (title, link).
    """
    url = "https://news.google.com/rss"
    try:
        resp = requests.get(url, timeout=6)
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
        items = []
        for item in root.findall(".//item")[:top_n]:
            title = item.find("title").text if item.find("title") is not None else ""
            link = item.find("link").text if item.find("link") is not None else ""
            items.append((title, link))
        return items
    except Exception as e:
        print("[WARN] fetch_news failed:", e)
        return []

def handle_news():
    news = fetch_news(5)
    if not news:
        return "I couldn't fetch news right now."
    msg_lines = ["Top headlines:"]
    for i, (t, l) in enumerate(news, start=1):
        msg_lines.append(f"{i}. {t}")
    return " ".join(msg_lines)

# -------------------------
# WEATHER (wttr.in JSON)
# -------------------------
def fetch_weather(location=""):
    """
    Use wttr.in JSON format. location optional (city name).
    """
    try:
        loc = location.strip() or ""
        url = f"http://wttr.in/{loc}?format=j1"
        resp = requests.get(url, timeout=6)
        resp.raise_for_status()
        data = resp.json()
        # best-effort parse
        current = data.get("current_condition", [{}])[0]
        temp_c = current.get("temp_C", "")
        weather_desc = current.get("weatherDesc", [{}])[0].get("value", "")
        feels_like = current.get("FeelsLikeC", "")
        return f"{weather_desc}. Temperature {temp_c}°C, feels like {feels_like}°C."
    except Exception as e:
        print("[WARN] fetch_weather failed:", e)
        return "I cannot fetch live weather right now."

# -------------------------
# DICTIONARY
# -------------------------
def define_word(word):
    """
    Uses dictionaryapi.dev free dictionary.
    """
    try:
        url = f"https://api.dictionaryapi.dev/api/v2/entries/en/{word}"
        resp = requests.get(url, timeout=6)
        resp.raise_for_status()
        data = resp.json()
        # take first meaning & definition
        meanings = data[0].get("meanings", [])
        if not meanings:
            return "No definition found."
        defs = meanings[0].get("definitions", [])
        if not defs:
            return "No definition found."
        definition = defs[0].get("definition", "")
        example = defs[0].get("example", "")
        out = f"{word}: {definition}"
        if example:
            out += f" Example: {example}"
        return out
    except Exception as e:
        print("[WARN] define_word failed:", e)
        return "I couldn't find the definition."

# -------------------------
# NOTES (save/list/delete)
# -------------------------
def save_note(text, title=None):
    notes = load_notes()
    if not title:
        title = f"note_{len(notes)+1}"
    notes[title] = {"text": text, "created": datetime.now().isoformat()}
    save_notes(notes)
    return f"Saved note as {title}."

def list_notes():
    notes = load_notes()
    if not notes:
        return "You have no notes."
    lines = []
    for k, v in notes.items():
        lines.append(f"{k}: {v.get('text')[:120]}")
    return "\n".join(lines)

def delete_note(title):
    notes = load_notes()
    if title in notes:
        notes.pop(title)
        save_notes(notes)
        return f"Deleted note {title}."
    else:
        return f"I couldn't find a note called {title}."

# -------------------------
# REMINDERS (background thread)
# -------------------------
_reminder_threads = []
_reminder_lock = threading.Lock()

def _reminder_worker(reminder):
    """
    reminder: dict with keys id, when (iso), message
    """
    try:
        target = datetime.fromisoformat(reminder["when"])
        now = datetime.now()
        delta = (target - now).total_seconds()
        if delta > 0:
            time.sleep(delta)
        # time to deliver
        speak(f"Reminder: {reminder['message']}")
    except Exception as e:
        print("[WARN] reminder worker error:", e)

def schedule_reminder(when_dt: datetime, message: str):
    reminders = load_reminders()
    reminder = {"id": int(time.time()*1000), "when": when_dt.isoformat(), "message": message}
    reminders.append(reminder)
    save_reminders(reminders)
    # start thread
    t = threading.Thread(target=_reminder_worker, args=(reminder,), daemon=True)
    t.start()
    with _reminder_lock:
        _reminder_threads.append((reminder, t))
    return f"Reminder set for {when_dt.strftime('%Y-%m-%d %H:%M:%S')}."

def list_reminders():
    reminders = load_reminders()
    if not reminders:
        return "No reminders set."
    lines = []
    for r in reminders:
        lines.append(f"{r['id']}: {r['message']} at {r['when']}")
    return "\n".join(lines)

def load_and_start_reminders_on_boot():
    reminders = load_reminders()
    for r in reminders:
        t = threading.Thread(target=_reminder_worker, args=(r,), daemon=True)
        t.start()
        with _reminder_lock:
            _reminder_threads.append((r, t))

# Start any persisted reminders
load_and_start_reminders_on_boot()

# -------------------------
# SYSTEM INFO
# -------------------------
def get_system_info():
    try:
        uname = platform.uname()
        info = [
            f"System: {uname.system} {uname.release}",
            f"Node: {uname.node}",
            f"Machine: {uname.machine}",
            f"Processor: {uname.processor}"
        ]
        if psutil:
            cpu = psutil.cpu_percent(interval=1)
            mem = psutil.virtual_memory()
            info.append(f"CPU usage: {cpu}%")
            info.append(f"Memory usage: {mem.percent}% ({int(mem.used/1024/1024)}MB used)")
        else:
            # fallback (limited)
            info.append("psutil not installed; install psutil for detailed stats.")
        return ". ".join(info)
    except Exception as e:
        print("[WARN] get_system_info failed:", e)
        return "Unable to fetch system information."

# -------------------------
# STORIES / FUN FACTS
# -------------------------
STORIES = [
    "Once upon a time, a curious robot learned to make coffee for humans and then decided to write poetry.",
    "Fun fact: Honey never spoils. Archaeologists have found edible honey in ancient tombs.",
    "A short fable: The tiny ant taught the giant elephant to dance - and they both found balance."
]

def tell_story():
    return random.choice(STORIES)

# -------------------------
# STOPWATCH / TIMER
# -------------------------
_stopwatch = {"running": False, "start": None, "elapsed": 0.0}
_stopwatch_lock = threading.Lock()

def stopwatch_start():
    with _stopwatch_lock:
        if _stopwatch["running"]:
            return "Stopwatch is already running."
        _stopwatch["running"] = True
        _stopwatch["start"] = time.time()
        return "Stopwatch started."

def stopwatch_stop():
    with _stopwatch_lock:
        if not _stopwatch["running"]:
            return f"Stopwatch not running. Last elapsed: {_stopwatch['elapsed']:.2f}s"
        elapsed = time.time() - _stopwatch["start"]
        _stopwatch["elapsed"] = elapsed
        _stopwatch["running"] = False
        _stopwatch["start"] = None
        return f"Stopwatch stopped. Elapsed: {elapsed:.2f} seconds."

def set_timer(seconds):
    def timer_worker(sec):
        time.sleep(sec)
        speak(f"Timer finished: {sec} seconds are up.")
    t = threading.Thread(target=timer_worker, args=(seconds,), daemon=True)
    t.start()
    return f"Timer set for {seconds} seconds."

# -------------------------
# MATH QUIZ (interactive)
# -------------------------
def math_quiz():
    # generate a simple random arithmetic
    a = random.randint(1, 12)
    b = random.randint(1, 12)
    op = random.choice(["+", "-", "*"])
    question = f"{a} {op} {b}"
    answer = eval(question)
    # ask user and wait for text response (synchronous)
    speak(f"What is {a} {op} {b}?")
    user_ans = input("Your answer: ").strip()
    try:
        user_val = float(user_ans)
        if abs(user_val - answer) < 1e-6:
            return "Correct! Nice work."
        else:
            return f"Not quite — the answer is {answer}."
    except:
        return f"Couldn't parse your answer. The correct answer is {answer}."

# -------------------------
# FILE SEARCH / OPEN
# -------------------------
def search_files(root_dir, pattern, max_results=10):
    matches = []
    pat = pattern.lower()
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for fn in filenames:
            if pat in fn.lower():
                full = os.path.join(dirpath, fn)
                matches.append(full)
                if len(matches) >= max_results:
                    return matches
    return matches

def open_file(path):
    try:
        system = platform.system().lower()
        if system == "windows":
            os.startfile(path)
        elif system == "darwin":
            subprocess.Popen(["open", path])
        else:
            subprocess.Popen(["xdg-open", path])
        return f"Opened {path}"
    except Exception as e:
        return f"Failed to open {path}: {e}"

# -------------------------
# CALCULATOR (existing functionality)
# -------------------------
def safe_calculate(user_text):
    try:
        expr = user_text.lower()
        # remove common words
        for token in ["calculate", "what is", "what's", "compute", "equals", "=", "please"]:
            expr = expr.replace(token, "")
        expr = expr.strip()
        # allow only digits, operators, parentheses, spaces, dot
        allowed = "0123456789+-*/(). %"
        cleaned = "".join(ch for ch in expr if ch in allowed)
        if not cleaned:
            return "Sorry, I couldn't parse the expression."
        # Evaluate safely (no builtins)
        result = eval(cleaned, {"__builtins__": {}})
        return f"The result is {result}"
    except Exception as e:
        print("[WARN] safe_calculate:", e)
        return "Sorry, I couldn't calculate that."

# -------------------------
# Handle intents (existing + new features)
# -------------------------
def handle_intent(intent_obj, user_text):
    tag = intent_obj["tag"]

    # Date
    if tag == "date":
        return f"Today's date is {datetime.now().strftime('%B %d, %Y')}."

    # Time
    elif tag == "time":
        return f"The current time is {datetime.now().strftime('%H:%M:%S')}."

    # Jokes
    elif tag == "joke":
        jokes = [
            "Why don’t skeletons fight each other? Because they don’t have the guts!",
            "I told my computer I needed a break, and it said 'No problem, I’ll go to sleep.'",
            "Why was the math book sad? Because it had too many problems."
        ]
        return random.choice(jokes)

    # Calculator (basic safe eval)
    elif tag == "calculator":
        return safe_calculate(user_text)

    # Memory (remember name)
    elif tag == "remember":
        # simple extraction: take last word as name
        words = user_text.split()
        if "name" in user_text:
            # try to find a name after 'is' or last token
            if "is" in words:
                try:
                    i = words.index("is")
                    name = words[i+1]
                except:
                    name = words[-1]
            else:
                name = words[-1]
            memory["name"] = name.capitalize()
            save_memory(memory)
            return f"Okay, I’ll remember your name is {memory['name']}."
        return "What should I remember?"

    # Recall memory
    elif tag == "recall":
        if "name" in memory:
            return f"Your name is {memory['name']}."
        else:
            return "I don’t remember your name yet."

    # Play YouTube
    elif tag == "play_youtube":
        query = user_text.lower().replace("play", "").replace("on youtube", "").strip()
        if query:
            speak(f"Playing {query} on YouTube.")
            try:
                pywhatkit.playonyt(query)
                return f"Now playing {query} on YouTube."
            except Exception as e:
                return f"Failed to play on YouTube: {e}"
        return "What video should I play?"

    # Open / Launch services and apps
    elif tag == "open_service":
        return open_service_from_text(user_text)

    # NEWS
    elif tag == "news":
        return handle_news()

    # WEATHER
    elif tag == "weather":
        # try to parse "in <city>"
        m = re.search(r"in\s+([A-Za-z\s]+)", user_text, re.IGNORECASE)
        city = m.group(1).strip() if m else ""
        return fetch_weather(city)

    # DEFINE
    elif tag == "define":
        # extract last word as the target word
        words = re.findall(r"[A-Za-z']+", user_text)
        # find the word after 'define' or 'meaning of' etc.
        word = ""
        if "define" in user_text.lower():
            parts = user_text.split()
            try:
                idx = [p.lower() for p in parts].index("define")
                word = parts[idx+1]
            except:
                word = parts[-1]
        elif "meaning" in user_text.lower() or "what does" in user_text.lower():
            # take last token
            word = words[-1] if words else ""
        else:
            word = words[-1] if words else ""
        if not word:
            return "Which word do you want me to define?"
        return define_word(word)

    # NOTES
    elif tag == "note_save":
        # expects user_text like "take a note buy milk"
        # naive parse: everything after 'note' or 'remember'...
        m = re.search(r"note (.*)", user_text, re.IGNORECASE)
        content = m.group(1).strip() if m else user_text
        # allow title if "title: <t>"
        title_match = re.search(r"title[:\-]\s*([^\n]+)", content, re.IGNORECASE)
        title = None
        if title_match:
            title = title_match.group(1).strip()
            content = re.sub(title_match.group(0), "", content).strip()
        return save_note(content, title)

    elif tag == "note_list":
        return list_notes()

    elif tag == "note_delete":
        m = re.search(r"delete note (.+)", user_text, re.IGNORECASE)
        if not m:
            return "Which note should I delete? Say 'delete note <note_title>'."
        title = m.group(1).strip()
        return delete_note(title)

    # REMINDERS
    elif tag == "reminder_set":
        # naive parsing: look for time patterns like "at 6pm", "in 10 minutes", "tomorrow 9am"
        txt = user_text.lower()
        # check "in X minutes/seconds/hours"
        m_in = re.search(r"in\s+(\d+)\s*(seconds?|minutes?|hours?)", txt)
        if m_in:
            val = int(m_in.group(1))
            unit = m_in.group(2)
            sec = val * (60 if "minute" in unit else 3600 if "hour" in unit else 1)
            when = datetime.now() + timedelta(seconds=sec)
            # message is after 'remind me to' or entire text
            msg_match = re.search(r"remind me to (.*?)(?: in \d+ )", user_text, re.IGNORECASE)
            msg = msg_match.group(1).strip() if msg_match else user_text
            return schedule_reminder(when, msg)
        # check "at HH:MM" or "at 6pm"
        m_at = re.search(r"at\s+(\d{1,2}(?::\d{2})?\s*(?:am|pm)?)", txt)
        if m_at:
            time_str = m_at.group(1)
            # parse roughly
            try:
                when_time = datetime.strptime(time_str.strip(), "%I%p") if re.match(r"^\d{1,2}\s*(am|pm)$", time_str.strip()) else None
            except Exception:
                when_time = None
            # fallback: set tomorrow at that time if already passed today
            try:
                # try common formats
                for fmt in ("%I%p", "%I:%M%p", "%H:%M"):
                    try:
                        t = datetime.strptime(time_str.strip(), fmt).time()
                        when = datetime.combine(datetime.now().date(), t)
                        if when < datetime.now():
                            when = when + timedelta(days=1)
                        return schedule_reminder(when, user_text)
                    except:
                        continue
            except Exception:
                pass
        # as last resort, ask for a clearer time (but per system instruction, don't ask — so create a quick reminder in 1 minute)
        fallback = datetime.now() + timedelta(minutes=1)
        return schedule_reminder(fallback, user_text)

    elif tag == "reminder_list":
        return list_reminders()

    # SYSTEM INFO
    elif tag == "system_info":
        return get_system_info()

    # STORY
    elif tag == "story":
        return tell_story()

    # STOPWATCH control
    elif tag == "stopwatch_start":
        return stopwatch_start()
    elif tag == "stopwatch_stop":
        return stopwatch_stop()

    # TIMER
    elif tag == "timer_set":
        # expect "set timer 5 minutes" or "set timer 10 seconds"
        m = re.search(r"(\d+)\s*(seconds?|minutes?|hours?)", user_text.lower())
        if not m:
            return "Please tell me how long the timer should be (e.g., 'timer 5 minutes')."
        val = int(m.group(1))
        unit = m.group(2)
        sec = val * (60 if "minute" in unit else 3600 if "hour" in unit else 1)
        return set_timer(sec)

    # MATH QUIZ
    elif tag == "math_quiz":
        return math_quiz()

    # FILE SEARCH
    elif tag == "file_search":
        # parse query
        m = re.search(r"find file (.+)", user_text, re.IGNORECASE)
        query = m.group(1).strip() if m else user_text
        # search home directory by default
        home = os.path.expanduser("~")
        results = search_files(home, query, max_results=5)
        if not results:
            return f"No files matching '{query}' found under {home}."
        # open the first one automatically and list others
        first = results[0]
        others = results[1:]
        msg = open_file(first)
        if others:
            msg += "\nOther matches:\n" + "\n".join(others)
        return msg

    # Default intent response (existing)
    else:
        return intent_obj["responses"][0]

# -------------------------
# MAIN LOOP
# -------------------------
def assistant():
    speak("Hello! I'm your upgraded assistant. You can ask me to open apps or websites. Say 'exit' anytime to quit.")

    while True:
        print("\nChoose input method: (1) Text  (2) Voice")
        choice = input("Enter 1 or 2: ").strip()

        if choice == "1":
            user_text = input("You: ").strip()
        elif choice == "2":
            wav = record()
            user_text = transcribe(wav)
            print("You (voice):", user_text)
        else:
            print("Invalid choice.")
            continue

        if not user_text:
            speak("Sorry, I didn’t catch that.")
            continue

        # check for forced open/play commands regardless of NLU
        low = user_text.lower()
        if low.startswith("play ") and "youtube" in low:
            # play youtube directly
            try:
                # reuse play_youtube behavior
                query = low.replace("play", "").replace("on youtube", "").strip()
                if query:
                    speak(f"Playing {query} on YouTube.")
                    pywhatkit.playonyt(query)
                    continue
            except Exception as e:
                speak(f"Couldn't play on YouTube: {e}")
                continue
        if low.startswith("open ") or low.startswith("launch ") or "open " in low or "launch " in low:
            # try to open service directly
            resp = open_service_from_text(low)
            speak(resp)
            continue

        # normal NLU flow
        intent_obj = get_intent(low)
        if intent_obj is None:
            # fallback to web search
            response = web_search(user_text)
            speak(response)
            continue

        response = handle_intent(intent_obj, user_text)
        speak(response)

        if intent_obj["tag"] == "exit":
            break

if __name__ == "__main__":
    assistant()
