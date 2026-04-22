# =============================================================================
# ZYNBOX v2.0 – Personal Productivity AI Agent
# Team: Utkarsh Sharma, Zeeshan, Utkarsh Pandey
# =============================================================================

import streamlit as st
import json
import os
import pickle
import re
import random
import datetime

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from gensim.models import Word2Vec

# =============================================================================
# SECTION 1: CONFIGURATION & CONSTANTS
# =============================================================================

MODEL_PATH = "zynbox_models.pkl"
TASKS_PATH = "zynbox_tasks.json"
NOTES_PATH = "zynbox_notes.json"

MOTIVATIONAL = [
    "Deep work beats busy work. Stay focused! 🎯",
    "Every small step counts. Stay consistent! 💪",
    "Progress, not perfection. Keep moving! ✨",
    "One task at a time. You're crushing it! 🔥",
    "Ideas without action are just dreams. Build! 💡",
    "The best time to start was yesterday. Second best: NOW! ⚡",
    "Small daily improvements lead to stunning results! 📈",
    "Your future self is watching. Make them proud! 👑",
    "Believe in yourself – you've got this! 🚀",
    "Stay curious. Stay hungry. Keep creating! 🌟",
]

# =============================================================================
# SECTION 2: EXPANDED TRAINING DATA
# =============================================================================

TRAINING_DATA = [
    # Task intents
    ("add a new task",                "add_task",     "neutral"),
    ("I want to add task",            "add_task",     "neutral"),
    ("create a new task",             "add_task",     "neutral"),
    ("add task to my list",           "add_task",     "neutral"),
    ("new task please",               "add_task",     "neutral"),
    ("remind me to do something",     "add_task",     "neutral"),
    ("I need to do something",        "add_task",     "neutral"),
    ("put something on my list",      "add_task",     "neutral"),
    ("schedule a task",               "add_task",     "neutral"),
    ("create to-do",                  "add_task",     "neutral"),

    ("show my tasks",                 "show_tasks",   "neutral"),
    ("list all tasks",                "show_tasks",   "neutral"),
    ("what are my tasks",             "show_tasks",   "neutral"),
    ("display tasks",                 "show_tasks",   "neutral"),
    ("show me my to do list",         "show_tasks",   "neutral"),
    ("what do I have to do",          "show_tasks",   "neutral"),
    ("pending tasks",                 "show_tasks",   "neutral"),
    ("view tasks",                    "show_tasks",   "neutral"),

    ("delete a task",                 "delete_task",  "neutral"),
    ("remove task",                   "delete_task",  "neutral"),
    ("I want to delete task",         "delete_task",  "neutral"),
    ("clear a task",                  "delete_task",  "neutral"),

    ("done task",                     "done_task",    "neutral"),
    ("mark task complete",            "done_task",    "neutral"),
    ("finish task",                   "done_task",    "neutral"),
    ("complete task",                 "done_task",    "neutral"),
    ("task is finished",              "done_task",    "neutral"),

    # Notes
    ("take a note",                   "add_note",     "neutral"),
    ("save this note",                "add_note",     "neutral"),
    ("note this down",                "add_note",     "neutral"),
    ("write a note",                  "add_note",     "neutral"),
    ("add a note",                    "add_note",     "neutral"),
    ("remember this",                 "add_note",     "neutral"),
    ("show my notes",                 "show_notes",   "neutral"),
    ("list notes",                    "show_notes",   "neutral"),
    ("view my notes",                 "show_notes",   "neutral"),

    # Mood – Study
    ("I want to study",               "mood",         "study"),
    ("let us study",                  "mood",         "study"),
    ("time to focus",                 "mood",         "study"),
    ("I need to concentrate",         "mood",         "study"),
    ("help me learn",                 "mood",         "study"),
    ("I want to be productive",       "mood",         "study"),
    ("I have an exam",                "mood",         "study"),
    ("I need to prepare",             "mood",         "study"),
    ("switch to study mode",          "mood",         "study"),
    ("enable study mode",             "mood",         "study"),
    ("I need to focus",               "mood",         "study"),
    ("deep work mode",                "mood",         "study"),
    ("focus mode on",                 "mood",         "study"),

    # Mood – Creative
    ("I feel creative",               "mood",         "creative"),
    ("I am bored",                    "mood",         "creative"),
    ("I want to be creative",         "mood",         "creative"),
    ("I have an idea",                "mood",         "creative"),
    ("let us brainstorm",             "mood",         "creative"),
    ("I want to create something",    "mood",         "creative"),
    ("feeling artistic",              "mood",         "creative"),
    ("I want to explore",             "mood",         "creative"),
    ("switch to creative mode",       "mood",         "creative"),
    ("enable creative mode",          "mood",         "creative"),
    ("I want to design",              "mood",         "creative"),
    ("inspiration time",              "mood",         "creative"),
    ("imagination mode",              "mood",         "creative"),

    # Greetings
    ("hello",                         "greeting",     "neutral"),
    ("hi there",                      "greeting",     "neutral"),
    ("hey",                           "greeting",     "neutral"),
    ("good morning",                  "greeting",     "neutral"),
    ("good evening",                  "greeting",     "neutral"),
    ("howdy",                         "greeting",     "neutral"),
    ("what is up",                    "greeting",     "neutral"),
    ("greetings",                     "greeting",     "neutral"),
    ("yo",                            "greeting",     "neutral"),
    ("sup",                           "greeting",     "neutral"),

    # Pomodoro
    ("start pomodoro",                "pomodoro",     "study"),
    ("start timer",                   "pomodoro",     "study"),
    ("focus timer",                   "pomodoro",     "study"),
    ("25 minute timer",               "pomodoro",     "study"),
    ("start focus session",           "pomodoro",     "study"),
    ("pomodoro timer",                "pomodoro",     "study"),

    # Stats
    ("show stats",                    "stats",        "neutral"),
    ("my productivity",               "stats",        "neutral"),
    ("progress report",               "stats",        "neutral"),
    ("how am I doing",                "stats",        "neutral"),
    ("show analytics",                "stats",        "neutral"),
    ("my performance",                "stats",        "neutral"),

    # Help
    ("help me",                       "help",         "neutral"),
    ("what can you do",               "help",         "neutral"),
    ("how does this work",            "help",         "neutral"),
    ("what are your features",        "help",         "neutral"),
    ("commands",                      "help",         "neutral"),
    ("show guide",                    "help",         "neutral"),

    # General AI conversation
    ("tell me about",                 "general",      "neutral"),
    ("what is",                       "general",      "neutral"),
    ("how do I",                      "general",      "neutral"),
    ("explain this",                  "general",      "neutral"),
    ("can you help me with",          "general",      "neutral"),
    ("I need advice",                 "general",      "neutral"),
    ("write for me",                  "general",      "creative"),
    ("generate ideas",                "general",      "creative"),
    ("create content",                "general",      "creative"),
    ("summarize this",                "general",      "neutral"),
    ("give me ideas",                 "general",      "creative"),
    ("what do you think about",       "general",      "neutral"),
    ("can you explain",               "general",      "neutral"),
    ("help me understand",            "general",      "neutral"),
]

# =============================================================================
# SECTION 3: THEMES – Study & Creative Only
# =============================================================================

THEMES = {
    "study": {
        "name":        "Study Mode",
        "bg":          "#07111f",
        "card":        "#0c1c30",
        "card2":       "#102338",
        "accent":      "#3b82f6",
        "accent2":     "#60a5fa",
        "text":        "#e2e8f0",
        "subtext":     "#94a3b8",
        "user_bubble": "#1d4ed8",
        "bot_bubble":  "#0c1c30",
        "border":      "#1a3556",
        "glow":        "rgba(59,130,246,0.18)",
        "tone":        "focused",
        "greeting":    "Study Mode activated. Let's get into deep focus! 📚",
        "emoji":       "📚",
        "gradient":    "linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%)",
        "grad_soft":   "linear-gradient(135deg, rgba(59,130,246,0.12), rgba(29,78,216,0.06))",
    },
    "creative": {
        "name":        "Creative Mode",
        "bg":          "#0c0418",
        "card":        "#160828",
        "card2":       "#1e0d38",
        "accent":      "#a855f7",
        "accent2":     "#c084fc",
        "text":        "#f3e8ff",
        "subtext":     "#c084fc",
        "user_bubble": "#6d28d9",
        "bot_bubble":  "#160828",
        "border":      "#361565",
        "glow":        "rgba(168,85,247,0.18)",
        "tone":        "creative",
        "greeting":    "Creative Mode on. Let your imagination run wild! 🎨",
        "emoji":       "🎨",
        "gradient":    "linear-gradient(135deg, #a855f7 0%, #6d28d9 100%)",
        "grad_soft":   "linear-gradient(135deg, rgba(168,85,247,0.12), rgba(109,40,217,0.06))",
    },
}

MOOD_TO_THEME = {
    "study":    "study",
    "creative": "creative",
    "neutral":  "study",
}

# =============================================================================
# SECTION 4: NLP PIPELINE
# =============================================================================

STOPWORDS = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves",
    "you", "your", "yours", "he", "him", "his", "she", "her", "hers",
    "it", "its", "they", "them", "their", "what", "which", "who",
    "is", "are", "was", "were", "be", "been", "being", "have", "has",
    "do", "does", "did", "a", "an", "the", "and", "but", "if", "or",
    "because", "as", "of", "at", "by", "for", "with", "about", "to",
    "from", "up", "in", "out", "on", "off", "then", "so", "no", "not",
    "very", "just", "can", "will", "should", "now", "that", "this",
    "these", "those", "am", "want", "feel", "feeling", "need",
}

def preprocess_text(text: str) -> str:
    """Lowercase → strip punctuation → tokenize → remove stopwords."""
    text   = text.lower()
    text   = re.sub(r"[^a-z\s]", "", text)
    tokens = [t for t in text.split() if t not in STOPWORDS]
    return " ".join(tokens)

def tokenize(text: str) -> list:
    return preprocess_text(text).split()

# =============================================================================
# SECTION 5: ML MODEL TRAINING & INFERENCE
# =============================================================================

def build_tfidf_vectorizer(corpus):
    v = TfidfVectorizer()
    v.fit(corpus)
    return v

def build_word2vec_model(tokenized_corpus):
    return Word2Vec(
        sentences=tokenized_corpus,
        vector_size=50, window=3, min_count=1, workers=1, seed=42
    )

def sentence_to_w2v_vector(sentence: str, w2v_model, vector_size=50):
    tokens  = tokenize(sentence)
    vectors = [w2v_model.wv[w] for w in tokens if w in w2v_model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(vector_size)

def train_models():
    raw_texts     = [r[0] for r in TRAINING_DATA]
    intent_labels = [r[1] for r in TRAINING_DATA]
    mood_labels   = [r[2] for r in TRAINING_DATA]
    clean_texts   = [preprocess_text(t) for t in raw_texts]

    tfidf   = build_tfidf_vectorizer(clean_texts)
    X_tfidf = tfidf.transform(clean_texts).toarray()
    w2v     = build_word2vec_model([tokenize(t) for t in raw_texts])

    intent_enc = LabelEncoder()
    mood_enc   = LabelEncoder()
    y_intent   = intent_enc.fit_transform(intent_labels)
    y_mood     = mood_enc.fit_transform(mood_labels)

    intent_clf = LogisticRegression(max_iter=500)
    intent_clf.fit(X_tfidf, y_intent)
    mood_clf   = LogisticRegression(max_iter=500)
    mood_clf.fit(X_tfidf, y_mood)

    bundle = {
        "tfidf": tfidf, "w2v": w2v,
        "intent_clf": intent_clf, "intent_enc": intent_enc,
        "mood_clf": mood_clf, "mood_enc": mood_enc,
    }
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(bundle, f)
    return bundle

def load_models():
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
    return train_models()

def predict_intent(text: str, bundle: dict) -> str:
    vec  = bundle["tfidf"].transform([preprocess_text(text)]).toarray()
    pred = bundle["intent_clf"].predict(vec)[0]
    return bundle["intent_enc"].inverse_transform([pred])[0]

def predict_mood(text: str, bundle: dict) -> str:
    vec  = bundle["tfidf"].transform([preprocess_text(text)]).toarray()
    pred = bundle["mood_clf"].predict(vec)[0]
    return bundle["mood_enc"].inverse_transform([pred])[0]

def predict_intent_w2v(text: str, bundle: dict) -> str:
    if "w2v_clf_intent" not in st.session_state:
        raw_texts     = [r[0] for r in TRAINING_DATA]
        intent_labels = [r[1] for r in TRAINING_DATA]
        X   = np.array([sentence_to_w2v_vector(t, bundle["w2v"]) for t in raw_texts])
        clf = LogisticRegression(max_iter=500)
        clf.fit(X, bundle["intent_enc"].transform(intent_labels))
        st.session_state["w2v_clf_intent"] = clf
    clf  = st.session_state["w2v_clf_intent"]
    vec  = sentence_to_w2v_vector(text, bundle["w2v"]).reshape(1, -1)
    pred = clf.predict(vec)[0]
    return bundle["intent_enc"].inverse_transform([pred])[0]

# =============================================================================
# SECTION 6: TASK MANAGER
# =============================================================================

def load_tasks() -> list:
    if os.path.exists(TASKS_PATH):
        with open(TASKS_PATH, "r") as f:
            return json.load(f)
    return []

def save_tasks(tasks: list):
    with open(TASKS_PATH, "w") as f:
        json.dump(tasks, f, indent=2)

def add_task(task_text: str, priority: str = "medium") -> str:
    tasks = load_tasks()
    task  = {
        "id":       len(tasks) + 1,
        "text":     task_text,
        "done":     False,
        "priority": priority,
        "created":  datetime.datetime.now().strftime("%d %b, %H:%M"),
    }
    tasks.append(task)
    save_tasks(tasks)
    pri_badge = {"high": "🔴 High", "medium": "🟡 Medium", "low": "🟢 Low"}.get(priority, "🟡 Medium")
    return f"✅ **Task added!**\n\n📌 {task_text}\n⚑ Priority: {pri_badge}"

def show_tasks() -> str:
    tasks = load_tasks()
    if not tasks:
        return (
            "📭 **No tasks yet!**\n\n"
            "Add one with:\n"
            "`add task: Buy groceries`\n"
            "`add task: Study physics [high]`"
        )
    order = {"high": 0, "medium": 1, "low": 2}
    sorted_tasks = sorted(
        tasks,
        key=lambda x: (x.get("done", False), order.get(x.get("priority", "medium"), 1))
    )
    lines = ["📋 **Your Tasks:**\n"]
    for task in sorted_tasks:
        status = "✅" if task["done"] else "⏳"
        pri    = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(task.get("priority", "medium"), "🟡")
        strike = "~~" if task["done"] else ""
        lines.append(f"{status} {pri} `#{task['id']}` {strike}{task['text']}{strike}")
    done_count = sum(1 for t in tasks if t["done"])
    lines.append(f"\n_Progress: **{done_count}/{len(tasks)}** tasks done_")
    return "\n".join(lines)

def delete_task(task_id: int) -> str:
    tasks     = load_tasks()
    new_tasks = [t for t in tasks if t["id"] != task_id]
    if len(new_tasks) == len(tasks):
        return f"❌ Task `#{task_id}` not found. Use `show tasks` to see IDs."
    save_tasks(new_tasks)
    return f"🗑️ Task `#{task_id}` removed successfully."

def mark_task_done(task_id: int) -> str:
    tasks = load_tasks()
    for t in tasks:
        if t["id"] == task_id:
            t["done"] = True
            save_tasks(tasks)
            score = get_productivity_score()
            return (
                f"🎉 **Task #{task_id} completed!**\n\n"
                f"Great job! Your productivity score is now **{score}%** 🚀"
            )
    return f"❌ Task `#{task_id}` not found."

def get_productivity_score() -> float:
    tasks = load_tasks()
    if not tasks:
        return 0.0
    done = sum(1 for t in tasks if t["done"])
    return round((done / len(tasks)) * 100, 1)

# =============================================================================
# SECTION 7: NOTES MANAGER
# =============================================================================

def load_notes() -> list:
    if os.path.exists(NOTES_PATH):
        with open(NOTES_PATH, "r") as f:
            return json.load(f)
    return []

def save_notes(notes: list):
    with open(NOTES_PATH, "w") as f:
        json.dump(notes, f, indent=2)

def add_note(note_text: str) -> str:
    notes = load_notes()
    note  = {
        "id":      len(notes) + 1,
        "text":    note_text,
        "created": datetime.datetime.now().strftime("%d %b, %H:%M"),
    }
    notes.append(note)
    save_notes(notes)
    return f"📝 **Note saved!**\n\n_{note_text}_"

def show_notes() -> str:
    notes = load_notes()
    if not notes:
        return "📭 **No notes yet!**\n\nTry: `note: Remember to review chapter 5`"
    lines = [f"📝 **Your Notes** _(last {min(10, len(notes))})_:\n"]
    for n in notes[-10:]:
        lines.append(f"• `#{n['id']}` [{n['created']}] — {n['text']}")
    return "\n".join(lines)

def delete_note(note_id: int) -> str:
    notes     = load_notes()
    new_notes = [n for n in notes if n["id"] != note_id]
    if len(new_notes) == len(notes):
        return f"❌ Note `#{note_id}` not found."
    save_notes(new_notes)
    return f"🗑️ Note `#{note_id}` deleted."

# =============================================================================
# SECTION 8: CLAUDE AI INTEGRATION
# =============================================================================

def get_claude_response(user_input: str, theme_key: str, chat_history: list, api_key: str) -> str:
    """
    Call the Anthropic Claude API for intelligent, context-aware responses.
    Passes the last 8 messages as conversation history for continuity.
    """
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)

        # Mode-specific system prompt
        if theme_key == "study":
            system_prompt = (
                "You are Zynbox, an intelligent study and productivity assistant. "
                "Help users learn concepts, plan study sessions, solve problems, and stay productive. "
                "Be concise, clear, and precise. Use bullet points or numbered lists when helpful. "
                "Provide examples where they add value. Keep answers focused and actionable. "
                "Format with markdown (bold, code blocks, lists) when it improves clarity."
            )
        else:
            system_prompt = (
                "You are Zynbox, a creative AI partner. "
                "Help users brainstorm ideas, write content, explore creative concepts, "
                "and unlock their imagination. Be inspiring, expressive, and enthusiastic. "
                "Offer fresh perspectives and imaginative solutions. "
                "Use vivid, energetic language that sparks creativity."
            )

        # Build message history (last 8 turns for context)
        messages = []
        for msg in chat_history[-8:]:
            role = "user" if msg["role"] == "user" else "assistant"
            messages.append({"role": role, "content": msg["text"]})

        messages.append({"role": "user", "content": user_input})

        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            system=system_prompt,
            messages=messages,
        )
        return response.content[0].text

    except ImportError:
        return (
            "⚠️ **Anthropic package not installed.**\n\n"
            "Run: `pip install anthropic` to enable AI responses."
        )
    except Exception as e:
        err = str(e)
        if "authentication" in err.lower() or "api_key" in err.lower() or "401" in err:
            return "❌ **Invalid API key.** Please check your Anthropic API key in the sidebar."
        if "rate" in err.lower():
            return "⏳ **Rate limit reached.** Please wait a moment and try again."
        return (
            f"⚠️ **AI temporarily unavailable.**\n\n"
            f"Local mode active. Type `help` for commands or check your API key."
        )

def get_creative_spark(api_key: str) -> str:
    """Generate a unique creative prompt using Claude or local fallback."""
    local_prompts = [
        "Write a story that starts with: 'The last algorithm made a decision nobody expected.'",
        "Design an app that solves the problem people don't know they have.",
        "What if your city had a secret underground creative district?",
        "Create a product that makes waiting enjoyable.",
        "Write a poem from the perspective of a forgotten bookshelf.",
        "Invent a new form of social media that makes people calmer, not anxious.",
        "Describe your dream workspace in exactly 50 words.",
    ]
    if not api_key:
        return f"💡 **Creative Spark:** {random.choice(local_prompts)}"
    try:
        import anthropic
        client   = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=120,
            messages=[{
                "role": "user",
                "content": "Give me one unique, inspiring creative prompt or challenge. One sentence only, no preamble."
            }]
        )
        return f"💡 **Creative Spark:** {response.content[0].text.strip()}"
    except Exception:
        return f"💡 **Creative Spark:** {random.choice(local_prompts)}"

# =============================================================================
# SECTION 9: CHATBOT RESPONSE ENGINE
# =============================================================================

def chatbot_response(user_input: str, bundle: dict, theme_key: str, api_key: str = "") -> dict:
    """
    Hybrid response engine:
    1. Rule-based pattern matching (highest priority – commands)
    2. ML-based intent/mood detection (theme switching)
    3. Claude AI fallback (general conversation)
    4. Local fallback (no API key)
    """
    text_lower = user_input.lower().strip()
    new_theme  = theme_key

    detected_mood = predict_mood(user_input, bundle)
    tfidf_intent  = predict_intent(user_input, bundle)
    w2v_intent    = predict_intent_w2v(user_input, bundle)

    def build(reply, intent=tfidf_intent, mood=detected_mood, theme=new_theme):
        return {
            "reply":        reply,
            "intent":       intent,
            "mood":         mood,
            "tfidf_intent": tfidf_intent,
            "w2v_intent":   w2v_intent,
            "theme_switch": theme,
        }

    # ── 1A. TASK: Add ─────────────────────────────────────────────────────────
    m = re.search(
        r"(?:add\s+task|new\s+task|create\s+task)\s*[:\-]?\s*(.+?)(?:\s*\[(high|medium|low)\])?$",
        text_lower
    )
    if m:
        task_text = m.group(1).strip().title()
        priority  = m.group(2) or "medium"
        return build(add_task(task_text, priority), "add_task")

    # ── 1B. TASK: Show ───────────────────────────────────────────────────────
    if any(p in text_lower for p in [
        "show task", "list task", "my task", "all task", "display task",
        "view task", "pending task", "what tasks", "see my task"
    ]):
        return build(show_tasks(), "show_tasks")

    # ── 1C. TASK: Delete ─────────────────────────────────────────────────────
    m = re.search(r"(?:delete|remove)\s+task\s*[:\-#]?\s*(\d+)", text_lower)
    if m:
        return build(delete_task(int(m.group(1))), "delete_task")

    # ── 1D. TASK: Done ───────────────────────────────────────────────────────
    m = re.search(r"(?:done|finish|mark|complete)\s+task\s*[:\-#]?\s*(\d+)", text_lower)
    if m:
        return build(mark_task_done(int(m.group(1))), "done_task")

    # ── 2A. NOTE: Add ────────────────────────────────────────────────────────
    m = re.search(r"(?:note|add\s+note|save\s+note|write\s+note|remember)\s*[:\-]?\s*(.+)", text_lower)
    if m:
        return build(add_note(m.group(1).strip()), "add_note")

    # ── 2B. NOTE: Show ───────────────────────────────────────────────────────
    if any(p in text_lower for p in ["show note", "list note", "my note", "view note", "all note"]):
        return build(show_notes(), "show_notes")

    # ── 2C. NOTE: Delete ─────────────────────────────────────────────────────
    m = re.search(r"delete\s+note\s*[:\-#]?\s*(\d+)", text_lower)
    if m:
        return build(delete_note(int(m.group(1))), "delete_note")

    # ── 3. GREETINGS ─────────────────────────────────────────────────────────
    if any(p in text_lower for p in [
        "hello", "hi", "hey", "good morning", "good afternoon",
        "good evening", "good night", "howdy", "yo", "sup", "greetings"
    ]):
        hour = datetime.datetime.now().hour
        time_greet = "Good morning" if hour < 12 else "Good afternoon" if hour < 17 else "Good evening"
        t = THEMES[theme_key]
        reply = (
            f"{time_greet}! 👋\n\n"
            f"{t['greeting']}\n\n"
            f"{'🟢 **AI Mode Active** — Ask me anything!' if api_key else '💡 Add your **Anthropic API key** in the sidebar for full AI chat.'}\n\n"
            f"Type `help` to see all commands."
        )
        return build(reply, "greeting")

    # ── 4. POMODORO ───────────────────────────────────────────────────────────
    if any(p in text_lower for p in [
        "pomodoro", "focus timer", "start timer", "25 min", "work timer", "start focus"
    ]):
        st.session_state.pomodoro_active = True
        st.session_state.pomodoro_start  = datetime.datetime.now()
        return build(
            "⏱️ **Pomodoro started!**\n\n"
            "Your focus session is running. No distractions — let's lock in. 🎯\n\n"
            "_Check the sidebar timer. I'll be here when you're done._",
            "pomodoro"
        )

    # ── 5. STATS ──────────────────────────────────────────────────────────────
    if any(p in text_lower for p in [
        "stats", "analytics", "productivity score", "how am i doing",
        "my performance", "progress report", "show analytics"
    ]):
        tasks  = load_tasks()
        notes  = load_notes()
        done   = sum(1 for t in tasks if t["done"])
        score  = get_productivity_score()
        high   = sum(1 for t in tasks if t.get("priority") == "high" and not t["done"])
        verdict = (
            "🌟 **Outstanding!** You're in top productivity mode!"
            if score >= 75 else
            "💪 **Good progress!** Keep completing those tasks."
            if score >= 40 else
            "🚀 **Just getting started!** Let's knock out some tasks."
        )
        sessions = st.session_state.get("sessions_done", 0)
        focus_m  = st.session_state.get("total_focus_mins", 0)
        return build(
            f"📊 **Productivity Dashboard**\n\n"
            f"✅ Tasks Done: **{done} / {len(tasks)}**\n"
            f"🔴 High-Priority Pending: **{high}**\n"
            f"📝 Notes Saved: **{len(notes)}**\n"
            f"⏱️ Focus Sessions: **{sessions}** ({focus_m} mins)\n"
            f"💬 Messages: **{len(st.session_state.chat_history)}**\n"
            f"🎯 Score: **{score}%**\n\n"
            f"{verdict}",
            "stats"
        )

    # ── 6. HELP ───────────────────────────────────────────────────────────────
    if any(p in text_lower for p in [
        "help", "what can you do", "features", "commands", "guide", "how to use"
    ]):
        return build(
            "🤖 **Zynbox Command Guide**\n\n"
            "**📋 Task Management**\n"
            "`add task: <text>` – Add a task\n"
            "`add task: <text> [high]` – Add with priority\n"
            "`show tasks` – View all tasks\n"
            "`done task <id>` – Mark complete\n"
            "`delete task <id>` – Remove task\n\n"
            "**📝 Notes**\n"
            "`note: <text>` – Quick note\n"
            "`show notes` – View notes\n"
            "`delete note <id>` – Remove note\n\n"
            "**🎨 Modes**\n"
            "`study mode` → Focus & productivity\n"
            "`creative mode` → Imagination & ideas\n\n"
            "**⏱️ Focus**\n"
            "`pomodoro` – Start focus timer\n"
            "`stats` – Your productivity report\n\n"
            "**🤖 AI Chat**\n"
            "Just type anything! With your API key connected, I can answer questions, "
            "explain concepts, brainstorm ideas, write content, and hold full conversations.\n\n"
            f"_Status: {'🟢 AI Active' if api_key else '🔴 Add API key in sidebar'}_",
            "help"
        )

    # ── 7. THEME SWITCHING (rule-based) ───────────────────────────────────────
    if any(p in text_lower for p in [
        "study mode", "switch to study", "enable study", "i want to study",
        "focus mode", "i need to focus", "deep work"
    ]):
        return build(f"📚 {THEMES['study']['greeting']}", "mood", "study", "study")

    if any(p in text_lower for p in [
        "creative mode", "switch to creative", "enable creative",
        "i want to create", "i feel creative", "imagination mode"
    ]):
        return build(f"🎨 {THEMES['creative']['greeting']}", "mood", "creative", "creative")

    # ── 8. ML-BASED MOOD / THEME SWITCHING ───────────────────────────────────
    if tfidf_intent == "mood" and detected_mood in MOOD_TO_THEME:
        new_theme = MOOD_TO_THEME[detected_mood]
        t_info    = THEMES[new_theme]
        replies   = {
            "study":    f"📚 Switching to **Study Mode**!\n\n{t_info['greeting']}",
            "creative": f"🎨 Switching to **Creative Mode**!\n\n{t_info['greeting']}",
            "neutral":  t_info["greeting"],
        }
        return build(replies.get(detected_mood, t_info["greeting"]), "mood", detected_mood, new_theme)

    # ── 9. ML-BASED INTENTS ───────────────────────────────────────────────────
    if tfidf_intent == "add_task":
        return build(
            "Sure! What's the task?\n\nUse: `add task: <your task>`\n"
            "Add priority with: `add task: <text> [high]`"
        )
    if tfidf_intent == "show_tasks":
        return build(show_tasks())
    if tfidf_intent == "delete_task":
        return build("Which task to delete? Use: `delete task <id>`\nSee IDs with `show tasks`.")
    if tfidf_intent == "stats":
        tasks = load_tasks()
        done  = sum(1 for t in tasks if t["done"])
        score = get_productivity_score()
        return build(f"📊 Score: **{score}%** | Tasks: {done}/{len(tasks)} done")
    if tfidf_intent == "pomodoro":
        st.session_state.pomodoro_active = True
        st.session_state.pomodoro_start  = datetime.datetime.now()
        return build("⏱️ Pomodoro started! Check the sidebar timer.", "pomodoro")

    # ── 10. CLAUDE AI FALLBACK ────────────────────────────────────────────────
    if api_key:
        ai_reply = get_claude_response(user_input, theme_key, st.session_state.chat_history, api_key)
        return build(ai_reply, "ai_response")

    # ── 11. LOCAL FALLBACK ────────────────────────────────────────────────────
    quote = random.choice(MOTIVATIONAL)
    return build(
        f"🤔 I'm not sure I caught that fully.\n\n"
        f"💡 **Pro tip:** Add your **Anthropic API key** in the sidebar to unlock full AI conversation mode — "
        f"then you can ask me literally anything!\n\n"
        f"Or type `help` to see available commands.\n\n"
        f"_{quote}_",
        "unknown"
    )

# =============================================================================
# SECTION 10: THEME CSS ENGINE
# =============================================================================

def get_theme_css(theme_key: str) -> str:
    t = THEMES[theme_key]
    return f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

    *, *::before, *::after {{
        font-family: 'Inter', 'Segoe UI', system-ui, sans-serif !important;
        box-sizing: border-box;
    }}

    /* ── Global ── */
    .stApp {{
        background-color: {t['bg']} !important;
        color: {t['text']} !important;
    }}
    .main .block-container {{
        padding-top: 1.5rem !important;
        padding-bottom: 1rem !important;
    }}

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, {t['card']} 0%, {t['bg']} 100%) !important;
        border-right: 1px solid {t['border']} !important;
    }}
    [data-testid="stSidebar"] * {{ color: {t['text']} !important; }}

    /* ── Inputs ── */
    .stTextInput input,
    .stTextArea textarea {{
        background-color: {t['card']} !important;
        color: {t['text']} !important;
        border: 1px solid {t['border']} !important;
        border-radius: 12px !important;
        padding: 10px 16px !important;
        font-size: 14px !important;
        transition: border-color 0.2s, box-shadow 0.2s;
    }}
    .stTextInput input:focus,
    .stTextArea textarea:focus {{
        border-color: {t['accent']} !important;
        box-shadow: 0 0 0 3px {t['glow']} !important;
        outline: none !important;
    }}
    .stTextInput input[type="password"] {{
        background-color: {t['card']} !important;
        color: {t['text']} !important;
        border: 1px solid {t['border']} !important;
    }}

    /* ── Buttons ── */
    .stButton > button,
    [data-testid="stFormSubmitButton"] > button {{
        background: {t['gradient']} !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
        font-size: 13px !important;
        padding: 9px 18px !important;
        letter-spacing: 0.3px;
        transition: opacity 0.2s, transform 0.15s, box-shadow 0.2s;
        box-shadow: 0 2px 12px {t['glow']};
        cursor: pointer;
    }}
    .stButton > button:hover,
    [data-testid="stFormSubmitButton"] > button:hover {{
        opacity: 0.88;
        transform: translateY(-1px);
        box-shadow: 0 6px 20px {t['glow']};
    }}
    .stButton > button:active,
    [data-testid="stFormSubmitButton"] > button:active {{
        transform: translateY(0) !important;
    }}

    /* ── Cards ── */
    .zy-card {{
        background: {t['card']};
        border: 1px solid {t['border']};
        border-radius: 16px;
        padding: 16px 20px;
        margin-bottom: 12px;
    }}
    .zy-card-sm {{
        background: {t['card2']};
        border: 1px solid {t['border']};
        border-radius: 12px;
        padding: 10px 14px;
        margin-bottom: 8px;
    }}

    /* ── Chat bubbles ── */
    .chat-row {{ display: flex; margin: 6px 0; }}
    .user-bubble {{
        background: {t['user_bubble']};
        color: #ffffff;
        padding: 11px 18px;
        border-radius: 20px 20px 4px 20px;
        max-width: 72%;
        font-size: 14px;
        line-height: 1.65;
        word-wrap: break-word;
        box-shadow: 0 2px 10px rgba(0,0,0,0.25);
        margin-left: auto;
    }}
    .bot-bubble {{
        background: {t['bot_bubble']};
        color: {t['text']};
        border: 1px solid {t['border']};
        padding: 11px 18px;
        border-radius: 20px 20px 20px 4px;
        max-width: 82%;
        font-size: 14px;
        line-height: 1.65;
        word-wrap: break-word;
        box-shadow: 0 2px 10px rgba(0,0,0,0.25);
    }}
    .bubble-label {{
        font-size: 10px;
        color: {t['subtext']};
        margin: 2px 6px;
        letter-spacing: 0.5px;
    }}

    /* ── Accent / labels ── */
    .zy-label {{
        color: {t['accent']};
        font-weight: 700;
        font-size: 11px;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-bottom: 8px;
    }}
    .zy-sub {{ color: {t['subtext']}; font-size: 12px; }}
    .zy-badge {{
        background: {t['glow']};
        color: {t['accent']};
        border: 1px solid {t['border']};
        border-radius: 20px;
        padding: 3px 12px;
        font-size: 11px;
        font-weight: 600;
        display: inline-block;
    }}

    /* ── Timer ── */
    .zy-timer {{
        font-size: 38px;
        font-weight: 900;
        color: {t['accent']};
        text-align: center;
        font-variant-numeric: tabular-nums;
        letter-spacing: 3px;
        line-height: 1;
        padding: 8px 0;
    }}

    /* ── Progress bar ── */
    .stProgress > div > div > div > div {{
        background: {t['gradient']} !important;
        border-radius: 10px !important;
    }}

    /* ── Selectbox ── */
    .stSelectbox > div > div {{
        background-color: {t['card']} !important;
        border: 1px solid {t['border']} !important;
        border-radius: 10px !important;
        color: {t['text']} !important;
    }}

    /* ── Radio ── */
    .stRadio label {{ color: {t['text']} !important; font-size: 13px !important; }}
    .stRadio > label {{ color: {t['accent']} !important; }}

    /* ── Divider ── */
    hr {{ border: none !important; border-top: 1px solid {t['border']} !important; opacity: 0.5 !important; margin: 12px 0 !important; }}

    /* ── Expander ── */
    .streamlit-expanderHeader {{
        background-color: {t['card']} !important;
        color: {t['text']} !important;
        border-radius: 10px !important;
        border: 1px solid {t['border']} !important;
    }}

    /* ── Metrics ── */
    [data-testid="stMetricValue"] {{
        color: {t['accent']} !important;
        font-size: 24px !important;
        font-weight: 800 !important;
    }}
    [data-testid="stMetricLabel"] {{
        color: {t['subtext']} !important;
        font-size: 11px !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }}

    /* ── Success / info messages ── */
    .stSuccess {{ background-color: rgba(34,197,94,0.12) !important; border-color: #22c55e !important; }}
    .stInfo    {{ background-color: {t['glow']} !important; border-color: {t['accent']} !important; }}

    /* ── Hide Streamlit chrome ── */
    #MainMenu {{ visibility: hidden; }}
    footer     {{ visibility: hidden; }}
    header     {{ visibility: hidden; }}

    /* ── Scrollbar ── */
    ::-webkit-scrollbar              {{ width: 5px; height: 5px; }}
    ::-webkit-scrollbar-track        {{ background: {t['bg']}; }}
    ::-webkit-scrollbar-thumb        {{ background: {t['border']}; border-radius: 3px; }}
    ::-webkit-scrollbar-thumb:hover  {{ background: {t['accent']}; }}
    </style>
    """

# =============================================================================
# SECTION 11: UI COMPONENTS
# =============================================================================

def init_session_state():
    defaults = {
        "chat_history":      [],
        "current_theme":     "study",
        "bundle":            None,
        "last_analysis":     None,
        "api_key":           "",
        "pomodoro_active":   False,
        "pomodoro_start":    None,
        "pomodoro_minutes":  25,
        "sessions_done":     0,
        "total_focus_mins":  0,
        "daily_quote":       random.choice(MOTIVATIONAL),
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
    if st.session_state.bundle is None:
        with st.spinner("⚡ Initialising Zynbox AI…"):
            st.session_state.bundle = load_models()


def render_chat_bubble(role: str, text: str):
    """Render a styled, markdown-aware chat bubble."""
    css_class = "user-bubble" if role == "user" else "bot-bubble"
    icon      = "🧑" if role == "user"  else "🤖"

    html = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    html = html.replace("\n", "<br>")
    # Bold **text**
    html = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', html)
    # Italic _text_
    html = re.sub(r'_(.+?)_', r'<em>\1</em>', html)
    # Strike ~~text~~
    html = re.sub(r'~~(.+?)~~', r'<s>\1</s>', html)
    # Inline code `text`
    html = re.sub(
        r'`(.+?)`',
        r'<code style="background:rgba(255,255,255,0.1);padding:1px 6px;border-radius:4px;'
        r'font-family:\'Fira Code\',monospace;font-size:12px;">\1</code>',
        html
    )

    st.markdown(
        f'<div class="chat-row">'
        f'<div class="{css_class}">'
        f'<span style="font-size:15px;">{icon}</span>&nbsp; {html}'
        f'</div></div>',
        unsafe_allow_html=True
    )


def render_pomodoro_widget(theme_key: str):
    t = THEMES[theme_key]
    st.markdown(f'<div class="zy-label">⏱ Focus Timer</div>', unsafe_allow_html=True)

    if st.session_state.pomodoro_active and st.session_state.pomodoro_start:
        elapsed   = (datetime.datetime.now() - st.session_state.pomodoro_start).total_seconds()
        total_sec = st.session_state.pomodoro_minutes * 60
        remaining = max(0, total_sec - int(elapsed))
        mins, secs = divmod(remaining, 60)

        if remaining == 0:
            st.session_state.pomodoro_active   = False
            st.session_state.sessions_done    += 1
            st.session_state.total_focus_mins += st.session_state.pomodoro_minutes
            st.success(f"🎉 Session #{st.session_state.sessions_done} complete! Take a break.")
        else:
            progress = elapsed / total_sec
            st.markdown(f'<div class="zy-timer">{mins:02d}:{secs:02d}</div>', unsafe_allow_html=True)
            st.progress(min(progress, 1.0))
            st.markdown(
                f'<div class="zy-sub" style="text-align:center;margin-top:4px;">🎯 Deep work in progress…</div>',
                unsafe_allow_html=True
            )
            if st.button("⏹ Stop", key="stop_pom", use_container_width=True):
                st.session_state.pomodoro_active = False
                st.rerun()
    else:
        c1, c2 = st.columns([3, 2])
        with c1:
            dur = st.selectbox(
                "min", [15, 20, 25, 30, 45, 60],
                index=2, key="pom_dur",
                label_visibility="collapsed"
            )
            st.session_state.pomodoro_minutes = dur
        with c2:
            if st.button("▶ Start", key="start_pom", use_container_width=True):
                st.session_state.pomodoro_active = True
                st.session_state.pomodoro_start  = datetime.datetime.now()
                st.rerun()

        if st.session_state.sessions_done > 0:
            st.markdown(
                f'<div class="zy-sub" style="margin-top:4px;">'
                f'✅ {st.session_state.sessions_done} sessions &nbsp;·&nbsp; '
                f'⏱ {st.session_state.total_focus_mins} mins total</div>',
                unsafe_allow_html=True
            )


def render_sidebar(theme_key: str):
    t = THEMES[theme_key]

    with st.sidebar:
        # ── Branding ─────────────────────────────────────────────────────────
        st.markdown(f"""
        <div style="text-align:center; padding: 14px 0 18px 0;">
            <div style="font-size:38px; margin-bottom:4px;">⚡</div>
            <div style="font-size:22px; font-weight:900; letter-spacing:3px;
                background:{t['gradient']}; -webkit-background-clip:text;
                -webkit-text-fill-color:transparent;">
                ZYNBOX
            </div>
            <div style="font-size:10px; color:{t['subtext']}; margin-top:3px;
                letter-spacing:1.5px; text-transform:uppercase;">
                AI Productivity Agent
            </div>
            <div style="margin-top:10px;">
                <span class="zy-badge">{t['emoji']} {t['name']}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        # ── API Key ───────────────────────────────────────────────────────────
        st.markdown('<div class="zy-label">🔑 Anthropic API Key</div>', unsafe_allow_html=True)
        api_input = st.text_input(
            "api_key_input", type="password",
            value=st.session_state.api_key,
            placeholder="sk-ant-api03-…",
            label_visibility="collapsed",
        )
        if api_input != st.session_state.api_key:
            st.session_state.api_key = api_input

        if st.session_state.api_key:
            st.markdown(
                '<div class="zy-sub" style="color:#22c55e;">🟢 AI Mode Active</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div class="zy-sub" style="color:#ef4444;">🔴 Local Mode — add key for AI chat</div>',
                unsafe_allow_html=True
            )

        st.divider()

        # ── Mode Selector ─────────────────────────────────────────────────────
        st.markdown('<div class="zy-label">🎨 Mode</div>', unsafe_allow_html=True)
        theme_options = {v["name"]: k for k, v in THEMES.items()}
        current_name  = THEMES[theme_key]["name"]
        selected_name = st.selectbox(
            "mode_select", list(theme_options.keys()),
            index=list(theme_options.keys()).index(current_name),
            label_visibility="collapsed",
        )
        if theme_options[selected_name] != st.session_state.current_theme:
            st.session_state.current_theme = theme_options[selected_name]
            st.rerun()

        st.divider()

        # ── Pomodoro ──────────────────────────────────────────────────────────
        render_pomodoro_widget(theme_key)

        st.divider()

        # ── Productivity Score ────────────────────────────────────────────────
        score = get_productivity_score()
        tasks = load_tasks()
        done  = sum(1 for task in tasks if task["done"])
        st.markdown('<div class="zy-label">📊 Productivity</div>', unsafe_allow_html=True)
        st.progress(score / 100 if score > 0 else 0.0)
        st.markdown(f"""
        <div style="display:flex; justify-content:space-between; align-items:center; margin-top:4px;">
            <span style="color:{t['accent']}; font-size:20px; font-weight:800;">{score}%</span>
            <span class="zy-sub">{done}/{len(tasks)} tasks done</span>
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        # ── Task Quick List ───────────────────────────────────────────────────
        st.markdown('<div class="zy-label">📋 Tasks</div>', unsafe_allow_html=True)
        if tasks:
            for task in tasks[-5:]:
                icon    = "✅" if task["done"] else "⏳"
                pri_col = {"high": "#ef4444", "medium": "#f59e0b", "low": "#22c55e"}.get(
                    task.get("priority", "medium"), "#f59e0b"
                )
                display = task["text"][:25] + "…" if len(task["text"]) > 25 else task["text"]
                st.markdown(
                    f'<div class="zy-sub" style="padding:3px 0;">'
                    f'{icon} <span style="color:{pri_col}; font-size:10px;">●</span> {display}</div>',
                    unsafe_allow_html=True
                )
        else:
            st.markdown('<div class="zy-sub">No tasks yet.</div>', unsafe_allow_html=True)

        st.divider()

        # ── Controls ──────────────────────────────────────────────────────────
        c1, c2 = st.columns(2)
        with c1:
            if st.button("🗑️ Clear Chat", key="clear_chat", use_container_width=True):
                st.session_state.chat_history  = []
                st.session_state.last_analysis = None
                st.rerun()
        with c2:
            if st.button("🔄 Retrain", key="retrain_btn", use_container_width=True):
                if os.path.exists(MODEL_PATH):
                    os.remove(MODEL_PATH)
                if "w2v_clf_intent" in st.session_state:
                    del st.session_state["w2v_clf_intent"]
                st.session_state.bundle = train_models()
                st.success("✅ Models retrained!")

        # ── Footer ────────────────────────────────────────────────────────────
        st.markdown(f"""
        <div style="text-align:center; margin-top:14px; color:{t['subtext']};
            font-size:10px; letter-spacing:0.5px; line-height:1.8;">
            Utkarsh Sharma · Zeeshan · Utkarsh Pandey<br>
            <span style="color:{t['border']};">Powered by Claude AI & Streamlit</span>
        </div>
        """, unsafe_allow_html=True)


def render_analysis_panel(analysis: dict, theme_key: str):
    t = THEMES[theme_key]
    if analysis is None:
        st.markdown(f"""
        <div class="zy-card" style="text-align:center; padding:24px 12px;">
            <div style="font-size:24px; margin-bottom:8px;">🔍</div>
            <div class="zy-sub">Send a message<br>to see AI analysis</div>
        </div>
        """, unsafe_allow_html=True)
        return

    rows = [
        ("TF-IDF Intent",  analysis.get("tfidf_intent", "—")),
        ("W2Vec Intent",   analysis.get("w2v_intent",   "—")),
        ("Mood Detected",  analysis.get("mood", "—").upper()),
        ("Active Mode",    THEMES[theme_key]["name"]),
    ]
    rows_html = "".join(
        f'<div style="display:flex;justify-content:space-between;padding:6px 0;'
        f'border-bottom:1px solid {t["border"]}30;">'
        f'<span class="zy-sub">{label}</span>'
        f'<span style="color:{t["accent"]};font-size:12px;font-weight:600;">{value}</span>'
        f'</div>'
        for label, value in rows
    )

    st.markdown(f"""
    <div class="zy-card">
        <div class="zy-label" style="margin-bottom:10px;">🔍 AI Analysis</div>
        {rows_html}
    </div>
    """, unsafe_allow_html=True)

# =============================================================================
# SECTION 12: MAIN APPLICATION
# =============================================================================

def main():
    st.set_page_config(
        page_title="Zynbox AI",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    init_session_state()
    theme_key = st.session_state.current_theme
    t         = THEMES[theme_key]

    st.markdown(get_theme_css(theme_key), unsafe_allow_html=True)
    render_sidebar(theme_key)

    # ── Header ────────────────────────────────────────────────────────────────
    h1, h2, h3 = st.columns([4, 1, 1])
    with h1:
        ai_badge = (
            '<span style="color:#22c55e;font-size:11px;font-weight:600;">● AI Active</span>'
            if st.session_state.api_key else
            '<span style="color:#ef4444;font-size:11px;font-weight:600;">● Local Mode</span>'
        )
        st.markdown(f"""
        <div style="padding:4px 0 14px 0;">
            <div style="font-size:26px;font-weight:900;letter-spacing:1px;
                background:{t['gradient']};-webkit-background-clip:text;
                -webkit-text-fill-color:transparent;">
                ⚡ ZYNBOX
            </div>
            <div style="color:{t['subtext']};font-size:12px;margin-top:2px;
                letter-spacing:0.4px;">
                Personal AI Productivity Agent &nbsp;·&nbsp;
                {t['emoji']} {t['name']} &nbsp;·&nbsp; {ai_badge}
            </div>
        </div>
        """, unsafe_allow_html=True)
    with h2:
        st.metric("Productivity", f"{get_productivity_score()}%")
    with h3:
        tasks = load_tasks()
        done  = sum(1 for task in tasks if task["done"])
        st.metric("Tasks", f"{done}/{len(tasks)}")

    # ── Two-column layout ─────────────────────────────────────────────────────
    chat_col, info_col = st.columns([3, 1])

    # ── RIGHT PANEL ───────────────────────────────────────────────────────────
    with info_col:
        render_analysis_panel(st.session_state.last_analysis, theme_key)

        st.markdown(f"""
        <div class="zy-card">
            <div class="zy-label" style="margin-bottom:8px;">💬 Daily Motivation</div>
            <div style="font-size:13px;color:{t['text']};line-height:1.55;font-style:italic;">
                "{st.session_state.daily_quote}"
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f'<div class="zy-label" style="margin-bottom:8px;">⚡ Quick Actions</div>', unsafe_allow_html=True)

        if st.button("📚 Study Mode",    key="qa_study",    use_container_width=True):
            st.session_state.current_theme = "study"
            st.session_state.chat_history.append({"role": "user", "text": "study mode"})
            st.session_state.chat_history.append({"role": "bot",  "text": f"📚 {THEMES['study']['greeting']}"})
            st.rerun()

        if st.button("🎨 Creative Mode", key="qa_creative", use_container_width=True):
            st.session_state.current_theme = "creative"
            st.session_state.chat_history.append({"role": "user", "text": "creative mode"})
            st.session_state.chat_history.append({"role": "bot",  "text": f"🎨 {THEMES['creative']['greeting']}"})
            st.rerun()

        if st.button("📋 My Tasks",      key="qa_tasks",    use_container_width=True):
            reply = show_tasks()
            st.session_state.chat_history.append({"role": "user", "text": "show tasks"})
            st.session_state.chat_history.append({"role": "bot",  "text": reply})
            st.rerun()

        if st.button("📊 My Stats",      key="qa_stats",    use_container_width=True):
            result = chatbot_response(
                "stats", st.session_state.bundle, theme_key, st.session_state.api_key
            )
            st.session_state.chat_history.append({"role": "user", "text": "stats"})
            st.session_state.chat_history.append({"role": "bot",  "text": result["reply"]})
            st.session_state.last_analysis = result
            st.rerun()

        if st.button("⏱️ Start Pomodoro", key="qa_pomo",   use_container_width=True):
            st.session_state.pomodoro_active = True
            st.session_state.pomodoro_start  = datetime.datetime.now()
            reply = (
                "⏱️ **Pomodoro started!**\n\n"
                "Focus session running. Stay locked in. 🎯\n"
                "_Check sidebar timer._"
            )
            st.session_state.chat_history.append({"role": "user", "text": "start pomodoro"})
            st.session_state.chat_history.append({"role": "bot",  "text": reply})
            st.rerun()

        if theme_key == "creative":
            if st.button("💡 Creative Spark", key="qa_spark", use_container_width=True):
                prompt = get_creative_spark(st.session_state.api_key)
                st.session_state.chat_history.append({"role": "user", "text": "give me a creative prompt"})
                st.session_state.chat_history.append({"role": "bot",  "text": prompt})
                st.rerun()

    # ── CHAT PANEL ────────────────────────────────────────────────────────────
    with chat_col:
        chat_container = st.container()
        with chat_container:
            if not st.session_state.chat_history:
                t_data = THEMES[theme_key]
                ai_hint = (
                    "🟢 <strong>AI Mode Active</strong> — Ask me anything! I understand context and remember our conversation."
                    if st.session_state.api_key else
                    "💡 <strong>Tip:</strong> Add your <strong>Anthropic API key</strong> in the sidebar to unlock full AI conversation mode!"
                )
                st.markdown(f"""
                <div class="bot-bubble" style="max-width:92%;border-left:3px solid {t_data['accent']};margin:8px 0;">
                    <span style="font-size:16px;">🤖</span>&nbsp;
                    <strong>Welcome to Zynbox v2.0!</strong><br><br>
                    {t_data['greeting']}<br><br>
                    {ai_hint}<br><br>
                    Type <code style="background:rgba(255,255,255,0.1);padding:1px 6px;border-radius:4px;">help</code>
                    to see all commands, or just start chatting.
                </div>
                """, unsafe_allow_html=True)
            else:
                for msg in st.session_state.chat_history:
                    render_chat_bubble(msg["role"], msg["text"])

        st.markdown("<div style='height:6px;'></div>", unsafe_allow_html=True)

        # ── Input form ────────────────────────────────────────────────────────
        with st.form(key="chat_form", clear_on_submit=True):
            inp_col, btn_col = st.columns([5, 1])
            with inp_col:
                user_input = st.text_input(
                    "input",
                    placeholder="Ask anything, add a task, start a timer, take a note…",
                    label_visibility="collapsed",
                )
            with btn_col:
                submitted = st.form_submit_button("Send ➤", use_container_width=True)

        if submitted and user_input.strip():
            st.session_state.chat_history.append({"role": "user", "text": user_input})

            result = chatbot_response(
                user_input,
                st.session_state.bundle,
                st.session_state.current_theme,
                st.session_state.api_key,
            )

            st.session_state.chat_history.append({"role": "bot", "text": result["reply"]})
            st.session_state.last_analysis = result

            if result["theme_switch"] != st.session_state.current_theme:
                st.session_state.current_theme = result["theme_switch"]

            st.rerun()

    # ── Footer ────────────────────────────────────────────────────────────────
    st.divider()
    st.markdown(f"""
    <div style="text-align:center;color:{t['subtext']};font-size:11px;padding:4px 0;letter-spacing:0.5px;">
        ⚡ ZYNBOX v2.0 &nbsp;·&nbsp; Streamlit · Scikit-learn · Gensim · Claude AI
        &nbsp;·&nbsp; Utkarsh Sharma · Zeeshan · Utkarsh Pandey
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    main()
