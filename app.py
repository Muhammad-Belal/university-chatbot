import streamlit as st
from groq import Groq
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from pymongo import MongoClient
from datetime import datetime
import bcrypt
import os

# ═══════════════════════════════════════
# CONNECTIONS
# ═══════════════════════════════════════
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
index = pc.Index(os.environ.get("PINECONE_INDEX"))
mongo = MongoClient(os.environ.get("MONGODB_URI"))
db = mongo["university_db"]
users_col = db["users"]
chats_col = db["chat_history"]

from knowledge_base import UNIVERSITY_KNOWLEDGE
from assets import IUB_LOGO, BZU_LOGO

# ═══════════════════════════════════════
# DATABASE HELPERS
# ═══════════════════════════════════════
def verify_user(username, password):
    user = users_col.find_one({"username": username})
    if not user:
        return False, None
    if bcrypt.checkpw(password.encode(), user["password"]):
        return True, user["full_name"]
    return False, None

def create_user(username, password, full_name):
    if users_col.find_one({"username": username}):
        return False, "Username already exists!"
    hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt())
    users_col.insert_one({
        "username": username,
        "password": hashed,
        "full_name": full_name,
        "created_at": datetime.now()
    })
    return True, "Account created successfully!"

def save_chat(username, question, answer):
    chats_col.insert_one({
        "username": username,
        "question": question,
        "answer": answer,
        "timestamp": datetime.now(),
        "feedback": None
    })

def get_history(username, limit=10):
    return list(chats_col.find({"username": username}).sort("timestamp", -1).limit(limit))

def update_feedback(username, answer, feedback_val):
    chats_col.update_one(
        {"username": username, "answer": answer},
        {"$set": {"feedback": feedback_val}}
    )

# ═══════════════════════════════════════
# AI HELPERS
# ═══════════════════════════════════════
def get_embedding(text):
    return embedding_model.encode(text).tolist()

def search_docs(question, uni_prefix, top_k=3):
    vec = get_embedding(question)
    results = index.query(vector=vec, top_k=top_k, include_metadata=True).matches
    filtered = [r for r in results if r.metadata.get("source", "").lower().startswith(uni_prefix)]
    return filtered if filtered else results

def get_answer(question, chunks, university):
    context = "\n\n".join([c.metadata.get("text", "") for c in chunks]) if chunks else ""
    sources = list(set([c.metadata.get("source", "") for c in chunks])) if chunks else []

    q_lower = question.lower()
    keywords = [
        "attendance", "exam", "fee", "hostel", "admission", "department",
        "program", "scholarship", "library", "semester", "result", "campus",
        "engineering", "medical", "computer", "science", "arts", "law"
    ]

    if university == "BZU":
        bzu_idx = UNIVERSITY_KNOWLEDGE.find("BZU")
        if bzu_idx == -1:
            bzu_idx = UNIVERSITY_KNOWLEDGE.find("Bahauddin")
        uni_section = UNIVERSITY_KNOWLEDGE[bzu_idx:] if bzu_idx != -1 else UNIVERSITY_KNOWLEDGE
    else:
        iub_end = UNIVERSITY_KNOWLEDGE.find("BZU")
        if iub_end == -1:
            iub_end = UNIVERSITY_KNOWLEDGE.find("Bahauddin")
        uni_section = UNIVERSITY_KNOWLEDGE[:iub_end] if iub_end != -1 else UNIVERSITY_KNOWLEDGE

    extra_knowledge = ""
    for kw in keywords:
        if kw in q_lower:
            idx = uni_section.lower().find(kw)
            if idx != -1:
                extra_knowledge = uni_section[max(0, idx - 200):idx + 3000]
                break
    if not extra_knowledge:
        extra_knowledge = uni_section[:3000]

    prompt = f"""You are a smart, friendly AI assistant for {university} university students.

LANGUAGE RULES - follow strictly, no mixing allowed:
- English question -> English reply only
- Roman Urdu question -> Roman Urdu reply only
- Urdu script question -> Urdu script reply only
- Never mix languages, never mention which language you are using

QUESTION TYPE RULES:
- If question is a greeting or general chat (like "kya hal ha", "hello", "how are you") -> reply naturally and friendly, DO NOT use university data
- If question is about university -> use Documents and Knowledge Base

FORMAT RULES:
- Give detailed answers using bullet points (*)
- Use sub-bullets (->) for extra details
- Bold the main heading like **Fee Structure:**
- Give 3-5 bullet points per answer
- End with a helpful tip starting with Tip:

ANSWER RULES:
- First check Documents, then Knowledge Base
- Never say "data not available" or "visit website" if any source has info
- Always extract and present whatever information is available
- Be helpful, friendly and to the point

Documents:
{context}

Knowledge Base:
{extra_knowledge}

Question: {question}

Answer:"""

    res = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=700
    )
    return res.choices[0].message.content, sources

# ═══════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════
st.set_page_config(
    page_title="University AI Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ═══════════════════════════════════════
# GLOBAL CSS
# ═══════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Serif+Display&display=swap');

/* ── Reset & Base ── */
*, *::before, *::after { box-sizing: border-box; }
#MainMenu, footer, header, .stDeployButton { display: none !important; }

html, body, [data-testid="stAppViewContainer"] {
    font-family: 'DM Sans', sans-serif;
    background: #f5f4f0 !important;
}

/* ── Main Content Area ── */
.main .block-container {
    max-width: 680px !important;
    margin: 0 auto !important;
    padding: 2rem 1.5rem 6rem !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #1c1c1e !important;
    border-right: none !important;
    min-width: 260px !important;
    max-width: 260px !important;
}

[data-testid="stSidebar"] * {
    color: #e8e8e6 !important;
}

[data-testid="stSidebar"] .stMarkdown h3 {
    font-family: 'DM Serif Display', serif !important;
    font-size: 13px !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: #888 !important;
    margin-bottom: 8px !important;
}

[data-testid="stSidebar"] hr {
    border-color: #333 !important;
    margin: 12px 0 !important;
}

[data-testid="stSidebar"] .stButton > button {
    background: #2a2a2c !important;
    color: #e8e8e6 !important;
    border: 1px solid #3a3a3c !important;
    border-radius: 8px !important;
    font-size: 12px !important;
    padding: 6px 12px !important;
    width: 100% !important;
    margin: 2px 0 !important;
    transition: background 0.2s !important;
}

[data-testid="stSidebar"] .stButton > button:hover {
    background: #3a3a3c !important;
    border-color: #555 !important;
}

/* Logout/Clear in sidebar - danger style */
[data-testid="stSidebar"] .stButton:last-child > button {
    background: #2c1a1a !important;
    border-color: #5c2a2a !important;
    color: #ff8a80 !important;
}

/* ── Sidebar toggle arrow ── */
[data-testid="stSidebarCollapsedControl"] {
    background: #1c1c1e !important;
    color: white !important;
}

/* ── Auth Page ── */
.auth-header {
    text-align: center;
    padding: 3rem 0 2rem;
}

.auth-header h1 {
    font-family: 'DM Serif Display', serif;
    font-size: 2.2rem;
    color: #1a1a1a;
    margin-bottom: 4px;
}

.auth-header p {
    color: #888;
    font-size: 15px;
}

/* ── Tabs ── */
[data-testid="stTabs"] button {
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    font-size: 14px !important;
}

/* ── Input fields ── */
.stTextInput > div > div > input {
    border-radius: 10px !important;
    border: 1.5px solid #e0e0e0 !important;
    padding: 10px 14px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 14px !important;
    background: white !important;
    transition: border-color 0.2s !important;
}

.stTextInput > div > div > input:focus {
    border-color: #1a1a1a !important;
    box-shadow: none !important;
}

/* ── Primary Buttons ── */
[data-testid="baseButton-primary"] {
    background: #1a1a1a !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    font-size: 14px !important;
    padding: 10px !important;
    transition: opacity 0.2s !important;
}

[data-testid="baseButton-primary"]:hover {
    opacity: 0.85 !important;
}

/* Secondary buttons */
[data-testid="baseButton-secondary"] {
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 13px !important;
}

/* ── University Selector Cards ── */
.uni-card {
    background: white;
    border: 1.5px solid #e8e8e6;
    border-radius: 16px;
    padding: 28px 20px 20px;
    text-align: center;
    transition: all 0.25s ease;
    cursor: pointer;
    margin-bottom: 12px;
}

.uni-card:hover {
    border-color: #1a1a1a;
    box-shadow: 0 8px 24px rgba(0,0,0,0.08);
    transform: translateY(-2px);
}

.uni-card img {
    width: 68px;
    height: 68px;
    object-fit: contain;
    margin-bottom: 14px;
    border-radius: 12px;
}

.uni-card h4 {
    font-family: 'DM Serif Display', serif;
    font-size: 16px;
    color: #1a1a1a;
    margin: 0 0 6px;
}

.uni-card p {
    color: #888;
    font-size: 13px;
    margin: 0;
}

/* ── Chat Header ── */
.chat-header {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 14px 18px;
    background: white;
    border: 1.5px solid #e8e8e6;
    border-radius: 14px;
    margin-bottom: 20px;
}

.chat-header img {
    width: 42px;
    height: 42px;
    border-radius: 10px;
    object-fit: contain;
}

.chat-header-info h3 {
    font-family: 'DM Serif Display', serif;
    font-size: 17px;
    color: #1a1a1a;
    margin: 0;
}

.chat-header-info p {
    font-size: 12px;
    color: #888;
    margin: 0;
}

/* ── Chat Messages ── */
.chat-wrap {
    display: flex;
    flex-direction: column;
    gap: 10px;
    margin-bottom: 16px;
}

.msg-user {
    display: flex;
    justify-content: flex-end;
}

.msg-bot {
    display: flex;
    justify-content: flex-start;
}

.bubble-user {
    background: #1a1a1a;
    color: white;
    padding: 11px 16px;
    border-radius: 18px 18px 4px 18px;
    max-width: 78%;
    font-size: 14px;
    line-height: 1.5;
    word-wrap: break-word;
}

.bubble-bot {
    background: white;
    color: #1a1a1a;
    padding: 11px 16px;
    border-radius: 18px 18px 18px 4px;
    max-width: 78%;
    font-size: 14px;
    line-height: 1.6;
    border: 1.5px solid #e8e8e6;
    word-wrap: break-word;
}

/* ── Quick Action Buttons ── */
.quick-label {
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #aaa;
    margin-bottom: 8px;
}

/* ── Sidebar History Item ── */
.hist-item {
    background: #2a2a2c;
    border: 1px solid #3a3a3c;
    border-radius: 8px;
    padding: 8px 10px;
    font-size: 12px;
    color: #ccc !important;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    margin-bottom: 4px;
    cursor: pointer;
}

.hist-item:hover {
    background: #333;
}

/* ── Sidebar User Info ── */
.sidebar-user {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 4px 0 8px;
}

.sidebar-avatar {
    width: 34px;
    height: 34px;
    background: #3a3a3c;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 14px;
    font-weight: 600;
    color: #e8e8e6;
    flex-shrink: 0;
}

.sidebar-name {
    font-size: 13px;
    font-weight: 600;
    color: #e8e8e6 !important;
}

.sidebar-uni {
    font-size: 11px;
    color: #888 !important;
}

/* ── Footer ── */
.footer {
    text-align: center;
    color: #bbb;
    font-size: 11px;
    padding: 20px 0 4px;
    letter-spacing: 0.02em;
}

/* ── Feedback row ── */
.feedback-row {
    display: flex;
    gap: 6px;
    margin-top: 4px;
    margin-left: 4px;
}

/* ── Spinner ── */
.stSpinner > div {
    border-top-color: #1a1a1a !important;
}

/* ── Alerts ── */
.stAlert {
    border-radius: 10px !important;
    font-size: 13px !important;
}

/* ── Chat input ── */
[data-testid="stChatInput"] {
    border-radius: 12px !important;
    border: 1.5px solid #ddd !important;
    font-family: 'DM Sans', sans-serif !important;
}

[data-testid="stChatInput"] textarea {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 14px !important;
}

/* ── Mobile ── */
@media (max-width: 768px) {
    .main .block-container {
        padding: 1rem 1rem 5rem !important;
    }
    .bubble-user, .bubble-bot {
        max-width: 90% !important;
    }
    [data-testid="stSidebar"] {
        min-width: 240px !important;
        max-width: 240px !important;
    }
}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════
# SESSION STATE INIT
# ═══════════════════════════════════════
defaults = {
    "logged_in": False,
    "username": "",
    "full_name": "",
    "messages": [],
    "university": None,
    "pending_q": None,
    "feedback_done": set()
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

def do_logout():
    for k, v in defaults.items():
        st.session_state[k] = v

# ═══════════════════════════════════════════════════
# PAGE 1 — LOGIN / REGISTER
# ═══════════════════════════════════════════════════
if not st.session_state.logged_in:

    st.markdown("""
    <div class='auth-header'>
        <h1>🎓 University AI</h1>
        <p>Your smart guide for IUB & BZU</p>
    </div>
    """, unsafe_allow_html=True)

    col_l, col_form, col_r = st.columns([1, 2, 1])
    with col_form:
        tab_login, tab_reg = st.tabs(["Login", "Register"])

        with tab_login:
            un = st.text_input("Username", key="l_un", placeholder="Enter username")
            pw = st.text_input("Password", key="l_pw", placeholder="Enter password", type="password")
            st.write("")
            if st.button("Login →", use_container_width=True, type="primary", key="btn_login"):
                if un.strip() and pw.strip():
                    ok, name = verify_user(un.strip(), pw.strip())
                    if ok:
                        st.session_state.logged_in = True
                        st.session_state.username = un.strip()
                        st.session_state.full_name = name
                        st.session_state.messages = []
                        st.session_state.university = None
                        st.rerun()
                    else:
                        st.error("❌ Incorrect username or password.")
                else:
                    st.warning("Please fill both fields.")

        with tab_reg:
            fn = st.text_input("Full Name", key="r_fn", placeholder="e.g. Muhammad Ali")
            ru = st.text_input("Username", key="r_un", placeholder="e.g. muhammadali")
            rp = st.text_input("Password", key="r_pw", placeholder="Min 6 characters", type="password")
            st.write("")
            if st.button("Create Account →", use_container_width=True, type="primary", key="btn_reg"):
                if fn.strip() and ru.strip() and rp.strip():
                    if len(rp) < 6:
                        st.error("Password must be at least 6 characters.")
                    else:
                        ok, msg = create_user(ru.strip(), rp.strip(), fn.strip())
                        if ok:
                            st.success(f"✅ {msg} Please login.")
                        else:
                            st.error(f"❌ {msg}")
                else:
                    st.warning("Please fill all fields.")

    st.markdown("<div class='footer'>© 2026 Muhammad Belal | AI University Assistant</div>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════
# PAGE 2 — UNIVERSITY SELECTOR
# ═══════════════════════════════════════════════════
elif st.session_state.university is None:

    # Sidebar for this page
    with st.sidebar:
        initials = st.session_state.full_name[:1].upper() if st.session_state.full_name else "U"
        st.markdown(f"""
        <div class='sidebar-user'>
            <div class='sidebar-avatar'>{initials}</div>
            <div>
                <div class='sidebar-name'>{st.session_state.full_name}</div>
                <div class='sidebar-uni'>Choose a university</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("### Navigation")
        if st.button("🚪  Logout", use_container_width=True, key="lo_sel"):
            do_logout()
            st.rerun()

    # Main content
    st.markdown(f"""
    <div style='text-align:center; padding: 2.5rem 0 1.5rem;'>
        <h2 style='font-family: DM Serif Display, serif; font-size:1.8rem; color:#1a1a1a; margin-bottom:6px;'>
            Welcome, {st.session_state.full_name}! 👋
        </h2>
        <p style='color:#888; font-size:15px;'>Select your university to get started</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown(f"""
        <div class='uni-card'>
            <img src='{IUB_LOGO}' />
            <h4>Islamia University Bahawalpur</h4>
            <p>IUB — Est. 1925, Bahawalpur</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Select IUB →", use_container_width=True, type="primary", key="sel_iub"):
            st.session_state.university = "IUB"
            st.session_state.messages = [{
                "role": "assistant",
                "content": f"Welcome {st.session_state.full_name}! 🎓 I'm your IUB AI Assistant. Ask me about fees, admissions, exams, hostel, scholarships, or any university topic."
            }]
            st.rerun()

    with col2:
        st.markdown(f"""
        <div class='uni-card'>
            <img src='{BZU_LOGO}' />
            <h4>Bahauddin Zakariya University</h4>
            <p>BZU — Est. 1975, Multan</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Select BZU →", use_container_width=True, type="primary", key="sel_bzu"):
            st.session_state.university = "BZU"
            st.session_state.messages = [{
                "role": "assistant",
                "content": f"Welcome {st.session_state.full_name}! 🎓 I'm your BZU AI Assistant. Ask me about fees, admissions, exams, hostel, scholarships, or any university topic."
            }]
            st.rerun()

    st.markdown("<div class='footer'>© 2026 Muhammad Belal | AI University Assistant</div>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════
# PAGE 3 — CHATBOT
# ═══════════════════════════════════════════════════
else:
    uni = st.session_state.university
    logo = IUB_LOGO if uni == "IUB" else BZU_LOGO
    uni_full = "Islamia University Bahawalpur" if uni == "IUB" else "Bahauddin Zakariya University"
    initials = st.session_state.full_name[:1].upper() if st.session_state.full_name else "U"

    # ── SIDEBAR ──
    with st.sidebar:
        # User info
        st.markdown(f"""
        <div class='sidebar-user'>
            <div class='sidebar-avatar'>{initials}</div>
            <div>
                <div class='sidebar-name'>{st.session_state.full_name}</div>
                <div class='sidebar-uni'>{uni} — {uni_full[:22]}…</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### Recent Chats")

        hist = get_history(st.session_state.username)
        if hist:
            for i, h in enumerate(hist):
                q_text = h["question"]
                q_short = q_text[:32] + "…" if len(q_text) > 32 else q_text
                col_h, col_d = st.columns([5, 1])
                with col_h:
                    st.markdown(f"<div class='hist-item' title='{q_text}'>💬 {q_short}</div>", unsafe_allow_html=True)
                with col_d:
                    if st.button("✕", key=f"del_{i}", help="Delete this chat"):
                        chats_col.delete_one({"_id": h["_id"]})
                        st.rerun()
        else:
            st.markdown("<span style='color:#666;font-size:12px;'>No history yet</span>", unsafe_allow_html=True)

        st.markdown("---")

        if st.button("🔄  Switch University", use_container_width=True, key="sw_uni"):
            st.session_state.university = None
            st.session_state.messages = []
            st.rerun()

        if st.button("🗑️  Clear Chat", use_container_width=True, key="clr_chat"):
            st.session_state.messages = [{
                "role": "assistant",
                "content": f"Chat cleared! 😊 Ask me anything about {uni}."
            }]
            st.rerun()

        if st.button("🚪  Logout", use_container_width=True, key="lo_chat"):
            do_logout()
            st.rerun()

    # ── HEADER ──
    st.markdown(f"""
    <div class='chat-header'>
        <img src='{logo}' />
        <div class='chat-header-info'>
            <h3>{uni} AI Assistant</h3>
            <p>{uni_full}</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── MESSAGES ──
    for i, msg in enumerate(st.session_state.messages):
        if msg["role"] == "user":
            st.markdown(f"""
            <div class='msg-user'>
                <div class='bubble-user'>{msg['content']}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class='msg-bot'>
                <div class='bubble-bot'>{msg['content']}</div>
            </div>
            """, unsafe_allow_html=True)

            # Feedback buttons only for non-welcome messages
            if i > 0:
                fb_key = f"fb_{i}"
                if fb_key not in st.session_state.feedback_done:
                    fc1, fc2, fc3 = st.columns([1, 1, 10])
                    with fc1:
                        if st.button("👍", key=f"like_{i}", help="Helpful"):
                            update_feedback(st.session_state.username, msg['content'], "good")
                            st.session_state.feedback_done.add(fb_key)
                            st.toast("Thanks! 😊")
                            st.rerun()
                    with fc2:
                        if st.button("👎", key=f"dislike_{i}", help="Not helpful"):
                            update_feedback(st.session_state.username, msg['content'], "bad")
                            st.session_state.feedback_done.add(fb_key)
                            st.toast("Thanks, we'll improve! 🙏")
                            st.rerun()

    # ── QUICK QUESTIONS ──
    st.markdown("")
    st.markdown("<div class='quick-label'>Quick Questions</div>", unsafe_allow_html=True)
    qc1, qc2, qc3 = st.columns(3)
    with qc1:
        if st.button("📋 Attendance", use_container_width=True, key="qq1"):
            st.session_state.pending_q = f"What is the attendance policy at {uni}?"
    with qc2:
        if st.button("📝 Exam Rules", use_container_width=True, key="qq2"):
            st.session_state.pending_q = f"What are the exam rules at {uni}?"
    with qc3:
        if st.button("💰 Fee Structure", use_container_width=True, key="qq3"):
            st.session_state.pending_q = f"What is the fee structure at {uni}?"

    # ── CHAT INPUT ──
    user_input = st.chat_input(f"Ask anything about {uni}…")

    # Determine what to process
    to_process = None
    if user_input and user_input.strip():
        to_process = user_input.strip()
    elif st.session_state.pending_q:
        to_process = st.session_state.pending_q
        st.session_state.pending_q = None

    if to_process:
        st.session_state.messages.append({"role": "user", "content": to_process})
        with st.spinner("Thinking…"):
            prefix = "iub" if uni == "IUB" else "bzu"
            docs = search_docs(to_process, prefix)
            ans, srcs = get_answer(to_process, docs, uni)

            uni_kw = ["fee", "admission", "hostel", "exam", "library",
                      "attendance", "scholarship", "result", "department", "semester"]
            is_uni_q = any(w in to_process.lower() for w in uni_kw)

            full_ans = ans
            if srcs and is_uni_q:
                full_ans += f"\n\n📄 *Sources: {', '.join(srcs)}*"

            st.session_state.messages.append({"role": "assistant", "content": full_ans})
            save_chat(st.session_state.username, to_process, full_ans)
            st.rerun()

    st.markdown("<div class='footer'>© 2026 Muhammad Belal | AI University Assistant</div>", unsafe_allow_html=True)
