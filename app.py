import streamlit as st
from groq import Groq
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from pymongo import MongoClient
from datetime import datetime
import bcrypt
import os

# -- Connections --
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

# -- DB helpers --
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
    users_col.insert_one({"username": username, "password": hashed, "full_name": full_name})
    return True, "Account created!"

def save_chat(username, question, answer):
    chats_col.insert_one({"username": username, "question": question, "answer": answer, "timestamp": datetime.now()})

def get_history(username):
    return list(chats_col.find({"username": username}).sort("timestamp", -1).limit(10))

# -- AI helpers --
def get_embedding(text):
    return embedding_model.encode(text).tolist()

def search_docs(question, uni_prefix, top_k=3):
    vec = get_embedding(question)
    results = index.query(vector=vec, top_k=top_k, include_metadata=True).matches
    filtered = [r for r in results if r.metadata.get("source", "").lower().startswith(uni_prefix)]
    return filtered if filtered else results

def get_answer(question, chunks, university):
    context = "\n\n".join([c.metadata.get("text", "") for c in chunks])
    sources = list(set([c.metadata.get("source", "") for c in chunks]))
    q_lower = question.lower()
    keywords = ["attendance", "exam", "fee", "hostel", "admission", "department",
                "program", "scholarship", "library", "semester", "result", "campus",
                "engineering", "medical", "computer", "science", "arts", "law"]

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

LANGUAGE RULES:
- English question -> English reply only
- Roman Urdu question -> Roman Urdu reply only
- Urdu script question -> Urdu script reply only

QUESTION RULES:
- Greeting/general chat -> reply friendly, do NOT use university data
- University question -> use Documents and Knowledge Base

FORMAT:
- Bullet points, bold headings, 3-5 points, end with Tip:

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

# -- Page config --
st.set_page_config(page_title="University AI Assistant", page_icon="🎓", layout="wide")

# -- Minimal CSS: fix layout only --
st.markdown("""
<style>
#MainMenu, footer, header { display: none !important; }

/* Sidebar dark */
[data-testid="stSidebar"] {
    background-color: #1e1e1e !important;
    padding: 1rem !important;
}
[data-testid="stSidebar"] .stMarkdown,
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] label {
    color: #dddddd !important;
}
[data-testid="stSidebar"] hr {
    border-color: #444 !important;
}
[data-testid="stSidebar"] .stButton > button {
    background: #2d2d2d !important;
    color: #dddddd !important;
    border: 1px solid #444 !important;
    border-radius: 8px !important;
    width: 100% !important;
    margin-bottom: 4px !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: #3d3d3d !important;
}

/* Main area max width so chat doesn't stretch full screen */
.main .block-container {
    max-width: 750px !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
}

/* Chat messages */
.user-msg {
    background: #1e1e1e;
    color: white;
    padding: 10px 14px;
    border-radius: 16px 16px 4px 16px;
    margin: 6px 0 6px auto;
    max-width: 65%;
    width: fit-content;
    font-size: 14px;
    word-wrap: break-word;
}
.bot-msg {
    background: #f0f0f0;
    color: #1e1e1e;
    padding: 10px 14px;
    border-radius: 16px 16px 16px 4px;
    margin: 6px auto 6px 0;
    max-width: 75%;
    width: fit-content;
    font-size: 14px;
    word-wrap: break-word;
    line-height: 1.6;
}
.hist-item {
    background: #2d2d2d;
    color: #cccccc !important;
    border-radius: 6px;
    padding: 6px 10px;
    font-size: 12px;
    margin-bottom: 4px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}
</style>
""", unsafe_allow_html=True)

# -- Session state --
for key, val in {"logged_in": False, "username": "", "full_name": "", "messages": [], "university": None, "pending": None}.items():
    if key not in st.session_state:
        st.session_state[key] = val

def logout():
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.session_state.full_name = ""
    st.session_state.messages = []
    st.session_state.university = None
    st.session_state.pending = None

# ══════════════════════
# PAGE 1 — LOGIN
# ══════════════════════
if not st.session_state.logged_in:
    st.markdown("<h2 style='text-align:center;margin-top:2rem;'>🎓 University AI Assistant</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;color:gray;'>IUB & BZU Smart Guide</p>", unsafe_allow_html=True)
    st.write("")

    col1, col2, col3 = st.columns([1, 1.5, 1])
    with col2:
        tab1, tab2 = st.tabs(["Login", "Register"])

        with tab1:
            un = st.text_input("Username", key="l_un")
            pw = st.text_input("Password", type="password", key="l_pw")
            if st.button("Login", type="primary", use_container_width=True):
                if un and pw:
                    ok, name = verify_user(un, pw)
                    if ok:
                        st.session_state.logged_in = True
                        st.session_state.username = un
                        st.session_state.full_name = name
                        st.rerun()
                    else:
                        st.error("Wrong username or password.")
                else:
                    st.warning("Fill both fields.")

        with tab2:
            fn = st.text_input("Full Name", key="r_fn")
            ru = st.text_input("Username", key="r_un")
            rp = st.text_input("Password", type="password", key="r_pw")
            if st.button("Register", type="primary", use_container_width=True):
                if fn and ru and rp:
                    if len(rp) < 6:
                        st.error("Password min 6 characters.")
                    else:
                        ok, msg = create_user(ru, rp, fn)
                        st.success(msg) if ok else st.error(msg)
                else:
                    st.warning("Fill all fields.")

# ══════════════════════
# PAGE 2 — UNI SELECT
# ══════════════════════
elif st.session_state.university is None:
    with st.sidebar:
        st.markdown(f"**👤 {st.session_state.full_name}**")
        st.markdown("---")
        if st.button("🚪 Logout"):
            logout()
            st.rerun()

    st.markdown(f"<h3 style='text-align:center;'>Welcome, {st.session_state.full_name}! 👋</h3>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;color:gray;'>Select your university</p>", unsafe_allow_html=True)
    st.write("")

    col1, col2 = st.columns(2, gap="large")

    with col1:
        with st.container(border=True):
            st.image(IUB_LOGO, width=70)
            st.markdown("#### Islamia University Bahawalpur")
            st.caption("IUB — Est. 1925, Bahawalpur")
            if st.button("Select IUB", use_container_width=True, type="primary", key="iub"):
                st.session_state.university = "IUB"
                st.session_state.messages = [{"role": "assistant", "content": f"Welcome {st.session_state.full_name}! 🎓 Ask me anything about IUB."}]
                st.rerun()

    with col2:
        with st.container(border=True):
            st.image(BZU_LOGO, width=70)
            st.markdown("#### Bahauddin Zakariya University")
            st.caption("BZU — Est. 1975, Multan")
            if st.button("Select BZU", use_container_width=True, type="primary", key="bzu"):
                st.session_state.university = "BZU"
                st.session_state.messages = [{"role": "assistant", "content": f"Welcome {st.session_state.full_name}! 🎓 Ask me anything about BZU."}]
                st.rerun()

# ══════════════════════
# PAGE 3 — CHAT
# ══════════════════════
else:
    uni = st.session_state.university
    logo = IUB_LOGO if uni == "IUB" else BZU_LOGO
    uni_full = "Islamia University Bahawalpur" if uni == "IUB" else "Bahauddin Zakariya University"

    # ── SIDEBAR ──
    with st.sidebar:
        st.image(logo, width=55)
        st.markdown(f"**{st.session_state.full_name}**")
        st.caption(f"{uni} Assistant")
        st.markdown("---")

        st.markdown("**Recent Chats**")
        hist = get_history(st.session_state.username)
        if hist:
            for i, h in enumerate(hist):
                q = h["question"]
                short = q[:30] + "..." if len(q) > 30 else q
                c1, c2 = st.columns([5, 1])
                with c1:
                    st.markdown(f"<div class='hist-item'>💬 {short}</div>", unsafe_allow_html=True)
                with c2:
                    if st.button("✕", key=f"del_{i}"):
                        chats_col.delete_one({"_id": h["_id"]})
                        st.rerun()
        else:
            st.caption("No history yet")

        st.markdown("---")
        if st.button("🔄 Switch University", use_container_width=True):
            st.session_state.university = None
            st.session_state.messages = []
            st.rerun()
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = [{"role": "assistant", "content": f"Chat cleared! Ask me anything about {uni} 😊"}]
            st.rerun()
        if st.button("🚪 Logout", use_container_width=True):
            logout()
            st.rerun()

    # ── HEADER ──
    c1, c2 = st.columns([1, 8])
    with c1:
        st.image(logo, width=45)
    with c2:
        st.markdown(f"**{uni} AI Assistant** — {uni_full}")
    st.divider()

    # ── MESSAGES ──
    for i, msg in enumerate(st.session_state.messages):
        if msg["role"] == "user":
            st.markdown(f"<div class='user-msg'>{msg['content']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='bot-msg'>{msg['content']}</div>", unsafe_allow_html=True)
            if i > 0:
                fc1, fc2, _ = st.columns([1, 1, 10])
                with fc1:
                    if st.button("👍", key=f"like_{i}"):
                        chats_col.update_one({"username": st.session_state.username, "answer": msg["content"]}, {"$set": {"feedback": "good"}})
                        st.toast("Thanks! 😊")
                with fc2:
                    if st.button("👎", key=f"dislike_{i}"):
                        chats_col.update_one({"username": st.session_state.username, "answer": msg["content"]}, {"$set": {"feedback": "bad"}})
                        st.toast("We'll improve! 🙏")

    # ── QUICK BUTTONS ──
    st.write("")
    q1, q2, q3 = st.columns(3)
    with q1:
        if st.button("📋 Attendance Policy", use_container_width=True):
            st.session_state.pending = f"What is the attendance policy at {uni}?"
    with q2:
        if st.button("📝 Exam Rules", use_container_width=True):
            st.session_state.pending = f"What are the exam rules at {uni}?"
    with q3:
        if st.button("💰 Fee Structure", use_container_width=True):
            st.session_state.pending = f"What is the fee structure at {uni}?"

    # ── INPUT ──
    user_input = st.chat_input(f"Ask anything about {uni}...")

    to_process = None
    if user_input and user_input.strip():
        to_process = user_input.strip()
    elif st.session_state.pending:
        to_process = st.session_state.pending
        st.session_state.pending = None

    if to_process:
        st.session_state.messages.append({"role": "user", "content": to_process})
        with st.spinner("Thinking..."):
            prefix = "iub" if uni == "IUB" else "bzu"
            docs = search_docs(to_process, prefix)
            ans, srcs = get_answer(to_process, docs, uni)
            uni_kw = ["fee", "admission", "hostel", "exam", "library", "attendance", "scholarship"]
            full_ans = ans
            if srcs and any(w in to_process.lower() for w in uni_kw):
                full_ans += f"\n\n📄 *Sources: {', '.join(srcs)}*"
            st.session_state.messages.append({"role": "assistant", "content": full_ans})
            save_chat(st.session_state.username, to_process, full_ans)
            st.rerun()

    st.markdown("<p style='text-align:center;color:#bbb;font-size:11px;margin-top:2rem;'>© 2026 Muhammad Belal | AI University Assistant</p>", unsafe_allow_html=True)
