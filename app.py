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

# -- Knowledge Base --
from knowledge_base import UNIVERSITY_KNOWLEDGE

# -- Logos --
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
    filtered = [r for r in results if r.metadata.get("source","").lower().startswith(uni_prefix)]
    return filtered if filtered else results

def get_answer(question, chunks, university):
    context = "\n\n".join([c.metadata["text"] for c in chunks])
    sources = list(set([c.metadata["source"] for c in chunks]))
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
                extra_knowledge = uni_section[max(0, idx-200):idx+3000]
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

FORMAT RULES - always follow this structure:
- Give detailed answers using bullet points (*)
- Use sub-bullets (->) for extra details
- Bold the main heading like **Fee Structure:**
- Give 3-5 bullet points per answer - not too short, not too long
- End with a helpful tip starting with Tip:

ANSWER RULES:
- First check Documents, then Knowledge Base
- Both IUB and BZU questions must get equally detailed answers
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

# -- Page config --
st.set_page_config(page_title="University AI Assistant", page_icon="🎓", layout="centered")

# -- CSS with movable sidebar --
st.markdown("""
<style>
@import url("https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap");

*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"], .stApp {
    background-color: #ffffff !important;
    color: #1a1a1a !important;
    font-family: "Inter", -apple-system, BlinkMacSystemFont, sans-serif !important;
}

#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

/* Sidebar styles - movable */
[data-testid="stSidebar"] {
    min-width: 280px !important;
    width: 280px !important;
    background-color: #f9f9f9 !important;
    border-right: 1px solid #e5e5e5 !important;
    transition: transform 0.3s ease-in-out !important;
    z-index: 999 !important;
}

/* When sidebar is collapsed */
[data-testid="stSidebar"][aria-expanded="false"] {
    transform: translateX(-100%) !important;
}

/* When sidebar is expanded */
[data-testid="stSidebar"][aria-expanded="true"] {
    transform: translateX(0) !important;
}

/* Sidebar toggle button */
[data-testid="stSidebarCollapsedControl"] {
    display: block !important;
    visibility: visible !important;
    background-color: #ffffff !important;
    border: 1px solid #e5e5e5 !important;
    border-radius: 8px !important;
    cursor: pointer !important;
    position: fixed !important;
    left: 10px !important;
    top: 70px !important;
    z-index: 1000 !important;
    padding: 8px 12px !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1) !important;
    font-size: 18px !important;
}

[data-testid="stSidebarCollapsedControl"]:hover {
    background-color: #f0f0f0 !important;
}

/* Mobile responsive */
@media screen and (max-width: 768px) {
    [data-testid="stSidebar"] {
        position: fixed !important;
        height: 100vh !important;
        top: 0 !important;
        left: 0 !important;
        box-shadow: 2px 0 12px rgba(0,0,0,0.15) !important;
    }
    
    [data-testid="stSidebarCollapsedControl"] {
        left: 10px !important;
        top: 70px !important;
        padding: 8px 12px !important;
    }
    
    .main .block-container {
        padding-left: 20px !important;
        padding-right: 20px !important;
    }
}

/* Desktop styles */
@media screen and (min-width: 769px) {
    [data-testid="stSidebarCollapsedControl"] {
        left: 15px !important;
        top: 80px !important;
    }
}

[data-testid="stSidebar"] * {
    color: #1a1a1a !important;
}

.stTextInput > div > div > input {
    background: #ffffff !important;
    border: 1px solid #e5e5e5 !important;
    border-radius: 8px !important;
    color: #1a1a1a !important;
    font-size: 14px !important;
    padding: 10px 14px !important;
    box-shadow: none !important;
}

.stTextInput > div > div > input:focus {
    border-color: #b4b4b4 !important;
    box-shadow: none !important;
    outline: none !important;
}

.stTextInput > label {
    color: #6b6b6b !important;
    font-size: 13px !important;
    font-weight: 500 !important;
}

.stButton > button {
    background: #ffffff !important;
    color: #1a1a1a !important;
    border: 1px solid #e5e5e5 !important;
    border-radius: 8px !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    padding: 8px 16px !important;
    transition: background 0.15s !important;
    box-shadow: none !important;
}

.stButton > button:hover {
    background: #f7f7f7 !important;
    border-color: #d0d0d0 !important;
}

.stButton > button[kind="primary"] {
    background: #1a1a1a !important;
    color: #ffffff !important;
    border: none !important;
}

.stButton > button[kind="primary"]:hover {
    background: #333333 !important;
}

.stTabs [data-baseweb="tab-list"] {
    border-bottom: 1px solid #e5e5e5 !important;
    gap: 0 !important;
    background: transparent !important;
}

.stTabs [data-baseweb="tab"] {
    font-size: 14px !important;
    font-weight: 500 !important;
    color: #6b6b6b !important;
    border-bottom: 2px solid transparent !important;
    padding: 10px 20px !important;
    background: transparent !important;
}

.stTabs [aria-selected="true"] {
    color: #1a1a1a !important;
    border-bottom: 2px solid #1a1a1a !important;
}

hr {
    border: none !important;
    border-top: 1px solid #e5e5e5 !important;
    margin: 12px 0 !important;
}

.user-msg {
    display: flex;
    justify-content: flex-end;
    margin: 12px 0;
}

.user-bubble {
    background: #1a1a1a;
    color: #ffffff;
    padding: 10px 16px;
    border-radius: 18px 18px 4px 18px;
    max-width: 72%;
    font-size: 14px;
    line-height: 1.6;
}

.bot-msg {
    display: flex;
    justify-content: flex-start;
    margin: 12px 0;
}

.bot-bubble {
    background: #f4f4f4;
    color: #1a1a1a;
    padding: 10px 16px;
    border-radius: 18px 18px 18px 4px;
    max-width: 72%;
    font-size: 14px;
    line-height: 1.6;
}

.hist-item {
    padding: 8px 12px;
    border-radius: 8px;
    margin: 4px 0;
    background: #ffffff;
    border: 1px solid #e5e5e5;
    color: #4b4b4b;
    font-size: 13px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

.footer {
    text-align: center;
    color: #b4b4b4;
    font-size: 12px;
    margin-top: 24px;
    padding-bottom: 8px;
}

.uni-card {
    border: 1px solid #e5e5e5;
    border-radius: 12px;
    padding: 24px 16px;
    text-align: center;
    background: #fafafa;
    height: 100%;
}

.uni-card img {
    width: 72px;
    height: 72px;
    object-fit: contain;
    margin-bottom: 12px;
    border-radius: 8px;
    background: transparent;
    mix-blend-mode: multiply;
}

.uni-card h4 {
    margin: 0 0 4px;
    font-size: 14px;
    font-weight: 600;
    color: #1a1a1a;
}

.uni-card p {
    margin: 0;
    font-size: 12px;
    color: #8a8a8a;
}

.stSpinner > div { border-top-color: #1a1a1a !important; }
</style>

<script>
// JavaScript to handle sidebar toggle
document.addEventListener('DOMContentLoaded', function() {
    const toggleBtn = parent.document.querySelector('[data-testid="stSidebarCollapsedControl"]');
    const sidebar = parent.document.querySelector('[data-testid="stSidebar"]');
    
    if (toggleBtn && sidebar) {
        toggleBtn.addEventListener('click', function() {
            const isExpanded = sidebar.getAttribute('aria-expanded') === 'true';
            sidebar.setAttribute('aria-expanded', !isExpanded);
        });
    }
});
</script>
""", unsafe_allow_html=True)

# -- Session state init --
for key, val in {"logged_in": False, "username": "", "full_name": "", "messages": [], "university": None}.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ════════════════════════════════
# PAGE 1 - LOGIN
# ════════════════════════════════
if not st.session_state.logged_in:
    st.write("")
    st.markdown("<h2 style='text-align:center;font-weight:600;font-size:26px;color:#1a1a1a;'>🎓 University AI Assistant</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;color:#8a8a8a;font-size:14px;margin-bottom:24px;'>Your Smart University Guide</p>", unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["Login", "Register"])
    
    with tab1:
        st.write("")
        un = st.text_input("Username", key="l_un", placeholder="Enter your username")
        pw = st.text_input("Password", key="l_pw", placeholder="Enter your password", type="password")
        st.write("")
        if st.button("Login", use_container_width=True, type="primary", key="login_btn"):
            if un and pw:
                ok, name = verify_user(un, pw)
                if ok:
                    st.session_state.logged_in = True
                    st.session_state.username = un
                    st.session_state.full_name = name
                    st.session_state.messages = []
                    st.session_state.university = None
                    st.rerun()
                else:
                    st.error("Wrong username or password.")
            else:
                st.warning("Please fill both fields.")
    
    with tab2:
        st.write("")
        fn = st.text_input("Full Name", key="r_fn", placeholder="e.g. Muhammad Ali")
        ru = st.text_input("Username", key="r_un", placeholder="e.g. muhammadali")
        rp = st.text_input("Password", key="r_pw", placeholder="Minimum 6 characters", type="password")
        st.write("")
        if st.button("Create Account", use_container_width=True, type="primary", key="reg_btn"):
            if fn and ru and rp:
                if len(rp) < 6:
                    st.error("Password must be at least 6 characters.")
                else:
                    ok, msg = create_user(ru, rp, fn)
                    if ok:
                        st.success(msg + " Please login.")
                    else:
                        st.error(msg)
            else:
                st.warning("Please fill all fields.")
    
    st.markdown("<div class='footer'>© 2026 Muhammad Belal | AI University Assistant</div>", unsafe_allow_html=True)

# ════════════════════════════════
# PAGE 2 - UNIVERSITY SELECTOR
# ════════════════════════════════
elif st.session_state.university is None:
    st.write("")
    st.markdown(f"<h3 style='text-align:center;font-weight:600;color:#1a1a1a;'>Welcome, {st.session_state.full_name}! 👋</h3>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;color:#8a8a8a;font-size:14px;'>Choose your university to get started</p>", unsafe_allow_html=True)
    st.write("")
    
    c1, c2 = st.columns(2, gap="medium")
    
    with c1:
        st.markdown(f"""<div class='uni-card'>
            <img src='{IUB_LOGO}' />
            <h4>Islamia University Bahawalpur</h4>
            <p>IUB — Est. 1925, Bahawalpur</p>
        </div>""", unsafe_allow_html=True)
        st.write("")
        if st.button("Select IUB", use_container_width=True, key="sel_iub"):
            st.session_state.university = "IUB"
            st.session_state.messages = [{"role": "assistant", "content": f"Welcome {st.session_state.full_name}! 🎓 Ask me anything about IUB — fees, admissions, exams, hostel, or library."}]
            st.rerun()
    
    with c2:
        st.markdown(f"""<div class='uni-card'>
            <img src='{BZU_LOGO}' />
            <h4>Bahauddin Zakariya University</h4>
            <p>BZU — Est. 1975, Multan</p>
        </div>""", unsafe_allow_html=True)
        st.write("")
        if st.button("Select BZU", use_container_width=True, key="sel_bzu"):
            st.session_state.university = "BZU"
            st.session_state.messages = [{"role": "assistant", "content": f"Welcome {st.session_state.full_name}! 🎓 Ask me anything about BZU — fees, admissions, exams, hostel, or scholarships."}]
            st.rerun()
    
    st.write("")
    st.write("")
    _, mid, _ = st.columns([3, 1, 3])
    with mid:
        if st.button("Logout", key="lo_sel"):
            for k in ["logged_in", "username", "full_name", "messages", "university"]:
                st.session_state[k] = False if k == "logged_in" else (None if k == "university" else ("" if k != "messages" else []))
            st.rerun()
    
    st.markdown("<div class='footer'>© 2026 Muhammad Belal | AI University Assistant</div>", unsafe_allow_html=True)

# ════════════════════════════════
# PAGE 3 - CHATBOT
# ════════════════════════════════
else:
    uni = st.session_state.university
    logo = IUB_LOGO if uni == "IUB" else BZU_LOGO
    uni_full = "Islamia University Bahawalpur" if uni == "IUB" else "Bahauddin Zakariya University"
    
    # Header
    h1, h2, h3 = st.columns([1, 5, 2])
    with h1:
        st.markdown(f"<img src='{logo}' style='width:48px;height:48px;object-fit:contain;border-radius:8px;margin-top:6px;mix-blend-mode:multiply;'>", unsafe_allow_html=True)
    with h2:
        st.markdown(f"<div style='padding-top:4px;'><p style='margin:0;font-weight:600;font-size:17px;color:#1a1a1a;'>{uni} AI Assistant</p><p style='margin:0;font-size:12px;color:#8a8a8a;'>{uni_full} — {st.session_state.full_name}</p></div>", unsafe_allow_html=True)
    with h3:
        if st.button("Switch Uni", key="sw"):
            st.session_state.university = None
            st.session_state.messages = []
            st.rerun()
    
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # Sidebar with recent chats
    with st.sidebar:
        st.markdown(f"<img src='{logo}' style='width:72px;height:72px;object-fit:contain;border-radius:8px;display:block;margin:0 auto 12px;mix-blend-mode:multiply;'>", unsafe_allow_html=True)
        st.markdown("<p style='font-weight:600;font-size:13px;color:#4b4b4b;margin-bottom:8px;'>📋 Recent Chats</p>", unsafe_allow_html=True)
        
        hist = get_history(st.session_state.username)
        if hist:
            for i, h in enumerate(hist):
                q = h["question"][:35] + "..." if len(h["question"]) > 35 else h["question"]
                c1, c2 = st.columns([5, 1])
                with c1:
                    st.markdown(f"<div class='hist-item'>{q}</div>", unsafe_allow_html=True)
                with c2:
                    if st.button("🗑️", key=f"del_{i}", help="Delete"):
                        chats_col.delete_one({"_id": h["_id"]})
                        st.rerun()
        else:
            st.markdown("<p style='color:#b4b4b4;font-size:13px;'>No chat history yet.</p>", unsafe_allow_html=True)
        
        st.markdown("<hr>", unsafe_allow_html=True)
        
        if st.button("🗑️ Clear Chat", use_container_width=True, key="clr"):
            st.session_state.messages = [{"role": "assistant", "content": f"Chat cleared! Ask me anything about {uni}. 😊"}]
            st.rerun()
        
        if st.button("🚪 Logout", use_container_width=True, key="lo_chat"):
            for k in ["logged_in", "username", "full_name", "messages", "university"]:
                st.session_state[k] = False if k == "logged_in" else (None if k == "university" else ("" if k != "messages" else []))
            st.rerun()
    
    # Display messages
    for i, msg in enumerate(st.session_state.messages):
        if msg["role"] == "user":
            st.markdown(f"<div class='user-msg'><div class='user-bubble'>{msg['content']}</div></div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='bot-msg'><div class='bot-bubble'>{msg['content']}</div></div>", unsafe_allow_html=True)
            
            # Feedback buttons for bot messages
            col1, col2, col3 = st.columns([1, 1, 10])
            with col1:
                if st.button("👍", key=f"like_{i}", help="Good response"):
                    chats_col.update_one(
                        {"username": st.session_state.username, "answer": msg['content']},
                        {"$set": {"feedback": "good"}}
                    )
                    st.toast("Thanks for feedback! 😊")
            with col2:
                if st.button("👎", key=f"dislike_{i}", help="Bad response"):
                    chats_col.update_one(
                        {"username": st.session_state.username, "answer": msg['content']},
                        {"$set": {"feedback": "bad"}}
                    )
                    st.toast("Sorry! We'll improve 🙏")
            with col3:
                st.empty()
    
    st.write("")
    
    # Quick buttons
    st.markdown("<p style='font-size:12px;color:#8a8a8a;font-weight:500;margin-bottom:6px;'>⚡ Quick Questions</p>", unsafe_allow_html=True)
    q1, q2, q3 = st.columns(3)
    with q1:
        if st.button("📋 Attendance Policy", use_container_width=True, key="qq_att"):
            st.session_state.pending = f"What is the attendance policy at {uni}?"
    with q2:
        if st.button("📝 Exam Rules", use_container_width=True, key="qq_exam"):
            st.session_state.pending = f"What are the exam rules at {uni}?"
    with q3:
        if st.button("💰 Fee Structure", use_container_width=True, key="qq_fee"):
            st.session_state.pending = f"What is the fee structure at {uni}?"
    
    # Chat input
    user_input = st.chat_input("💬 Message University AI Assistant...")
    to_process = None
    
    if user_input:
        to_process = user_input
    elif hasattr(st.session_state, "pending"):
        to_process = st.session_state.pending
        del st.session_state.pending
    
    if to_process:
        st.session_state.messages.append({"role": "user", "content": to_process})
        with st.spinner("🤔 Thinking..."):
            prefix = "iub" if uni == "IUB" else "bzu"
            docs = search_docs(to_process, prefix)
            ans, srcs = get_answer(to_process, docs, uni)
            
            uni_kw = ["fee", "admission", "hostel", "exam", "library",
                      "attendance", "scholarship", "department",
                      "iub", "bzu", "university", "policy",
                      "semester", "result"]
            is_uni_q = any(w in to_process.lower() for w in uni_kw)
            
            full_ans = ans
            if srcs and is_uni_q:
                full_ans += f"\n\n📄 *Sources: {', '.join(srcs)}*"
            
            st.session_state.messages.append({"role": "assistant", "content": full_ans})
            save_chat(st.session_state.username, to_process, full_ans)
            st.rerun()
    
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<div class='footer'>© 2026 Muhammad Belal | AI University Assistant</div>", unsafe_allow_html=True)
