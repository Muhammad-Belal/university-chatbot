import streamlit as st
from groq import Groq
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from pymongo import MongoClient
from datetime import datetime
import bcrypt
import os

groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
index = pc.Index(os.environ.get("PINECONE_INDEX"))
client = MongoClient(os.environ.get("MONGODB_URI"))
db = client["university_db"]
users_collection = db["users"]
chat_collection = db["chat_history"]

IUB_LOGO = "iub_logo.png"
BZU_LOGO = "bzu_logo.png"

def get_embedding(text):
    return embedding_model.encode(text).tolist()

def search_pinecone(question, university_filter=None, top_k=3):
    question_embedding = get_embedding(question)
    results = index.query(vector=question_embedding, top_k=top_k, include_metadata=True)
    matches = results.matches
    if university_filter:
        prefix = university_filter.lower()
        filtered = [m for m in matches if m.metadata.get('source', '').lower().startswith(prefix)]
        if filtered:
            return filtered
    return matches

def generate_answer(question, relevant_chunks, university):
    context = "\n\n".join([chunk.metadata['text'] for chunk in relevant_chunks])
    sources = list(set([chunk.metadata['source'] for chunk in relevant_chunks]))
    prompt = f"""You are a highly intelligent, friendly, and helpful AI assistant for {university} students.
Detect the language of the question automatically.
If the question is in Urdu script, reply in Urdu.
If the question is in Roman Urdu, reply in Roman Urdu.
If the question is in English, reply in English.
For university related questions about admissions, fees, exams, hostel, library, or policies, use the university documents provided below to give accurate answers.
For all other questions including general knowledge, science, history, coding, math, relationships, career advice, everyday problems, or casual conversation, answer using your own knowledge. Be accurate, helpful, and conversational.
Never say you cannot answer. Always try your best to help.

University Documents:
{context}

Student Question: {question}

Answer:"""
    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500
    )
    return response.choices[0].message.content, sources

def ask_chatbot(question, university):
    prefix = "iub" if university == "IUB" else "bzu"
    relevant_chunks = search_pinecone(question, university_filter=prefix)
    if not relevant_chunks:
        return "Sorry, I could not find any relevant information.", []
    answer, sources = generate_answer(question, relevant_chunks, university)
    return answer, sources

def verify_user(username, password):
    user = users_collection.find_one({"username": username})
    if not user:
        return False, None
    if bcrypt.checkpw(password.encode('utf-8'), user['password']):
        return True, user['full_name']
    return False, None

def create_user(username, password, full_name):
    if users_collection.find_one({"username": username}):
        return False, "Username already exists!"
    hashed = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    users_collection.insert_one({"username": username, "password": hashed, "full_name": full_name})
    return True, "Account created successfully!"

def save_chat(username, question, answer):
    chat_collection.insert_one({"username": username, "question": question, "answer": answer, "timestamp": datetime.now()})

def get_chat_history(username):
    history = chat_collection.find({"username": username}).sort("timestamp", -1).limit(10)
    return list(history)

st.set_page_config(page_title="University AI Chatbot", page_icon="🎓", layout="centered")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .stApp { background-color: #ffffff; color: #111827; }
    header[data-testid="stHeader"] { background-color: #ffffff; }
    .user-message { display: flex; justify-content: flex-end; margin: 8px 0; }
    .user-bubble { background-color: #2563EB; color: white; padding: 12px 16px; border-radius: 18px 18px 4px 18px; max-width: 75%; font-size: 14px; line-height: 1.5; }
    .bot-message { display: flex; justify-content: flex-start; margin: 8px 0; }
    .bot-bubble { background-color: #F3F4F6; color: #111827; padding: 12px 16px; border-radius: 18px 18px 18px 4px; max-width: 75%; font-size: 14px; line-height: 1.5; }
    [data-testid="stSidebar"] { background-color: #F9FAFB; border-right: 1px solid #E5E7EB; }
    .history-item { padding: 10px 12px; border-radius: 8px; margin: 4px 0; background-color: #FFFFFF; border: 1px solid #E5E7EB; color: #374151; font-size: 13px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .stButton > button { border-radius: 8px; font-weight: 500; font-size: 14px; }
    .stTextInput > div > div > input { border-radius: 8px; border: 1px solid #D1D5DB; font-size: 14px; }
    hr { border: none; border-top: 1px solid #E5E7EB; margin: 16px 0; }
    .footer { text-align: center; color: #9CA3AF; font-size: 12px; padding: 16px 0 8px 0; }
    .uni-row { display: flex; gap: 16px; margin: 24px 0; }
    .uni-card { flex: 1; border: 1.5px solid #E5E7EB; border-radius: 14px; padding: 20px 12px; text-align: center; background: #FAFAFA; }
    .uni-card img { width: 60px; height: 60px; object-fit: contain; margin-bottom: 10px; }
    .uni-card h4 { margin: 0 0 4px; font-size: 13px; font-weight: 600; color: #111827; }
    .uni-card p { margin: 0; font-size: 11px; color: #6B7280; }
</style>
""", unsafe_allow_html=True)

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""
if "full_name" not in st.session_state:
    st.session_state.full_name = ""
if "messages" not in st.session_state:
    st.session_state.messages = []
if "university_selected" not in st.session_state:
    st.session_state.university_selected = None

# LOGIN PAGE
if not st.session_state.logged_in:
    st.write("")
    st.markdown("<h2 style='text-align:center; font-weight:600; color:#111827;'>🎓 University AI Assistant</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:#6B7280; font-size:14px;'>IUB & BZU Smart Chatbot</p>", unsafe_allow_html=True)
    st.write("")
    tab1, tab2 = st.tabs(["Login", "Register"])
    with tab1:
        st.markdown("<p style='font-weight:500; color:#374151;'>Welcome back</p>", unsafe_allow_html=True)
        username = st.text_input("Username", key="login_user", placeholder="Enter your username")
        password = st.text_input("Password", type="password", key="login_pass", placeholder="Enter your password")
        if st.button("Login", use_container_width=True, type="primary"):
            if username and password:
                success, full_name = verify_user(username, password)
                if success:
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.session_state.full_name = full_name
                    st.session_state.messages = []
                    st.session_state.university_selected = None
                    st.rerun()
                else:
                    st.error("Wrong username or password!")
            else:
                st.warning("Please enter username and password!")
    with tab2:
        st.markdown("<p style='font-weight:500; color:#374151;'>Create your account</p>", unsafe_allow_html=True)
        new_name = st.text_input("Full Name", key="reg_name", placeholder="Muhammad Ali")
        new_user = st.text_input("Username", key="reg_user", placeholder="muhammadali123")
        new_pass = st.text_input("Password", type="password", key="reg_pass", placeholder="Minimum 6 characters")
        if st.button("Register", use_container_width=True, type="primary"):
            if new_name and new_user and new_pass:
                if len(new_pass) < 6:
                    st.error("Password must be at least 6 characters!")
                else:
                    success, msg = create_user(new_user, new_pass, new_name)
                    if success:
                        st.success(msg + " Please login now!")
                    else:
                        st.error(msg)
            else:
                st.warning("Please fill all fields!")
    st.markdown("<div class='footer'>&#169; 2026 Muhammad Belal | AI University Assistant</div>", unsafe_allow_html=True)

# UNIVERSITY SELECTOR PAGE
elif st.session_state.university_selected is None:
    st.write("")
    st.markdown(f"<h3 style='text-align:center; font-weight:600; color:#111827;'>Welcome, {st.session_state.full_name}! 👋</h3>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:#6B7280; font-size:14px;'>Select your university to continue</p>", unsafe_allow_html=True)
    st.write("")
    col_l, col_m, col_r = st.columns([1, 4, 1])
    with col_m:
        st.markdown(f"""
        <div class="uni-row">
            <div class="uni-card">
                <img src="{IUB_LOGO}" onerror="this.style.fontSize='40px'; this.outerHTML='<div style=font-size:40px>🏛️</div>'" />
                <h4>Islamia University Bahawalpur</h4>
                <p>IUB &mdash; Est. 1925, Bahawalpur</p>
            </div>
            <div class="uni-card">
                <img src="{BZU_LOGO}" onerror="this.style.fontSize='40px'; this.outerHTML='<div style=font-size:40px>🏫</div>'" />
                <h4>Bahauddin Zakariya University</h4>
                <p>BZU &mdash; Est. 1975, Multan</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Select IUB", use_container_width=True, type="secondary", key="select_iub"):
                st.session_state.university_selected = "IUB"
                st.session_state.messages = [{"role": "assistant", "content": f"Welcome {st.session_state.full_name}! 🎓 I am your IUB Smart Assistant. Ask me anything about Islamia University Bahawalpur — admissions, fees, exams, hostel, or library!"}]
                st.rerun()
        with col2:
            if st.button("Select BZU", use_container_width=True, type="secondary", key="select_bzu"):
                st.session_state.university_selected = "BZU"
                st.session_state.messages = [{"role": "assistant", "content": f"Welcome {st.session_state.full_name}! 🎓 I am your BZU Smart Assistant. Ask me anything about Bahauddin Zakariya University — admissions, fees, exams, hostel, or scholarships!"}]
                st.rerun()
        st.write("")
        col_a, col_b, col_c = st.columns([2, 1, 2])
        with col_b:
            if st.button("Logout", key="logout_selector", type="secondary"):
                st.session_state.logged_in = False
                st.session_state.university_selected = None
                st.session_state.messages = []
                st.rerun()
    st.markdown("<div class='footer'>&#169; 2026 Muhammad Belal | AI University Assistant</div>", unsafe_allow_html=True)

# CHATBOT PAGE
else:
    university = st.session_state.university_selected
    uni_color = "#1D4ED8" if university == "IUB" else "#065F46"
    uni_logo = IUB_LOGO if university == "IUB" else BZU_LOGO
    uni_full = "Islamia University Bahawalpur" if university == "IUB" else "Bahauddin Zakariya University"

    col1, col2, col3 = st.columns([1, 5, 1])
    with col1:
        st.image(uni_logo, width=52)
    with col2:
        st.markdown(f"<h3 style='margin:0; font-weight:600; color:{uni_color};'>{university} AI Assistant</h3>", unsafe_allow_html=True)
        st.markdown(f"<p style='margin:0; color:#6B7280; font-size:12px;'>{uni_full} — {st.session_state.full_name}</p>", unsafe_allow_html=True)
    with col3:
        if st.button("↩ Switch"):
            st.session_state.university_selected = None
            st.session_state.messages = []
            st.rerun()

    st.markdown("<hr>", unsafe_allow_html=True)

    with st.sidebar:
        st.image(uni_logo, width=80)
        st.markdown("<p style='font-weight:600; color:#111827; font-size:14px; margin-top:8px;'>📜 Recent Chats</p>", unsafe_allow_html=True)
        history = get_chat_history(st.session_state.username)
        if history:
            for item in history:
                q = item['question'][:45] + "..." if len(item['question']) > 45 else item['question']
                st.markdown(f"<div class='history-item'>💬 {q}</div>", unsafe_allow_html=True)
        else:
            st.markdown("<p style='color:#9CA3AF; font-size:13px;'>No history yet</p>", unsafe_allow_html=True)
        st.markdown("<hr>", unsafe_allow_html=True)
        if st.button("🔓 Logout", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.university_selected = None
            st.session_state.messages = []
            st.rerun()
        if st.button("🔄 Clear Chat", use_container_width=True):
            st.session_state.messages = [{"role": "assistant", "content": f"Chat cleared! Ask me anything about {university}. 😊"}]
            st.rerun()

    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f"<div class='user-message'><div class='user-bubble'>{message['content']}</div></div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='bot-message'><div class='bot-bubble'>{message['content']}</div></div>", unsafe_allow_html=True)

    st.write("")
    st.markdown("<p style='color:#6B7280; font-size:12px; font-weight:500;'>💡 Quick Questions</p>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("📋 Attendance Policy", use_container_width=True, type="secondary"):
            st.session_state.pending_question = f"What is the attendance policy at {university}?"
    with col2:
        if st.button("📝 Exam Rules", use_container_width=True, type="secondary"):
            st.session_state.pending_question = f"What are the exam rules at {university}?"
    with col3:
        if st.button("💰 Fee Structure", use_container_width=True, type="secondary"):
            st.session_state.pending_question = f"What is the fee structure at {university}?"

    user_input = st.chat_input("Ask me anything...")
    question_to_process = None
    if user_input:
        question_to_process = user_input
    elif hasattr(st.session_state, 'pending_question'):
        question_to_process = st.session_state.pending_question
        del st.session_state.pending_question

    if question_to_process:
        st.session_state.messages.append({"role": "user", "content": question_to_process})
        with st.spinner("Thinking..."):
            answer, sources = ask_chatbot(question_to_process, university)
        uni_keywords = ["fee", "admission", "hostel", "exam", "library", "attendance", "scholarship", "department", "iub", "bzu", "university", "policy", "semester", "result"]
        is_uni_question = any(word in question_to_process.lower() for word in uni_keywords)
        full_answer = answer
        if sources and is_uni_question:
            full_answer += f"\n\n📄 *Sources: {', '.join(sources)}*"
        st.session_state.messages.append({"role": "assistant", "content": full_answer})
        save_chat(st.session_state.username, question_to_process, full_answer)
        st.rerun()

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<div class='footer'>&#169; 2026 Muhammad Belal | AI University Assistant</div>", unsafe_allow_html=True)
