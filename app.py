
import streamlit as st
from groq import Groq
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from pymongo import MongoClient
from datetime import datetime
import bcrypt
import os

# Initialize
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
index = pc.Index(os.environ.get("PINECONE_INDEX"))
client = MongoClient(os.environ.get("MONGODB_URI"))
db = client["university_db"]
users_collection = db["users"]
chat_collection = db["chat_history"]

# Functions
def get_embedding(text):
    return embedding_model.encode(text).tolist()

def search_pinecone(question, top_k=3):
    question_embedding = get_embedding(question)
    results = index.query(vector=question_embedding, top_k=top_k, include_metadata=True)
    return results.matches

def generate_answer(question, relevant_chunks):
    context = "\n\n".join([chunk.metadata['text'] for chunk in relevant_chunks])
    sources = list(set([chunk.metadata['source'] for chunk in relevant_chunks]))
    prompt = f"""You are a helpful university chatbot assistant for Islamia University Bahawalpur.
Use ONLY the following university document information to answer the student's question.
If the answer is not in the documents, say I don't have information about this.
Be clear, helpful and professional.

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

def ask_chatbot(question):
    relevant_chunks = search_pinecone(question)
    if not relevant_chunks:
        return "Sorry, I couldn't find any relevant information.", []
    answer, sources = generate_answer(question, relevant_chunks)
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
    users_collection.insert_one({
        "username": username,
        "password": hashed,
        "full_name": full_name
    })
    return True, "Account created successfully!"

def save_chat(username, question, answer):
    chat_collection.insert_one({
        "username": username,
        "question": question,
        "answer": answer,
        "timestamp": datetime.now()
    })

def get_chat_history(username):
    history = chat_collection.find({"username": username}).sort("timestamp", -1).limit(10)
    return list(history)

# Page config
st.set_page_config(page_title="University AI Chatbot", page_icon="🎓", layout="centered")

st.markdown("""
<style>
    .chat-message { padding: 15px; border-radius: 10px; margin: 10px 0; }
    .user-message { background-color: #1565c0; border-left: 4px solid #90caf9; color: white; }
    .bot-message { background-color: #4a148c; border-left: 4px solid #ce93d8; color: white; }
    .user-message b { color: #90caf9; }
    .bot-message b { color: #ce93d8; }
    .history-item { padding: 8px; border-radius: 8px; margin: 5px 0; background-color: #1a1a2e; color: white; font-size: 13px; }
</style>
""", unsafe_allow_html=True)

# Session state
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""
if "full_name" not in st.session_state:
    st.session_state.full_name = ""
if "messages" not in st.session_state:
    st.session_state.messages = []

# LOGIN PAGE
if not st.session_state.logged_in:
    st.markdown("<h1 style='text-align:center; color:#7c4dff;'>🎓 University AI Chatbot</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center'>Islamia University Bahawalpur</p>", unsafe_allow_html=True)
    st.divider()

    tab1, tab2 = st.tabs(["Login", "Register"])

    with tab1:
        st.subheader("Login to your account")
        username = st.text_input("Username", key="login_user")
        password = st.text_input("Password", type="password", key="login_pass")

        if st.button("Login", use_container_width=True):
            if username and password:
                success, full_name = verify_user(username, password)
                if success:
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.session_state.full_name = full_name
                    st.session_state.messages = []
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"Hello {full_name}! I am your University AI Assistant 👋 Ask me about attendance, exams, fees, hostel, or library policies!"
                    })
                    st.rerun()
                else:
                    st.error("Wrong username or password!")
            else:
                st.warning("Please enter username and password!")

    with tab2:
        st.subheader("Create new account")
        new_name = st.text_input("Full Name", key="reg_name")
        new_user = st.text_input("Username", key="reg_user")
        new_pass = st.text_input("Password", type="password", key="reg_pass")

        if st.button("Register", use_container_width=True):
            if new_name and new_user and new_pass:
                success, msg = create_user(new_user, new_pass, new_name)
                if success:
                    st.success(msg + " Please login now!")
                else:
                    st.error(msg)
            else:
                st.warning("Please fill all fields!")

# CHATBOT PAGE
else:
    # Header
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(f"<h2 style='color:#7c4dff;'>🎓 University AI Chatbot</h2>", unsafe_allow_html=True)
        st.markdown(f"<p style='color:gray;'>Welcome, {st.session_state.full_name}!</p>", unsafe_allow_html=True)
    with col2:
        if st.button("Logout"):
            st.session_state.logged_in = False
            st.session_state.username = ""
            st.session_state.full_name = ""
            st.session_state.messages = []
            st.rerun()

    st.divider()

    # Chat history sidebar
    with st.sidebar:
        st.markdown("### 📜 Your Chat History")
        history = get_chat_history(st.session_state.username)
        if history:
            for item in history:
                st.markdown(f"<div class='history-item'>Q: {item['question'][:50]}...</div>", unsafe_allow_html=True)
        else:
            st.info("No chat history yet!")

    # Messages
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f"<div class='chat-message user-message'><b>You:</b> {message['content']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='chat-message bot-message'><b>🤖 Assistant:</b> {message['content']}</div>", unsafe_allow_html=True)

    # Suggested questions
    st.markdown("**💡 Suggested Questions:**")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("📋 Attendance Policy"):
            st.session_state.pending_question = "What is the attendance policy?"
    with col2:
        if st.button("📝 Exam Rules"):
            st.session_state.pending_question = "What are the exam rules?"
    with col3:
        if st.button("💰 Fee Structure"):
            st.session_state.pending_question = "What is the fee structure?"

    # Input
    user_input = st.chat_input("Type your question here...")

    question_to_process = None
    if user_input:
        question_to_process = user_input
    elif hasattr(st.session_state, 'pending_question'):
        question_to_process = st.session_state.pending_question
        del st.session_state.pending_question

    if question_to_process:
        st.session_state.messages.append({"role": "user", "content": question_to_process})
        with st.spinner("🔍 Searching university documents..."):
            answer, sources = ask_chatbot(question_to_process)
        full_answer = answer
        if sources:
            full_answer += f"\n\n📄 *Sources: {', '.join(sources)}*"
        st.session_state.messages.append({"role": "assistant", "content": full_answer})
        save_chat(st.session_state.username, question_to_process, full_answer)
        st.rerun()

    st.divider()
    st.markdown("<p style='text-align:center; color:gray; font-size:12px'>Powered by Groq + Pinecone + MongoDB</p>", unsafe_allow_html=True)
