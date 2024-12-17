import os
import re
import pdfplumber
import pyttsx3
import speech_recognition as sr
from dotenv import load_dotenv
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters.character import CharacterTextSplitter
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from gtts import gTTS  # Google Text-to-Speech for human-like voice
import io

# Load environment variables
load_dotenv()
working_dir = os.path.dirname(os.path.abspath(__file__))

# Streamlit UI setup
st.set_page_config(
    page_title="Welcome to MitraBot",
    page_icon="📄",
    layout="centered"
)

# Custom CSS for styling the app
st.markdown("""
    <style>
    body {
        background: linear-gradient(135deg, #f2f2f2, #e0e0e0);
        font-family: 'Arial', sans-serif;
    }
    .title {
        text-align: center;
        font-size: 3em;
        color: #4A90E2;
        margin-bottom: 30px;
    }
    .chat-message {
        border-radius: 15px;
        padding: 10px 20px;
        margin-bottom: 10px;
        max-width: 70%;
    }
    .user {
        background-color: #4CAF50;
        color: white;
        align-self: flex-start;
    }
    .assistant {
        background-color: #2196F3;
        color: white;
        align-self: flex-end;
    }
    button {
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 12px 25px;
        font-size: 16px;
        cursor: pointer;
        border-radius: 8px;
        transition: background-color 0.3s ease;
    }
    button:hover {
        background-color: #45a049;
    }
    .success {
        background-color: #4CAF50;
        color: white;
        padding: 10px;
        border-radius: 5px;
        font-weight: bold;
    }
    .file-uploader {
        padding: 20px;
        background-color: #ffffff;
        border-radius: 8px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        text-align: center;
        border: 2px dashed #2196F3;
    }
    </style>
""", unsafe_allow_html=True)

# Title and branding
st.markdown('<h1 class="title">🦙 Chat with MitraBot powered by LLAMA 3.1</h1>', unsafe_allow_html=True)

# Initialize session states
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "input_mode" not in st.session_state:
    st.session_state.input_mode = "Text"  # Default input mode

# Input mode selection
st.session_state.input_mode = st.radio(
    "Choose input mode", 
    ("Text", "Voice"), 
    key="input_mode_selector"
)

# Function to load and preprocess the PDF document
def load_document(file_path):
    try:
        documents = []
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    text = re.sub(r'\s+', ' ', text).strip()
                    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
                    documents.append({"text": text})
        return documents
    except Exception as e:
        st.error(f"Failed to load PDF document: {e}")
        return None

# Function to set up the vectorstore
def setup_vectorstore(documents):
    try:
        embeddings = HuggingFaceEmbeddings()
        text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200)
        doc_chunks = []
        for doc in documents:
            chunks = text_splitter.split_text(doc["text"])
            doc_chunks.extend(chunks)
        vectorstore = FAISS.from_texts(doc_chunks, embeddings)
        st.info("Vectorstore created successfully!")
        return vectorstore
    except Exception as e:
        st.error(f"Error creating vectorstore: {e}")
        return None

# Function to create the conversation chain
def create_chain(vectorstore):
    try:
        llm = ChatGroq(model="llama-3.1-70b-versatile", temperature=0)
        retriever = vectorstore.as_retriever()
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory)
        st.info("Conversation chain initialized successfully!")
        return chain
    except Exception as e:
        st.error(f"Failed to create conversation chain: {e}")
        return None

# Text-to-speech functionality
def speak(text):
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    engine.setProperty('volume', 1)
    engine.say(text)
    engine.runAndWait()

# Voice recognition functionality
def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Listening for your question... Please speak.")
        audio = recognizer.listen(source)
    try:
        query = recognizer.recognize_google(audio)
        return query
    except Exception as e:
        st.error(f"Sorry, I couldn't understand your question. Error: {e}")
        return None

# PDF Upload Section
uploaded_file = st.file_uploader("Upload your PDF file", type=["pdf"])
if uploaded_file:
    file_path = os.path.join(working_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    documents = load_document(file_path)
    if documents:
        if "vectorstore" not in st.session_state:
            st.session_state.vectorstore = setup_vectorstore(documents)
        if "conversation_chain" not in st.session_state:
            st.session_state.conversation_chain = create_chain(st.session_state.vectorstore)
        st.success("PDF uploaded and processed successfully!", icon="✅")
    else:
        st.error("Failed to load document.")

# Display chat history
for message in st.session_state.chat_history:
    class_name = "user" if message["role"] == "user" else "assistant"
    st.markdown(f'<div class="chat-message {class_name}">{message["content"]}</div>', unsafe_allow_html=True)

# Clear chat button
if st.button("Clear Chat"):
    st.session_state.chat_history = []
    st.success("Chat history cleared!")

# Text Input Mode
if st.session_state.input_mode == "Text":
    user_input = st.chat_input("Ask MitraBot...")
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        with st.chat_message("assistant"):
            response = st.session_state.conversation_chain({"question": user_input})
            assistant_response = response.get("answer", "Sorry, I couldn't process that.")
            st.markdown(assistant_response)
            st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})
            speak(assistant_response)

# Voice Input Mode
if st.session_state.input_mode == "Voice" and st.button("Ask via Voice"):
    query = recognize_speech()
    if query:
        st.session_state.chat_history.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)
        with st.chat_message("assistant"):
            response = st.session_state.conversation_chain({"question": query})
            assistant_response = response.get("answer", "Sorry, I couldn't process that.")
            st.markdown(assistant_response)
            st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})
            speak(assistant_response)
