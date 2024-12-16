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

# Streamlit UI setup - this should be the first Streamlit function
st.set_page_config(
    page_title="Welcome to MitraBot",
    page_icon="📄",
    layout="centered"
)

# Custom CSS for styling the app
st.markdown("""
    <style>
    /* Customizing the main background */
    body {
        background: linear-gradient(135deg, #f2f2f2, #e0e0e0);
        font-family: 'Arial', sans-serif;
    }

    /* Styling for the title */
    .title {
        text-align: center;
        font-size: 3em;
        color: #4A90E2;
        margin-bottom: 30px;
    }

    /* Chat message bubbles */
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

    /* Customizing buttons */
    button {
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 12px 25px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 8px;
        transition: background-color 0.3s ease;
    }

    button:hover {
        background-color: #45a049;
    }

    /* Success message styling */
    .success {
        background-color: #4CAF50;
        color: white;
        padding: 10px;
        border-radius: 5px;
        font-weight: bold;
    }

    /* File uploader styles */
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

# Title and Branding
st.markdown('<h1 class="title">🦙 Chat with MitraBot powered by LLAMA 3.1</h1>', unsafe_allow_html=True)

# Initialize chat history in Streamlit session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Function to load and preprocess the PDF document
def load_document(file_path):
    try:
        documents = []
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    # Preprocess the text: remove extra spaces and non-ASCII characters
                    text = re.sub(r'\s+', ' ', text).strip()
                    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
                    documents.append({"text": text})
        print("PDF document loaded and processed successfully.")
        return documents
    except Exception as e:
        print(f"Failed to load PDF document: {e}")
        return None

# Function to set up the vector store
def setup_vectorstore(documents):
    try:
        embeddings = HuggingFaceEmbeddings()
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200
        )
        
        # Split documents into plain text chunks
        doc_chunks = []
        for doc in documents:
            chunks = text_splitter.split_text(doc["text"])
            doc_chunks.extend(chunks)

        # Create vector store from plain text chunks
        vectorstore = FAISS.from_texts(doc_chunks, embeddings)
        print("Vectorstore created successfully.")
        return vectorstore
    except Exception as e:
        print(f"Error creating vectorstore: {e}")
        return None

# Function to create the conversation chain
def create_chain(vectorstore):
    try:
        llm = ChatGroq(
            model="llama-3.1-70b-versatile",
            temperature=0
        )
        retriever = vectorstore.as_retriever()
        memory = ConversationBufferMemory(
            llm=llm,
            output_key="answer",
            memory_key="chat_history",
            return_messages=True
        )
        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            chain_type="map_reduce",
            memory=memory,
            verbose=True
        )
        print("Conversation chain created successfully.")
        return chain
    except Exception as e:
        st.error(f"Failed to create the conversation chain: {e}")
        return None

# Function to initialize text-to-speech engine
def speak(text):
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)  # Speed of speech
    engine.setProperty('volume', 1)  # Volume level (0.0 to 1.0)
    engine.say(text)
    engine.runAndWait()

# Function to recognize speech using speech recognition
def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Listening for your question... Please speak.")
        audio = recognizer.listen(source)
    try:
        query = recognizer.recognize_google(audio)
        st.write(f"You said: {query}")
        return query
    except Exception as e:
        st.error(f"Sorry, I couldn't understand your question. Error: {e}")
        return None

# File uploader to upload PDF
uploaded_file = st.file_uploader(label="Upload your PDF file", type=["pdf"])

# Uploading the file and processing
if uploaded_file:
    file_path = os.path.join(working_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Load and process the PDF file
    documents = load_document(file_path)
    if documents:
        if "vectorstore" not in st.session_state:
            st.session_state.vectorstore = setup_vectorstore(documents)
        if st.session_state.vectorstore and "conversation_chain" not in st.session_state:
            st.session_state.conversation_chain = create_chain(st.session_state.vectorstore)
        
        if not st.session_state.conversation_chain:
            st.error("Failed to initialize the conversation chain.")
        else:
            # Display success message after PDF is processed
            st.success("PDF uploaded and processed successfully!", icon="✅")
    else:
        st.error("Failed to load document. Please check the file format.")

# Display chat history with custom message bubbles
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        class_name = "user" if message["role"] == "user" else "assistant"
        st.markdown(f'<div class="chat-message {class_name}">{message["content"]}</div>', unsafe_allow_html=True)

# Radio button to select input mode (Text or Voice)
input_mode = st.radio("Choose input mode", ("Text", "Voice"))

# User input for chat (Text)
if input_mode == "Text":
    user_input = st.chat_input("Ask Llama...")

    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        with st.chat_message("user"):
            st.markdown(user_input)

        if "conversation_chain" in st.session_state and st.session_state.conversation_chain:
            with st.chat_message("assistant"):
                try:
                    # Run the input through the conversation chain
                    response = st.session_state.conversation_chain({"question": user_input})
                    assistant_response = response.get("answer", "Sorry, I couldn't process that.")
                    st.markdown(assistant_response)
                    st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})
                    speak(assistant_response)  # Respond with human-like voice after every answer
                except Exception as e:
                    st.error(f"Error during processing: {e}")
        else:
            st.error("Conversation chain not initialized. Please check if the document and vectorstore loaded correctly.")

# Button to enable voice input
if input_mode == "Voice" and st.button("Ask via Voice"):
    query = recognize_speech()  # Listen for user speech input
    if query:
        st.session_state.chat_history.append({"role": "user", "content": query})

        with st.chat_message("user"):
            st.markdown(query)

        if "conversation_chain" in st.session_state and st.session_state.conversation_chain:
            with st.chat_message("assistant"):
                try:
                    response = st.session_state.conversation_chain({"question": query})
                    assistant_response = response.get("answer", "Sorry, I couldn't process that.")
                    st.markdown(assistant_response)
                    st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})
                    speak(assistant_response)  # Speak out the response
                except Exception as e:
                    st.error(f"Error during voice response: {e}")