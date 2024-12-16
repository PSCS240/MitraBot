import os
import re
import pdfplumber
import pyttsx3
import whisper  # For Speech-to-Text
import speech_recognition as sr  # For audio recording
from dotenv import load_dotenv
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters.character import CharacterTextSplitter
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from transformers import pipeline  # NLP pipeline for translation
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException
import io

# Load environment variables
load_dotenv()

# Set up translation pipeline for multilingual support
translator = pipeline("translation", model="Helsinki-NLP/opus-mt-mul-en")

# Streamlit UI setup - this should be the first Streamlit function
st.set_page_config(
    page_title="Welcome to MitraBot",
    page_icon="📄",
    layout="centered"
)

# Custom CSS for styling the app
st.markdown("""<style>...</style>""", unsafe_allow_html=True)  # Your existing CSS here

# Title and Branding
st.markdown('<h1 class="title">🦙 Chat with MitraBot powered by LLAMA 3.1</h1>', unsafe_allow_html=True)

# Initialize chat history in Streamlit session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "last_detected_language" not in st.session_state:
    st.session_state.last_detected_language = "en"

# Clear chat history button
if st.button("Clear Chat"):
    st.session_state.chat_history = []

# Function to detect language
def detect_language(text):
    try:
        return detect(text)
    except LangDetectException:
        st.error("Could not detect language. Defaulting to English.")
        return "en"

# Function to translate text
def translate_text(text, target_lang):
    try:
        result = translator(text, target_lang=target_lang)
        return result[0]["translation_text"]
    except Exception as e:
        st.error(f"Translation error: {e}")
        return text

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

# Function to set up the vector store
def setup_vectorstore(documents):
    try:
        embeddings = HuggingFaceEmbeddings()
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200
        )
        doc_chunks = []
        for doc in documents:
            chunks = text_splitter.split_text(doc["text"])
            doc_chunks.extend(chunks)
        vectorstore = FAISS.from_texts(doc_chunks, embeddings)
        return vectorstore
    except Exception as e:
        st.error(f"Error creating vectorstore: {e}")
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
        return chain
    except Exception as e:
        st.error(f"Failed to create the conversation chain: {e}")
        return None

# Function to initialize text-to-speech engine
def speak(text):
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    engine.setProperty('volume', 1)
    engine.say(text)
    engine.runAndWait()

# Function to recognize speech using Whisper
def recognize_speech():
    st.info("Listening for your question... Please speak.")
    model = whisper.load_model("base")  # Whisper model size: "tiny", "base", "small", "medium", or "large"
    audio_path = "user_audio.wav"

    # Recording audio
    try:
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            st.info("Recording audio... Please speak now.")
            audio = recognizer.listen(source)
            with open(audio_path, "wb") as f:
                f.write(audio.get_wav_data())

        # Transcribing audio with Whisper
        result = model.transcribe(audio_path)
        query = result["text"]
        st.write(f"You said: {query}")
        return query
    except Exception as e:
        st.error(f"Error during voice recognition: {e}")
        return None

# Function to generate dual responses
def generate_dual_response(assistant_response_english, output_language):
    try:
        assistant_response_translated = (
            translate_text(assistant_response_english, target_lang=output_language)
            if output_language != "en"
            else assistant_response_english
        )
        # Text-to-speech for translated response
        speak(assistant_response_translated)
        return assistant_response_translated
    except Exception as e:
        st.error(f"Error during dual response generation: {e}")
        return assistant_response_english

# File uploader to upload PDF
uploaded_file = st.file_uploader(label="Upload your PDF file", type=["pdf"])

if uploaded_file:
    file_path = os.path.join(os.getcwd(), uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    documents = load_document(file_path)
    if documents:
        if "vectorstore" not in st.session_state:
            st.session_state.vectorstore = setup_vectorstore(documents)
        if st.session_state.vectorstore and "conversation_chain" not in st.session_state:
            st.session_state.conversation_chain = create_chain(st.session_state.vectorstore)
        if not st.session_state.conversation_chain:
            st.error("Failed to initialize the conversation chain.")
        else:
            st.success("PDF uploaded and processed successfully!", icon="✅")
    else:
        st.error("Failed to load document. Please check the file format.")

# Display chat history with custom message bubbles
for message in st.session_state.chat_history:
    class_name = "user" if message["role"] == "user" else "assistant"
    st.markdown(f'<div class="chat-message {class_name}">{message["content"]}</div>', unsafe_allow_html=True)

# Radio button to select input mode (Text or Voice)
input_mode = st.radio("Choose input mode", ("Text", "Voice"))

# Text input for chat
if input_mode == "Text":
    user_input = st.chat_input("Ask MitraBot...")
    if user_input:
        input_language = detect_language(user_input)
        user_query_english = translate_text(user_input, target_lang="en") if input_language != "en" else user_input
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        if "conversation_chain" in st.session_state and st.session_state.conversation_chain:
            try:
                response = st.session_state.conversation_chain({"question": user_query_english})
                assistant_response_english = response.get("answer", "Sorry, I couldn't process that.")
                assistant_response_translated = generate_dual_response(assistant_response_english, input_language)
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": assistant_response_translated}
                )
                st.markdown(f"**English**: {assistant_response_english}")
                st.markdown(f"**{input_language.upper()}**: {assistant_response_translated}")
            except Exception as e:
                st.error(f"Error during processing: {e}")

# Voice input for chat
if input_mode == "Voice" and st.button("Ask via Voice"):
    query = recognize_speech()
    if query:
        input_language = detect_language(query)
        user_query_english = translate_text(query, target_lang="en") if input_language != "en" else query
        st.session_state.chat_history.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)
        if "conversation_chain" in st.session_state and st.session_state.conversation_chain:
            try:
                response = st.session_state.conversation_chain({"question": user_query_english})
                assistant_response_english = response.get("answer", "Sorry, I couldn't process that.")
                assistant_response_translated = generate_dual_response(assistant_response_english, input_language)
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": assistant_response_translated}
                )
                st.markdown(f"**English**: {assistant_response_english}")
                st.markdown(f"**{input_language.upper()}**: {assistant_response_translated}")
            except Exception as e:
                st.error(f"Error during processing: {e}")
