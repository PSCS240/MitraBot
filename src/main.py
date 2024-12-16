import os
import pdfplumber
from dotenv import load_dotenv
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters.character import CharacterTextSplitter
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# Load environment variables
load_dotenv()

working_dir = os.path.dirname(os.path.abspath(__file__))

def load_document(file_path):
    try:
        documents = []
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    documents.append({"text": text})
        print("PDF document loaded and processed successfully.")
        return documents
    except Exception as e:
        print(f"Failed to load PDF document: {e}")
        return None

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

st.set_page_config(
    page_title="Welcome to MitraBot",
    page_icon="📄",
    layout="centered"
)

st.title("🦙 Chat with MitraBot powered by - LLAMA 3.1")

# Initialize chat history in Streamlit session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# File uploader to upload PDF
uploaded_file = st.file_uploader(label="Upload your PDF file", type=["pdf"])

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
        st.error("Failed to load document. Please check the file format.")

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input for chat
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
            except Exception as e:
                st.error(f"Error during processing: {e}")
    else:
        st.error("Conversation chain not initialized. Please check if the document and vectorstore loaded correctly.")
