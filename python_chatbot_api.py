# File: chatbot_api.py

import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import UnstructuredMarkdownLoader

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# loader = UnstructuredMarkdownLoader("main.md", mode="single")
# docs = loader.load()

# Embedding model and FAISS index path
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
DB_PATH = "faiss_index"
TEXT_FILE = "main2.md"

# Load and split content into chunks
def load_and_split_text(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return text_splitter.split_text(text)

# Initialize or load FAISS vector store
def get_or_create_faiss(chunks):
    if os.path.exists(DB_PATH):
        print("Loading existing FAISS index...")
        db = FAISS.load_local(DB_PATH, embedding_model, allow_dangerous_deserialization=True)
    else:
        print("Creating new FAISS index...")
        db = FAISS.from_texts(chunks, embedding_model)
        db.save_local(DB_PATH)
    return db

# Gemini response function
def gemini_resp(context, query):
    prompt = f"""
You are Saanvi, an intelligent and friendly AI assistant designed to help parents learn more about Sanmati H.S. School.

Your task is to understand the parent’s question and decide:

1. If the question is general in nature (like parenting tips, learning styles, etc.), answer directly without using the school's context.
2. If the question is specifically about Sanmati H.S. School (like admission process, facilities, timings, mission, curriculum, etc.), use the provided school information (context) to generate a detailed and helpful answer.

Always be polite, informative, and supportive in your tone. Use clear and complete sentences. If the answer is about the school, ensure it is based on the provided school information.

---

Context (use this only if the question is about the school):
{context}

---

Parent's Question:
{query}

---

Saanvi's Response:
"""
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=GOOGLE_API_KEY)
    response = llm.invoke(prompt)
    return response.content

# FAISS search + Gemini answer
chunks = load_and_split_text(TEXT_FILE)
db = get_or_create_faiss(chunks)

def query_faiss(query):
    docs = db.similarity_search(query, k=3)
    context = "\n\n".join([doc.page_content for doc in docs])
    return gemini_resp(context, query)

# --- Flask API setup ---
app = Flask(__name__)
CORS(app)

@app.route("/", methods=["GET"])
def index():
    return jsonify({"message": "Saanvi Chatbot is running!"}), 200


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_query = data.get("message")
    
    if not user_query:
        return jsonify({"error": "No message provided"}), 400

    try:
        response = query_faiss(user_query)
        return jsonify({"reply": response})
    except Exception as e:
        print(f"Error processing query: {e}")
        return jsonify({"error": "Internal server error", "details": str(e)}), 500

# Optional reset (if needed for future caching)
@app.route("/reset", methods=["POST"])
def reset():
    if os.path.exists(DB_PATH):
        try:
            os.remove(DB_PATH)
            return jsonify({"message": "FAISS index reset successfully."})
        except Exception as e:
            return jsonify({"error": "Failed to delete FAISS index", "details": str(e)}), 500
    return jsonify({"message": "No FAISS index to delete."})

from twilio.twiml.messaging_response import MessagingResponse

@app.route("/whatsapp", methods=["POST"])
def whatsapp_webhook():
    incoming_msg = request.form.get("Body")
    phone_number = request.form.get("From")

    if not incoming_msg:
        return "No message", 400

    try:
        bot_response = query_faiss(incoming_msg)

        resp = MessagingResponse()
        msg = resp.message()
        msg.body(bot_response)
        return str(resp)

    except Exception as e:
        print(f"Error: {e}")
        resp = MessagingResponse()
        resp.message("Something went wrong. Please try again later.")
        return str(resp)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
