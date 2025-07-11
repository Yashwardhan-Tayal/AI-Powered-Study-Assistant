import os
from dotenv import load_dotenv
import fitz
import faiss
import numpy as np
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_mistralai import ChatMistralAI
from langchain.prompts import ChatPromptTemplate
load_dotenv()
embedModel = HuggingFaceEmbeddings(
    model_name='sentence-transformers/all-MiniLM-L6-v2',
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True})

def TextFromPDF(uploaded_file):
    file = uploaded_file.read()
    doc = fitz.open(stream=file, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text
    
def chunkText(text, chunk_size=450):
    return [text[i:i + chunk_size+50] for i in range(0, len(text), chunk_size)]

@csrf_exempt
def home(request):
    answer = ""
    if request.method == "POST" and "file" in request.FILES:
        uploaded_file = request.FILES["file"]
        text = TextFromPDF(uploaded_file)
        chunks = chunkText(text)
        embeddings = embedModel.embed_documents(chunks)
        request.session["chunks"] = chunks
        request.session["index_data"] = embeddings
        request.session["chat_history"] = []
        answer = "File uploaded. Now ask your question below."

    elif request.method == "POST" and "question" in request.POST:
        question = request.POST["question"]
        chunks = request.session.get("chunks",[])
        index_data = request.session.get("index_data",[])
        if not chunks or not index_data:
            answer = "Please upload a PDF file first."
        else:
            chat_history = request.session.get("chat_history",[])
            embeddings = np.array(index_data).astype('float32')
            dimension = embeddings.shape[1]
            index = faiss.IndexFlatL2(dimension)
            index.add(embeddings)
            q_embedding = embedModel.embed_query(question)
            q_embedding = np.array([q_embedding]).astype('float32')
            k = 5
            distances, indices = index.search(q_embedding, k)
            context = "\n".join(chunks[i] for i in indices[0])
            history_context = "\n".join(
                f"User: {msg}" if role == "user" else f"Assistant: {msg}"
                for role, msg in chat_history[-10:])
            prompt = ChatPromptTemplate.from_messages([
                ("system", f"You are a helpful study assistant chatbot. Use the following material to help the student based on their query:\n\Study Material:{context}\nChat History for Context:{history_context}\n\n"),
                ("human", f"{question}")])
            try:
                llm = ChatMistralAI()
                message=prompt.format_messages()
                response = llm.invoke(message)
                answer = response.content
                chat_history.append(("user", question))
                chat_history.append(("assistant", answer))
                request.session["chat_history"] = chat_history
            except Exception as e:
                answer = f"Error: {str(e)}"

    return render(request, "home.html", {
        "answer": answer,
        "chat_history": request.session.get("chat_history", [])})

