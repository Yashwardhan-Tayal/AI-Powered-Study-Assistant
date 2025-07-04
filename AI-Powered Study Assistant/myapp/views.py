import os
from dotenv import load_dotenv
import fitz
from langchain_huggingface import HuggingFaceEmbeddings
import faiss
import numpy as np
from langchain_mistralai import ChatMistralAI
from langchain.prompts import ChatPromptTemplate
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
load_dotenv()

embedModel = HuggingFaceEmbeddings(
    model_name='sentence-transformers/all-MiniLM-L6-v2',
    model_kwargs={'device':'cpu'},
    encode_kwargs={'normalize_embeddings':True})


def extractTextFromPDF(file_path):
    try:
        doc = fitz.open(file_path)
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    except Exception as e:
        return ""

def chunkText(text, chunk_size=500):
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i+chunk_size])
    return chunks

@csrf_exempt
def home(request):
    answer = ""
    if request.method == "POST":
        uploadedFile = request.FILES.get("file")
        question = request.POST.get("question")

        if not uploadedFile or not question:
            return render(request, "home.html", {"answer": "Please upload PDF and enter a question."})

        # Saves PDF temporarily
        tempPath = f"temp_{uploadedFile.name}"
        with open(tempPath, "wb") as f:
            for chunk in uploadedFile.chunks():
                f.write(chunk)

        # Extracts and chunks text
        text = extractTextFromPDF(tempPath)
        chunks = chunkText(text)
        
        # Creates embeddings for chunks
        embeddings = embedModel.embed_documents(chunks)
        embeddings=np.array(embeddings).astype('float32')

        # FAISS
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)

        # Embed question
        qEmbedding = embedModel.embed_query(question)
        qEmbedding=np.array([qEmbedding]).astype('float32')

        # Search top 5 relevant chunks
        k = 5
        distances, indices = index.search(qEmbedding, k)

        context = "\n\n".join(chunks[i] for i in indices[0])

        # Prompt
        prompt = ChatPromptTemplate.from_messages([
            ('system',f'''You are a study companion bot. Help the student out in whatever they need. Keep your answers short and to the point. Your only goal is to help them. Format the text to make it easily readable.
            Study Material:{context}'''),
            ('human',f'{question}')
        ])

        # Query mistral
        try:
            llm=ChatMistralAI()
            message=prompt.format_messages()
            response=llm.invoke(message)
            answer=response.content


        except Exception as e:
            answer = f"Error from Mistral API: {str(e)}"

        # Cleanup
        os.remove(tempPath)

    return render(request, "home.html", {"answer": answer})


