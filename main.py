import os
import io
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Milvus
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.schema import Document
import PyPDF2
import docx2txt
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="RAG Bot API", description="A RAG bot with Gemini AI and Milvus")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QuestionRequest(BaseModel):
    question: str
    session_id: Optional[str] = "default"

class QuestionResponse(BaseModel):
    answer: str
    session_id: str

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "your_gemini_api_key_here")
MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = int(os.getenv("MILVUS_PORT", "19530"))

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=GEMINI_API_KEY
)

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.3,
    google_api_key=GEMINI_API_KEY
)

vector_store = None
conversation_chains = {}

def initialize_milvus():
    global vector_store
    try:
        from langchain_community.vectorstores import FAISS
        vector_store = FAISS.from_texts(
            ["Initial empty document"], 
            embeddings
        )
        print("Using FAISS as fallback vector store")
        return True
    except Exception as e:
        try:
            vector_store = Milvus(
                embedding_function=embeddings,
                connection_args={"host": MILVUS_HOST, "port": MILVUS_PORT},
                collection_name="rag_documents"
            )
            print("Connected to Milvus")
            return True
        except Exception as milvus_error:
            print(f"Failed to connect to Milvus: {milvus_error}")
            print(f"FAISS fallback also failed: {e}")
            return False

def extract_text_from_file(file: UploadFile) -> str:
    file_extension = file.filename.split('.')[-1].lower()
    
    if file_extension == 'pdf':
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(file.file.read()))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    
    elif file_extension in ['docx', 'doc']:
        return docx2txt.process(io.BytesIO(file.file.read()))
    
    elif file_extension == 'txt':
        return file.file.read().decode('utf-8')
    
    elif file_extension in ['csv']:
        df = pd.read_csv(io.BytesIO(file.file.read()))
        return df.to_string()
    
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {file_extension}")

def chunk_text(text: str) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunks = text_splitter.split_text(text)
    documents = [Document(page_content=chunk) for chunk in chunks]
    
    return documents

def get_conversation_chain(session_id: str):
    if session_id not in conversation_chains:
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        
        conversation_chains[session_id] = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
            memory=memory,
            return_source_documents=True
        )
    
    return conversation_chains[session_id]

@app.on_event("startup")
async def startup_event():
    if not initialize_milvus():
        print("Warning: Could not connect to Milvus. Make sure Docker Compose is running.")

@app.get("/")
async def root():
    return {"message": "RAG Bot API is running"}

@app.get("/health")
async def health_check():
    milvus_status = "connected" if vector_store else "disconnected"
    return {
        "status": "healthy",
        "milvus": milvus_status,
        "gemini_api_configured": bool(GEMINI_API_KEY and GEMINI_API_KEY != "your_gemini_api_key_here")
    }

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    if not vector_store:
        raise HTTPException(status_code=503, detail="Vector store not initialized. Check Milvus connection.")
    
    try:
        text = extract_text_from_file(file)
        
        if not text.strip():
            raise HTTPException(status_code=400, detail="No text could be extracted from the file")
        
        documents = chunk_text(text)
        
        vector_store.add_documents(documents)
        
        return {
            "message": f"Successfully processed and stored {len(documents)} chunks from {file.filename}",
            "chunks_count": len(documents),
            "filename": file.filename
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

@app.post("/question", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    if not vector_store:
        raise HTTPException(status_code=503, detail="Vector store not initialized. Check Milvus connection.")
    
    try:
        conversation_chain = get_conversation_chain(request.session_id)
        
        response = conversation_chain({"question": request.question})
        
        return QuestionResponse(
            answer=response["answer"],
            session_id=request.session_id
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")

@app.delete("/session/{session_id}")
async def clear_session(session_id: str):
    if session_id in conversation_chains:
        del conversation_chains[session_id]
        return {"message": f"Session {session_id} cleared"}
    else:
        raise HTTPException(status_code=404, detail="Session not found")

@app.get("/sessions")
async def list_sessions():
    return {"active_sessions": list(conversation_chains.keys())}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)