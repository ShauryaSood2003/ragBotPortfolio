import os
import io
import time
import random
from typing import List, Optional, Dict, Any
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
import logging

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="RAG Bot API", description="A RAG bot with Gemini AI and Smart Model Fallback")

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
    model_used: Optional[str] = None

# Gemini model configurations ordered by preference (best first)
GEMINI_MODELS = [
    "gemini-2.5-flash-lite",  # 1000 RPD - Best option
    "gemini-2.5-flash",       # 250 RPD
    "gemini-2.0-flash",       # 200 RPD  
    "gemini-2.0-flash-lite",  # 200 RPD
    "gemini-2.5-pro",        # 100 RPD
    "gemini-1.5-flash"       # 50 RPD - Last resort
]

# Global variables
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "your_gemini_api_key_here")
MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = int(os.getenv("MILVUS_PORT", "19530"))

# Current model tracking
current_model_index = 0
model_request_count = 0
failed_models = set()  # Track models that have failed today

def get_current_model():
    """Get the currently selected model"""
    global current_model_index
    if current_model_index < len(GEMINI_MODELS):
        return GEMINI_MODELS[current_model_index]
    else:
        # Reset to first model if we've exhausted all
        current_model_index = 0
        failed_models.clear()
        return GEMINI_MODELS[0]

def switch_to_next_model():
    """Switch to the next available model"""
    global current_model_index
    
    # Mark current model as failed
    if current_model_index < len(GEMINI_MODELS):
        failed_models.add(GEMINI_MODELS[current_model_index])
    
    # Find next available model
    for i in range(current_model_index + 1, len(GEMINI_MODELS)):
        if GEMINI_MODELS[i] not in failed_models:
            current_model_index = i
            logger.info(f"Switched to model: {GEMINI_MODELS[i]}")
            return GEMINI_MODELS[i]
    
    # If no models available, reset and try first model
    logger.warning("All models exhausted, resetting to first model")
    current_model_index = 0
    failed_models.clear()
    return GEMINI_MODELS[0]

def create_llm(model_name: str = None):
    """Create LLM instance with specified model"""
    if model_name is None:
        model_name = get_current_model()
    
    try:
        llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=0.3,
            google_api_key=GEMINI_API_KEY
        )
        logger.info(f"Created LLM with model: {model_name}")
        return llm, model_name
    except Exception as e:
        logger.error(f"Error creating LLM with model {model_name}: {e}")
        raise

def process_question_with_fallback(question: str, session_id: str, max_retries: int = 3):
    """Process question with automatic model fallback on rate limits"""
    global model_request_count
    
    for attempt in range(max_retries):
        current_model = get_current_model()
        
        try:
            # Create LLM with current model
            llm, model_name = create_llm(current_model)
            
            # Create conversation chain
            if session_id not in conversation_chains or attempt > 0:
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
            
            # Process the question
            response = conversation_chains[session_id]({"question": question})
            
            # Track successful usage
            model_request_count += 1
            logger.info(f"Successfully processed question with {model_name} (total requests: {model_request_count})")
            
            return response["answer"], model_name
            
        except Exception as e:
            error_message = str(e)
            logger.error(f"Attempt {attempt + 1} failed with model {current_model}: {error_message}")
            
            # Check if it's a rate limit error
            if any(keyword in error_message.lower() for keyword in ["429", "quota", "rate limit", "exceeded"]):
                logger.info(f"Rate limit detected for {current_model}, switching models...")
                
                # Switch to next model
                next_model = switch_to_next_model()
                if next_model == current_model:
                    # All models exhausted
                    logger.error("All models exhausted")
                    break
                
                # Clear conversation chain to use new model
                if session_id in conversation_chains:
                    del conversation_chains[session_id]
                
                # Wait before retrying
                time.sleep(random.uniform(1, 3))
                continue
            else:
                # Non-rate-limit error, don't retry
                logger.error(f"Non-rate-limit error: {error_message}")
                break
    
    # If all retries failed
    raise HTTPException(
        status_code=503, 
        detail=f"Service temporarily unavailable. Error: {error_message}"
    )

# Initialize embeddings
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
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
        logger.info("Using FAISS as fallback vector store")
        return True
    except Exception as e:
        try:
            vector_store = Milvus(
                embedding_function=embeddings,
                connection_args={"host": MILVUS_HOST, "port": MILVUS_PORT},
                collection_name="rag_documents"
            )
            logger.info("Connected to Milvus")
            return True
        except Exception as milvus_error:
            logger.error(f"Failed to connect to Milvus: {milvus_error}")
            logger.error(f"FAISS fallback also failed: {e}")
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

@app.on_event("startup")
async def startup_event():
    if not initialize_milvus():
        logger.warning("Could not connect to Milvus. Make sure Docker Compose is running.")
    
    # Log initial status
    logger.info(f"Starting with model: {get_current_model()}")
    logger.info(f"Available models: {GEMINI_MODELS}")

@app.get("/")
async def root():
    return {"message": "RAG Bot API with Smart Model Fallback is running"}

@app.get("/health")
async def health_check():
    milvus_status = "connected" if vector_store else "disconnected"
    
    return {
        "status": "healthy",
        "milvus": milvus_status,
        "gemini_api_configured": bool(GEMINI_API_KEY and GEMINI_API_KEY != "your_gemini_api_key_here"),
        "current_model": get_current_model(),
        "total_requests": model_request_count,
        "failed_models": list(failed_models),
        "available_models": [m for m in GEMINI_MODELS if m not in failed_models]
    }

@app.get("/model-status")
async def model_status():
    """Get detailed model status"""
    current = get_current_model()
    
    return {
        "current_model": current,
        "current_model_index": current_model_index,
        "total_requests_today": model_request_count,
        "available_models": GEMINI_MODELS,
        "failed_models": list(failed_models),
        "working_models": [m for m in GEMINI_MODELS if m not in failed_models]
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
        answer, model_used = process_question_with_fallback(request.question, request.session_id)
        
        return QuestionResponse(
            answer=answer,
            session_id=request.session_id,
            model_used=model_used
        )
    
    except HTTPException:
        raise
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