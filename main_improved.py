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

# Gemini model configurations with daily limits
GEMINI_MODELS = [
    {
        "name": "gemini-2.5-flash-lite",
        "daily_limit": 1000,
        "rpm": 15,
        "priority": 1  # Highest priority (most requests per day)
    },
    {
        "name": "gemini-2.5-flash", 
        "daily_limit": 250,
        "rpm": 10,
        "priority": 2
    },
    {
        "name": "gemini-2.0-flash",
        "daily_limit": 200,
        "rpm": 15,
        "priority": 3
    },
    {
        "name": "gemini-2.0-flash-lite",
        "daily_limit": 200,
        "rpm": 30,
        "priority": 4
    },
    {
        "name": "gemini-2.5-pro",
        "daily_limit": 100,
        "rpm": 5,
        "priority": 5
    },
    {
        "name": "gemini-1.5-flash",
        "daily_limit": 50,
        "rpm": 15,
        "priority": 6  # Lowest priority (deprecated, fallback only)
    }
]

# Global variables
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "your_gemini_api_key_here")
MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = int(os.getenv("MILVUS_PORT", "19530"))

# Model usage tracking
model_usage = {}
current_model_index = 0
last_reset_date = None

def reset_daily_usage():
    """Reset usage counters daily"""
    global model_usage, last_reset_date
    current_date = time.strftime("%Y-%m-%d")
    
    if last_reset_date != current_date:
        model_usage = {}
        last_reset_date = current_date
        logger.info(f"Daily usage counters reset for {current_date}")

def get_available_model():
    """Get the best available model that hasn't hit its daily limit"""
    reset_daily_usage()
    
    # Sort models by priority (lowest number = highest priority)
    sorted_models = sorted(GEMINI_MODELS, key=lambda x: x["priority"])
    
    for model in sorted_models:
        model_name = model["name"]
        daily_used = model_usage.get(model_name, 0)
        
        if daily_used < model["daily_limit"]:
            logger.info(f"Selected model: {model_name} (used: {daily_used}/{model['daily_limit']})")
            return model
    
    # If all models are exhausted, use the one with highest limit as fallback
    fallback_model = max(GEMINI_MODELS, key=lambda x: x["daily_limit"])
    logger.warning(f"All models exhausted, using fallback: {fallback_model['name']}")
    return fallback_model

def create_llm_with_fallback():
    """Create LLM instance with automatic model selection"""
    model_config = get_available_model()
    
    try:
        llm = ChatGoogleGenerativeAI(
            model=model_config["name"],
            temperature=0.3,
            google_api_key=GEMINI_API_KEY
        )
        return llm, model_config["name"]
    except Exception as e:
        logger.error(f"Error creating LLM with model {model_config['name']}: {e}")
        raise

def increment_model_usage(model_name: str):
    """Track model usage"""
    model_usage[model_name] = model_usage.get(model_name, 0) + 1
    logger.info(f"Model {model_name} usage: {model_usage[model_name]}")

def handle_rate_limit_error(error_message: str, current_model: str):
    """Handle rate limit errors by switching models"""
    logger.warning(f"Rate limit hit for model {current_model}: {error_message}")
    
    # Mark current model as exhausted for today
    for model in GEMINI_MODELS:
        if model["name"] == current_model:
            model_usage[current_model] = model["daily_limit"]
            break
    
    # Try to get a new model
    try:
        new_model_config = get_available_model()
        if new_model_config["name"] != current_model:
            logger.info(f"Switching from {current_model} to {new_model_config['name']}")
            return create_llm_with_fallback()
        else:
            logger.error("No alternative models available")
            return None, None
    except Exception as e:
        logger.error(f"Error switching models: {e}")
        return None, None

# Initialize embeddings (these usually have higher limits)
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=GEMINI_API_KEY
)

# Initialize with best available model
llm, current_model_name = create_llm_with_fallback()

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

def get_conversation_chain_with_retry(session_id: str, max_retries: int = 3):
    """Get conversation chain with automatic model switching on rate limits"""
    global llm, current_model_name
    
    if session_id not in conversation_chains:
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        
        # Create conversation chain with current LLM
        conversation_chains[session_id] = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
            memory=memory,
            return_source_documents=True
        )
    
    return conversation_chains[session_id]

def process_question_with_fallback(question: str, session_id: str, max_retries: int = 3):
    """Process question with automatic model fallback on rate limits"""
    global llm, current_model_name
    
    for attempt in range(max_retries):
        try:
            # Get conversation chain
            conversation_chain = get_conversation_chain_with_retry(session_id)
            
            # Process the question
            response = conversation_chain({"question": question})
            
            # Track successful usage
            increment_model_usage(current_model_name)
            
            return response["answer"], current_model_name
            
        except Exception as e:
            error_message = str(e)
            logger.error(f"Attempt {attempt + 1} failed with model {current_model_name}: {error_message}")
            
            # Check if it's a rate limit error
            if "429" in error_message or "quota" in error_message.lower() or "rate" in error_message.lower():
                logger.info(f"Rate limit detected, attempting to switch models...")
                
                # Try to switch to a different model
                new_llm, new_model_name = handle_rate_limit_error(error_message, current_model_name)
                
                if new_llm and new_model_name:
                    llm = new_llm
                    current_model_name = new_model_name
                    
                    # Clear conversation chains to use new model
                    if session_id in conversation_chains:
                        del conversation_chains[session_id]
                    
                    # Wait a bit before retrying
                    time.sleep(random.uniform(1, 3))
                    continue
                else:
                    logger.error("No alternative models available")
                    break
            else:
                # Non-rate-limit error, don't retry
                logger.error(f"Non-rate-limit error: {error_message}")
                break
    
    # If all retries failed
    raise HTTPException(
        status_code=503, 
        detail=f"Service temporarily unavailable. All models exhausted or error persists. Last error with {current_model_name}: {error_message}"
    )

@app.on_event("startup")
async def startup_event():
    if not initialize_milvus():
        logger.warning("Could not connect to Milvus. Make sure Docker Compose is running.")
    
    # Log initial model status
    logger.info(f"Starting with model: {current_model_name}")
    logger.info(f"Available models: {[m['name'] for m in GEMINI_MODELS]}")

@app.get("/")
async def root():
    return {"message": "RAG Bot API with Smart Model Fallback is running"}

@app.get("/health")
async def health_check():
    milvus_status = "connected" if vector_store else "disconnected"
    
    # Get current model status
    model_status = {}
    for model in GEMINI_MODELS:
        model_name = model["name"]
        used = model_usage.get(model_name, 0)
        model_status[model_name] = {
            "used": used,
            "limit": model["daily_limit"],
            "available": used < model["daily_limit"]
        }
    
    return {
        "status": "healthy",
        "milvus": milvus_status,
        "gemini_api_configured": bool(GEMINI_API_KEY and GEMINI_API_KEY != "your_gemini_api_key_here"),
        "current_model": current_model_name,
        "model_usage": model_status,
        "total_daily_capacity": sum(m["daily_limit"] for m in GEMINI_MODELS)
    }

@app.get("/model-status")
async def model_status():
    """Get detailed model usage information"""
    reset_daily_usage()  # Ensure we have current data
    
    model_info = []
    for model in sorted(GEMINI_MODELS, key=lambda x: x["priority"]):
        used = model_usage.get(model["name"], 0)
        model_info.append({
            "name": model["name"],
            "priority": model["priority"],
            "used_today": used,
            "daily_limit": model["daily_limit"],
            "remaining": model["daily_limit"] - used,
            "available": used < model["daily_limit"],
            "rpm": model["rpm"]
        })
    
    return {
        "current_model": current_model_name,
        "models": model_info,
        "total_requests_today": sum(model_usage.values()),
        "total_daily_capacity": sum(m["daily_limit"] for m in GEMINI_MODELS),
        "reset_date": last_reset_date
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