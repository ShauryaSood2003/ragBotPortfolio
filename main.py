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
# from langchain_community.vectorstores import Milvus  # Not needed for Render deployment
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.schema import Document
import PyPDF2
import docx2txt
import csv
import io
from dotenv import load_dotenv
import logging

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="RAG Bot API", description="A RAG bot with Gemini AI and Smart Model Fallback")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://shauryasportfolio.netlify.app",
        "http://localhost:3000",  # For local development
        "http://127.0.0.1:3000"   # For local development
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)

class QuestionRequest(BaseModel):
    question: str
    session_id: Optional[str] = "default"
    language: Optional[str] = "english"  # "english" or "hindi"

class QuestionResponse(BaseModel):
    answer: str
    session_id: str
    model_used: Optional[str] = None
    language: Optional[str] = None

# Current (Non-Deprecated) Gemini models ordered by preference (best first)
GEMINI_MODELS = [
    "gemini-2.5-flash-lite-preview-06-17",  # Default - Best for high volume
    "gemini-2.5-flash",                     # High performance model
    "gemini-2.0-flash",                     # Fast processing
    "gemini-2.0-flash-lite",                # Lightweight option
    "gemini-2.5-pro",                       # Most capable model
    "gemini-1.5-flash",                     # Deprecated but still accessible
    "gemini-1.5-pro",                       # Deprecated fallback
]

# Global variables
GEMINI_API_KEY_PRIMARY = os.getenv("GEMINI_API_KEY_PRIMARY", "your_primary_key_here")
GEMINI_API_KEY_FALLBACK = os.getenv("GEMINI_API_KEY_FALLBACK", "your_fallback_key_here")

# API Key and model tracking
current_api_key_index = 0  # 0 = primary, 1 = fallback
current_model_index = 0
model_request_count = 0
failed_models = set()  # Track models that have failed today
failed_api_keys = set()  # Track API keys that have failed

# API Keys list for fallback
API_KEYS = [GEMINI_API_KEY_PRIMARY, GEMINI_API_KEY_FALLBACK]

def get_current_api_key():
    """Get the currently selected API key"""
    global current_api_key_index
    if current_api_key_index < len(API_KEYS):
        return API_KEYS[current_api_key_index]
    else:
        # Reset to first API key if exhausted
        current_api_key_index = 0
        return API_KEYS[0]

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

def switch_to_next_api_key():
    """Switch to the next available API key"""
    global current_api_key_index
    
    # Mark current API key as failed
    if current_api_key_index < len(API_KEYS):
        failed_api_keys.add(current_api_key_index)
    
    # Try next API key
    if current_api_key_index + 1 < len(API_KEYS):
        current_api_key_index += 1
        logger.info(f"Switched to fallback API key (index: {current_api_key_index})")
        return get_current_api_key()
    
    # All API keys exhausted
    logger.warning("All API keys exhausted")
    return None

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
    
    # If no models available, try switching API key
    logger.info("All models exhausted for current API key, trying next API key")
    next_api_key = switch_to_next_api_key()
    if next_api_key:
        # Reset model tracking for new API key
        current_model_index = 0
        failed_models.clear()
        return GEMINI_MODELS[0]
    
    # Everything exhausted
    logger.warning("All models and API keys exhausted")
    return None

def create_llm(model_name: str = None):
    """Create LLM instance with specified model and current API key"""
    if model_name is None:
        model_name = get_current_model()
    
    current_api_key = get_current_api_key()
    
    try:
        llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=0.3,
            google_api_key=current_api_key
        )
        logger.info(f"Created LLM with model: {model_name}, API key index: {current_api_key_index}")
        return llm, model_name
    except Exception as e:
        logger.error(f"Error creating LLM with model {model_name}, API key index {current_api_key_index}: {e}")
        raise

def process_question_with_fallback(question: str, session_id: str, language: str = "english", max_retries: int = 3):
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
            
            # Add language instruction to the question
            language_instruction = ""
            if language.lower() == "hindi":
                language_instruction = "कृपया हिंदी में उत्तर दें। Please respond in Hindi language: "
            else:
                language_instruction = "Please respond in English: "
            
            enhanced_question = language_instruction + question
            
            # Process the question
            response = conversation_chains[session_id]({"question": enhanced_question})
            
            # Track successful usage
            model_request_count += 1
            logger.info(f"Successfully processed question with {model_name} (total requests: {model_request_count})")
            
            return response["answer"], model_name, language
            
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

# Initialize embeddings with primary API key
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=GEMINI_API_KEY_PRIMARY
)

vector_store = None
conversation_chains = {}

def load_training_data():
    """Load and chunk the training data file"""
    try:
        with open("training_data.txt", "r", encoding="utf-8") as f:
            content = f.read()
        
        if not content.strip():
            logger.warning("Training data file is empty")
            return ["No training data available"]
        
        # Split into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        chunks = text_splitter.split_text(content)
        logger.info(f"Loaded {len(chunks)} chunks from training data")
        return chunks
        
    except FileNotFoundError:
        logger.warning("training_data.txt not found, using fallback content")
        return ["Shaurya Sood is a Senior Full Stack Developer with expertise in React, Node.js, and AI-powered applications."]
    except Exception as e:
        logger.error(f"Error loading training data: {e}")
        return ["Training data could not be loaded"]

def initialize_vector_store():
    global vector_store
    try:
        # Try FAISS first for Render deployment
        from langchain_community.vectorstores import FAISS
        
        # Load training data and create FAISS with it
        training_chunks = load_training_data()
        vector_store = FAISS.from_texts(training_chunks, embeddings)
        
        logger.info(f"Using FAISS with {len(training_chunks)} pre-loaded chunks")
        return True
    except Exception as e:
        logger.error(f"FAISS initialization failed: {e}")
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
        content = file.file.read().decode('utf-8')
        csv_reader = csv.reader(io.StringIO(content))
        rows = list(csv_reader)
        # Convert CSV to readable text format
        text = ""
        for row in rows:
            text += ", ".join(row) + "\n"
        return text
    
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
    if not initialize_vector_store():
        logger.error("Could not initialize vector store.")
    
    # Log initial status
    logger.info(f"Starting with model: {get_current_model()}")
    logger.info(f"Available models: {GEMINI_MODELS}")

@app.get("/")
async def root():
    return {"message": "RAG Bot API with Smart Model Fallback is running"}

@app.get("/health")
async def health_check():
    vector_store_status = "connected" if vector_store else "disconnected"
    
    return {
        "status": "healthy",
        "vector_store": vector_store_status,
        "api_keys_configured": {
            "primary": bool(GEMINI_API_KEY_PRIMARY and GEMINI_API_KEY_PRIMARY != "your_primary_key_here"),
            "fallback": bool(GEMINI_API_KEY_FALLBACK and GEMINI_API_KEY_FALLBACK != "your_fallback_key_here")
        },
        "current_api_key_index": current_api_key_index,
        "current_model": get_current_model(),
        "total_requests": model_request_count,
        "failed_models": list(failed_models),
        "failed_api_keys": list(failed_api_keys),
        "available_models": [m for m in GEMINI_MODELS if m not in failed_models]
    }

@app.get("/model-status")
async def model_status():
    """Get detailed model and API key status"""
    current = get_current_model()
    
    return {
        "current_model": current,
        "current_model_index": current_model_index,
        "current_api_key_index": current_api_key_index,
        "total_requests_today": model_request_count,
        "available_models": GEMINI_MODELS,
        "failed_models": list(failed_models),
        "failed_api_keys": list(failed_api_keys),
        "working_models": [m for m in GEMINI_MODELS if m not in failed_models],
        "api_keys_status": {
            "primary_configured": bool(GEMINI_API_KEY_PRIMARY and GEMINI_API_KEY_PRIMARY != "your_primary_key_here"),
            "fallback_configured": bool(GEMINI_API_KEY_FALLBACK and GEMINI_API_KEY_FALLBACK != "your_fallback_key_here"),
            "total_capacity": f"{len(GEMINI_MODELS)} models × {len(API_KEYS)} API keys = {len(GEMINI_MODELS) * len(API_KEYS)} total configurations"
        }
    }

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    if not vector_store:
        raise HTTPException(status_code=503, detail="Vector store not initialized.")
    
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
        raise HTTPException(status_code=503, detail="Vector store not initialized.")
    
    try:
        answer, model_used, language = process_question_with_fallback(
            request.question, 
            request.session_id, 
            request.language
        )
        
        return QuestionResponse(
            answer=answer,
            session_id=request.session_id,
            model_used=model_used,
            language=language
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