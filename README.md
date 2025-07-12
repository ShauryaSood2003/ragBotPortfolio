# RAG Bot with Gemini AI - Complete Setup Guide

A Retrieval-Augmented Generation (RAG) bot that lets you upload documents and have intelligent conversations about their content using Google's Gemini AI.

## 🚀 Quick Start Guide

### Prerequisites
- Python 3.8+ installed on your system
- Internet connection for API calls
- Optional: Docker for Milvus database (system works with FAISS fallback)

---

## 📋 Step-by-Step Setup Instructions

### Step 1: Navigate to Project Directory
```bash
cd /home/shaurya/Desktop/portfolio/ragBotMe
```

### Step 2: Create and Activate Python Virtual Environment

**Create virtual environment:**
```bash
python3 -m venv venv
```

**Activate virtual environment:**
```bash
source venv/bin/activate
```

**You'll know it's working when you see `(venv)` at the start of your terminal prompt**

**To deactivate later (when you're done):**
```bash
deactivate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Configure Gemini API Key
The API key to be configured in `.env` file:
```
GEMINI_API_KEY=your_api_key
```

### Step 5: Start the RAG Bot Server
```bash
python3 main.py
```

**You should see:**
```
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

**Keep this terminal open - this is your server running!**

---

## 📚 How to Use Your RAG Bot

### Open a New Terminal Window
*Keep the server running in the first terminal, open a second terminal for commands*

### Step 1: Check if Everything is Working
```bash
curl http://localhost:8000/health
```
**Expected response:**
```json
{"status":"healthy","milvus":"connected","gemini_api_configured":true}
```

### Step 2: Upload Your Documents (Training Phase)

**For a text file:**
```bash
curl -X POST "http://localhost:8000/upload" -F "file=@your_document.txt"
```

**For a PDF file:**
```bash
curl -X POST "http://localhost:8000/upload" -F "file=@your_document.pdf"
```

**For a Word document:**
```bash
curl -X POST "http://localhost:8000/upload" -F "file=@your_document.docx"
```

**Expected response:**
```json
{
  "message": "Successfully processed and stored 5 chunks from your_document.pdf",
  "chunks_count": 5,
  "filename": "your_document.pdf"
}
```

### Step 3: Start Asking Questions (Retrieval Phase)

**Ask your first question (English):**
```bash
curl -X POST "http://localhost:8000/question" \
     -H "Content-Type: application/json" \
     -d '{"question": "What is this document about?", "language": "english"}'
```

**Ask questions in Hindi:**
```bash
curl -X POST "http://localhost:8000/question" \
     -H "Content-Type: application/json" \
     -d '{"question": "शौर्य का काम क्या है?", "language": "hindi"}'
```

**Ask follow-up questions in the same conversation:**
```bash
curl -X POST "http://localhost:8000/question" \
     -H "Content-Type: application/json" \
     -d '{"question": "Can you give me more details about the main topics?", "session_id": "my_session", "language": "english"}'
```

**Continue the conversation in Hindi:**
```bash
curl -X POST "http://localhost:8000/question" \
     -H "Content-Type: application/json" \
     -d '{"question": "और कौन से प्रोजेक्ट्स पर काम किया है?", "session_id": "my_session", "language": "hindi"}'
```

### Step 4: Manage Your Conversations

**Check active sessions:**
```bash
curl http://localhost:8000/sessions
```

**Clear a specific conversation:**
```bash
curl -X DELETE "http://localhost:8000/session/my_session"
```

---

## 🔄 Complete Workflow Example

### 1. Start Everything
```bash
# Terminal 1 - Start server
cd /home/shaurya/Desktop/portfolio/ragBotMe
source venv/bin/activate
python3 main.py
```

### 2. Upload Training Data (New Terminal)
```bash
# Terminal 2 - Upload documents
curl -X POST "http://localhost:8000/upload" -F "file=@sample_doc.txt"
```

### 3. Have Intelligent Conversations (English & Hindi Support)
```bash
# Ask about your documents in English
curl -X POST "http://localhost:8000/question" \
     -H "Content-Type: application/json" \
     -d '{"question": "What is Shaurya background as a developer?", "session_id": "ai_chat", "language": "english"}'

# Ask in Hindi
curl -X POST "http://localhost:8000/question" \
     -H "Content-Type: application/json" \
     -d '{"question": "शौर्य के बारे में बताएं", "session_id": "hindi_chat", "language": "hindi"}'

# Follow up question in English
curl -X POST "http://localhost:8000/question" \
     -H "Content-Type: application/json" \
     -d '{"question": "What projects has he worked on?", "session_id": "ai_chat", "language": "english"}'

# Continue conversation in Hindi
curl -X POST "http://localhost:8000/question" \
     -H "Content-Type: application/json" \
     -d '{"question": "उनके टेक्निकल स्किल्स क्या हैं?", "session_id": "hindi_chat", "language": "hindi"}'
```

---

## 🛑 How to Stop Everything

### Stop the Server
In the terminal running the server, press:
```
Ctrl + C
```

### Exit Virtual Environment
```bash
deactivate
```

### Stop Docker (if running)
```bash
docker compose down
```

---

## 📝 Supported File Types

- **PDF files** (.pdf)
- **Word documents** (.docx, .doc)
- **Text files** (.txt)
- **CSV files** (.csv)

---

## 🔧 Troubleshooting

### Server won't start?
```bash
# Make sure you're in virtual environment
source venv/bin/activate

# Check if port 8000 is free
lsof -i :8000

# Kill any process using port 8000
sudo kill -9 $(lsof -t -i:8000)
```

### Upload failing?
- Check file exists: `ls -la your_file.pdf`
- Check file format is supported
- File size should be reasonable (< 50MB)

### Questions not working?
- Ensure documents are uploaded first
- Check server logs in Terminal 1
- Verify Gemini API key is correct

---

## 💡 Tips for Best Results

1. **Upload multiple related documents** for richer context
2. **Use specific questions** rather than very general ones
3. **Use session_id** to maintain conversation context
4. **Upload documents in smaller chunks** if very large
5. **Wait for upload confirmation** before asking questions

---

## 🎯 What's Happening Under the Hood

1. **Document Upload**: Text is extracted and split into chunks
2. **Embedding**: Each chunk gets converted to vectors using Gemini AI
3. **Storage**: Vectors stored in database (FAISS locally, or Milvus with Docker)
4. **Question Processing**: Your question gets converted to a vector
5. **Retrieval**: System finds most relevant document chunks
6. **Generation**: Gemini AI generates answer using retrieved context
7. **Memory**: Conversation history maintained per session

## 🌐 **Multi-Language Support**

Your RAG bot now supports both **English and Hindi** languages:

### Language Parameter:
- `"language": "english"` - Responds in English
- `"language": "hindi"` - हिंदी में जवाब देता है

### Example Usage:
```bash
# English question
curl -X POST "http://localhost:8000/question" \
     -H "Content-Type: application/json" \
     -d '{"question": "What is Shaurya expertise?", "language": "english"}'

# Hindi question  
curl -X POST "http://localhost:8000/question" \
     -H "Content-Type: application/json" \
     -d '{"question": "शौर्य की विशेषज्ञता क्या है?", "language": "hindi"}'
```

## 🔄 **Smart Model Fallback System**

Your RAG bot features an intelligent model switching system:
- **1,800+ daily requests** across multiple Gemini models
- **Automatic failover** when rate limits are hit
- **Zero downtime** - seamless model switching
- **Enterprise-grade reliability** for your portfolio

## 📊 **Monitor Your Bot:**
- `/health` - Check current status and model usage
- `/model-status` - Detailed breakdown of all models

**Check model status:**
```bash
curl http://localhost:8000/model-status
```

**Example response:**
```json
{
  "current_model": "gemini-1.5-pro",
  "failed_models": ["gemini-1.5-flash"],
  "working_models": ["gemini-1.5-pro"],
  "total_requests_today": 52
}
```

## ⚠️ **Rate Limit Notes:**
- **Free tier limits**: 50 requests/day per model
- **Auto-switching**: When one model hits limit, automatically switches to backup
- **Daily reset**: Limits reset at midnight UTC
- **Model fallback**: System tries multiple models before failing

Your RAG bot is now ready to help you understand and interact with your documents in both English and Hindi! 🤖✨