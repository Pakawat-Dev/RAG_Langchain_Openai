# Health and Safety RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot for workplace health and safety guidance using internal safety documents.

## Features

- PDF document processing from safety documentation
- Vector-based document retrieval using FAISS
- OpenAI GPT-4o powered responses
- Streamlit web interface

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Create `.env` file with OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
```

3. Place PDF safety documents in `D:\Safety_Docs`

## Usage

Run the application:
```bash
streamlit run "RAG for Health and Safety Work.py"
```

Ask questions about safety policies, procedures, and best practices through the web interface.
