from fastapi import FastAPI, APIRouter, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM
from langchain.prompts import ChatPromptTemplate
import asyncio

app = FastAPI()
router = APIRouter()

# Pydantic models
class Message(BaseModel):
    role: str
    content: str

class ChatPayload(BaseModel):
    messages: List[Message]

    class Config:
        schema_extra = {
            "example": {
                "messages": [{"role": "user", "content": "What does leaf instance segmentation mean?"}]
            }
        }

# Pre-load documents and build vector store
pdf_path = "pdfs/"
docs = []
for pdf in os.listdir(pdf_path):
    loader = PyPDFLoader(os.path.join(pdf_path, pdf))
    doc = loader.load()
    docs.extend(doc)

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=300, chunk_overlap=50)
splits = text_splitter.split_documents(docs)

embd = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(documents=splits, embedding=embd)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Set up model and prompt template
chat = OllamaLLM(model="llama2")
template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# Streaming endpoint
@router.post("/api/completion")
async def stream(request: Request, payload: ChatPayload):
    # Only use the latest user message
    last_user_msg = next((m.content for m in reversed(payload.messages) if m.role == "user"), None)
    if not last_user_msg:
        return {"error": "No user message found."}

    async def event_stream():
        # Retrieve docs
        relevant_docs = retriever.get_relevant_documents(last_user_msg)
        context = "\n\n".join([doc.page_content for doc in relevant_docs])

        # Build and run the chain
        chain = prompt | chat

        # Use async streaming if supported
        async for chunk in chain.astream({"context": context, "question": last_user_msg}):
            yield f"data: {chunk}\n\n"
            await asyncio.sleep(0)  # Yield control

        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")

app.include_router(router)

