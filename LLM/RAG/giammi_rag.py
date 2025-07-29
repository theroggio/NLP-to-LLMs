import os
import bs4
from tqdm.auto import tqdm
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM
from langchain.prompts import ChatPromptTemplate

# data processing
pdf_path = "pdfs/"
docs = []

for pdf in os.listdir(pdf_path):
    loader = PyPDFLoader(os.path.join(pdf_path, pdf))
    doc = loader.load()
    docs.extend(doc)

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=300,
    chunk_overlap=50)

splits = text_splitter.split_documents(docs)

# set up model, embedding model, template, retriever
llm = OllamaLLM(model="llama2")
embd = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(documents=splits,
                                    embedding=embd)

template = """Answer the question based only on the following context:
{context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

retriever = vectorstore.as_retriever(search_kwargs={"k": 1})

question = "something"
while True:
    question = input("Ask something... Leave empty to terminate.`\n")
    if question == "":
        exit(0)
    docs = retriever.invoke(question)

    chain = prompt | llm
    res = chain.invoke({"context":docs,"question":question})

    print(res)

