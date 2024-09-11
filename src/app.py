import os
import time
import math
import streamlit as st
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from langchain_community.llms import LlamaCpp
from langchain.embeddings.base import Embeddings
from langchain_community.vectorstores import Chroma
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter

from settings import *


class LocalEmbeddings(Embeddings):
    def __init__(self, model_name: str):
        # Load the model locally
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        # Convert embeddings to Python lists before returning
        embeddings = self.model.encode(texts, convert_to_tensor=False)
        return [embedding.tolist() for embedding in embeddings]

    def embed_query(self, text):
        # Convert a single embedding to a Python list
        embedding = self.model.encode([text], convert_to_tensor=False)
        return embedding[0].tolist()
        
def create_or_load_db():
    embeddings = LocalEmbeddings(model_name=TOKENIZER_PATH)

    if not os.path.exists(PERSIST_DIRECTORY):
        os.mkdir(PERSIST_DIRECTORY)
    db = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings)
    return db

def initiate_llm(model_path):
  llm = LlamaCpp(
    model_path=model_path,
    n_gpu_layers=N_GPU_LAYERS,
    n_batch=N_BATCH,
    n_ctx=2048,
    f16_kv=True, # MUST set to True, otherwise you will run into problem after a couple of calls
    verbose=True,
  )
  return llm

def render_model_uploader(existing_models):
  model_file = st.file_uploader("Choose your model file:")
  if model_file is not None and model_file.name not in existing_models:
    # To read file as bytes:
    bytes_data = model_file.getvalue()
    model_file_path = f"{MODELS_PATH}{model_file.name}"
    with open(model_file_path, "wb") as write_file:
      # Write bytes to file
      write_file.write(bytes_data)
      print("File upload has been done.")
      # Refresh page after upload is done
      st.experimental_rerun()

def render_side_bar():
  with st.sidebar:
    st.info("This application allows you to use LLMs for a range of tasks. Please choose your usecase.")
    task_type = st.radio("Choose your task:", ["Base", "Creative", "Summarization", "Few Shot"])
    models =  list(os.listdir(MODELS_PATH))
    models = [model for model in models if model!=".DS_Store"]
    chosen_model = st.radio("Choose your model:", models)
    model_path = f"{MODELS_PATH}{chosen_model}"
    llm = initiate_llm(model_path)
    render_model_uploader(models)
    return task_type, llm

def extract_text_from_pdf(pdf_reader):
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def split_text(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_text(text)
    return chunks

def add_to_vector_store(db, file_name, chunks):
    ids = [f"{file_name}-{i}" for i in range(len(chunks))]
    db.add_texts(texts=chunks, ids=ids)
    db.persist()

def add_pdf_to_db(db, pdf):
    pdf_reader = PdfReader(pdf)
    text = extract_text_from_pdf(pdf_reader)
    chunks = split_text(text)
    file_name = pdf.name[:-4]
    print("============Upload PDF Logs=============")
    print(f"Befor Upload PDF DB Docs Count: {db._collection.count()}")
    add_to_vector_store(db, file_name, chunks)
    print(f"After Upload PDF DB Docs Count: {db._collection.count()}")
    print(f"{file_name} has been added to database successfully.")
    print("========================================")
   
def upload_pdf(db):
    st.header("Add Documents 💬")
    pdf = st.file_uploader("Upload your PDF", type="pdf")
    if pdf:
        add_pdf_to_db(db, pdf)

def add_local_docs_to_db(db):
   for file_name in os.listdir(LOCAL_DOCS_FOLDER):
    file_path = os.path.join(LOCAL_DOCS_FOLDER, file_name)
    file_extension = file_name.split(".")[-1]
    if os.path.isfile(file_path) and file_extension in LOCAL_DOCS_ALLOWED_FORMATS:
        add_pdf_to_db(db, open(file_path, "rb"))

def initiate_session_state():
  if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
      AIMessage(content="Hello, I am a bot. How can I help you?"),
    ]

def show_previous_chats():
  for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
      with st.chat_message("AI"):
        st.write(message.content)
    elif isinstance(message, HumanMessage):
      with st.chat_message("Human"):
        st.write(message.content)

def render_prompt(task_type):
    st.info(TYPE_INFO_TEXT[task_type])
    examples = None
    if task_type == "Few Shot": 
        examples = st.text_area("Plug in your examples!")
    user_query = st.chat_input("Type your message here...")
    return user_query, examples

def get_chain(llm, task_type):
    template = TYPE_PROMPT_TEMPELATES[task_type]
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()
    return chain

def find_relative_docs(db, question):
    print("============DB Search Logs=============")
    start = time.time_ns()
    results = db.similarity_search_with_score(question, k=1)
    end = time.time_ns()
    print(f"Similarity Search Time: {(end - start)/1000000000}s")
    print("========================================")
    docs = [item[0] for item in results]
    scores = [item[1] for item in results]
    return docs, scores

def get_response_from_llm(llm, task_type, user_query, examples, chat_history):
    chain = get_chain(llm, task_type)
    return chain.stream({
        "chat_history": chat_history,
        "user_question": user_query,
        "examples": examples,
    })

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def ask_question_from_document(llm, question, db, chat_history):
    print("============LLM Response Logs=============")
    start = time.time_ns()

    retriever=db.as_retriever()
    prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    response = rag_chain.stream({"question": question, "chat_history": chat_history})
    end = time.time_ns()
    print(f"LLM Response Time: {(end - start)/1000000000}s")
    print("========================================")
    return response

def chat_with_user(llm, task_type, db):
  user_query, examples = render_prompt(task_type)
  if user_query is not None and user_query != "":
    st.session_state.chat_history.append(HumanMessage(content=user_query))

    with st.chat_message("Human"):
      st.markdown(user_query)

    _, scores = find_relative_docs(db, user_query)
    if scores:
        first_doc_score = scores[0]
    else: first_doc_score = math.inf
    with st.chat_message("AI"):
      if first_doc_score < MAX_ACCEPTABLE_RELEVANCE_SCORE:
        print("Answer from localdocs")
        response = st.write_stream(ask_question_from_document(llm, user_query, db, st.session_state.chat_history))
      else:
        print("Answer from llm")
        response = st.write_stream(get_response_from_llm(llm, task_type, user_query, examples, st.session_state.chat_history))

    st.session_state.chat_history.append(AIMessage(content=response))

def run():
  st.set_page_config(page_title="Local Chatbot", page_icon="🤖")
  st.title("LocalGPT 💬")
  db = create_or_load_db()
  add_local_docs_to_db(db)
  task_type, llm = render_side_bar()
  upload_pdf(db)
  initiate_session_state()
  show_previous_chats()
  chat_with_user(llm, task_type, db)

if __name__ == "__main__":
  run()
