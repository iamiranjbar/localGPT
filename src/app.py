import os
import streamlit as st
from langchain_community.llms import LlamaCpp
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from settings import *

def initiate_llm(model_path):
  llm = LlamaCpp(
    model_path=model_path,
    n_gpu_layers=n_gpu_layers,
    n_batch=n_batch,
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

def get_response(llm, task_type, user_query, examples, chat_history):
    chain = get_chain(llm, task_type)
    return chain.stream({
        "chat_history": chat_history,
        "user_question": user_query,
        "examples": examples,
    })

def chat_with_user(llm, task_type):
  user_query, examples = render_prompt(task_type)
  if user_query is not None and user_query != "":
    st.session_state.chat_history.append(HumanMessage(content=user_query))

    with st.chat_message("Human"):
      st.markdown(user_query)

    with st.chat_message("AI"):
      response = st.write_stream(get_response(llm, task_type, user_query, examples, st.session_state.chat_history))

    st.session_state.chat_history.append(AIMessage(content=response))

def run():
  st.set_page_config(page_title="Local Chatbot", page_icon="🤖")
  st.title("LocalGPT 💬")
  task_type, llm = render_side_bar()
  initiate_session_state()
  show_previous_chats()
  chat_with_user(llm, task_type)

if __name__ == "__main__":
  run()
