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

def get_response(llm, user_query, chat_history):
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    chain = prompt | llm | StrOutputParser()

    return chain.stream({
        "chat_history": chat_history,
        "user_question": user_query,
    })

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

def chat_with_user(llm):
  user_query = st.chat_input("Type your message here...")
  if user_query is not None and user_query != "":
    st.session_state.chat_history.append(HumanMessage(content=user_query))

    with st.chat_message("Human"):
      st.markdown(user_query)

    with st.chat_message("AI"):
      response = st.write_stream(get_response(llm, user_query, st.session_state.chat_history))

    st.session_state.chat_history.append(AIMessage(content=response))

def run():
  st.set_page_config(page_title="Local Chatbot", page_icon="🤖")
  st.title("LocalGPT 💬")
  llm = initiate_llm(MODEL_PATH)
  initiate_session_state()
  show_previous_chats()
  chat_with_user(llm)

if __name__ == "__main__":
  run()
