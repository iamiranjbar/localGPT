import uvicorn
from fastapi import FastAPI
from langserve import add_routes

from app import get_chain, initiate_llm
from settings import MODELS_PATH


def run_server():
    app = FastAPI(title="Retrieval App")

    chosen_model = "llama-2-7b-chat.Q2_K.gguf"
    task_type = "Base"

    model_path = f"{MODELS_PATH}{chosen_model}"
    llm = initiate_llm(model_path)

    chain = get_chain(llm, task_type)
    add_routes(app, chain)
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    run_server()
