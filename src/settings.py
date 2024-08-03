MODEL_PATH = "/Users/amirranjbar/Desktop/Military/Code/models/mistral-7b-openorca.gguf"
# "/Users/amirranjbar/Desktop/Military/Code/models/llama-2-7b-chat.Q2_K.gguf"
n_gpu_layers = 1  # Metal set to 1 is enough.
n_batch = 256  # Should be between 1 and n_ctx, consider the amount of RAM of your Apple Silicon Chip.

PROMPT_TEMPLATE = """
    You are a helpful assistant. Answer the following questions considering the history of the conversation:

    Chat history: {chat_history}

    User question: {user_question}
    """