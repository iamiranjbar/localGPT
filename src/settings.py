# For local
MODELS_PATH = "/Users/amirranjbar/Desktop/Military/Code/models/"
# For Docker
# MODELS_PATH = "/app/models/"

N_GPU_LAYERS = 1  # Metal set to 1 is enough.
N_BATCH = 256  # Should be between 1 and n_ctx, consider the amount of RAM of your Apple Silicon Chip.
PERSIST_DIRECTORY = "./db"
MAX_ACCEPTABLE_RELEVANCE_SCORE = 1.2

TYPE_INFO_TEXT = {
    "Base": "Use this type to perform standard chat generation tasks.",
    "Creative": "Use this application to perform creative tasks like writing stories and poems.",
    "Summarization": "Use this application to perform summarization on blocks of text.",
    "Few Shot": "Pass through some examples of task-output to perform few-shot prompting.",
}

TYPE_PROMPT_TEMPELATES = {
    "Base": 
        """
        ### Instruction: 
        The prompt below is a question to answer, a task to complete, or a conversation to respond to; 
        decide which and write an appropriate response. Considering the history of the conversation!
        
        ### Chat history: 
        {chat_history}
        ### Prompt: 
        {user_question}
        ### Response:
        """,
    "Creative": 
        """
        As a creative agent, let's {user_question} Considering the history of the conversation!
        ### Chat history: 
        {chat_history}
        """,
    "Summarization": 
        """
        ### Instruction: 
        The prompt below is a passage to summarize. Using the prompt, provide a summarized response. Considering the history of the conversation!

        ### Chat history: 
        {chat_history}
        ### Prompt: 
        {user_question}
        ### Summary:
        """,
    "Few Shot": 
        """
        ### Instruction: 
        The prompt below is a question to answer, a task to complete, or a conversation to respond to; decide which and write an appropriate response. Considering the history of the conversation!

        ### Chat history: 
        {chat_history}
        ### Examples: 
        {examples}
        ### Prompt: 
        {user_question}
        ### Response:"""
}
