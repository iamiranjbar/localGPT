# localGPT

A local chatbot using Langchain and GPT4All libraries

## Features

- Local language model execution using LlamaCpp
- Document embedding and retrieval with Chroma vector store
- PDF upload and processing
- Streamlit-based user interface
- Support for multiple LLM models
- Task-specific prompts (Base, Creative, Summarization, Few Shot)

## Prerequisites

- Python 3.8+
- CUDA-compatible GPU (optional, but recommended for better performance)

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/localGPT.git
   cd localGPT
   ```

2. Install requirements and libraries:
   ```
   bash install.sh
   ```

## Usage

Run the application with:

` streamlit run src/app.py `

# Settings Parameter:
* n_gpu_layers: 
You should choose the number of layers to offload to the GPU based on your GPU's VRAM capacity. A value like 20 to 40 might be a good start if you have 10-12 GB of VRAM, but experiment based on your specific hardware.

* CUDA Environment: 
Make sure you have CUDA installed and your environment is set up for GPU acceleration.

# How to make a Cronjob to update datbase with new Local Documents periodically
1. How to create or edit Crontab File
`crontab -e`
2. choose editor (e.g. 1)
3. Create the Cronjob in the editor with this line
- Every day at midnight
`0 0 * * * /{path_to_project}/update_db.sh`
4. Save and Exit (Based on editor)
5. Check active crontabs
`crontab -l`
