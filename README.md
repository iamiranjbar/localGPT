# localGPT
A local chatbot using Langchain and GPT4All libraries

# How to run
Run the below command:

` streamlit run src/app.py `

# How to install requirements and libraries
Run the below command:

` bash install.sh`

# Settings Parameter:
* n_gpu_layers: 
You should choose the number of layers to offload to the GPU based on your GPU's VRAM capacity. A value like 20 to 40 might be a good start if you have 10-12 GB of VRAM, but experiment based on your specific hardware.

* CUDA Environment: 
Make sure you have CUDA installed and your environment is set up for GPU acceleration.