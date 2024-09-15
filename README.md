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
