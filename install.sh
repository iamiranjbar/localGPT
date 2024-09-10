pip uninstall llama-cpp-python -y
CMAKE_ARGS="-DGGML_METAL=on" pip install -U llama-cpp-python --no-cache-dir
# For using GPU
# CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install -U llama-cpp-python --no-cache-dir
pip install 'llama-cpp-python[server]'
pip install -r requirements.txt
