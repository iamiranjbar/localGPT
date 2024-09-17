# Use an official Python runtime as a parent image
ARG PYTHON_VERSION=3.12.2
FROM python:${PYTHON_VERSION}-slim as base
# Install base dependencies using apt
RUN apt update && \
    apt install -y gcc && \
    apt install -y build-essential && \
    apt install -y cmake \
    && apt-get clean
# Set the working directory in the container
WORKDIR /app
# Copy the current directory contents into the container at /app
COPY . /app
# Make Persist Directory as environment variable for docker use
ENV PERSIST_DIRECTORY=/app/db
# Make sure the install.sh script is executable
RUN chmod +x /app/install.sh
# Run the install.sh script to install dependencies
RUN /bin/bash /app/install.sh
# Expose the port that Streamlit uses
EXPOSE 8501
# Expose the port that Uvicorn uses
EXPOSE 8000
# Define the command to start the Streamlit app
# CMD ["streamlit", "run", "src/app.py"]
# Define the command to start the Server app
# CMD ["python", "src/server.py"]
# Keep the container running with a shell
CMD ["tail", "-f", "/dev/null"]
