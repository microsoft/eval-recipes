# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Start Ollama service in the background and pull the embedding model
RUN nohup ollama serve > /dev/null 2>&1 & \
    sleep 5 && \
    ollama pull embeddinggemma:300m-qat-q8_0
