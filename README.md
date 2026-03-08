# Pokémon Agentic RAG Assistant

An intelligent AI assistant built to analyze and retrieve Pokémon data using a sophisticated **Agentic RAG** (Retrieval-Augmented Generation) architecture. This system intelligently routes queries between semantic search and structured data analysis.

## 🧠 Key Architecture: The Agentic Router
This project implements an **Agentic Router** that classifies user intent to provide the most accurate response:
* **FAISS Route**: Handles semantic, descriptive, and background queries by retrieving context from a vector database.
* **Pandas Route**: Handles statistical, mathematical, and comparative queries by executing dynamic analysis on the structured dataset.

## 🛠 Tech Stack
* **Backend**: Flask (Python)
* **AI/ML**: Google Gemini (LLM), FAISS (Vector DB), Sentence-Transformers
* **Data Processing**: Pandas, NumPy
* **Infrastructure**: Docker, AWS EC2

## 🚀 Deployment
The application is containerized and ready for production deployment. 

```bash
docker run -p 5001:5001 \
  -e GEMINI_API_KEY="your_api_key_here" \
  --name pokemon-rag \
  adammes/pokemon-rag:1.0
```

## 📂 Project Structure
* app.py: Main application logic, including the Agentic Router and RAG chains.
* data/: Contains the source **"Pokemon.csv"** file used as the knowledge base.
* static/ & templates/: Web interface assets and HTML layouts.
* Dockerfile: Configuration for building the container image (linux/amd64).
* requirements.txt: List of Python dependencies.

# 📊 Data Credit
The dataset used in this project was sourced from [Kaggle](https://www.kaggle.com/datasets/abcsds/pokemon).