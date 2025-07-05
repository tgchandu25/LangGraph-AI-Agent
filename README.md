# LangGraph AI Agent – Weather + PDF Querying Assistant

## 🧠 Objective
The goal of this project is to develop a LangGraph-powered AI Agent capable of answering both unstructured document-based queries using RAG (Retrieval-Augmented Generation) and structured weather-based queries using a weather API. This dual-query agent is designed for intelligent reasoning, traceability through LangSmith, and real-world usability via a polished Streamlit UI.

This project showcases a multi-functional AI agent built using **LangGraph**, **LangChain**, and **Streamlit**, capable of answering user queries from two sources:

- Live **Weather Data** via the OpenWeatherMap API.
- **Document-Based RAG** using a sample PDF.

The project also integrates **LangSmith** for experiment tracking and performance evaluation.

---

## 🚀 Features

- 🌤️ Understands and responds to city-specific weather questions with detailed parameters (temperature, rain, humidity, snow, wind, etc.).
- 📄 Handles PDF-based Q&A using Retrieval-Augmented Generation (RAG) from uploaded documents.
- 🧠 Uses LangGraph to manage decision logic for question routing between Weather or PDF nodes.
- 📊 Tracks and evaluates responses using LangSmith.

---

## 💻 Technologies Used
- **LangGraph** – For graph-based multi-step agent orchestration
- **LangChain & LangSmith** – For RAG and trace logging
- **OpenAI API** – For natural language understanding and generation
- **Qdrant** – As the vector store for document chunk retrieval
- **Streamlit** – UI development and interactive frontend
- **OpenWeatherMap API** – For fetching real-time and forecast weather data
- **Python** – Core implementation language

## ⚙️ How the Agent Works (Design & Architecture)
The system is built as a LangGraph that routes user questions through two distinct paths based on intent:
- If the question is weather-related, the agent uses a dedicated weather node to extract city and context (temperature, humidity, rain, etc.) and calls OpenWeatherMap API to get a structured response.
- If the question is document-related, it passes through a RAG node that uses LangChain and Qdrant to retrieve relevant document chunks and respond using OpenAI completions.

The routing logic ensures the correct execution path with context-aware processing.

## 📁 Project Structure
```
├── app.py                                # Main application code for Streamlit app
├── requirements.txt                      # Python dependencies required for the app
├── README.md                             # Project overview and setup instructions
├── sample_document.pdf                   # Example PDF used for RAG-based querying
├── RAG_Weather_Evaluation_Questions.pdf  # Dataset of questions for evaluation
├── LangSmith_logs                        # Trace logs for LangSmith experiments
├── test_app.py                           # API handling, LLM processing, and Retrieval logic
├── test_results                          # Test results
├── Documentation                         # Full architecture + implementation report
```

## 🔧 Setup Instructions

1. **Clone the Repository**
```bash
git clone https://github.com/tgchandu25/LangGraph-AI-Agent.git
cd LangGraph-AI-Agent
```

2. **Install Dependencies**
```bash
pip install -r requirements.txt
```

3. **Setup Environment Variables**
Set the following in your Hugging Face or `.env` file:
```bash
OPENAI_API_KEY=your-openai-key
OPENWEATHER_API_KEY=your-openweather-key
LANGCHAIN_API_KEY=your-langsmith-api-key
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=LangGraph-AI-Agent
```

4. **Run the App**
```bash
streamlit run app.py
```

## 🚀 Live Demo
Click to Try the Agent on Hugging Face Spaces - https://huggingface.co/spaces/TGChandu/LangGraph-AI-Agent

## ✅ Conclusion
This LangGraph-powered AI Agent demonstrates seamless integration of structured (weather) and unstructured (PDF) data querying. Its LangSmith logs provide traceability, and the refined UI ensures a professional user experience. This project highlights advanced orchestration, real-world reasoning, and production-ready engineering.

---

For detailed usage, test cases, and evaluation results, refer to the respective folders in the repository.