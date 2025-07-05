import time
import os
import requests
import streamlit as st
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_core.runnables import RunnableLambda
from langchain.callbacks.tracers.langchain import LangChainTracer
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from typing import TypedDict, Literal, Optional

# ✅ Environment Variables
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = "LangGraph-AI-Agent"
os.environ["OPENWEATHER_API_KEY"] = os.getenv("OPENWEATHER_API_KEY")

# Access LangSmith environment variables
langsmith_api_key = os.getenv("LANGCHAIN_API_KEY")
project_name = os.getenv("LANGCHAIN_PROJECT", "LangGraph-AI-Agent")

tracer = LangChainTracer(project_name=os.getenv("LANGCHAIN_PROJECT"))

print("LangSmith API Key found:", bool(langsmith_api_key))
print("LangSmith Project:", project_name)


# ✅ Load and process PDF
pdf_name = "sample_document.pdf"
PDF_PATH = os.path.join(os.path.dirname(__file__), pdf_name)

if not os.path.exists(PDF_PATH):
    raise FileNotFoundError(f"PDF not found at: {PDF_PATH}")

# Load and process PDF
loader = PyPDFLoader(PDF_PATH)
pages = loader.load_and_split()
splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(pages)

# ✅ Embedding model and Qdrant in-memory store
embedding = OpenAIEmbeddings()
qdrant = Qdrant.from_documents(
    documents=chunks,
    embedding=embedding,
    location=":memory:",
    collection_name="rag_data"
)
retriever = qdrant.as_retriever()

# ✅ LLM
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# ✅ Graph State
class GraphState(TypedDict):
    query: str
    route: Literal["weather", "document"]
    weather_data: Optional[str]
    document_answer: Optional[str]
    final_response: Optional[str]

# ✅ Routing Logic
def router_node(state: GraphState) -> dict:
    query = state["query"].lower()
    weather_keywords = [
        "weather", "temperature", "forecast", "rain", "raining", "storm", "heat", "hot", "cold", "cool", "forecasting",
        "humidity", "humid", "wind", "windspeed", "snow", "clouds", "sunshine", "fog", "alert", "climate", "sunrise",
        "sunset", "dew", "minimum", "maximum"
    ]
    if any(word in query for word in weather_keywords):
        return {"route": "weather"}
    return {"route": "document"}


# ✅ Weather Node
def weather_node(state: GraphState) -> dict:
    import re
    import os
    import requests

    query = state["query"].lower()
    print(f"[DEBUG] Incoming weather query: {query}")

    # Extract city name
    city_match = re.search(r"(?:in|for)\s+([A-Za-z\s]+)", query)
    city = city_match.group(1).strip() if city_match else ""
    city = re.sub(r"(today|tomorrow|right now|currently|this weekend)", "", city, flags=re.IGNORECASE).strip().title()

    if not city:
        return {"weather_data": "Could not identify city in the query."}

    api_key = os.getenv("OPENWEATHER_API_KEY")
    if not api_key:
        return {"weather_data": "API key missing. Please set OPENWEATHER_API_KEY in Hugging Face secrets."}

    # Decide endpoint
    is_forecast = "tomorrow" in query or "weekend" in query
    endpoint = "forecast" if is_forecast else "weather"
    url = f"http://api.openweathermap.org/data/2.5/{endpoint}?q={city}&appid={api_key}&units=metric"
    
    try:
        response = requests.get(url)
        if response.status_code != 200:
            return {"weather_data": f"Failed to fetch weather for {city}. Please check spelling or try again."}
        data = response.json()

        print(f"[DEBUG] Fetched data for {city}: {data}")

        # Handle tomorrow forecast
        if is_forecast:
            for item in data.get("list", []):
                if "12:00:00" in item["dt_txt"]:
                    forecast_desc = item["weather"][0]["description"]
                    forecast_temp = item["main"]["temp"]
                    return {"weather_data": f"The forecast temperature in {city} tomorrow at noon is {forecast_temp}°C with {forecast_desc}."}
            return {"weather_data": "Could not retrieve forecast for tomorrow."}

        # Determine intent
        desc = data["weather"][0]["description"].capitalize()
        temp = data["main"]["temp"]
        wind_speed = data.get("wind", {}).get("speed", "N/A")
        is_rain = "rain" in desc.lower()
        is_snow = "snow" in desc.lower()

        # Answering logic
        if "rain" in query:
            return {"weather_data": f"Yes, it is raining in {city}." if is_rain else f"No, it's not raining in {city}."}
        elif "snow" in query:
            return {"weather_data": f"Yes, it's snowing in {city}." if is_snow else f"No, there is no snow in {city}."}
        elif "wind" in query:
            return {"weather_data": f"The current wind speed in {city} is {wind_speed} m/s with {desc}."}
        elif "summary" in query or "report" in query:
            humidity = data["main"].get("humidity", "N/A")
            feels_like = data["main"].get("feels_like", "N/A")
            return {"weather_data": f"Weather in {city}: {desc}, Temp: {temp}°C, Feels Like: {feels_like}°C, Humidity: {humidity}%, Wind: {wind_speed} m/s."}
        else:
            return {"weather_data": f"The current temperature in {city} is {temp}°C with {desc}."}

    except Exception as e:
        return {"weather_data": f"Weather fetch error: {str(e)}"}

# ✅ RAG Node
def rag_node(state: GraphState) -> dict:
    qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever)
    result = qa_chain.invoke({"query": state["query"]})
    return {"document_answer": result["result"]}

# ✅ Final Node
def final_node(state: GraphState) -> dict:
    return {
        "final_response": state["weather_data"] if state["route"] == "weather" else state["document_answer"]
    }

# ✅ Build LangGraph
def build_graph():
    builder = StateGraph(GraphState)
    builder.add_node("router", router_node)
    builder.add_node("weather", weather_node)
    builder.add_node("rag", rag_node)
    builder.add_node("llm", final_node)

    builder.set_entry_point("router")

    condition = RunnableLambda(lambda state: state["route"])
    condition.name = "route_selector"
    builder.add_conditional_edges("router", condition, {
        "weather": "weather",
        "document": "rag"
    })

    builder.add_edge("weather", "llm")
    builder.add_edge("rag", "llm")
    builder.add_edge("llm", END)

    return builder.compile()

graph = build_graph()

# ✅ Streamlit UI
st.set_page_config(page_title="LangGraph AI Agent", layout="wide")
st.title("🧠 LangGraph AI Agent")

st.markdown("Ask any question about **Weather** or from the **PDF content**.")

query = st.text_input("🔍 Enter your query here", key="query_input")

if 'history' not in st.session_state:
    st.session_state.history = []

col_submit, col_timer = st.columns([1, 3])

def get_reason(route: str) -> str:
    if route == "weather":
        return "Detected weather-related terms like 'weather', 'temperature', or city names."
    else:
        return "No weather keywords found. Using RAG PDF data for semantic answer."

if col_submit.button("Submit") and query:
    start = time.perf_counter()
    with st.spinner("Thinking..."):
        result = graph.invoke({"query": query})
        elapsed = time.perf_counter() - start
        route = result["route"]
        final_answer = result["final_response"]
        source = "🌤️ OPENWEATHER" if route == "weather" else "📄 RAG PDF"
        reason = get_reason(route)

        st.session_state.history.append({
            "query": query,
            "answer": final_answer,
            "source": source,
            "reason": reason,
            "time": f"{elapsed:.2f} sec"
        })
    st.rerun()

with col_timer:
    if st.session_state.history:
        last_time = st.session_state.history[-1]["time"]
        st.info(f"Time taken: {last_time}")

# ✅ Display History in Rows
if st.session_state.history:
    st.markdown("## 🔁 Previous Queries")
    for i, item in enumerate(reversed(st.session_state.history)):
        st.markdown(f"**{i+1}. Query**: {item['query']}")
        st.markdown(f"**Answer**: {item['answer']}")
        st.markdown(f"**Source**: {item['source']}")
        st.markdown(f"**Reason**: {item['reason']}")
        st.divider()
