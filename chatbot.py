import streamlit as st
import os
import requests
import time
import pandas as pd
import datetime
import yfinance as yf
import matplotlib.pyplot as plt
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import openai
import chromadb
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from io import BytesIO

# Load API Keys
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Base directory for storing reports
BASE_DIR = "financial_reports"
os.makedirs(BASE_DIR, exist_ok=True)

# Load BSE data
equity_df = pd.read_csv("Equity.csv", dtype=str)
eqt0_df = pd.read_csv("EQT0.csv", dtype=str)

# Streamlit UI
st.title("📊 AI-powered Financial Assistant")

# Sidebar - Upload Documents
st.sidebar.header("📂 Upload Annual Reports & Documents")
documents = st.sidebar.file_uploader("Upload PDF/Text Files", accept_multiple_files=True, type=["pdf", "txt"])

# Select Language for AI Chatbot
language = st.sidebar.selectbox("🌍 Select Chatbot Language", ["English", "Hindi", "French", "Spanish", "German"])

# Market Selection
option = st.radio("📈 Select Market:", ("NSE", "BSE"))
user_input = st.text_input("🔍 Enter Stock Symbol (NSE) or Security Code/ID (BSE):").upper()

# Function to generate BSE report URL
def generate_bse_url(user_input):
    match = equity_df[equity_df["Security Code"] == user_input] if user_input.isdigit() else equity_df[equity_df["Security Id"] == user_input]
    if match.empty:
        match = eqt0_df[eqt0_df["Security Code"] == user_input] if user_input.isdigit() else eqt0_df[eqt0_df["Security Id"] == user_input]
    if match.empty:
        return None
    security_code = match.iloc[0]["Security Code"]
    security_id = match.iloc[0]["Security Id"].lower()
    issuer_name = match.iloc[0]["Issuer Name"].lower().replace(" ", "-").replace("&", "-").rstrip(".")
    return f"https://www.bseindia.com/stock-share-price/{issuer_name}/{security_id}/{security_code}/financials-annual-reports/"

# Function to scrape BSE reports
def scrape_bse_reports(url):
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    driver.get(url)

    reports = []
    try:
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.XPATH, "//table[@ng-if=\"loader.ARState=='loaded'\"]")))
        rows = driver.find_elements(By.XPATH, "//table[@ng-if=\"loader.ARState=='loaded'\"]/tbody/tr")
        for row in rows:
            cols = row.find_elements(By.TAG_NAME, "td")
            if len(cols) >= 6:
                year, pdf_link = cols[0].text.strip(), cols[5].find_element(By.TAG_NAME, "a").get_attribute("href")
                reports.append(f"{year}: {pdf_link}")
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
    driver.quit()
    return reports

# Function to fetch NSE reports
def fetch_nse_reports(symbol):
    url = f"https://www.nseindia.com/api/annual-reports?index=equities&symbol={symbol}"
    headers = {"User-Agent": "Mozilla/5.0", "Accept": "application/json", "Referer": "https://www.nseindia.com/"}
    session = requests.Session()
    session.get("https://www.nseindia.com", headers=headers)
    time.sleep(2)
    response = session.get(url, headers=headers)
    if response.status_code != 200:
        return None
    json_data = response.json()
    return [f"{item['fromYr']}-{item['toYr']}: {item['fileName']}" for item in json_data.get("data", [])]

# Function to plot stock data
def plot_stock_data(symbol):
    end_date = datetime.datetime.today()
    start_date = end_date - datetime.timedelta(days=5 * 365)
    try:
        stock = yf.Ticker(symbol)
        df = stock.history(start=start_date, end=end_date)
        if df.empty:
            st.error("❌ No stock data found.")
            return
        st.subheader(f"📊 {symbol} - Stock Price Trend (Last 5 Years)")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(df.index, df["Close"], label="Closing Price", color="blue")
        ax.set_xlabel("Year")
        ax.set_ylabel("Stock Price (INR)")
        ax.set_title(f"{symbol} - Price Movement")
        ax.legend()
        ax.grid()
        st.pyplot(fig)
    except Exception as e:
        st.error(f"❌ Error fetching stock data: {str(e)}")

# Fetch reports
if st.button("📂 Fetch Reports"):
    if user_input:
        if option == "NSE":
            reports = fetch_nse_reports(user_input)
        elif option == "BSE":
            url = generate_bse_url(user_input)
            if url:
                st.success(f"🔗 Generated BSE URL: {url}")
                reports = scrape_bse_reports(url)
            else:
                reports = None
        if reports:
            st.subheader("📄 Annual Reports")
            for report in reports:
                st.write(f"📅 {report}")
        else:
            st.warning("❌ No reports found.")

        # Fetch stock data
        plot_stock_data(user_input + (".NS" if option == "NSE" else ".BO"))

# AI-powered Chatbot
vector_db = None
if documents:
    all_text = ""
    for doc in documents:
        if doc.type == "application/pdf":
            temp_pdf_path = f"temp_{doc.name.replace(' ', '_')}"
            try:
                with open(temp_pdf_path, "wb") as f:
                    f.write(doc.getbuffer())
                pdf_loader = PyPDFLoader(temp_pdf_path)
                text = "\n".join([page.page_content for page in pdf_loader.load()])
            except Exception as e:
                st.error(f"❌ Error processing {doc.name}: {e}")
                continue
            if os.path.exists(temp_pdf_path):
                os.remove(temp_pdf_path)
        else:
            text = doc.read().decode("utf-8")
        all_text += text + "\n"
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_text(all_text)
    embeddings = OpenAIEmbeddings()
    vector_db = FAISS.from_texts(chunks, embeddings)
    st.sidebar.success("✅ Documents Indexed!")

st.write("### 🤖 Ask AI About the Company")
user_query = st.text_input("💬 Ask a question:")

if st.button("🔍 Get Answer") and user_query:
    if vector_db:
        retriever = vector_db.as_retriever()
        qa_chain = RetrievalQA.from_chain_type(llm=ChatOpenAI(model_name="gpt-4o-mini"), chain_type="stuff", retriever=retriever)
        response = qa_chain.run(user_query)
        st.write("💡 **AI Assistant:**", response)
    else:
        st.warning("⚠️ Upload documents first!")
