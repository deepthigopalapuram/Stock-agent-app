import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from google import genai
from google.genai.errors import ServerError, ClientError

# Page Setup & Performance Optimization
st.set_page_config(page_title="Multi-Agent Stock Research Desk", layout="wide")

st.title("🤖 Multi-Agent Stock Analysis & Fundamental Research Desk")
st.markdown("Run automated quantitative, fundamental, and risk evaluations with comprehensive valuation grades and downloadable PDF reports.")

# Cache the complete Indian Stock Exchange equity list to optimize performance
@st.cache_data(ttl=86400)
def load_indian_tickers():
    try:
        # Fetching official NSE equity list
        df = pd.read_csv("https://archives.nseindia.com/content/equities/EQUITY_L.csv")
        # Format tickers with .NS suffix for yfinance compatibility
        tickers = sorted([f"{sym}.NS" for sym in df['SYMBOL'].dropna().unique()])
        return tickers
    except Exception:
        # Fallback default universe if network fetch fails
        return ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "ITC.NS", "BHARTIARTL.NS"]

# Cache market price downloads to maximize execution speed
@st.cache_data(ttl=3600)
def fetch_stock_data(ticker):
    return yf.download(ticker, period="6mo", interval="1d")

# Create Two-Column Browser Layout
col1, col2 = st.columns([1, 2], gap="large")

with col1:
    st.subheader("Input Panel")
    all_tickers = load_indian_tickers()
    
    # Allow searching or selecting from the full Indian market universe
    ticker = st.selectbox("Select or Search Stock Ticker:", options=all_tickers, index=all_tickers.index("RELIANCE.NS") if "RELIANCE.NS" in all_tickers else 0)
    run_btn = st.button("Run Multi-Agent Analysis", type="primary")

with col2:
    st.subheader("Analysis Output Dashboard")
    
    if run_btn:
        api_key = st.secrets.get("GEMINI_API_KEY")
        if not api_key:
            st.error("Please configure your GEMINI_API_KEY in your Streamlit app secrets.")
        else:
            client = genai.Client(api_key=api_key)
            
            # 1. Fetch Optimized Data & Render 6-Month Chart
            with st.spinner("Fetching market price data..."):
                df = fetch_stock_data(ticker)
                
            if df.empty:
                st.error("Invalid ticker or no data found.")
            else:
                st.write(f"📈 **6-Month Price Action & Trend: {ticker}**")
                fig, ax = plt.subplots(figsize=(8, 3.5))
                ax.plot(df.index, df['Close'], label="Close Price", color="#1f77b4", linewidth=2)
                ax.set_title(f"6-Month Historical Performance: {ticker}")
                ax.set_ylabel("Price")
                ax.grid(True, linestyle="--", alpha=0.5)
                ax.legend()
                st.pyplot(fig)
                
                # 2. Run Multi-Agent Analysis with DURGA Framework & Valuation Grading
                with st.spinner("Multi-agents evaluating fundamentals (DURGA framework) & risk..."):
                    prompt = f"""
                    You are a financial research team consisting of a Fundamental Analyst (using the DURGA Evaluation Framework) and a Risk Auditor.
                    Analyze the stock {ticker}. Provide:
                    1. Fundamental Valuation & Credit Assessment (Economic Moat, Balance Sheet Health, Governance Rating under the DURGA framework).
                    2. Explicit Valuation Grade (Choose strictly between Grade A: Undervalued/Deep Value, Grade B: Fairly Valued, or Grade C: Overvalued/Speculative) along with a clear, concise fundamental Rationale (2-3 sentences).
                    3. Quantitative & Technical Summary (Trend direction, support/resistance levels).
                    4. Final Risk Assessment Score (1-10).
                    Ensure all nomenclature references DURGA instead of any other rating agency. Keep output structured with markdown headings.
                    """
                    
                    # Resilient fallback sequence
                    models_to_try = ["gemini-3.8-flash", "gemini-2.5-flash", "gemini-2.0-flash"]
                    response = None
                    
                    for model_name in models_to_try:
                        try:
                            response = client.models.generate_content(
                                model=model_name,
                                contents=prompt
                            )
                            break  # Success
                        except (ServerError, ClientError):
                            continue
                            
                if response:
                    report_text = response.text
                    st.success("Analysis Complete!")
                    st.markdown(report_text)
                    
                    # 3. Generate Downloadable PDF Report
                    pdf_buffer = BytesIO()
                    doc = SimpleDocTemplate(pdf_buffer, pagesize=letter)
                    styles = getSampleStyleSheet()
                    
                    story = [
                        Paragraph(f"Comprehensive Research Report: {ticker}", styles['Title']),
                        Spacer(1, 12),
                        Paragraph(report_text.replace('\n', '<br/>'), styles['Normal'])
                    ]
                    doc.build(story)
                    pdf_buffer.seek(0)
                    
                    st.download_button(
                        label="📥 Download Detailed Report as PDF",
                        data=pdf_buffer,
                        file_name=f"{ticker}_comprehensive_report.pdf",
                        mime="application/pdf"
                    )
                else:
                    st.error("⚠️ All server endpoints are currently experiencing heavy traffic. Please click **Run Multi-Agent Analysis** again.")
    else:
        st.info("👈 Select any traded Indian stock security from the dropdown on the left and click **Run Multi-Agent Analysis**.")
