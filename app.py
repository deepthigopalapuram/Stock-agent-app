import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from google import genai

# Page Setup
st.set_page_config(page_title="Multi-Agent Stock Research Desk", layout="wide")

st.title("🤖 Multi-Agent Stock Analysis & Fundamental Research Desk")
st.markdown("Run automated quantitative, fundamental, and risk evaluations with downloadable PDF reports.")

# Create Two-Column Browser Layout
col1, col2 = st.columns([1, 2], gap="large")

with col1:
    st.subheader("Input Panel")
    # Dropdown for Stock Tickers
    ticker_options = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS"]
    ticker = st.selectbox("Select Stock Ticker:", options=ticker_options)
    run_btn = st.button("Run Multi-Agent Analysis", type="primary")

with col2:
    st.subheader("Analysis Output Dashboard")
    
    if run_btn:
        api_key = st.secrets.get("GEMINI_API_KEY")
        if not api_key:
            st.error("Please configure your GEMINI_API_KEY in your Streamlit app secrets (see Step 5).")
        else:
            client = genai.Client(api_key=api_key)
            
            # 1. Fetch Data & Render 6-Month Chart
            with st.spinner("Fetching 6-month market price data..."):
                df = yf.download(ticker, period="6mo", interval="1d")
                
            if df.empty:
                st.error("Invalid ticker or no data found.")
            else:
                st.write("📈 **6-Month Price Action & Trend**")
                fig, ax = plt.subplots(figsize=(8, 3.5))
                ax.plot(df.index, df['Close'], label="Close Price", color="#1f77b4", linewidth=2)
                ax.set_title(f"6-Month Historical Performance: {ticker}")
                ax.set_ylabel("Price")
                ax.grid(True, linestyle="--", alpha=0.5)
                ax.legend()
                st.pyplot(fig)
                
                # 2. Run Multi-Agent Analysis
                with st.spinner("Multi-agents evaluating fundamentals (CRISIL-style) & risk..."):
                    prompt = f"""
                    You are a financial research team consisting of a Fundamental Analyst (CRISIL rating style) and a Risk Auditor.
                    Analyze the stock {ticker}. Provide:
                    1. Fundamental Valuation (Economic Moat, Balance Sheet Health, Governance Rating).
                    2. Quantitative & Technical Summary (Trend direction, support/resistance levels).
                    3. Final Risk Assessment Score (1-10).
                    Keep the output structured with clear markdown headings.
                    """
                    response = client.models.generate_content(
                        model="gemini-2.5-flash",
                        contents=prompt
                    )
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
        st.info("👈 Select a stock ticker from the dropdown on the left and click **Run Multi-Agent Analysis**.")
