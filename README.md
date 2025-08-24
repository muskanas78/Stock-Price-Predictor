Stock Analysis Platform
Problem Statement
Individual investors struggle with stock analysis due to:

Scattered Information - Data spread across multiple platforms and sources
Complex Technical Analysis - Difficulty interpreting charts and indicators
Time-Intensive Research - Hours spent gathering and analyzing data
Lack of AI Assistance - No intelligent guidance for investment decisions
Limited Global Access - Restricted coverage of international markets

Solution
A unified AI-powered platform that provides:

Centralized Dashboard - All stock data and analysis in one place
Intelligent AI Assistant - Natural language queries for instant insights
Automated Technical Analysis - Real-time indicators and predictions
Machine Learning Predictions - Price forecasting with multiple models
Global Market Coverage - Stocks from 6 major international markets
Interactive Visualizations - Easy-to-understand charts and metrics

AI-powered stock analysis platform with real-time data, technical indicators, ML price predictions, and intelligent chatbot assistant for comprehensive investment research.
Features

AI-Powered Chat Assistant - Natural language stock analysis
Machine Learning Predictions - Linear Regression + Random Forest models
Technical Indicators - RSI, MACD, Moving Averages, Bollinger Bands
Global Markets - US, Pakistan, India, UK, Germany, Japan stocks
Real-time Data - Live prices and market updates
Interactive Charts - Candlestick and line visualizations
Financial Ratios - P/E, ROE, debt ratios, liquidity metrics
News Integration - Company news and congressional tracking

Installation
bashgit clone https://github.com/yourusername/stock-analysis-platform.git
cd stock-analysis-platform
pip install streamlit yfinance pandas numpy plotly scikit-learn requests python-dotenv
Setup
Create .env file:
envCONGRESS_API_KEY=your_congress_api_key
FMP_API_KEY=your_fmp_api_key
Usage
bashstreamlit run app.py
Open http://localhost:8501 in your browser.
API Keys

Congress API: api.congress.gov
Financial Modeling Prep: financialmodelingprep.com

Technology Stack

Frontend: Streamlit, Plotly
Data: Yahoo Finance, Financial Modeling Prep API, Congress.gov API
ML: scikit-learn, NumPy, Pandas
Analysis: Technical indicators, price predictions

Supported Markets

US: AAPL, GOOGL, MSFT, AMZN, TSLA, NVDA
Pakistan: HBL.KA, UBL.KA, ENGRO.KA, LUCK.KA
India: RELIANCE.NS, TCS.NS, INFY.NS
UK: SHEL.L, AZN.L, BP.L
Germany: SAP.DE, SIE.DE, ALV.DE
Japan: 7203.T, 6758.T, 9984.T

AI Assistant
Ask questions like:

"What's the current price?"
"Is this stock overbought?"
"Should I buy this stock?"
"How risky is this investment?"

Contributing

Fork the repository
Create feature branch
Make changes
Submit pull request

License
MIT License
Disclaimer
This platform is for educational purposes only. Not financial advice. Always do your own research and consult financial advisors before investing.
