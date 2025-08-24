# **Stock Analysis Platform**

AI-powered stock analysis platform with real-time data, technical indicators, ML price predictions, and intelligent chatbot assistant for comprehensive investment research.

## **Features**

- **AI-Powered Chat Assistant** - Natural language stock analysis
- **Machine Learning Predictions** - Linear Regression + Random Forest models
- **Technical Indicators** - RSI, MACD, Moving Averages, Bollinger Bands
- **Global Markets** - US, Pakistan, India, UK, Germany, Japan stocks
- **Real-time Data** - Live prices and market updates
- **Interactive Charts** - Candlestick and line visualizations
- **Financial Ratios** - P/E, ROE, debt ratios, liquidity metrics
- **News Integration** - Company news and congressional tracking

## **Installation**

```bash
git clone https://github.com/yourusername/stock-analysis-platform.git
cd stock-analysis-platform
pip install streamlit yfinance pandas numpy plotly scikit-learn requests python-dotenv
```

## **Setup**

Create `.env` file:
```env
CONGRESS_API_KEY=your_congress_api_key
FMP_API_KEY=your_fmp_api_key
```

## **Usage**

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

## **API Keys**

- **Congress API**: [api.congress.gov](https://api.congress.gov/)
- **Financial Modeling Prep**: [financialmodelingprep.com](https://financialmodelingprep.com/developer/docs)

## **Technology Stack**

- **Frontend**: Streamlit, Plotly
- **Data**: Yahoo Finance, Financial Modeling Prep API, Congress.gov API
- **ML**: scikit-learn, NumPy, Pandas
- **Analysis**: Technical indicators, price predictions

## **Supported Markets**

- **US**: AAPL, GOOGL, MSFT, AMZN, TSLA, NVDA
- **Pakistan**: HBL.KA, UBL.KA, ENGRO.KA, LUCK.KA
- **India**: RELIANCE.NS, TCS.NS, INFY.NS
- **UK**: SHEL.L, AZN.L, BP.L
- **Germany**: SAP.DE, SIE.DE, ALV.DE
- **Japan**: 7203.T, 6758.T, 9984.T

## **AI Assistant**

Ask questions like:
- "What's the current price?"
- "Is this stock overbought?"
- "Should I buy this stock?"
- "How risky is this investment?"

## **Contributing**

1. Fork the repository
2. Create feature branch
3. Make changes
4. Submit pull request

## **License**

MIT License

## **Disclaimer**

This platform is for educational purposes only. Not financial advice. Always do your own research and consult financial advisors before investing.
