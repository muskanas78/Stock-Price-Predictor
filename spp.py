import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from datetime import datetime, timedelta
import requests
import json
import os
from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore')

# API Keys
load_dotenv()
CONGRESS_API_KEY = os.getenv('CONGRESS_API_KEY')
FMP_API_KEY = os.getenv('FMP_API_KEY')

# Configure the page
st.set_page_config(
    page_title="Advanced Stock Analysis Platform", 
    page_icon="📊", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f4e79;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f4e79;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .user-message {
        background-color: #e3f2fd;
        text-align: right;
    }
    .bot-message {
        background-color: #f5f5f5;
        text-align: left;
    }
    .ai-response {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .news-item {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #28a745;
    }
</style>
""", unsafe_allow_html=True)

# Stock database with countries
STOCK_DATABASE = {
    "United States": {
        "AAPL": "Apple Inc.",
        "GOOGL": "Alphabet Inc.",
        "MSFT": "Microsoft Corporation",
        "AMZN": "Amazon.com Inc.",
        "TSLA": "Tesla Inc.",
        "NVDA": "NVIDIA Corporation",
        "META": "Meta Platforms Inc.",
        "JPM": "JPMorgan Chase & Co.",
        "JNJ": "Johnson & Johnson",
        "V": "Visa Inc.",
        "PG": "Procter & Gamble Co.",
        "UNH": "UnitedHealth Group Inc.",
        "HD": "Home Depot Inc.",
        "MA": "Mastercard Inc.",
        "BAC": "Bank of America Corp",
        "DIS": "Walt Disney Co.",
        "ADBE": "Adobe Inc.",
        "CRM": "Salesforce Inc.",
        "NFLX": "Netflix Inc.",
        "PYPL": "PayPal Holdings Inc."
    },
    "Pakistan": {
        "HBL.KA": "Habib Bank Limited",
        "UBL.KA": "United Bank Limited",
        "BAFL.KA": "Bank Alfalah Limited",
        "MCB.KA": "MCB Bank Limited",
        "ENGRO.KA": "Engro Corporation",
        "LUCK.KA": "Lucky Cement Limited",
        "PSO.KA": "Pakistan State Oil",
        "OGDC.KA": "Oil & Gas Development Company",
        "PPL.KA": "Pakistan Petroleum Limited",
        "FCCL.KA": "Fauji Cement Company Limited",
        "HUBC.KA": "Hub Power Company Limited",
        "TRG.KA": "TRG Pakistan Limited",
        "SSGC.KA": "Sui Southern Gas Company",
        "SNGP.KA": "Sui Northern Gas Pipelines",
        "MEBL.KA": "Meezan Bank Limited"
    },
    "India": {
        "RELIANCE.NS": "Reliance Industries Limited",
        "TCS.NS": "Tata Consultancy Services",
        "INFY.NS": "Infosys Limited",
        "HINDUNILVR.NS": "Hindustan Unilever Limited",
        "ICICIBANK.NS": "ICICI Bank Limited",
        "HDFCBANK.NS": "HDFC Bank Limited",
        "ITC.NS": "ITC Limited",
        "SBIN.NS": "State Bank of India",
        "BHARTIARTL.NS": "Bharti Airtel Limited",
        "KOTAKBANK.NS": "Kotak Mahindra Bank"
    },
    "United Kingdom": {
        "SHEL.L": "Shell plc",
        "AZN.L": "AstraZeneca PLC",
        "BP.L": "BP p.l.c.",
        "ULVR.L": "Unilever PLC",
        "HSBA.L": "HSBC Holdings plc",
        "VOD.L": "Vodafone Group Plc",
        "GSK.L": "GSK plc",
        "LLOY.L": "Lloyds Banking Group plc",
        "BT-A.L": "BT Group plc",
        "BARC.L": "Barclays PLC"
    },
    "Germany": {
        "SAP.DE": "SAP SE",
        "SIE.DE": "Siemens AG",
        "ASME.DE": "ASML Holding N.V.",
        "ALV.DE": "Allianz SE",
        "BAS.DE": "BASF SE",
        "BMW.DE": "Bayerische Motoren Werke AG",
        "VOW3.DE": "Volkswagen AG",
        "DTE.DE": "Deutsche Telekom AG",
        "MUV2.DE": "Munich Re",
        "ADS.DE": "Adidas AG"
    },
    "Japan": {
        "7203.T": "Toyota Motor Corporation",
        "6758.T": "Sony Group Corporation",
        "9984.T": "SoftBank Group Corp.",
        "8306.T": "Mitsubishi UFJ Financial Group",
        "6861.T": "Keyence Corporation",
        "9434.T": "SoftBank Corp.",
        "4689.T": "Yahoo Japan Corporation",
        "8316.T": "Sumitomo Mitsui Financial Group",
        "6954.T": "Fanuc Corporation",
        "7974.T": "Nintendo Co., Ltd."
    }
}

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'analysis_data' not in st.session_state:
    st.session_state.analysis_data = {}
if 'ai_context' not in st.session_state:
    st.session_state.ai_context = {}

# API Integration Functions
@st.cache_data(ttl=3600)
def get_financial_data_fmp(symbol):
    """Fetch detailed financial data from Financial Modeling Prep"""
    try:
        # Clean symbol for API (remove exchange suffixes)
        clean_symbol = symbol.split('.')[0]
        
        # Company profile
        profile_url = f"https://financialmodelingprep.com/api/v3/profile/{clean_symbol}?apikey={FMP_API_KEY}"
        profile_response = requests.get(profile_url)
        
        # Financial ratios
        ratios_url = f"https://financialmodelingprep.com/api/v3/ratios/{clean_symbol}?apikey={FMP_API_KEY}"
        ratios_response = requests.get(ratios_url)
        
        # Key metrics
        metrics_url = f"https://financialmodelingprep.com/api/v3/key-metrics/{clean_symbol}?apikey={FMP_API_KEY}"
        metrics_response = requests.get(metrics_url)
        
        # News
        news_url = f"https://financialmodelingprep.com/api/v3/stock_news?tickers={clean_symbol}&limit=10&apikey={FMP_API_KEY}"
        news_response = requests.get(news_url)
        
        profile_data = profile_response.json() if profile_response.status_code == 200 else []
        ratios_data = ratios_response.json() if ratios_response.status_code == 200 else []
        metrics_data = metrics_response.json() if metrics_response.status_code == 200 else []
        news_data = news_response.json() if news_response.status_code == 200 else []
        
        return {
            'profile': profile_data,
            'ratios': ratios_data,
            'metrics': metrics_data,
            'news': news_data
        }
        
    except Exception as e:
        st.error(f"Error fetching FMP data: {str(e)}")
        return {'profile': [], 'ratios': [], 'metrics': [], 'news': []}

@st.cache_data(ttl=86400)
def get_congress_data():
    """Fetch relevant financial/economic bills from Congress API"""
    try:
        # Search for financial/economic bills
        url = f"https://api.congress.gov/v3/bill?api_key={CONGRESS_API_KEY}&limit=20&format=json"
        response = requests.get(url)
        
        if response.status_code == 200:
            data = response.json()
            bills = data.get('bills', [])
            
            # Filter for financial/economic related bills
            financial_keywords = ['tax', 'financial', 'economic', 'banking', 'securities', 'investment', 'market', 'trade']
            relevant_bills = []
            
            for bill in bills:
                title = bill.get('title', '').lower()
                if any(keyword in title for keyword in financial_keywords):
                    relevant_bills.append(bill)
                    
            return relevant_bills[:10]  # Return top 10 relevant bills
            
    except Exception as e:
        st.error(f"Error fetching Congress data: {str(e)}")
        return []

@st.cache_data
def fetch_stock_data(symbol, period):
    """Fetch comprehensive stock data from Yahoo Finance"""
    try:
        stock = yf.Ticker(symbol)
        data = stock.history(period=period)
        
        if data.empty:
            st.error(f"No data found for symbol {symbol}")
            return None, None, None, None, None
            
        info = stock.info
        
        # Get additional data with error handling
        try:
            financials = stock.financials
        except:
            financials = pd.DataFrame()
            
        try:
            balance_sheet = stock.balance_sheet
        except:
            balance_sheet = pd.DataFrame()
            
        try:
            cashflow = stock.cashflow
        except:
            cashflow = pd.DataFrame()
        
        return data, info, financials, balance_sheet, cashflow
        
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {str(e)}")
        return None, None, None, None, None

def calculate_technical_indicators(data):
    """Calculate various technical indicators with error handling"""
    try:
        df = data.copy()
        
        # Moving Averages
        df['MA_5'] = df['Close'].rolling(window=5).mean()
        df['MA_10'] = df['Close'].rolling(window=10).mean()
        df['MA_20'] = df['Close'].rolling(window=20).mean()
        df['MA_50'] = df['Close'].rolling(window=50).mean()
        
        # RSI with error handling
        try:
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
        except:
            df['RSI'] = np.nan
        
        # MACD with error handling
        try:
            df['EMA_12'] = df['Close'].ewm(span=12).mean()
            df['EMA_26'] = df['Close'].ewm(span=26).mean()
            df['MACD'] = df['EMA_12'] - df['EMA_26']
            df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        except:
            df['MACD'] = np.nan
            df['MACD_Signal'] = np.nan
        
        # Bollinger Bands with error handling
        try:
            df['BB_Middle'] = df['Close'].rolling(window=20).mean()
            bb_std = df['Close'].rolling(window=20).std()
            df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
            df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        except:
            df['BB_Middle'] = np.nan
            df['BB_Upper'] = np.nan
            df['BB_Lower'] = np.nan
        
        # Volume indicators
        df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
        df['Price_Change'] = df['Close'].pct_change()
        df['Volume_Change'] = df['Volume'].pct_change()
        
        return df
        
    except Exception as e:
        st.error(f"Error calculating technical indicators: {str(e)}")
        return data

def create_interactive_chart(data, symbol, chart_type="candlestick"):
    """Create interactive charts with hover information"""
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=(f'{symbol} Price Chart', 'Volume', 'RSI'),
        row_heights=[0.6, 0.2, 0.2]
    )
    
    if chart_type == "candlestick":
        # Candlestick chart - use simple hoverinfo for compatibility
        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name="OHLC",
                hoverinfo='x+y'
            ),
            row=1, col=1
        )
    else:
        # Line chart
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['Close'],
                mode='lines',
                name='Close Price',
                line=dict(color='#1f77b4', width=2),
                hoverinfo='x+y'
            ),
            row=1, col=1
        )
    
    # Add moving averages
    for ma_period, color in [(20, 'orange'), (50, 'red')]:
        if f'MA_{ma_period}' in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data[f'MA_{ma_period}'],
                    mode='lines',
                    name=f'MA {ma_period}',
                    line=dict(color=color, width=1),
                    hoverinfo='x+y+name'
                ),
                row=1, col=1
            )
    
    # Volume chart
    colors = ['red' if close < open else 'green' for close, open in zip(data['Close'], data['Open'])]
    fig.add_trace(
        go.Bar(
            x=data.index,
            y=data['Volume'],
            name='Volume',
            marker_color=colors,
            hoverinfo='x+y+name'
        ),
        row=2, col=1
    )
    
    # RSI chart
    if 'RSI' in data.columns:
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['RSI'],
                mode='lines',
                name='RSI',
                line=dict(color='purple'),
                hoverinfo='x+y+name'
            ),
            row=3, col=1
        )
        
        # Add RSI levels
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
    
    fig.update_layout(
        title=f"{symbol} Stock Analysis",
        xaxis_title="Date",
        template='plotly_white',
        height=800,
        showlegend=True,
        hovermode='x unified'
    )
    
    fig.update_xaxes(rangeslider_visible=False, row=1, col=1)
    
    return fig

def generate_stock_report(symbol, data, info, prediction_data=None, fmp_data=None):
    """Generate comprehensive stock analysis report"""
    if data is None or data.empty:
        return "Unable to generate report due to insufficient data."
    
    try:
        current_price = data['Close'].iloc[-1]
        prev_price = data['Close'].iloc[-2] if len(data) > 1 else current_price
        price_change = current_price - prev_price
        price_change_pct = (price_change / prev_price) * 100
        
        # Technical analysis with safe handling
        rsi = data['RSI'].iloc[-1] if 'RSI' in data.columns and not pd.isna(data['RSI'].iloc[-1]) else None
        ma_20 = data['MA_20'].iloc[-1] if 'MA_20' in data.columns and not pd.isna(data['MA_20'].iloc[-1]) else None
        
        # Format RSI and MA20 safely
        rsi_display = f"{rsi:.2f}" if rsi is not None else "N/A"
        ma_20_display = f"${ma_20:.2f}" if ma_20 is not None else "N/A"
        
        report = f"""
# Stock Analysis Report: {symbol}

## Executive Summary
**Current Price:** ${current_price:.2f}  
**Daily Change:** ${price_change:.2f} ({price_change_pct:+.2f}%)  
**Analysis Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Company Overview
"""
        
        # Use FMP data if available, otherwise use Yahoo Finance data
        if fmp_data and fmp_data.get('profile'):
            profile = fmp_data['profile'][0] if isinstance(fmp_data['profile'], list) and fmp_data['profile'] else {}
            
            company_name = profile.get('companyName', 'N/A')
            sector = profile.get('sector', 'N/A')
            industry = profile.get('industry', 'N/A')
            market_cap = profile.get('mktCap', 'N/A')
            website = profile.get('website', 'N/A')
            description = profile.get('description', 'N/A')
            
            # Format market cap safely
            if isinstance(market_cap, (int, float)) and market_cap > 0:
                if market_cap >= 1e12:
                    market_cap_display = f"${market_cap/1e12:.2f}T"
                elif market_cap >= 1e9:
                    market_cap_display = f"${market_cap/1e9:.2f}B"
                elif market_cap >= 1e6:
                    market_cap_display = f"${market_cap/1e6:.2f}M"
                else:
                    market_cap_display = f"${market_cap:,.0f}"
            else:
                market_cap_display = "N/A"
            
            report += f"""
**Company Name:** {company_name}  
**Sector:** {sector}  
**Industry:** {industry}  
**Market Cap:** {market_cap_display}  
**Website:** {website}  
**Description:** {description[:200]}...  
"""
        elif info and isinstance(info, dict):
            # Safe extraction of company info
            company_name = info.get('longName', 'N/A')
            sector = info.get('sector', 'N/A')
            industry = info.get('industry', 'N/A')
            market_cap = info.get('marketCap', 'N/A')
            pe_ratio = info.get('trailingPE', 'N/A')
            dividend_yield = info.get('dividendYield', 'N/A')
            
            # Format market cap safely
            if isinstance(market_cap, (int, float)) and market_cap > 0:
                market_cap_display = f"{market_cap:,}"
            else:
                market_cap_display = "N/A"
            
            # Format P/E ratio safely
            pe_display = f"{pe_ratio:.2f}" if isinstance(pe_ratio, (int, float)) else "N/A"
            
            # Format dividend yield safely
            dividend_display = f"{dividend_yield:.4f}" if isinstance(dividend_yield, (int, float)) else "N/A"
            
            report += f"""
**Company Name:** {company_name}  
**Sector:** {sector}  
**Industry:** {industry}  
**Market Cap:** {market_cap_display} USD  
**P/E Ratio:** {pe_display}  
**Dividend Yield:** {dividend_display}  
"""
        
        # Add FMP financial ratios if available
        if fmp_data and fmp_data.get('ratios'):
            ratios = fmp_data['ratios'][0] if isinstance(fmp_data['ratios'], list) and fmp_data['ratios'] else {}
            
            report += f"""
## Advanced Financial Ratios (FMP Data)
**Return on Equity:** {ratios.get('returnOnEquity', 'N/A')}  
**Return on Assets:** {ratios.get('returnOnAssets', 'N/A')}  
**Debt Ratio:** {ratios.get('debtRatio', 'N/A')}  
**Current Ratio:** {ratios.get('currentRatio', 'N/A')}  
**Quick Ratio:** {ratios.get('quickRatio', 'N/A')}  
**Gross Profit Margin:** {ratios.get('grossProfitMargin', 'N/A')}  
**Operating Profit Margin:** {ratios.get('operatingProfitMargin', 'N/A')}  
**Net Profit Margin:** {ratios.get('netProfitMargin', 'N/A')}  
"""
        
        # Calculate performance metrics safely
        week_return = 0
        month_return = 0
        
        try:
            if len(data) >= 5:
                week_return = ((current_price - data['Close'].iloc[-5]) / data['Close'].iloc[-5] * 100)
        except:
            week_return = 0
            
        try:
            if len(data) >= 22:
                month_return = ((current_price - data['Close'].iloc[-22]) / data['Close'].iloc[-22] * 100)
        except:
            month_return = 0
        
        # Calculate volatility safely
        volatility = 0
        try:
            volatility = data['Close'].pct_change().tail(30).std() * np.sqrt(252) * 100
        except:
            volatility = 0
        
        beta = info.get('beta', 'N/A') if info and isinstance(info, dict) else 'N/A'
        beta_display = f"{beta:.2f}" if isinstance(beta, (int, float)) else "N/A"
        
        report += f"""
## Technical Analysis
**Current RSI:** {rsi_display}  
**20-Day Moving Average:** {ma_20_display}  
**52-Week High:** ${data['High'].max():.2f}  
**52-Week Low:** ${data['Low'].min():.2f}  

## Price Performance
**1-Day Return:** {price_change_pct:+.2f}%  
**1-Week Return:** {week_return:+.2f}%  
**1-Month Return:** {month_return:+.2f}%  

## Risk Assessment
**Volatility (30-day):** {volatility:.2f}%  
**Beta:** {beta_display}  

## Investment Recommendation
"""
        
        # Simple recommendation logic with safe RSI handling
        if rsi is not None:
            if rsi > 70:
                report += "**Signal:** OVERBOUGHT - Consider taking profits or waiting for a pullback.\n"
            elif rsi < 30:
                report += "**Signal:** OVERSOLD - Potential buying opportunity if fundamentals are strong.\n"
            else:
                report += "**Signal:** NEUTRAL - Monitor for trend continuation or reversal signals.\n"
        else:
            report += "**Signal:** INSUFFICIENT DATA - Unable to generate RSI-based recommendation.\n"
        
        if prediction_data:
            try:
                predicted_price = prediction_data['predictions'][6]
                expected_return = ((predicted_price - current_price) / current_price * 100)
                r2_score = prediction_data['r2_score']
                
                report += f"""
## Price Prediction
**Predicted Price (7 days):** ${predicted_price:.2f}  
**Expected Return:** {expected_return:+.2f}%  
**Model Accuracy (R²):** {r2_score:.3f}  
"""
            except:
                report += """
## Price Prediction
**Status:** Prediction data unavailable or incomplete.  
"""
        
        report += """
## Disclaimer
This analysis is for educational purposes only and should not be considered as financial advice. 
Always conduct your own research and consult with financial professionals before making investment decisions.
"""
        
        return report
        
    except Exception as e:
        return f"Error generating report: {str(e)}. Please try again or contact support."

def enhanced_stock_chatbot(user_question, symbol, data, info, fmp_data=None):
    """Enhanced AI chatbot for stock queries with context awareness"""
    if data is None or data.empty:
        return "🤖 I don't have data for this stock yet. Please select a valid stock from the sidebar first!"
    
    user_question = user_question.lower().strip()
    
    try:
        current_price = data['Close'].iloc[-1]
        prev_price = data['Close'].iloc[-2] if len(data) > 1 else current_price
        price_change = current_price - prev_price
        price_change_pct = (price_change / prev_price) * 100
        
        # Enhanced context from multiple sources
        context = {
            'current_price': current_price,
            'price_change': price_change,
            'price_change_pct': price_change_pct,
            'volume': data['Volume'].iloc[-1],
            'high_52w': data['High'].max(),
            'low_52w': data['Low'].min(),
            'rsi': data['RSI'].iloc[-1] if 'RSI' in data.columns and not pd.isna(data['RSI'].iloc[-1]) else None,
            'ma_20': data['MA_20'].iloc[-1] if 'MA_20' in data.columns and not pd.isna(data['MA_20'].iloc[-1]) else None
        }
        
        # Add info data safely
        if info and isinstance(info, dict):
            context.update({
                'market_cap': info.get('marketCap'),
                'pe_ratio': info.get('trailingPE'),
                'dividend_yield': info.get('dividendYield'),
                'sector': info.get('sector', 'Unknown'),
                'company_name': info.get('longName', symbol),
                'beta': info.get('beta')
            })
        
        # Add FMP data if available
        if fmp_data:
            if fmp_data.get('profile') and isinstance(fmp_data['profile'], list) and fmp_data['profile']:
                profile = fmp_data['profile'][0]
                context['company_description'] = profile.get('description', '')
                context['website'] = profile.get('website', '')
                context['industry'] = profile.get('industry', '')
            
            if fmp_data.get('ratios') and isinstance(fmp_data['ratios'], list) and fmp_data['ratios']:
                ratios = fmp_data['ratios'][0]
                context['roe'] = ratios.get('returnOnEquity')
                context['roa'] = ratios.get('returnOnAssets')
                context['debt_ratio'] = ratios.get('debtRatio')
                context['current_ratio'] = ratios.get('currentRatio')
        
        # Store context for future use
        st.session_state.ai_context = context
        
        # Intelligent response generation
        if any(word in user_question for word in ['price', 'cost', 'trading', 'current']):
            if abs(price_change_pct) > 5:
                trend = "📈 significantly up" if price_change_pct > 0 else "📉 significantly down"
                response = f"💰 {context['company_name']} ({symbol}) is currently trading at **${current_price:.2f}**, which is {trend} by **{abs(price_change_pct):.2f}%** from yesterday. This is quite a notable move!"
            else:
                trend = "📈 up" if price_change_pct > 0 else "📉 down"
                response = f"💰 {context['company_name']} ({symbol}) is currently trading at **${current_price:.2f}**, {trend} **{abs(price_change_pct):.2f}%** from yesterday."
            
            # Add volume context
            if context['volume'] > 1e6:
                response += f" Trading volume is **{context['volume']/1e6:.1f}M shares** today."
            
            return response
        
        elif any(word in user_question for word in ['high', 'low', 'range', '52', 'week']):
            response = f"📊 **52-Week Trading Range for {symbol}:**\n"
            response += f"• **High:** ${context['high_52w']:.2f}\n"
            response += f"• **Low:** ${context['low_52w']:.2f}\n"
            response += f"• **Current:** ${current_price:.2f}\n\n"
            
            # Calculate position in range
            range_position = (current_price - context['low_52w']) / (context['high_52w'] - context['low_52w']) * 100
            
            if range_position > 80:
                response += "🔥 The stock is trading near its 52-week high! This could indicate strong momentum or potential resistance."
            elif range_position < 20:
                response += "💎 The stock is trading near its 52-week low. This could be a value opportunity or indicate ongoing challenges."
            else:
                response += f"📍 The stock is trading at about {range_position:.0f}% of its 52-week range."
            
            return response
        
        elif any(word in user_question for word in ['volume', 'trading', 'activity']):
            volume_display = f"{context['volume']/1e6:.1f}M" if context['volume'] >= 1e6 else f"{context['volume']/1e3:.1f}K"
            response = f"📊 **Trading Activity for {symbol}:**\n"
            response += f"• **Current Volume:** {volume_display} shares\n"
            
            # Volume analysis
            if 'Volume_MA' in data.columns:
                avg_volume = data['Volume_MA'].iloc[-1]
                if not pd.isna(avg_volume) and avg_volume > 0:
                    volume_ratio = context['volume'] / avg_volume
                    if volume_ratio > 2:
                        response += "🚀 Volume is **significantly higher** than average - indicating high interest!"
                    elif volume_ratio > 1.5:
                        response += "📈 Volume is **above average** - showing increased activity."
                    elif volume_ratio < 0.5:
                        response += "😴 Volume is **below average** - relatively quiet trading day."
                    else:
                        response += "📊 Volume is **around average** for this stock."
            
            return response
        
        elif any(word in user_question for word in ['rsi', 'overbought', 'oversold', 'technical']):
            if context['rsi'] is not None:
                response = f"📈 **Technical Analysis for {symbol}:**\n"
                response += f"• **RSI:** {context['rsi']:.1f}\n\n"
                
                if context['rsi'] > 70:
                    response += "⚠️ **OVERBOUGHT ZONE** - The stock may be due for a pullback. Consider taking profits or waiting for a better entry."
                elif context['rsi'] < 30:
                    response += "💎 **OVERSOLD ZONE** - The stock may be undervalued. Could be a buying opportunity if fundamentals are strong."
                elif context['rsi'] > 60:
                    response += "🔥 **Strong Momentum** - Bullish territory but watch for overbought conditions."
                elif context['rsi'] < 40:
                    response += "🐻 **Weak Momentum** - Bearish territory, consider the broader trend."
                else:
                    response += "📊 **Neutral Zone** - RSI indicates balanced buying and selling pressure."
                
                # Add moving average context
                if context['ma_20'] is not None:
                    if current_price > context['ma_20']:
                        response += f"\n\n📈 Price (${current_price:.2f}) is **above** the 20-day MA (${context['ma_20']:.2f}) - bullish signal."
                    else:
                        response += f"\n\n📉 Price (${current_price:.2f}) is **below** the 20-day MA (${context['ma_20']:.2f}) - bearish signal."
            else:
                response = "📊 I don't have enough technical data to calculate RSI for this stock yet. Try selecting a longer time period!"
            
            return response
        
        elif any(word in user_question for word in ['market cap', 'marketcap', 'valuation', 'size']):
            if context.get('market_cap') and isinstance(context['market_cap'], (int, float)):
                market_cap = context['market_cap']
                if market_cap >= 200e9:
                    size_category = "🏢 **Mega Cap** (>$200B)"
                elif market_cap >= 10e9:
                    size_category = "🏬 **Large Cap** ($10B-$200B)"
                elif market_cap >= 2e9:
                    size_category = "🏪 **Mid Cap** ($2B-$10B)"
                elif market_cap >= 300e6:
                    size_category = "🏠 **Small Cap** ($300M-$2B)"
                else:
                    size_category = "🏘️ **Micro Cap** (<$300M)"
                
                if market_cap >= 1e12:
                    market_cap_display = f"${market_cap/1e12:.2f}T"
                elif market_cap >= 1e9:
                    market_cap_display = f"${market_cap/1e9:.2f}B"
                elif market_cap >= 1e6:
                    market_cap_display = f"${market_cap/1e6:.2f}M"
                else:
                    market_cap_display = f"${market_cap:,.0f}"
                
                response = f"🏦 **Market Valuation for {symbol}:**\n"
                response += f"• **Market Cap:** {market_cap_display}\n"
                response += f"• **Category:** {size_category}\n\n"
                
                if market_cap >= 10e9:
                    response += "🔵 This is a large, established company with stable operations."
                elif market_cap >= 2e9:
                    response += "🟡 This is a mid-sized company with growth potential and moderate risk."
                else:
                    response += "🔴 This is a smaller company with higher growth potential but also higher risk."
            else:
                response = f"📊 Market cap data is not available for {symbol} in my current dataset."
            
            return response
        
        elif any(word in user_question for word in ['dividend', 'yield', 'income', 'payout']):
            if context.get('dividend_yield') and isinstance(context['dividend_yield'], (int, float)):
                yield_pct = context['dividend_yield'] * 100
                response = f"💰 **Dividend Information for {symbol}:**\n"
                response += f"• **Dividend Yield:** {yield_pct:.2f}%\n\n"
                
                if yield_pct > 4:
                    response += "🎯 **High Yield** - Great for income investors, but verify sustainability."
                elif yield_pct > 2:
                    response += "📊 **Moderate Yield** - Balanced approach to growth and income."
                elif yield_pct > 0:
                    response += "📈 **Low Yield** - Company likely focuses more on growth than dividends."
                else:
                    response += "📉 **No Current Dividend** - Growth-focused company."
            else:
                response = f"📊 {symbol} doesn't appear to pay dividends currently, or the data isn't available."
            
            return response
        
        elif any(word in user_question for word in ['pe', 'p/e', 'ratio', 'valuation', 'expensive', 'cheap']):
            if context.get('pe_ratio') and isinstance(context['pe_ratio'], (int, float)):
                pe = context['pe_ratio']
                response = f"📊 **Valuation Metrics for {symbol}:**\n"
                response += f"• **P/E Ratio:** {pe:.2f}\n\n"
                
                if pe > 30:
                    response += "⚠️ **High P/E** - Stock may be expensive or has high growth expectations."
                elif pe > 20:
                    response += "📊 **Moderate P/E** - Reasonable valuation, depends on growth prospects."
                elif pe > 15:
                    response += "💎 **Fair P/E** - Well-valued, good balance of price and earnings."
                elif pe > 0:
                    response += "🎯 **Low P/E** - Potentially undervalued or company facing challenges."
                else:
                    response += "📉 **Negative P/E** - Company currently unprofitable."
                
                # Add sector context if available
                if context.get('sector'):
                    response += f"\n\n🏢 **Sector:** {context['sector']}"
            else:
                response = f"📊 P/E ratio data is not available for {symbol} (may be unprofitable or data unavailable)."
            
            return response
        
        elif any(word in user_question for word in ['sector', 'industry', 'business', 'company', 'what does']):
            response = f"🏢 **Company Information for {symbol}:**\n"
            response += f"• **Company:** {context.get('company_name', symbol)}\n"
            
            if context.get('sector'):
                response += f"• **Sector:** {context['sector']}\n"
            
            if context.get('industry'):
                response += f"• **Industry:** {context['industry']}\n"
            
            if context.get('website'):
                response += f"• **Website:** {context['website']}\n"
            
            if context.get('company_description'):
                description = context['company_description'][:300]
                response += f"\n📝 **About:** {description}...\n"
            
            # Add some sector-specific insights
            sector = context.get('sector', '').lower()
            if 'technology' in sector:
                response += "\n💻 **Tech Sector** - Often higher growth potential but more volatile."
            elif 'financial' in sector:
                response += "\n🏦 **Financial Sector** - Sensitive to interest rates and economic cycles."
            elif 'healthcare' in sector:
                response += "\n🏥 **Healthcare Sector** - Defensive sector, less economic sensitivity."
            elif 'energy' in sector:
                response += "\n⚡ **Energy Sector** - Cyclical, commodity price sensitive."
            
            return response
        
        elif any(word in user_question for word in ['buy', 'sell', 'hold', 'invest', 'recommendation', 'should i']):
            response = f"🤖 **AI Analysis for {symbol}:**\n\n"
            
            # Collect signals
            signals = []
            
            # RSI Signal
            if context['rsi'] is not None:
                if context['rsi'] > 70:
                    signals.append(("🔴 Overbought", "Consider selling or waiting"))
                elif context['rsi'] < 30:
                    signals.append(("🟢 Oversold", "Potential buying opportunity"))
                else:
                    signals.append(("🟡 Neutral RSI", "No clear signal"))
            
            # Price vs MA Signal
            if context['ma_20'] is not None:
                if current_price > context['ma_20']:
                    signals.append(("🟢 Above MA20", "Bullish trend"))
                else:
                    signals.append(("🔴 Below MA20", "Bearish trend"))
            
            # Valuation Signal
            if context.get('pe_ratio') and isinstance(context['pe_ratio'], (int, float)):
                pe = context['pe_ratio']
                if pe > 25:
                    signals.append(("🔴 High P/E", "Potentially overvalued"))
                elif 15 <= pe <= 25:
                    signals.append(("🟡 Fair P/E", "Reasonably valued"))
                elif pe > 0:
                    signals.append(("🟢 Low P/E", "Potentially undervalued"))
            
            # Volume Signal
            if abs(price_change_pct) > 3 and context['volume'] > 1e6:
                signals.append(("⚡ High Volume Move", "Significant market interest"))
            
            # Display signals
            for signal, description in signals:
                response += f"• {signal}: {description}\n"
            
            response += f"\n⚠️ **Disclaimer:** This is not financial advice! Always do your own research and consult with financial professionals before making investment decisions.\n\n"
            response += "🎯 **Key Factors to Consider:**\n"
            response += "• Company fundamentals and financial health\n"
            response += "• Market conditions and economic environment\n"
            response += "• Your risk tolerance and investment timeline\n"
            response += "• Diversification in your portfolio"
            
            return response
        
        elif any(word in user_question for word in ['news', 'latest', 'recent', 'update', 'what happened']):
            if fmp_data and fmp_data.get('news'):
                response = f"📰 **Latest News for {symbol}:**\n\n"
                for i, news_item in enumerate(fmp_data['news'][:3]):  # Show top 3 news items
                    title = news_item.get('title', 'No title')
                    published = news_item.get('publishedDate', 'Unknown date')
                    response += f"{i+1}. **{title}**\n   *Published: {published[:10]}*\n\n"
                
                response += "💡 Check the News section for more detailed articles and links!"
            else:
                response = f"📰 I don't have recent news data for {symbol} available right now. Try checking financial news websites or the company's official announcements."
            
            return response
        
        elif any(word in user_question for word in ['compare', 'vs', 'versus', 'better', 'worse']):
            response = "🤔 **Comparison Analysis:**\n\n"
            response += f"I'd love to help you compare {symbol} with other stocks! However, I can only analyze one stock at a time in the current session.\n\n"
            response += "📊 **To compare stocks, you can:**\n"
            response += "• Select different stocks from the sidebar and note their key metrics\n"
            response += "• Compare P/E ratios, market caps, and technical indicators\n"
            response += "• Look at sector performance and industry trends\n\n"
            response += f"📈 **Current {symbol} Key Metrics:**\n"
            response += f"• Price: ${current_price:.2f} ({price_change_pct:+.2f}%)\n"
            if context.get('pe_ratio'):
                response += f"• P/E Ratio: {context['pe_ratio']:.2f}\n"
            if context.get('market_cap'):
                market_cap = context['market_cap']
                if market_cap >= 1e9:
                    response += f"• Market Cap: ${market_cap/1e9:.1f}B\n"
                else:
                    response += f"• Market Cap: ${market_cap/1e6:.1f}M\n"
            
            return response
        
        elif any(word in user_question for word in ['risk', 'risky', 'safe', 'volatile', 'volatility']):
            response = f"⚠️ **Risk Assessment for {symbol}:**\n\n"
            
            # Beta risk
            if context.get('beta') and isinstance(context['beta'], (int, float)):
                beta = context['beta']
                response += f"• **Beta:** {beta:.2f}\n"
                if beta > 1.5:
                    response += "  🔴 **High Risk** - More volatile than the market\n"
                elif beta > 1.0:
                    response += "  🟡 **Moderate Risk** - Slightly more volatile than market\n"
                elif beta > 0.5:
                    response += "  🟢 **Lower Risk** - Less volatile than the market\n"
                else:
                    response += "  💎 **Low Risk** - Very stable compared to market\n"
            
            # Price volatility
            if len(data) >= 30:
                volatility = data['Close'].pct_change().tail(30).std() * np.sqrt(252) * 100
                response += f"• **30-Day Volatility:** {volatility:.1f}%\n"
                if volatility > 40:
                    response += "  🔴 **Very High Volatility** - Expect large price swings\n"
                elif volatility > 25:
                    response += "  🟡 **Moderate Volatility** - Normal price fluctuations\n"
                else:
                    response += "  🟢 **Low Volatility** - Relatively stable price movement\n"
            
            # Market cap risk
            if context.get('market_cap'):
                market_cap = context['market_cap']
                if market_cap < 2e9:
                    response += "• 🔴 **Small Cap Risk** - Higher growth potential but more volatile\n"
                elif market_cap < 10e9:
                    response += "• 🟡 **Mid Cap** - Balanced risk/reward profile\n"
                else:
                    response += "• 🟢 **Large Cap** - More stable, established company\n"
            
            response += "\n🎯 **Risk Management Tips:**\n"
            response += "• Diversify across sectors and asset classes\n"
            response += "• Only invest what you can afford to lose\n"
            response += "• Consider your investment timeline\n"
            response += "• Regular portfolio rebalancing"
            
            return response
        
        elif any(word in user_question for word in ['help', 'what can you do', 'capabilities', 'features']):
            response = "🤖 **AI Stock Assistant Capabilities:**\n\n"
            response += "I can help you analyze stocks in multiple ways:\n\n"
            response += "📊 **Price & Performance:**\n"
            response += "• Current price, changes, and trends\n"
            response += "• 52-week high/low analysis\n"
            response += "• Volume and trading activity\n\n"
            response += "📈 **Technical Analysis:**\n"
            response += "• RSI and overbought/oversold conditions\n"
            response += "• Moving average signals\n"
            response += "• MACD and momentum indicators\n\n"
            response += "🏢 **Company Information:**\n"
            response += "• Business description and sector\n"
            response += "• Market cap and company size\n"
            response += "• Financial ratios and valuation\n\n"
            response += "⚠️ **Risk Assessment:**\n"
            response += "• Volatility and beta analysis\n"
            response += "• Investment risk evaluation\n"
            response += "• Portfolio considerations\n\n"
            response += "📰 **News & Updates:**\n"
            response += "• Latest company news\n"
            response += "• Market developments\n\n"
            response += "💡 **Just ask me questions like:**\n"
            response += "• 'What's the current price?'\n"
            response += "• 'Is this stock overbought?'\n"
            response += "• 'What sector is this company in?'\n"
            response += "• 'Should I buy this stock?'\n"
            response += "• 'How risky is this investment?'"
            
            return response
        
        else:
            # Default response with context
            response = f"🤖 I'm here to help with {symbol} analysis!\n\n"
            response += f"📊 **Quick Stats:**\n"
            response += f"• **Current Price:** ${current_price:.2f} ({price_change_pct:+.2f}%)\n"
            response += f"• **Company:** {context.get('company_name', symbol)}\n"
            if context.get('sector'):
                response += f"• **Sector:** {context['sector']}\n"
            
            response += f"\n💡 **Try asking me about:**\n"
            response += "• Price and performance\n"
            response += "• Technical indicators (RSI, moving averages)\n"
            response += "• Company information and sector\n"
            response += "• Investment recommendations\n"
            response += "• Risk assessment\n"
            response += "• Latest news and updates\n\n"
            response += "🗣️ *Ask me anything about this stock!*"
            
            return response
    
    except Exception as e:
        return f"🤖 Sorry, I encountered an error while analyzing {symbol}. Please try asking your question differently, or make sure you have selected a valid stock. Error details: {str(e)}"

# Main App Layout
st.markdown('<h1 class="main-header">Advanced Stock Analysis Platform</h1>', unsafe_allow_html=True)

# Sidebar
st.sidebar.header("Stock Selection")

# Country selection
selected_country = st.sidebar.selectbox(
    "Select Country",
    options=list(STOCK_DATABASE.keys()),
    index=0
)

# Stock selection based on country
stocks_in_country = STOCK_DATABASE[selected_country]
stock_options = [f"{symbol} - {name}" for symbol, name in stocks_in_country.items()]

selected_stock_display = st.sidebar.selectbox(
    f"Select Stock ({selected_country})",
    options=stock_options,
    index=0
)

selected_symbol = selected_stock_display.split(" - ")[0]

# Analysis parameters
st.sidebar.subheader("Analysis Parameters")
time_period = st.sidebar.selectbox(
    "Time Period",
    options=["1mo", "3mo", "6mo", "1y", "2y", "5y", "max"],
    index=3
)

prediction_days = st.sidebar.slider("Prediction Days", 1, 30, 7)

chart_type = st.sidebar.radio(
    "Chart Type",
    options=["candlestick", "line"],
    index=0
)

view_mode = st.sidebar.radio(
    "View Mode",
    options=["Graphical", "Tabular", "Both"],
    index=2
)

# Main content
if selected_symbol:
    # Load data
    with st.spinner(f"📊 Fetching comprehensive data for {selected_symbol}..."):
        # Yahoo Finance data
        data, info, financials, balance_sheet, cashflow = fetch_stock_data(selected_symbol, time_period)
        
        # FMP data
        fmp_data = get_financial_data_fmp(selected_symbol)
    
    if data is not None and not data.empty:
        # Calculate technical indicators
        data_with_indicators = calculate_technical_indicators(data)
        
        # Store in session state for chatbot
        st.session_state.analysis_data = {
            'symbol': selected_symbol,
            'data': data_with_indicators,
            'info': info,
            'fmp_data': fmp_data
        }
        
        # Key metrics
        current_price = data['Close'].iloc[-1]
        prev_price = data['Close'].iloc[-2] if len(data) > 1 else current_price
        price_change = current_price - prev_price
        price_change_pct = (price_change / prev_price) * 100
        
        # Display key metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Current Price", f"${current_price:.2f}", f"{price_change_pct:+.2f}%")
        
        with col2:
            if info and isinstance(info, dict):
                market_cap = info.get('marketCap', 0)
                if market_cap and isinstance(market_cap, (int, float)):
                    if market_cap >= 1e12:
                        market_cap_display = f"${market_cap/1e12:.2f}T"
                    elif market_cap >= 1e9:
                        market_cap_display = f"${market_cap/1e9:.2f}B"
                    elif market_cap >= 1e6:
                        market_cap_display = f"${market_cap/1e6:.2f}M"
                    else:
                        market_cap_display = f"${market_cap:,.0f}"
                else:
                    market_cap_display = "N/A"
                st.metric("Market Cap", market_cap_display)
            else:
                st.metric("Market Cap", "N/A")
        
        with col3:
            volume = data['Volume'].iloc[-1]
            volume_display = f"{volume/1e6:.2f}M" if volume >= 1e6 else f"{volume/1e3:.2f}K"
            st.metric("Volume", volume_display)
        
        with col4:
            high_52w = data['High'].max()
            st.metric("52W High", f"${high_52w:.2f}")
        
        with col5:
            low_52w = data['Low'].min()
            st.metric("52W Low", f"${low_52w:.2f}")
        
        # Tabs for different sections
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["📈 Chart Analysis", "📊 Data Table", "🔮 Predictions", "📄 Financial Report", "🤖 AI Assistant", "📰 News & Congress", "🏛️ Advanced Analytics"])
        
        with tab1:
            st.subheader(f"{selected_symbol} Price Analysis")
            
            if view_mode in ["Graphical", "Both"]:
                # Interactive chart
                fig = create_interactive_chart(data_with_indicators, selected_symbol, chart_type)
                st.plotly_chart(fig, use_container_width=True)
            
            if view_mode in ["Tabular", "Both"]:
                st.subheader("Recent Price Data")
                display_data = data_with_indicators[['Open', 'High', 'Low', 'Close', 'Volume', 'MA_20', 'RSI']].tail(20)
                display_data = display_data.round(2)
                st.dataframe(display_data, use_container_width=True)
        
        with tab2:
            st.subheader("Detailed Stock Data")
            
            # Technical indicators table
            tech_indicators = data_with_indicators[['Close', 'MA_5', 'MA_10', 'MA_20', 'MA_50', 'RSI', 'MACD', 'MACD_Signal']].tail(50)
            tech_indicators = tech_indicators.round(4)
            
            st.write("**Technical Indicators (Last 50 Days)**")
            st.dataframe(tech_indicators, use_container_width=True)
            
            # Company fundamentals
            if info:
                st.subheader("Company Fundamentals")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fundamentals_1 = {
                        "Metric": ["P/E Ratio", "PEG Ratio", "Price to Book", "Price to Sales", "Enterprise Value"],
                        "Value": [
                            info.get('trailingPE', 'N/A'),
                            info.get('pegRatio', 'N/A'),
                            info.get('priceToBook', 'N/A'),
                            info.get('priceToSalesTrailing12Months', 'N/A'),
                            info.get('enterpriseValue', 'N/A')
                        ]
                    }
                    st.dataframe(pd.DataFrame(fundamentals_1), use_container_width=True)
                
                with col2:
                    fundamentals_2 = {
                        "Metric": ["ROE", "ROA", "Debt to Equity", "Current Ratio", "Quick Ratio"],
                        "Value": [
                            info.get('returnOnEquity', 'N/A'),
                            info.get('returnOnAssets', 'N/A'),
                            info.get('debtToEquity', 'N/A'),
                            info.get('currentRatio', 'N/A'),
                            info.get('quickRatio', 'N/A')
                        ]
                    }
                    st.dataframe(pd.DataFrame(fundamentals_2), use_container_width=True)
        
        with tab3:
            st.subheader("Price Predictions")
            
            # Machine Learning Prediction
            with st.spinner("🤖 Training prediction model..."):
                # Prepare features
                feature_data = data_with_indicators[['Close', 'Volume', 'MA_20', 'RSI', 'MACD']].dropna()
                
                if len(feature_data) > 60:
                    # Create features and targets
                    lookback = 30
                    X, y = [], []
                    
                    for i in range(lookback, len(feature_data)):
                        X.append(feature_data.iloc[i-lookback:i].values.flatten())
                        y.append(feature_data['Close'].iloc[i])
                    
                    X, y = np.array(X), np.array(y)
                    
                    # Split data
                    split_idx = int(len(X) * 0.8)
                    X_train, X_test = X[:split_idx], X[split_idx:]
                    y_train, y_test = y[:split_idx], y[split_idx:]
                    
                    # Train models
                    lr_model = LinearRegression()
                    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
                    
                    lr_model.fit(X_train, y_train)
                    rf_model.fit(X_train, y_train)
                    
                    # Predictions
                    lr_pred = lr_model.predict(X_test)
                    rf_pred = rf_model.predict(X_test)
                    
                    # Metrics
                    lr_r2 = r2_score(y_test, lr_pred)
                    rf_r2 = r2_score(y_test, rf_pred)
                    
                    lr_mae = mean_absolute_error(y_test, lr_pred)
                    rf_mae = mean_absolute_error(y_test, rf_pred)
                    
                    # Display model performance
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Linear Regression Model")
                        st.metric("R² Score", f"{lr_r2:.3f}")
                        st.metric("Mean Absolute Error", f"${lr_mae:.2f}")
                    
                    with col2:
                        st.subheader("Random Forest Model")
                        st.metric("R² Score", f"{rf_r2:.3f}")
                        st.metric("Mean Absolute Error", f"${rf_mae:.2f}")
                    
                    # Future predictions
                    best_model = rf_model if rf_r2 > lr_r2 else lr_model
                    model_name = "Random Forest" if rf_r2 > lr_r2 else "Linear Regression"
                    
                    st.subheader(f"Future Predictions ({model_name} Model)")
                    
                    # Generate future predictions
                    last_sequence = X[-1].reshape(1, -1)
                    future_predictions = []
                    
                    for _ in range(prediction_days):
                        pred = best_model.predict(last_sequence)[0]
                        future_predictions.append(pred)
                        
                        # Update sequence (simplified)
                        new_features = np.array([pred, feature_data['Volume'].iloc[-1], pred, 50, 0]).reshape(1, -1)
                        last_sequence = np.roll(last_sequence, -5, axis=1)
                        last_sequence[0, -5:] = new_features
                    
                    # Create prediction dataframe
                    future_dates = [data.index[-1] + timedelta(days=i+1) for i in range(prediction_days)]
                    pred_df = pd.DataFrame({
                        'Date': future_dates,
                        'Predicted Price': [f"${p:.2f}" for p in future_predictions],
                        'Change from Current': [f"${p - current_price:.2f}" for p in future_predictions],
                        'Percentage Change': [f"{((p - current_price) / current_price * 100):+.2f}%" for p in future_predictions]
                    })
                    
                    st.dataframe(pred_df, use_container_width=True)
                    
                    # Prediction chart
                    fig_pred = go.Figure()
                    
                    # Historical data (last 30 days)
                    recent_data = data.tail(30)
                    fig_pred.add_trace(go.Scatter(
                        x=recent_data.index,
                        y=recent_data['Close'],
                        mode='lines',
                        name='Historical Price',
                        line=dict(color='blue')
                    ))
                    
                    # Predictions
                    fig_pred.add_trace(go.Scatter(
                        x=future_dates,
                        y=future_predictions,
                        mode='lines+markers',
                        name='Predicted Price',
                        line=dict(color='red', dash='dash')
                    ))
                    
                    fig_pred.update_layout(
                        title=f"Price Prediction for {selected_symbol}",
                        xaxis_title="Date",
                        yaxis_title="Price ($)",
                        template='plotly_white',
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig_pred, use_container_width=True)
                    
                    # Store prediction data for report
                    prediction_data = {
                        'predictions': future_predictions,
                        'r2_score': rf_r2 if rf_r2 > lr_r2 else lr_r2,
                        'model': model_name
                    }
                    
                else:
                    st.warning("⚠️ Insufficient data for prediction model. Please select a longer time period.")
                    prediction_data = None
        
        with tab4:
            st.subheader("📄 Comprehensive Financial Report")
            
            # Generate and display report
            report = generate_stock_report(
                selected_symbol, 
                data_with_indicators, 
                info, 
                prediction_data if 'prediction_data' in locals() else None,
                fmp_data
            )
            
            st.markdown(report)
            
            # Download report button
            st.download_button(
                label="📥 Download Report as Text",
                data=report,
                file_name=f"{selected_symbol}_analysis_report_{datetime.now().strftime('%Y%m%d')}.txt",
                mime="text/plain"
            )
            
            # Additional financial charts
            if financials is not None and not financials.empty:
                st.subheader("📊 Financial Statements Overview")
                
                # Revenue and profit chart
                if 'Total Revenue' in financials.index:
                    revenue_data = financials.loc['Total Revenue'].dropna()
                    
                    fig_fin = go.Figure()
                    fig_fin.add_trace(go.Bar(
                        x=revenue_data.index,
                        y=revenue_data.values,
                        name='Total Revenue',
                        marker_color='lightblue'
                    ))
                    
                    fig_fin.update_layout(
                        title="Annual Revenue Trend",
                        xaxis_title="Year",
                        yaxis_title="Revenue ($)",
                        template='plotly_white'
                    )
                    
                    st.plotly_chart(fig_fin, use_container_width=True)
        
        with tab5:
            st.subheader("🤖 AI Stock Assistant")
            
            # Enhanced Chat interface with better styling and context awareness
            st.markdown("### 💬 Intelligent Stock Analysis Chat")
            
            # Context display - show what stock we're analyzing
            if st.session_state.analysis_data:
                current_symbol = st.session_state.analysis_data['symbol']
                current_price = st.session_state.analysis_data['data']['Close'].iloc[-1]
                price_change_pct = ((current_price - st.session_state.analysis_data['data']['Close'].iloc[-2]) / st.session_state.analysis_data['data']['Close'].iloc[-2] * 100) if len(st.session_state.analysis_data['data']) > 1 else 0
                
                st.info(f"🎯 **Currently Analyzing:** {current_symbol} | **Price:** ${current_price:.2f} ({price_change_pct:+.2f}%)")
            
            # Enhanced chat history display with better conversation flow
            chat_container = st.container()
            with chat_container:
                if st.session_state.chat_history:
                    st.markdown("**📜 Conversation History:**")
                    
                    # Show recent conversations (last 10 to avoid overwhelming)
                    recent_chats = st.session_state.chat_history[-10:] if len(st.session_state.chat_history) > 10 else st.session_state.chat_history
                    
                    for i, (question, answer) in enumerate(recent_chats):
                        # User message with improved styling
                        st.markdown(f"""
                        <div style="background-color: #e3f2fd; color: black; padding: 10px; border-radius: 10px; margin: 5px 0; border-left: 4px solid #1976d2;">
                            <strong>👤 You:</strong> {question}
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # AI response with enhanced styling
                        st.markdown(f"""
                        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 15px; border-radius: 10px; margin: 10px 0; border-left: 4px solid #4527a0;">
                            <strong>🤖 AI Assistant:</strong><br><br>{answer}
                        </div>
                        """, unsafe_allow_html=True)
                        
                        if i < len(recent_chats) - 1:  # Don't add separator after last item
                            st.markdown("---")
                
                else:
                    st.markdown("""
                    <div style="text-align: center; padding: 30px; background-color: #f8f9fa; border-radius: 10px; border: 2px dashed #dee2e6;">
                        <h4>👋 Welcome to the AI Stock Assistant!</h4>
                        <p>I'm ready to help you analyze your selected stock. Ask me anything about:</p>
                        <div style="display: flex; flex-wrap: wrap; justify-content: center; gap: 10px; margin-top: 15px;">
                            <span style="background: #e3f2fd; padding: 5px 10px; border-radius: 15px; font-size: 0.9em;">📊 Price Analysis</span>
                            <span style="background: #e8f5e8; padding: 5px 10px; border-radius: 15px; font-size: 0.9em;">📈 Technical Indicators</span>
                            <span style="background: #fff3e0; padding: 5px 10px; border-radius: 15px; font-size: 0.9em;">🏢 Company Info</span>
                            <span style="background: #fce4ec; padding: 5px 10px; border-radius: 15px; font-size: 0.9em;">💰 Investment Advice</span>
                        </div>
                        <p style="margin-top: 15px; font-style: italic;">Start by asking a question below! 👇</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Enhanced chat input with better UX and suggestions
            st.markdown("### ✍️ Ask Your Question")
            
            # Create input area with better layout
            input_col1, input_col2 = st.columns([4, 1])
            
            with input_col1:
                user_question = st.text_input(
                    "💭 What would you like to know about the stock?", 
                    key="chat_input", 
                    placeholder="e.g., What's the current price and trend? Is this stock overbought? Should I buy now?",
                    help="Ask me anything about price, technical analysis, company info, risks, or investment advice!"
                )
            
            with input_col2:
                send_button = st.button("📤 Ask AI", key="send_btn", use_container_width=True, type="primary")
            
            # Process chat input with enhanced feedback
            if send_button and user_question:
                if st.session_state.analysis_data:
                    with st.spinner("🤖 AI is analyzing and thinking..."):
                        # Add a small delay for better UX (makes it feel more thoughtful)
                        import time
                        time.sleep(0.5)
                        
                        bot_response = enhanced_stock_chatbot(
                            user_question,
                            st.session_state.analysis_data['symbol'],
                            st.session_state.analysis_data['data'],
                            st.session_state.analysis_data['info'],
                            st.session_state.analysis_data['fmp_data']
                        )
                    
                    # Add to chat history with timestamp
                    st.session_state.chat_history.append((user_question, bot_response))
                    
                    # Show success message
                    st.success("✅ Response generated! Check the conversation above.")
                    st.rerun()
                else:
                    st.error("❌ Please select a stock first to start the conversation.")
            
            # Enhanced action buttons with better organization
            st.markdown("### 🎛️ Quick Actions")
            
            action_col1, action_col2, action_col3, action_col4 = st.columns(4)
            
            with action_col1:
                if st.button("🗑️ Clear Chat", help="Clear all conversation history"):
                    st.session_state.chat_history = []
                    st.success("Chat history cleared!")
                    st.rerun()
            
            with action_col2:
                if st.button("❓ Help Guide", help="Show AI capabilities and sample questions"):
                    if st.session_state.analysis_data:
                        help_response = enhanced_stock_chatbot(
                            "help",
                            st.session_state.analysis_data['symbol'],
                            st.session_state.analysis_data['data'],
                            st.session_state.analysis_data['info'],
                            st.session_state.analysis_data['fmp_data']
                        )
                        st.session_state.chat_history.append(("What can you help me with?", help_response))
                        st.rerun()
            
            with action_col3:
                if st.button("📊 Stock Summary", help="Get comprehensive analysis summary"):
                    if st.session_state.analysis_data:
                        summary_response = enhanced_stock_chatbot(
                            "comprehensive summary",
                            st.session_state.analysis_data['symbol'],
                            st.session_state.analysis_data['data'],
                            st.session_state.analysis_data['info'],
                            st.session_state.analysis_data['fmp_data']
                        )
                        st.session_state.chat_history.append(("Give me a comprehensive stock summary", summary_response))
                        st.rerun()
            
            with action_col4:
                if st.button("🎯 Investment Analysis", help="Get detailed investment recommendation"):
                    if st.session_state.analysis_data:
                        investment_response = enhanced_stock_chatbot(
                            "should i invest in this stock",
                            st.session_state.analysis_data['symbol'],
                            st.session_state.analysis_data['data'],
                            st.session_state.analysis_data['info'],
                            st.session_state.analysis_data['fmp_data']
                        )
                        st.session_state.chat_history.append(("Should I invest in this stock?", investment_response))
                        st.rerun()
            
            # Enhanced quick questions with categorization and better layout
            st.markdown("### ⚡ Quick Questions by Category")
            
            # Create expandable sections for different question categories
            with st.expander("📊 **Price & Performance Questions**", expanded=False):
                price_questions = [
                    "What's the current price and how is it performing?",
                    "How has the stock performed in the past week?",
                    "What's the 52-week high and low?",
                    "How is the trading volume today?"
                ]
                
                price_cols = st.columns(2)
                for i, question in enumerate(price_questions):
                    with price_cols[i % 2]:
                        if st.button(f"💡 {question}", key=f"price_{i}"):
                            if st.session_state.analysis_data:
                                with st.spinner("🤖 Processing..."):
                                    bot_response = enhanced_stock_chatbot(
                                        question,
                                        st.session_state.analysis_data['symbol'],
                                        st.session_state.analysis_data['data'],
                                        st.session_state.analysis_data['info'],
                                        st.session_state.analysis_data['fmp_data']
                                    )
                                    st.session_state.chat_history.append((question, bot_response))
                                    st.rerun()
            
            with st.expander("📈 **Technical Analysis Questions**", expanded=False):
                technical_questions = [
                    "Is this stock overbought or oversold?",
                    "What do the moving averages indicate?",
                    "Show me the technical indicators summary",
                    "What's the RSI telling us?"
                ]
                
                tech_cols = st.columns(2)
                for i, question in enumerate(technical_questions):
                    with tech_cols[i % 2]:
                        if st.button(f"📈 {question}", key=f"tech_{i}"):
                            if st.session_state.analysis_data:
                                with st.spinner("🤖 Processing..."):
                                    bot_response = enhanced_stock_chatbot(
                                        question,
                                        st.session_state.analysis_data['symbol'],
                                        st.session_state.analysis_data['data'],
                                        st.session_state.analysis_data['info'],
                                        st.session_state.analysis_data['fmp_data']
                                    )
                                    st.session_state.chat_history.append((question, bot_response))
                                    st.rerun()
            
            with st.expander("🏢 **Company & Business Questions**", expanded=False):
                company_questions = [
                    "Tell me about this company and its business",
                    "What sector and industry is this?",
                    "What's the market cap and company size?",
                    "Does this company pay dividends?"
                ]
                
                company_cols = st.columns(2)
                for i, question in enumerate(company_questions):
                    with company_cols[i % 2]:
                        if st.button(f"🏢 {question}", key=f"company_{i}"):
                            if st.session_state.analysis_data:
                                with st.spinner("🤖 Processing..."):
                                    bot_response = enhanced_stock_chatbot(
                                        question,
                                        st.session_state.analysis_data['symbol'],
                                        st.session_state.analysis_data['data'],
                                        st.session_state.analysis_data['info'],
                                        st.session_state.analysis_data['fmp_data']
                                    )
                                    st.session_state.chat_history.append((question, bot_response))
                                    st.rerun()
            
            with st.expander("💰 **Investment & Risk Questions**", expanded=False):
                investment_questions = [
                    "Should I buy this stock right now?",
                    "How risky is this investment?",
                    "What are the key risks to consider?",
                    "Is this stock good for my portfolio?"
                ]
                
                invest_cols = st.columns(2)
                for i, question in enumerate(investment_questions):
                    with invest_cols[i % 2]:
                        if st.button(f"💰 {question}", key=f"invest_{i}"):
                            if st.session_state.analysis_data:
                                with st.spinner("🤖 Processing..."):
                                    bot_response = enhanced_stock_chatbot(
                                        question,
                                        st.session_state.analysis_data['symbol'],
                                        st.session_state.analysis_data['data'],
                                        st.session_state.analysis_data['info'],
                                        st.session_state.analysis_data['fmp_data']
                                    )
                                    st.session_state.chat_history.append((question, bot_response))
                                    st.rerun()
            
            # Enhanced conversation statistics and insights
            if st.session_state.chat_history:
                st.markdown("---")
                st.markdown("### 📈 Conversation Insights")
                
                stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
                
                with stats_col1:
                    st.metric("Total Questions", len(st.session_state.chat_history))
                
                with stats_col2:
                    # Count different types of questions asked
                    question_types = []
                    for q, _ in st.session_state.chat_history:
                        if any(word in q.lower() for word in ['price', 'cost', 'trading']):
                            question_types.append('Price')
                        elif any(word in q.lower() for word in ['rsi', 'technical', 'overbought']):
                            question_types.append('Technical')
                        elif any(word in q.lower() for word in ['company', 'business', 'sector']):
                            question_types.append('Company')
                        elif any(word in q.lower() for word in ['buy', 'sell', 'invest', 'risk']):
                            question_types.append('Investment')
                        else:
                            question_types.append('Other')
                    
                    most_common = max(set(question_types), key=question_types.count) if question_types else "None"
                    st.metric("Top Interest", most_common)
                
                with stats_col3:
                    # Show current stock being analyzed
                    current_stock = st.session_state.analysis_data['symbol'] if st.session_state.analysis_data else "None"
                    st.metric("Current Stock", current_stock)
                
                with stats_col4:
                    # Calculate session duration
                    if 'session_start' not in st.session_state:
                        st.session_state.session_start = datetime.now()
                    
                    duration = datetime.now() - st.session_state.session_start
                    minutes = int(duration.total_seconds() / 60)
                    st.metric("Session Time", f"{minutes}m")
            
            # Enhanced disclaimer and tips
            st.markdown("---")
            st.markdown("""
            <div style="background-color: #fff3cd; border: 1px solid #ffeeba; border-radius: 8px; padding: 15px; margin-top: 20px;">
                <h4 style="color: #856404; margin-top: 0;">⚠️ Important Disclaimer & Tips</h4>
                <div style="color: #856404;">
                    <strong>🎓 Educational Purpose:</strong> This AI assistant provides educational analysis only, not financial advice.<br>
                    <strong>🔍 Do Your Research:</strong> Always conduct independent research and due diligence.<br>
                    <strong>👨‍💼 Consult Professionals:</strong> Consider consulting with qualified financial advisors.<br>
                    <strong>💡 Best Practices:</strong> Use this as one tool in your research toolkit, not the only source.<br><br>
                    
                    <strong>🚀 Pro Tips for Better Interactions:</strong><br>
                    • Be specific with your questions for more detailed answers<br>
                    • Ask follow-up questions to dive deeper into topics<br>
                    • Use the quick question buttons for common queries<br>
                    • Review the conversation history for comprehensive insights
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with tab6:
            st.subheader("📰 Latest News & Congressional Updates")
            
            # Company News from FMP
            if fmp_data and fmp_data.get('news'):
                st.markdown("### 📈 Company News")
                
                for i, news_item in enumerate(fmp_data['news'][:5]):
                    title = news_item.get('title', 'No title available')
                    published = news_item.get('publishedDate', 'Unknown date')
                    url = news_item.get('url', '')
                    text = news_item.get('text', 'No summary available')
                    
                    with st.container():
                        st.markdown(f"""
                        <div class="news-item">
                            <h4>{title}</h4>
                            <p><em>Published: {published}</em></p>
                            <p>{text[:200]}...</p>
                            <a href="{url}" target="_blank">Read Full Article</a>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.info("📰 News data not available for this stock")
            
            # Congressional Bills
            st.markdown("### 🏛️ Relevant Congressional Activity")
            
            congress_bills = get_congress_data()
            
            if congress_bills:
                st.info(f"Found {len(congress_bills)} relevant financial/economic bills in Congress")
                
                for bill in congress_bills:
                    title = bill.get('title', 'No title')
                    bill_number = bill.get('number', 'Unknown')
                    bill_type = bill.get('type', 'Unknown')
                    congress = bill.get('congress', 'Unknown')
                    update_date = bill.get('updateDate', 'Unknown')
                    
                    st.markdown(f"""
                    **{bill_type} {bill_number}** - Congress {congress}  
                    **Title:** {title}  
                    **Last Updated:** {update_date}  
                    ---
                    """)
            else:
                st.warning("🏛️ Unable to fetch Congressional data at this time")
        
        with tab7:
            st.subheader("🏛️ Advanced Financial Analytics")
            
            # Enhanced FMP Data Display
            if fmp_data:
                # Company Profile
                if fmp_data.get('profile'):
                    profile = fmp_data['profile'][0] if isinstance(fmp_data['profile'], list) and fmp_data['profile'] else {}
                    
                    st.markdown("### 🏢 Company Profile (FMP Enhanced Data)")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Market Cap", f"${profile.get('mktCap', 0)/1e9:.2f}B" if profile.get('mktCap') else "N/A")
                        full_time_employees = profile.get('fullTimeEmployees')
                        if full_time_employees and isinstance(full_time_employees, (int, float)):
                            st.metric("Full Time Employees", f"{int(full_time_employees):,}")
                        else:
                            st.metric("Full Time Employees", "N/A")
                        st.metric("IPO Date", profile.get('ipoDate', 'N/A'))
                        st.metric("Exchange", profile.get('exchangeShortName', 'N/A'))
                    
                    with col2:
                        st.metric("Country", profile.get('country', 'N/A'))
                        st.metric("Currency", profile.get('currency', 'N/A'))
                        st.metric("Is ETF", "Yes" if profile.get('isEtf') else "No")
                        st.metric("Is Fund", "Yes" if profile.get('isFund') else "No")
                    
                    if profile.get('description'):
                        st.markdown("**Company Description:**")
                        st.write(profile['description'][:500] + "...")
                
                # Advanced Ratios
                if fmp_data.get('ratios'):
                    ratios = fmp_data['ratios'][0] if isinstance(fmp_data['ratios'], list) and fmp_data['ratios'] else {}
                    
                    st.markdown("### 📊 Advanced Financial Ratios")
                    
                    # Profitability Ratios
                    st.markdown("#### 💰 Profitability Ratios")
                    prof_col1, prof_col2, prof_col3, prof_col4 = st.columns(4)
                    
                    with prof_col1:
                        st.metric("Gross Profit Margin", f"{ratios.get('grossProfitMargin', 0)*100:.2f}%" if ratios.get('grossProfitMargin') else "N/A")
                    
                    with prof_col2:
                        st.metric("Operating Profit Margin", f"{ratios.get('operatingProfitMargin', 0)*100:.2f}%" if ratios.get('operatingProfitMargin') else "N/A")
                    
                    with prof_col3:
                        st.metric("Net Profit Margin", f"{ratios.get('netProfitMargin', 0)*100:.2f}%" if ratios.get('netProfitMargin') else "N/A")
                    
                    with prof_col4:
                        st.metric("Return on Equity", f"{ratios.get('returnOnEquity', 0)*100:.2f}%" if ratios.get('returnOnEquity') else "N/A")
                    
                    # Liquidity Ratios
                    st.markdown("#### 💧 Liquidity Ratios")
                    liq_col1, liq_col2, liq_col3, liq_col4 = st.columns(4)
                    
                    with liq_col1:
                        st.metric("Current Ratio", f"{ratios.get('currentRatio', 0):.2f}" if ratios.get('currentRatio') else "N/A")
                    
                    with liq_col2:
                        st.metric("Quick Ratio", f"{ratios.get('quickRatio', 0):.2f}" if ratios.get('quickRatio') else "N/A")
                    
                    with liq_col3:
                        st.metric("Cash Ratio", f"{ratios.get('cashRatio', 0):.2f}" if ratios.get('cashRatio') else "N/A")
                    
                    with liq_col4:
                        st.metric("Operating Cash Flow Ratio", f"{ratios.get('operatingCashFlowRatio', 0):.2f}" if ratios.get('operatingCashFlowRatio') else "N/A")
                    
                    # Leverage Ratios
                    st.markdown("#### ⚖️ Leverage Ratios")
                    lev_col1, lev_col2, lev_col3, lev_col4 = st.columns(4)
                    
                    with lev_col1:
                        st.metric("Debt Ratio", f"{ratios.get('debtRatio', 0):.3f}" if ratios.get('debtRatio') else "N/A")
                    
                    with lev_col2:
                        st.metric("Debt to Equity", f"{ratios.get('debtEquityRatio', 0):.3f}" if ratios.get('debtEquityRatio') else "N/A")
                    
                    with lev_col3:
                        st.metric("Long Term Debt to Capitalization", f"{ratios.get('longTermDebtToCapitalization', 0):.3f}" if ratios.get('longTermDebtToCapitalization') else "N/A")
                    
                    with lev_col4:
                        st.metric("Times Interest Earned", f"{ratios.get('timesInterestEarnedRatio', 0):.2f}" if ratios.get('timesInterestEarnedRatio') else "N/A")
                
                # Key Metrics
                if fmp_data.get('metrics'):
                    metrics = fmp_data['metrics'][0] if isinstance(fmp_data['metrics'], list) and fmp_data['metrics'] else {}
                    
                    st.markdown("### 🔑 Key Financial Metrics")
                    
                    met_col1, met_col2, met_col3, met_col4 = st.columns(4)
                    
                    with met_col1:
                        st.metric("Enterprise Value", f"${metrics.get('enterpriseValue', 0)/1e9:.2f}B" if metrics.get('enterpriseValue') else "N/A")
                    
                    with met_col2:
                        st.metric("EV/Revenue", f"{metrics.get('enterpriseValueOverRevenue', 0):.2f}" if metrics.get('enterpriseValueOverRevenue') else "N/A")
                    
                    with met_col3:
                        st.metric("EV/EBITDA", f"{metrics.get('evToEbitda', 0):.2f}" if metrics.get('evToEbitda') else "N/A")
                    
                    with met_col4:
                        st.metric("Free Cash Flow Yield", f"{metrics.get('freeCashFlowYield', 0)*100:.2f}%" if metrics.get('freeCashFlowYield') else "N/A")
                    
                    # Create advanced metrics chart
                    if any(metrics.get(key) for key in ['peRatio', 'pegRatio', 'priceToBookRatio', 'priceToSalesRatio']):
                        fig_metrics = go.Figure()
                        
                        metric_names = []
                        metric_values = []
                        
                        metric_map = {
                            'P/E Ratio': metrics.get('peRatio'),
                            'PEG Ratio': metrics.get('pegRatio'),
                            'Price to Book': metrics.get('priceToBookRatio'),
                            'Price to Sales': metrics.get('priceToSalesRatio')
                        }
                        
                        for name, value in metric_map.items():
                            if value and isinstance(value, (int, float)) and value > 0:
                                metric_names.append(name)
                                metric_values.append(value)
                        
                        if metric_names and metric_values:
                            fig_metrics.add_trace(go.Bar(
                                x=metric_names,
                                y=metric_values,
                                marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
                            ))
                            
                            fig_metrics.update_layout(
                                title="Valuation Metrics Comparison",
                                xaxis_title="Metric",
                                yaxis_title="Ratio Value",
                                template='plotly_white'
                            )
                            
                            st.plotly_chart(fig_metrics, use_container_width=True)
            
            else:
                st.info("🔍 Advanced analytics data not available for this stock")
        
        # Technical Analysis Summary (Bottom section)
        st.markdown("---")
        st.subheader("📊 Technical Analysis Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            try:
                rsi = data_with_indicators['RSI'].iloc[-1] if 'RSI' in data_with_indicators.columns else None
                if rsi and not pd.isna(rsi):
                    if rsi > 70:
                        rsi_signal = "🔴 Overbought"
                        rsi_color = "red"
                    elif rsi < 30:
                        rsi_signal = "🟢 Oversold" 
                        rsi_color = "green"
                    else:
                        rsi_signal = "🟡 Neutral"
                        rsi_color = "orange"
                    
                    st.metric("RSI Signal", rsi_signal, f"{rsi:.1f}")
                else:
                    st.metric("RSI Signal", "N/A", "No Data")
            except:
                st.metric("RSI Signal", "Error", "Calc Failed")
        
        with col2:
            try:
                ma_20 = data_with_indicators['MA_20'].iloc[-1] if 'MA_20' in data_with_indicators.columns else None
                if ma_20 and not pd.isna(ma_20):
                    if current_price > ma_20:
                        ma_signal = "📈 Above MA20"
                    else:
                        ma_signal = "📉 Below MA20"
                    
                    st.metric("Price vs MA20", ma_signal, f"${ma_20:.2f}")
                else:
                    st.metric("Price vs MA20", "N/A", "No Data")
            except:
                st.metric("Price vs MA20", "Error", "Calc Failed")
        
        with col3:
            try:
                if ('MACD' in data_with_indicators.columns and 
                    'MACD_Signal' in data_with_indicators.columns):
                    macd = data_with_indicators['MACD'].iloc[-1]
                    macd_signal = data_with_indicators['MACD_Signal'].iloc[-1]
                    
                    if not pd.isna(macd) and not pd.isna(macd_signal):
                        if macd > macd_signal:
                            macd_trend = "📈 Bullish"
                        else:
                            macd_trend = "📉 Bearish"
                        
                        st.metric("MACD Signal", macd_trend, f"{macd:.3f}")
                    else:
                        st.metric("MACD Signal", "N/A", "No Data")
                else:
                    st.metric("MACD Signal", "N/A", "No Data")
            except:
                st.metric("MACD Signal", "Error", "Calc Failed")
        
        with col4:
            try:
                # Volume analysis
                volume_ma = data_with_indicators['Volume_MA'].iloc[-1] if 'Volume_MA' in data_with_indicators.columns else None
                current_volume = data['Volume'].iloc[-1]
                
                if volume_ma and not pd.isna(volume_ma) and current_volume:
                    if current_volume > volume_ma:
                        volume_signal = "📊 High Volume"
                    else:
                        volume_signal = "📊 Low Volume"
                    
                    st.metric("Volume Signal", volume_signal, f"{current_volume/1e6:.1f}M")
                else:
                    st.metric("Volume Signal", "N/A", f"{current_volume/1e6:.1f}M")
            except:
                st.metric("Volume Signal", "Error", "Calc Failed")
    
    else:
        st.error(f"❌ Unable to fetch data for {selected_symbol}. Please check the symbol and try again.")

# Footer
st.markdown("---")
st.markdown("""
### 🎯 **Enhanced Stock Analysis Platform**

**Features:**
- 🤖 **AI-Powered Assistant**: Get intelligent answers about your stocks
- 📊 **Advanced Analytics**: Comprehensive technical and fundamental analysis  
- 🔮 **ML Predictions**: Machine learning-based price forecasting
- 📰 **Real-time News**: Latest company news and market updates
- 🏛️ **Congressional Tracking**: Monitor relevant financial legislation
- 🌍 **Global Markets**: Coverage across US, Pakistan, India, UK, Germany, Japan

**Data Sources:**
- 📈 Yahoo Finance (Real-time market data)
- 💼 Financial Modeling Prep API (Advanced financials)
- 🏛️ Congress.gov API (Legislative tracking)

**⚠️ Disclaimer:** This platform is for educational and informational purposes only. 
Stock market investments involve risk and past performance does not guarantee future results. 
Always consult with qualified financial advisors before making investment decisions.
""")

# Enhanced sidebar features
st.sidebar.markdown("---")
st.sidebar.subheader("🚀 Platform Features")

feature_status = {
    "Real-time Data": "✅",
    "AI Assistant": "✅", 
    "Technical Analysis": "✅",
    "Price Predictions": "✅",
    "News Integration": "✅",
    "Congress Tracking": "✅",
    "Advanced Ratios": "✅"
}

for feature, status in feature_status.items():
    st.sidebar.write(f"{status} {feature}")

st.sidebar.markdown("---")
st.sidebar.subheader("📊 Platform Stats")
st.sidebar.write(f"📈 Total Stocks: {sum(len(stocks) for stocks in STOCK_DATABASE.values())}")
st.sidebar.write(f"🌍 Countries: {len(STOCK_DATABASE)}")
st.sidebar.write(f"🔌 API Integrations: 3")
st.sidebar.write(f"🤖 AI Responses: {len(st.session_state.chat_history)}")

# API Status Check
st.sidebar.markdown("---")
st.sidebar.subheader("🔌 API Status")

# Test API connectivity
try:
    # Test FMP API
    test_url = f"https://financialmodelingprep.com/api/v3/profile/AAPL?apikey={FMP_API_KEY}"
    test_response = requests.get(test_url, timeout=5)
    if test_response.status_code == 200:
        st.sidebar.write("✅ Financial Modeling Prep API")
    else:
        st.sidebar.write("⚠️ FMP API Issues")
except:
    st.sidebar.write("❌ FMP API Offline")

try:
    # Test Congress API
    congress_url = f"https://api.congress.gov/v3/bill?api_key={CONGRESS_API_KEY}&limit=1"
    congress_response = requests.get(congress_url, timeout=5)
    if congress_response.status_code == 200:
        st.sidebar.write("✅ Congress.gov API")
    else:
        st.sidebar.write("⚠️ Congress API Issues")
except:
    st.sidebar.write("❌ Congress API Offline")

st.sidebar.write("✅ Yahoo Finance API")  # Always available through yfinance