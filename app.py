import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
import torch
import torch.nn as nn
import os
from datetime import datetime

# --- 1. CONFIGURAZIONE E ARCHITETTURA ---
st.set_page_config(page_title="V8 Trading AI", layout="centered")

class IrisTransformer(nn.Module):
    def __init__(self, input_dim=16, d_model=256, nhead=8, num_layers=4, dropout=0.2):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, 10, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True, 
            norm_first=True, dropout=dropout
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 4)
        )

    def forward(self, x):
        x = self.embedding(x) + self.pos_embedding
        x = self.transformer(x)
        return self.fc_out(x[:, -1, :])

# --- 2. FUNZIONI TECNICHE (Download & Features) ---
def clean_columns(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [str(c).strip() for c in df.columns]
    return df

@st.cache_data(ttl=3600)
def process_tickers(ticker_list):
    all_data = []
    vix = yf.download("^VIX", period="1mo", progress=False, auto_adjust=True)
    vix_close = clean_columns(vix)['Close']
    
    progress_bar = st.progress(0)
    for idx, symbol in enumerate(ticker_list):
        try:
            df = yf.download(symbol.replace('.', '-'), period="1y", progress=False, auto_adjust=True)
            df = clean_columns(df)
            if len(df) < 250: continue
            
            # Feature Engineering (Sintetizzata)
            df['MA21'] = ta.sma(df['Close'], length=21)
            df['MA50'] = ta.sma(df['Close'], length=50)
            df['MA200'] = ta.sma(df['Close'], length=200)
            df['RSI'] = ta.rsi(df['Close'], length=14)
            macd = ta.macd(df['Close'])
            df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = macd.iloc[:,0], macd.iloc[:,1], macd.iloc[:,2]
            df['T_open'] = df['Open'].pct_change() * 100
            df['T_close'] = df['Close'].pct_change() * 100
            df['T_min'] = df['Low'].pct_change() * 100
            df['T_max'] = df['High'].pct_change() * 100
            df['Ratio_MA21'] = (df['Close'] / df['MA21'])
            df['Ratio_MA50'] = (df['Close'] / df['MA50'])
            df['Ratio_MA200'] = (df['Close'] / df['MA200'])
            df['Vol_Ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
            df['Is_Quarter_End'] = df.index.is_quarter_end.astype(float)
            df['Recent_Div'] = 0.0
            df = df.join(vix_close.rename("VIX_Index"), how='left').ffill()
            
            df_clean = df.dropna().tail(10).copy()
            df_clean['Lookback_Day'] = np.arange(1, 11)
            df_clean['Ticker'] = symbol
            
            cols = ['Ticker', 'Lookback_Day','T_close','T_open','T_min','T_max',
                    'Ratio_MA21','Ratio_MA50','Ratio_MA200','RSI','MACD',
                    'MACD_Signal','MACD_Hist','Vol_Ratio','Is_Quarter_End','Recent_Div','VIX_Index']
            all_data.append(df_clean[cols])
        except: continue
        progress_bar.progress((idx + 1) / len(ticker_list))
    
    return pd.concat(all_data) if all_data else None

# --- 3. INTERFACCIA STREAMLIT ---
st.title("🚀 V8 Predictor Mobile")
model_path = "transformer_v8_epoch10_deep.pth"

if not os.path.exists(model_path):
    st.error(f"Manca il file del modello: {model_path}")
else:
    # Caricamento Tickers (Assicurati di avere il CSV o usa una lista fissa)
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "AMD", "NFLX", "PYPL"] # Esempio ridotto

    if st.button("🔄 1. Aggiorna Dati e Analizza", use_container_width=True):
        with st.spinner("Analisi di mercato in corso..."):
            data = process_tickers(tickers)
            if data is not None:
                # Task Predict
                model = IrisTransformer()
                model.load_state_dict(torch.load(model_path, map_location="cpu"))
                model.train() # MC Dropout attivo
                
                results = []
                for t in data['Ticker'].unique():
                    t_data = data[data['Ticker'] == t].sort_values('Lookback_Day')
                    feat_cols = [c for c in t_data.columns if c not in ['Ticker']]
                    features = t_data[feat_cols].values
                    input_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
                    
                    mc_preds = []
                    with torch.no_grad():
                        for _ in range(30):
                            mc_preds.append(model(input_tensor).numpy())
                    
                    mc_preds = np.array(mc_preds)
                    p_max = np.mean(mc_preds[:,:,3])
                    std_max = np.std(mc_preds[:,:,3])
                    conf = 1 / (1 + std_max)
                    evi = conf * p_max
                    
                    results.append({'Ticker': t, 'EVI': evi, 'CONF': conf, 'P_MAX': p_max})
                
                res_df = pd.DataFrame(results).sort_values('EVI', ascending=False)
                
                # Visualizzazione Mobile
                top = res_df.iloc[0]
                st.metric(label=f"TOP PICK: {top['Ticker']}", value=f"{top['P_MAX']:.2f}%", delta=f"EVI: {top['EVI']:.2f}")
                
                st.write("### 📊 Classifica EVI")
                st.dataframe(res_df.style.format({'EVI': '{:.2f}', 'CONF': '{:.2f}', 'P_MAX': '{:.2f}%'}), use_container_width=True)
            else:
                st.error("Errore nel download dei dati.")