import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
import torch
import torch.nn as nn
import os

# --- 1. ARCHITETTURA SPECIFICA PER EPOCH 11 (10 LAYERS) ---
class IrisTransformer(nn.Module):
    def __init__(self, input_dim=16, d_model=256, nhead=8, num_layers=10, dropout=0.2):
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

# --- 2. LOGICA DI CALCOLO ---
def clean_columns(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [str(c).strip() for c in df.columns]
    return df

@st.cache_data(ttl=3600)
def fetch_and_predict(ticker_list, _model, cycles):
    vix = yf.download("^VIX", period="1mo", progress=False, auto_adjust=True)
    vix_close = clean_columns(vix)['Close']
    
    all_results = []
    progress_bar = st.progress(0)
    
    for idx, symbol in enumerate(ticker_list):
        try:
            df = yf.download(symbol.replace('.', '-'), period="1y", progress=False, auto_adjust=True)
            df = clean_columns(df)
            if len(df) < 250: continue
            
            # Indicatori minimi necessari
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
            
            # Preparazione Tensor
            df_input = df.dropna().tail(10)
            cols = ['T_close','T_open','T_min','T_max','Ratio_MA21','Ratio_MA50','Ratio_MA200',
                    'RSI','MACD','MACD_Signal','MACD_Hist','Vol_Ratio','Is_Quarter_End','Recent_Div','VIX_Index']
            # Aggiungiamo 'Lookback_Day' per arrivare a 16 feature come richiesto dal modello
            features = df_input[cols].values
            lookback_days = np.arange(1, 11).reshape(-1, 1)
            final_features = np.hstack([lookback_days, features]) 
            
            input_tensor = torch.tensor(final_features, dtype=torch.float32).unsqueeze(0)
            
            # Monte Carlo
            mc_preds = []
            for _ in range(cycles):
                mc_preds.append(_model(input_tensor).detach().numpy())
            
            mc_preds = np.array(mc_preds)
            m_max = np.mean(mc_preds[:,:,3])
            m_min = np.mean(mc_preds[:,:,2])
            std_max = np.std(mc_preds[:,:,3])
            conf = 1 / (1 + std_max)
            
            all_results.append({
                'Ticker': symbol, 
                'EVI': conf * m_max, 
                'P_MAX': m_max, 
                'P_MIN': m_min, 
                'CONF': conf
            })
        except: continue
        progress_bar.progress((idx + 1) / len(ticker_list))
    
    return pd.DataFrame(all_results)

# --- 3. UI PRINCIPALE ---
st.set_page_config(page_title="V8 Deep Predictor", layout="centered")
st.title("🎯 V8 Analysis (Epoch 09)")

model_file = "transformer_v8_epoch09.pth" # Assicurati che il nome sia corretto su GitHub

if os.path.exists(model_file):
    # Caricamento Modello a 4 Layer
    @st.cache_resource
    def load_model():
        m = IrisTransformer(num_layers=4) # Forziamo 10 layer per Epoch 11
        m.load_state_dict(torch.load(model_file, map_location="cpu"))
        m.train() # Per dropout
        return m

    model = load_model()
    
    # Caricamento Tickers
    if os.path.exists("tickers_SP500_2026.csv"):
        df_t = pd.read_csv("tickers_SP500_2026.csv")
        ticker_list = df_t['Ticker'].tolist()
        
        st.write(f"Pronto ad analizzare {len(ticker_list)} titoli.")
        
        if st.button("🚀 AVVIA ANALISI COMPLETA", use_container_width=True):
            res = fetch_and_predict(ticker_list, model, 30)
            
            if not res.empty:
                res = res.sort_values('EVI', ascending=False)
                
                # Top Pick in evidenza
                top = res.iloc[0]
                c1, c2 = st.columns(2)
                c1.metric("TOP TICKER", top['Ticker'])
                c2.metric("EXPECTED MAX", f"{top['P_MAX']:.2f}%")
                
                st.write("### 📈 Classifica Risultati")
                # Tabella con P_MAX e P_MIN
                st.dataframe(
                    res[['Ticker', 'EVI', 'P_MAX', 'P_MIN', 'CONF']].style.format({
                        'EVI': '{:.2f}', 'P_MAX': '{:.2f}%', 'P_MIN': '{:.2f}%', 'CONF': '{:.2f}'
                    }), 
                    use_container_width=True
                )
            else:
                st.error("Nessun dato prodotto. Controlla la connessione o il file tickers.")
    else:
        st.error("File tickers_SP500_2026.csv non trovato.")
else:
    st.error(f"File {model_file} non trovato nel repository.")