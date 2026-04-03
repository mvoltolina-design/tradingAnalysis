import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
import torch
import torch.nn as nn
import os
from datetime import datetime
from streamlit_gsheets import GSheetsConnection

# --- 1. CONFIGURAZIONE E ARCHITETTURA MODELLO ---
class IrisTransformer(nn.Module):
    def __init__(self, input_dim=16, d_model=256, nhead=8, num_layers=4, dropout=0.2):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, 10, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True, norm_first=True, dropout=dropout
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Sequential(
            nn.Linear(d_model, d_model // 2), 
            nn.ReLU(), 
            nn.Linear(d_model // 2, 4)
        )
        self.output_dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.embedding(x) + self.pos_embedding
        x = self.transformer(x)
        x = self.fc_out(x[:, -1, :])
        return self.output_dropout(x)

# Ordine esatto delle feature (DNA del Parquet)
COLS_ORDER = [
    'Lookback_Day', 'T_close', 'T_open', 'T_min', 'T_max', 
    'Ratio_MA21', 'Ratio_MA50', 'Ratio_MA200', 'RSI', 
    'MACD', 'MACD_Signal', 'MACD_Hist', 'Vol_Ratio', 
    'Is_Quarter_End', 'Recent_Div', 'VIX_Index'
]

PORTFOLIO_FILE = "portfolio_v8.csv"

# --- 2. FUNZIONI DI UTILITÀ ---
def clean_columns(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [str(c).strip() for c in df.columns]
    return df

def get_gsheet_connection():
    return st.connection("gsheets", type=GSheetsConnection)

def load_portfolio():
    try:
        conn = st.connection("gsheets", type=GSheetsConnection)
        # Carica i dati ignorando la cache (ttl=0) per avere dati sempre freschi
        df = conn.read(worksheet="Sheet1", ttl=0)
        return df.dropna(how="all")
    except Exception as e:
        # Se il foglio è vuoto o non connesso, restituisce lo schema base
        return pd.DataFrame(columns=['Ticker', 'Data_Acquisto', 'Prezzo_Carico', 'Max_Raggiunto', 'Data_Max', 'Min_Raggiunto', 'Data_Min', 'Stato'])

def save_portfolio(df):
    try:
        conn = st.connection("gsheets", type=GSheetsConnection)
        # Sovrascrive il foglio con il nuovo DataFrame
        conn.update(worksheet="Sheet1", data=df)
        # Piccolo trucco: svuota la cache di lettura per riflettere subito le modifiche
        st.cache_data.clear() 
    except Exception as e:
        st.error(f"Errore nel salvataggio su Google Sheets: {e}")

@st.cache_resource
def load_v8_model():
    path = "transformer_v8_epoch09.pth"
    if os.path.exists(path):
        m = IrisTransformer(num_layers=4)
        m.load_state_dict(torch.load(path, map_location="cpu"))
        m.train() # Attivo Dropout per MC
        return m
    return None

# --- 3. ENGINE DI PREDIZIONE ---
def fetch_and_predict(ticker_list, model, cycles):
    vix = yf.download("^VIX", period="1mo", progress=False, auto_adjust=True)
    vix_close = clean_columns(vix)['Close']
    
    results = []
    prog_bar = st.progress(0, text="Inizializzazione...")
    
    for idx, t in enumerate(ticker_list):
        try:
            df = yf.download(t.replace('.', '-'), period="1y", progress=False, auto_adjust=True)
            df = clean_columns(df)
            if len(df) < 250: continue

            # Feature Engineering
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
            df['Ratio_MA21'] = df['Close'] / df['MA21']
            df['Ratio_MA50'] = df['Close'] / df['MA50']
            df['Ratio_MA200'] = df['Close'] / df['MA200']
            df['Vol_Ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
            df['Is_Quarter_End'] = df.index.is_quarter_end.astype(float)
            df['Recent_Div'] = 0.0
            df = df.join(vix_close.rename("VIX_Index"), how='left').ffill()

            df_in = df.dropna().tail(10).copy()
            df_in['Lookback_Day'] = np.arange(1, 11).astype(float)
            
            input_tensor = torch.tensor(df_in[COLS_ORDER].values, dtype=torch.float32).unsqueeze(0)

            mc_preds = []
            with torch.no_grad():
                for _ in range(cycles):
                    mc_preds.append(model(input_tensor).numpy())
            
            mc_preds = np.array(mc_preds)
            p_max, p_min = np.mean(mc_preds[:,:,3]), np.mean(mc_preds[:,:,2])
            conf = 1 / (1 + np.std(mc_preds[:,:,3]))
            
            results.append({'Ticker': t, 'EVI': conf * p_max, 'P_MAX': p_max, 'P_MIN': p_min, 'CONF': conf})
        except: continue
        prog_bar.progress((idx + 1) / len(ticker_list), text=f"Analisi {t}...")
    
    return pd.DataFrame(results)

# --- 4. LOGICA PORTFOLIO ---
def update_portfolio_metrics():
    df = load_portfolio()
    if df.empty: return df
    
    # Pulizia tipi
    for col in ['Prezzo_Carico', 'Max_Raggiunto', 'Min_Raggiunto']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    active_indices = df[df['Stato'] == 'OPEN'].index

    for idx in active_indices:
        # Usiamo variabili LOCALI per ogni iterazione per evitare incroci
        row_ticker = str(df.at[idx, 'Ticker']).strip().replace('.', '-')
        row_start_date = pd.to_datetime(df.at[idx, 'Data_Acquisto']).strftime('%Y-%m-%d')
        row_carico = float(df.at[idx, 'Prezzo_Carico'])

        try:
            # Scarichiamo SOLO per questo specifico ticker
            h_data = yf.download(row_ticker, start=row_start_date, progress=False, auto_adjust=True)
            
            if h_data.empty: continue
            h_data = clean_columns(h_data)
            
            # FILTRO RIGIDO: Solo date >= acquisto
            h_data = h_data[h_data.index >= pd.to_datetime(row_start_date)]
            if h_data.empty: continue

            # --- CALCOLO PROTETTO ---
            # Il massimo non può essere inferiore al carico al momento dell'acquisto
            current_max = float(h_data['High'].max())
            current_min = float(h_data['Low'].min())
            
            # PROTEZIONE: Se yfinance sballa, il Min/Max reale non può ignorare il prezzo di carico
            # (evita i minimi assurdi a 21.87 se il carico è 255)
            real_max = max(current_max, row_carico)
            real_min = min(current_min, row_carico)
            
            # Date dei picchi
            date_max_idx = h_data['High'].idxmax().strftime("%Y-%m-%d")
            date_min_idx = h_data['Low'].idxmin().strftime("%Y-%m-%d")

            # SCRITTURA PUNTUALE NELLA RIGA CORRETTA
            df.at[idx, 'Max_Raggiunto'] = real_max
            df.at[idx, 'Max_Raggiunto%'] = (real_max - row_carico) / row_carico
            df.at[idx, 'Data_Max'] = date_max_idx
            
            df.at[idx, 'Min_Raggiunto'] = real_min
            df.at[idx, 'Min_Raggiunto%'] = (real_min - row_carico) / row_carico
            df.at[idx, 'Data_Min'] = date_min_idx

        except Exception as e:
            continue
            
    save_portfolio(df)
    return df

# --- 5. INTERFACCIA STREAMLIT ---
st.set_page_config(page_title="V8 Predictor", layout="wide")
menu = st.sidebar.selectbox("Menu", ["Dashboard Portafoglio", "Aggiungi Titolo", "Analisi V8"])

if menu == "Dashboard Portafoglio":
    st.header("📈 Portafoglio Attivo")
    
    # 1. Esegue l'aggiornamento (scarica i dati da yfinance dal giorno d'acquisto)
    with st.spinner("Sincronizzazione dati storici con Google Sheets..."):
        df_p = update_portfolio_metrics()
    
    if df_p.empty or df_p[df_p['Stato'] == 'OPEN'].empty:
        st.info("Nessun titolo attivo in portafoglio.")
    else:
        # Ordiniamo per data di acquisto (più recenti in alto)
        # Assicuriamoci che Data_Acquisto sia trattata come stringa per il raggruppamento
        df_p['Data_Acquisto'] = df_p['Data_Acquisto'].astype(str)
        dates = sorted(df_p[df_p['Stato'] == 'OPEN']['Data_Acquisto'].unique(), reverse=True)

        for d in dates:
            with st.expander(f"📅 Acquisti del {d}", expanded=True):
                sub = df_p[(df_p['Data_Acquisto'] == d) & (df_p['Stato'] == 'OPEN')]
                
                for i, row in sub.iterrows():
                    c1, c2, c3 = st.columns([1.2, 2, 2])
                    
                    # --- COLONNA 1: INFO TITOLO ---
                    with c1:
                        st.subheader(row['Ticker'])
                        st.caption(f"Carico: ${float(row['Prezzo_Carico']):.2f}")
                    
                    # --- COLONNA 2: MASSIMO STORICO ---
                    with c2:
                        # Usiamo i valori già calcolati da update_portfolio_metrics
                        max_val = float(row['Max_Raggiunto'])
                        # Moltiplichiamo per 100 se nel foglio sono decimali (es. 0.08 -> 8%)
                        max_perc = float(row['Max_Raggiunto%']) * 100
                        
                        st.write(f"🚀 **Max: ${max_val:.2f}**")
                        st.write(f"({max_perc:+.2f}%)")
                        st.caption(f"🕒 {row['Data_Max']}")
                    
                    # --- COLONNA 3: MINIMO STORICO ---
                    with c3:
                        min_val = float(row['Min_Raggiunto'])
                        min_perc = float(row['Min_Raggiunto%']) * 100
                        
                        st.write(f"⚠️ **Min: ${min_val:.2f}**")
                        st.write(f"({min_perc:+.2f}%)")
                        st.caption(f"🕒 {row['Data_Min']}")
                    
                    # --- AZIONI ---
                    if st.button(f"Chiudi Posizione {row['Ticker']}", key=f"cl_{i}"):
                        df_p.at[i, 'Stato'] = 'CLOSED'
                        save_portfolio(df_p)
                        st.success(f"Posizione su {row['Ticker']} chiusa!")
                        st.rerun()
                    
                    st.divider()

elif menu == "Aggiungi Titolo":
    st.header("🛒 Nuovo Ingresso")
    t_in = st.text_input("Ticker:").upper()
    if t_in:
        d_yf = yf.download(t_in.replace('.', '-'), period="1d", progress=False, auto_adjust=True)
        if not d_yf.empty:
            if isinstance(d_yf.columns, pd.MultiIndex): d_yf.columns = d_yf.columns.get_level_values(0)
            curr = float(d_yf['Close'].iloc[-1])
            st.metric("Prezzo Attuale", f"${curr:.2f}")
            entry = st.number_input("Prezzo d'acquisto:", value=curr)
            if st.button("Salva in Portafoglio"):
                df_p = load_portfolio()
                new = {'Ticker': t_in, 'Data_Acquisto': datetime.now().strftime("%Y-%m-%d"), 'Prezzo_Carico': entry, 
                       'Max_Raggiunto': curr, 'Data_Max': "In attesa", 'Min_Raggiunto': curr, 'Data_Min': "In attesa", 'Stato': 'OPEN'}
                save_portfolio(pd.concat([df_p, pd.DataFrame([new])], ignore_index=True))
                st.success("Salvato!")
        else: st.error("Ticker non trovato.")

elif menu == "Analisi V8":
    st.header("🎯 Analisi Predittiva V8")
    model = load_v8_model()
    if model and os.path.exists("tickers_SP500_2026.csv"):
        t_list = pd.read_csv("tickers_SP500_2026.csv")['Ticker'].tolist()
        if st.button("🚀 AVVIA ANALISI S&P 500"):
            res = fetch_and_predict(t_list, model, 30)
            if not res.empty:
                st.dataframe(res.sort_values('EVI', ascending=False), use_container_width=True)
