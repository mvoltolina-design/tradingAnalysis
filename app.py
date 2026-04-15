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

# --- CONFIGURAZIONE COSTANTI ---
COLONNE_PORTAFOGLIO = [
    'Ticker', 'Data_Acquisto', 'Prezzo_Carico', 
    'Max_Raggiunto', 'Max_Raggiunto%', 'Data_Max', 
    'Min_Raggiunto', 'Min_Raggiunto%', 'Data_Min', 
    'Stato', 'Est_Max', 'Est_Min', 'Confidence'
]

COLS_ORDER = [
    'Lookback_Day', 'T_close', 'T_open', 'T_min', 'T_max', 
    'Ratio_MA21', 'Ratio_MA50', 'Ratio_MA200', 'RSI', 
    'MACD', 'MACD_Signal', 'MACD_Hist', 'Vol_Ratio', 
    'Is_Quarter_End', 'Recent_Div', 'VIX_Index'
]

# --- 1. ARCHITETTURA MODELLO ---
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

    def forward(self, x):
        x = self.embedding(x) + self.pos_embedding
        x = self.transformer(x)
        return self.fc_out(x[:, -1, :])

# --- 2. FUNZIONI DI UTILITÀ & DATI ---
def clean_columns(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [str(c).strip() for c in df.columns]
    return df

def get_gsheet_connection():
    return st.connection("gsheets", type=GSheetsConnection)

def load_portfolio():
    try:
        conn = get_gsheet_connection()
        df = conn.read(worksheet="Sheet1", ttl=0)
        if df is None or df.empty:
            return pd.DataFrame(columns=COLONNE_PORTAFOGLIO)
        
        df = clean_columns(df)
        df['Stato'] = df['Stato'].astype(str).str.strip()
        df['Ticker'] = df['Ticker'].astype(str).str.strip()
        
        for col in COLONNE_PORTAFOGLIO:
            if col not in df.columns:
                df[col] = 0.0
        return df
    except Exception as e:
        st.error(f"Errore caricamento: {e}")
        return pd.DataFrame(columns=COLONNE_PORTAFOGLIO)

def save_portfolio(df):
    conn = get_gsheet_connection()
    df_to_save = df[COLONNE_PORTAFOGLIO].copy()
    conn.update(worksheet="Sheet1", data=df_to_save)
    st.cache_data.clear()

@st.cache_data(ttl=3600)
def load_analisi_data():
    try:
        conn = get_gsheet_connection()
        df = conn.read(worksheet="candidati", ttl=3600)
        if df is not None and not df.empty:
            df.columns = [str(c).strip() for c in df.columns]
            return df
        return pd.DataFrame(columns=['Ticker', 'P_MAX', 'P_MIN', 'CONF'])
    except:
        return pd.DataFrame(columns=['Ticker', 'P_MAX', 'P_MIN', 'CONF'])

# --- 3. ENGINE DI ANALISI (V8.08 LOGIC) ---
def get_market_data(symbol, vix_data):
    try:
        df = yf.download(symbol.replace('.', '-'), period="1y", progress=False, auto_adjust=True)
        df = clean_columns(df)
        if len(df) < 250: return None

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
        
        df = df.join(vix_data.rename("VIX_Index"), how='left').ffill()
        df_clean = df.dropna()
        if len(df_clean) < 10: return None
        
        df_in = df_clean.tail(10).copy()
        df_in['Lookback_Day'] = np.arange(1, 11).astype(float)
        return df_in[COLS_ORDER].values
    except:
        return None

def task_predict(model_path, ticker_list):
    try:
        vix = yf.download("^VIX", period="1y", progress=False, auto_adjust=True)
        vix = clean_columns(vix)
        vix_close = vix['Close']
    except:
        vix_close = pd.Series(20.0, index=pd.date_range(end=datetime.now(), periods=500))

    model = IrisTransformer()
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.train() 
    
    results = []
    prog_bar = st.progress(0, text="Inizializzazione...")
    for idx, symbol in enumerate(ticker_list):
        prog_bar.progress((idx+1) / len(ticker_list), text=f"Analisi: {symbol}")
        features = get_market_data(symbol, vix_close)
        if features is not None:
            input_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
            mc_preds = []
            with torch.no_grad():
                for _ in range(20):
                    mc_preds.append(model(input_tensor).numpy())
            
            mc_preds = np.array(mc_preds)
            p_max, p_min = np.mean(mc_preds[:,:,3]), np.mean(mc_preds[:,:,2])
            std_max = np.std(mc_preds[:,:,3])
            conf = 1 / (1 + std_max)
            results.append({'Ticker': symbol, 'EVI': conf * p_max, 'P_MAX': p_max, 'P_MIN': p_min, 'CONF': conf})
    
    return pd.DataFrame(results).sort_values('EVI', ascending=False)

# --- 4. INTERFACCIA UTENTE ---
st.set_page_config(page_title="V8 Predictor", layout="wide")
menu = st.sidebar.selectbox("Menu", ["Dashboard Portafoglio", "Aggiungi Titolo", "Analisi V8"])

if menu == "Dashboard Portafoglio":
    st.header("📊 Analisi Portafoglio Attivo")
    df_port_raw = load_portfolio()
    
    if df_port_raw.empty:
        st.info("📭 Portafoglio vuoto.")
    else:
        filtro_stato = st.radio("Filtra:", ["Solo OPEN", "Solo CLOSE", "Tutti"], horizontal=True)
        
        # Filtriamo mantenendo gli indici originali!
        if filtro_stato == "Solo OPEN":
            df_port = df_port_raw[df_port_raw['Stato'] == 'OPEN'].copy()
        elif filtro_stato == "Solo CLOSE":
            df_port = df_port_raw[df_port_raw['Stato'] == 'CLOSE'].copy()
        else:
            df_port = df_port_raw.copy()

        if not df_port.empty and filtro_stato != "Solo CLOSE":
            st.write("🔄 Aggiornamento in corso...")
            for index, row in df_port.iterrows():
                if row['Stato'] == 'OPEN':
                    try:
                        ticker = str(row['Ticker']).strip()
                        # Usiamo la data acquisto specifica della riga
                        acquisto_dt = pd.to_datetime(row['Data_Acquisto'])
                        start_date = (acquisto_dt + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
                        
                        data_yf = yf.download(ticker, start=start_date, progress=False)
                        
                        if not data_yf.empty:
                            data_yf = clean_columns(data_yf)
                            val_max = float(data_yf['High'].max())
                            val_min = float(data_yf['Low'].min())
                            # Aggiorniamo SOLO questa riga tramite indice
                            df_port.at[index, 'Max_Raggiunto'] = val_max
                            df_port.at[index, 'Min_Raggiunto'] = val_min
                            df_port.at[index, 'Data_Max'] = data_yf['High'].idxmax().strftime('%Y-%m-%d')
                            df_port.at[index, 'Data_Min'] = data_yf['Low'].idxmin().strftime('%Y-%m-%d')
                            
                            p_carico = float(row['Prezzo_Carico'])
                            if p_carico > 0:
                                df_port.at[index, 'Max_Raggiunto%'] = (val_max - p_carico) / p_carico
                                df_port.at[index, 'Min_Raggiunto%'] = (val_min - p_carico) / p_carico
                    except: continue

        # Visualizzazione con percentuali moltiplicate per 100 solo per il display
        df_display = df_port.copy()
        target_perc = ["Est_Max", "Est_Min", "Confidence", "Max_Raggiunto%", "Min_Raggiunto%"]
        for col in target_perc:
            if col in df_display.columns:
                df_display[col] = pd.to_numeric(df_display[col], errors='coerce') * 100

        st.dataframe(df_display, use_container_width=True, hide_index=True, column_config={
            "Prezzo_Carico": st.column_config.NumberColumn("Carico $", format="$ %.2f"),
            "Max_Raggiunto": st.column_config.NumberColumn("Max $", format="$ %.2f"),
            "Min_Raggiunto": st.column_config.NumberColumn("Min $", format="$ %.2f"),
            "Max_Raggiunto%": st.column_config.NumberColumn("Max %", format="%.2f%%"),
            "Min_Raggiunto%": st.column_config.NumberColumn("Min %", format="%.2f%%"),
        })

        if st.button("💾 Salva modifiche su Database"):
            # Fondamentale: sovrascriviamo le righe in df_port_raw usando gli indici di df_port
            for index in df_port.index:
                df_port_raw.loc[index] = df_port.loc[index]
            
            save_portfolio(df_port_raw)
            st.success("✅ Database aggiornato con successo!")
            st.rerun()

elif menu == "Aggiungi Titolo":
    st.header("🆕 Nuova Posizione")
    df_analisi = load_analisi_data()
    t_in = st.text_input("Ticker:").upper().strip()
    
    if t_in:
        # Prezzo corrente
        if "market_price" not in st.session_state or st.session_state.get("last_t") != t_in:
            try:
                p = yf.Ticker(t_in).history(period="1d")['Close'].iloc[-1]
                st.session_state.market_price = float(p)
                st.session_state.last_t = t_in
            except: st.session_state.market_price = 0.0

        st.write(f"Prezzo attuale: **{st.session_state.market_price:.2f} $**")
        
        with st.form("add_form"):
            p_carico = st.number_input("Prezzo Carico", value=st.session_state.market_price)
            # Dati da analisi
            match = df_analisi[df_analisi['Ticker'] == t_in]
            e_max = st.number_input("Est Max", value=float(match['P_MAX'].values[0]) if not match.empty else 0.0)
            e_min = st.number_input("Est Min", value=float(match['P_MIN'].values[0]) if not match.empty else 0.0)
            conf = st.number_input("Confidence", value=float(match['CONF'].values[0]) if not match.empty else 0.0)
            
            if st.form_submit_button("Salva"):
                df_p = load_portfolio()
                new_row = {
                    'Ticker': t_in, 'Data_Acquisto': datetime.now().strftime("%Y-%m-%d"),
                    'Prezzo_Carico': p_carico, 'Stato': 'OPEN', 'Max_Raggiunto': p_carico,
                    'Min_Raggiunto': p_carico, 'Max_Raggiunto%': 0.0, 'Min_Raggiunto%': 0.0,
                    'Est_Max': e_max / 100, 'Est_Min': e_min / 100, 'Confidence': conf
                }
                df_p = pd.concat([df_p, pd.DataFrame([new_row])], ignore_index=True)
                save_portfolio(df_p)
                st.success("Salvato!")

elif menu == "Analisi V8":
    st.header("🎯 Analisi Predittiva")
    df_ans = load_analisi_data()
    if not df_ans.empty: st.dataframe(df_ans)
    
    if st.button("🚀 Avvia Analisi S&P 500"):
        if os.path.exists("tickers_SP500_2026.csv"):
            tkrs = pd.read_csv("tickers_SP500_2026.csv")['Ticker'].tolist()
            res = task_predict("transformer_v8.1_refine_epoch8.pth", tkrs)
            if not res.empty:
                get_gsheet_connection().update(worksheet="candidati", data=res)
                st.success("Analisi completata!")
                st.rerun()
