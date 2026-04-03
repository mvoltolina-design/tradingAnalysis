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

COLONNE_PORTAFOGLIO = [
    'Ticker', 'Data_Acquisto', 'Prezzo_Carico', 
    'Max_Raggiunto', 'Max_Raggiunto%', 'Data_Max', 
    'Min_Raggiunto', 'Min_Raggiunto%', 'Data_Min', 
    'Stato', 'Est_Max', 'Est_Min', 'Confidence'
]
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
    'MACD', 'MACD_Signal', 'MACD_Hi', 'Vol_Ratio', 
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
        df = conn.read(worksheet="Sheet1", ttl=0)
        
        if df is None or df.empty:
            return pd.DataFrame(columns=COLONNE_PORTAFOGLIO)

        # RIMOZIONE SPAZI BIANCHI: Spesso il problema è qui
        df['Stato'] = df['Stato'].astype(str).str.strip()
        df['Ticker'] = df['Ticker'].astype(str).str.strip()

        # RIPARAZIONE COLONNE MANCANTI: 
        # Se i titoli sono vecchi e mancano le nuove colonne, le aggiungiamo noi al volo
        for col in COLONNE_PORTAFOGLIO:
            if col not in df.columns:
                df[col] = 0.0 # O None, per i vecchi titoli

        return df
    except Exception as e:
        st.error(f"Errore caricamento: {e}")
        return pd.DataFrame(columns=COLONNE_PORTAFOGLIO)

def save_portfolio(df):
    conn = st.connection("gsheets", type=GSheetsConnection)
    # Assicurati di salvare nell'ordine esatto
    df_to_save = df[COLONNE_PORTAFOGLIO].copy()
    conn.update(worksheet="Sheet1", data=df_to_save)
    st.cache_data.clear()

def load_analisi_data():
    # Carica il foglio dove tieni i risultati di Analisi V8 (es. 'candidati')
    conn = st.connection("gsheets", type=GSheetsConnection)
    df = conn.read(worksheet="candidati", ttl=0) # Nome del tab dell'analisi
    return df

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
    
    with st.spinner("Aggiornamento dati storici..."):
        df_p = update_portfolio_metrics()
    
    if df_p.empty or df_p[df_p['Stato'] == 'OPEN'].empty:
        st.info("Nessun titolo attivo.")
    else:
        dates = sorted(df_p[df_p['Stato'] == 'OPEN']['Data_Acquisto'].unique(), reverse=True)

        for d in dates:
            with st.expander(f"📅 Acquisti del {d}", expanded=True):
                sub = df_p[(df_p['Data_Acquisto'] == d) & (df_p['Stato'] == 'OPEN')]
                
                for i, row in sub.iterrows():
                    # --- RIGA 1: PERFORMANCE REALE (VALORI ASSOLUTI) ---
                    c1, c2, c3 = st.columns([1.2, 2, 2])
                    
                    with c1:
                        st.subheader(row['Ticker'])
                        st.caption(f"Carico: **${float(row['Prezzo_Carico']):.2f}**")
                        # Badge Confidence
                        conf = str(row['Confidence'])
                        c_color = "green" if "Alta" in conf else "orange" if "Media" in conf else "red"
                        st.markdown(f"**Conf:** :{c_color}[{conf}]")
                    
                    with c2:
                        max_r = float(row['Max_Raggiunto'])
                        max_p = float(row['Max_Raggiunto%']) * 100
                        st.write(f"🚀 **Max Reale: ${max_r:.2f}**")
                        st.write(f"({max_p:+.2f}%)")
                        st.caption(f"🕒 {row['Data_Max']}")
                    
                    with c3:
                        min_r = float(row['Min_Raggiunto'])
                        min_p = float(row['Min_Raggiunto%']) * 100
                        st.write(f"⚠️ **Min Reale: ${min_r:.2f}**")
                        st.write(f"({min_p:+.2f}%)")
                        st.caption(f"🕒 {row['Data_Min']}")

                    # --- RIGA 2: CONFRONTO TARGET PERCENTUALI (ESTIMATED VS REAL) ---
                    st.divider()
                    st.caption("🎯 **Analisi Target (Previsioni % vs Realtà %)**")
                    ca, cb, cc = st.columns([1.2, 2, 2])
                    
                    # Calcolo distanze dai target
                    target_max_p = float(row['Est_Max']) * 100 # Es: 0.08 -> 8%
                    target_min_p = float(row['Est_Min']) * 100 # Es: -0.05 -> -5%
                    
                    with cb:
                        # Quanto manca al target di guadagno?
                        dist_max = target_max_p - max_p
                        color_max = "green" if dist_max <= 0 else "gray" # Verde se raggiunto/superato
                        st.write(f"📈 Target Max: **{target_max_p:+.2f}%**")
                        if dist_max <= 0:
                            st.success("✅ Target raggiunto!")
                        else:
                            st.info(f"Mancano {dist_max:.2f} punti %")

                    with cc:
                        # Quanto siamo vicini al limite minimo previsto?
                        dist_min = min_p - target_min_p
                        st.write(f"📉 Target Min: **{target_min_p:+.2f}%**")
                        if min_p <= target_min_p:
                            st.error("🚨 Sotto il minimo stimato!")
                        else:
                            st.write(f"Margine: {dist_min:.2f} punti %")

                    # --- AZIONI ---
                    if st.button(f"Chiudi {row['Ticker']}", key=f"cl_{i}"):
                        df_p.at[i, 'Stato'] = 'CLOSED'
                        save_portfolio(df_p)
                        st.rerun()
                    
                    st.divider()
                    
if menu == "Aggiungi Titolo":
    st.header("🆕 Inserimento Nuova Posizione")
    # --- RECUPERO DATI ANALISI ---
    df_analisi = load_analisi_data()
    
    # 1. Input del Ticker
    t_in = st.text_input("Ticker (es. AAPL, NVDA):").upper().strip()
    
    if t_in:
        # Recuperiamo i dati dall'Analisi V8 (assumendo che df_analisi sia il risultato del tuo modello)
        # Cerchiamo se il ticker esiste nei 'candidati'
        match = df_analisi[df_analisi['Ticker'] == t_in]
        
        # Valori di default
        default_max = 0.0
        default_min = 0.0
        default_conf = "N/D"
        
        if not match.empty:
            st.success(f"✅ Titolo trovato nell'Analisi V8!")
            # Prendiamo i valori dalle colonne P_MAX, P_MIN e CONF dell'analisi
            default_max = float(match['P_MAX'].values[0])
            default_min = float(match['P_MIN'].values[0])
            default_conf = str(match['CONF'].values[0])
        else:
            st.warning("⚠️ Titolo non presente nell'ultima Analisi V8. Inserisci i parametri manualmente.")

        # 2. Form di inserimento con valori pre-compilati
        with st.form("form_aggiunta"):
            c1, c2 = st.columns(2)
            entry_price = c1.number_input("Prezzo di Carico ($)", min_value=0.01, step=0.01)
            
            st.divider()
            st.subheader("Target di Analisi")
            col_a, col_b, col_c = st.columns(3)
            
            # Qui l'utente può modificare i valori suggeriti
            est_max = col_a.number_input("Estimated Max ($)", value=default_max)
            est_min = col_b.number_input("Estimated Min ($)", value=default_min)
            conf = col_c.text_input("Confidence Score", value=default_conf)
            
            submit = st.form_submit_button("Conferma Acquisto")
            
            if submit:
                df_p = load_portfolio()
                
                # Creazione riga con le 3 nuove colonne
                new_row = {
                    'Ticker': t_in,
                    'Data_Acquisto': datetime.now().strftime("%Y-%m-%d"),
                    'Prezzo_Carico': entry_price,
                    'Max_Raggiunto': entry_price, # Inizializzato al prezzo di carico
                    'Max_Raggiunto%': 0.0,
                    'Data_Max': datetime.now().strftime("%Y-%m-%d"),
                    'Min_Raggiunto': entry_price,
                    'Min_Raggiunto%': 0.0,
                    'Data_Min': datetime.now().strftime("%Y-%m-%d"),
                    'Stato': 'OPEN',
                    'Est_Max': est_max,       # Nuova colonna
                    'Est_Min': est_min,       # Nuova colonna
                    'Confidence': conf        # Nuova colonna
                }
                
                df_p = pd.concat([df_p, pd.DataFrame([new_row])], ignore_index=True)
                save_portfolio(df_p)
                st.balloons()
                st.success(f"{t_in} salvato con successo!")

elif menu == "Analisi V8":
    st.header("🎯 Analisi Predittiva V8")
    model = load_v8_model()
    
    if model and os.path.exists("tickers_SP500_2026.csv"):
        t_list = pd.read_csv("tickers_SP500_2026.csv")['Ticker'].tolist()
        
        if st.button("🚀 AVVIA ANALISI S&P 500"):
            res = fetch_and_predict(t_list, model, 30)
            
            if not res.empty:
                # --- NUOVA LOGICA DI SALVATAGGIO ---
                st.subheader("Top Opportunità Rilevate")
                res_sorted = res.sort_values('EVI', ascending=False)
                st.dataframe(res_sorted, use_container_width=True)
                
                try:
                    conn = get_gsheet_connection()
                    # Sovrascrive il tab 'candidati' con i nuovi risultati
                    conn.update(worksheet="candidati", data=res_sorted)
                    st.success("✅ Risultati analisi salvati nel tab 'candidati'!")
                    # Puliamo la cache così il menu "Aggiungi Titolo" vede i nuovi dati
                    st.cache_data.clear()
                except Exception as e:
                    st.error(f"Errore durante il salvataggio dei candidati: {e}")
