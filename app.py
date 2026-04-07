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

@st.cache_data(ttl=3600)
def load_analisi_data():
    try:
        conn = st.connection("gsheets", type=GSheetsConnection)
        df = conn.read(worksheet="candidati", ttl=3600)
        
        if df is not None and not df.empty:
            # Pulizia automatica: rimuove spazi bianchi dai nomi delle colonne
            df.columns = [str(c).strip() for c in df.columns]
            return df
        
        # Se il foglio è vuoto, restituiamo un DataFrame con le colonne giuste
        return pd.DataFrame(columns=['Ticker', 'P_MAX', 'P_MIN', 'CONF'])
    except Exception as e:
        # In caso di errore, restituiamo comunque la struttura minima per non rompere l'app
        return pd.DataFrame(columns=['Ticker', 'P_MAX', 'P_MIN', 'CONF'])


@st.cache_resource
def load_v8_model():
    #path = "transformer_v8_epoch09.pth"
    path = "transformer_v8.1_refine_epoch8.pth"
    if os.path.exists(path):
        m = IrisTransformer(num_layers=4)
        m.load_state_dict(torch.load(path, map_location="cpu"))
        m.train() # Attivo Dropout per MC
        return m
    return None

# --- 3. ENGINE DI PREDIZIONE ---
def fetch_and_predict(ticker_list, model, cycles):
    st.write("🔍 **DEBUG START**: Inizio scaricamento dati VIX...")
    
    # 1. Recupero VIX (Benchmark per il modello)
    try:
        vix = yf.download("^VIX", period="1mo", progress=False, auto_adjust=True)
        if isinstance(vix.columns, pd.MultiIndex):
            vix.columns = vix.columns.get_level_values(0)
        vix = clean_columns(vix)
        vix_close = vix['Close']
        st.write("✅ VIX scaricato con successo.")
    except Exception as e:
        st.error(f"❌ Errore critico download VIX: {e}")
        # Fallback per non bloccare l'intera analisi se il VIX ha problemi temporanei
        vix_close = pd.Series(20.0, index=pd.date_range(end=datetime.now(), periods=30))

    results = []
    prog_bar = st.progress(0, text="Inizializzazione...")
    
    # Per il debug, puoi limitare la lista: debug_list = ticker_list[:10]
    for idx, t in enumerate(ticker_list):
        prog_bar.progress((idx + 1) / len(ticker_list), text=f"Analisi in corso: {t}...")
        
        try:
            # 2. Download Dati Titolo
            # Gestione ticker con punti (es. BRK.B -> BRK-B)
            tk_fixed = str(t).replace('.', '-')
            df = yf.download(tk_fixed, period="1y", progress=False, auto_adjust=True)
            
            if df.empty:
                continue
            
            # Pulizia MultiIndex (Necessaria per le ultime versioni di yfinance)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = clean_columns(df)

            if len(df) < 250:
                continue

            # 3. Feature Engineering (Punto critico per il KeyError)
            # Calcoliamo gli indicatori usando pandas_ta
            df['MA21'] = ta.sma(df['Close'], length=21)
            df['MA50'] = ta.sma(df['Close'], length=50)
            df['MA200'] = ta.sma(df['Close'], length=200)
            df['RSI'] = ta.rsi(df['Close'], length=14)
            
            macd_df = ta.macd(df['Close'])
            if macd_df is not None and not macd_df.empty:
                # Usiamo iloc per evitare errori se i nomi delle colonne variano
                df['MACD'] = macd_df.iloc[:, 0]
                df['MACD_Signal'] = macd_df.iloc[:, 1]
                df['MACD_Hi'] = macd_df.iloc[:, 2]
            else:
                continue # Salta se il MACD fallisce

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
            
            # Allineamento VIX
            df = df.join(vix_close.rename("VIX_Index"), how='left')
            df['VIX_Index'] = df['VIX_Index'].ffill().fillna(20.0)

            # 4. Preparazione Tensor per il Modello
            # Dropna rimuove le righe iniziali necessarie alle medie mobili (es. prime 200)
            df_clean = df.dropna().copy()
            if len(df_clean) < 10:
                continue

            df_in = df_clean.tail(10).copy()
            df_in['Lookback_Day'] = np.arange(1, 11).astype(float)
            
            # Verifica finale colonne: COLS_ORDER deve corrispondere al DNA del modello
            input_data = df_in[COLS_ORDER].values
            input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0)

            # 5. Inferenza Monte Carlo
            mc_preds = []
            with torch.no_grad():
                for _ in range(cycles):
                    # Il modello deve essere in model.train() per attivare il Dropout
                    pred = model(input_tensor).numpy()
                    mc_preds.append(pred)
            
            mc_preds = np.array(mc_preds)
            
            # Supponendo che l'output del modello sia [Batch, 4]
            # dove index 3 = Max, index 2 = Min
            p_max = np.mean(mc_preds[:, :, 3])
            p_min = np.mean(mc_preds[:, :, 2])
            
            # La confidenza è l'inverso della deviazione standard del Max
            std_max = np.std(mc_preds[:, :, 3])
            conf = 1 / (1 + std_max)
            
            # EVI (Expected Value Index) per il ranking
            evi = conf * p_max
            
            results.append({
                'Ticker': t, 
                'EVI': evi, 
                'P_MAX': p_max, 
                'P_MIN': p_min, 
                'CONF': conf
            })

        except Exception as e:
            # Se un ticker fallisce, stampiamo l'errore specifico ma proseguiamo
            st.warning(f"⚠️ Errore su {t}: {e}")
            continue
            
    # Restituisce il DataFrame finale
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

# ==========================================
# 1. SEZIONE DASHBOARD
# ==========================================
if menu == "Dashboard Portafoglio":
    st.header("📊 Analisi Portafoglio Attivo")
    
    df_port = load_portfolio().copy()
    
    if df_port.empty:
        st.info("📭 Il portafoglio è attualmente vuoto.")
    else:
        target_perc = ["Est_Max", "Est_Min", "Confidence"]
        found_to_multiply = []

        for col in df_port.columns:
            nome_pulito = col.strip()
            if nome_pulito.endswith('%') or nome_pulito in target_perc:
                try:
                    df_port[col] = pd.to_numeric(df_port[col], errors='coerce') * 100
                    found_to_multiply.append(col)
                except:
                    continue

        config_visuale = {
            "Ticker": st.column_config.TextColumn("Ticker", width="small"),
            "Prezzo_Ingresso": st.column_config.NumberColumn("Ingresso $", format="$ %.2f"),
            "Max_Raggiunto": st.column_config.NumberColumn("Max Assoluto", format="$ %.2f"),
        }
        
        for col in found_to_multiply:
            config_visuale[col] = st.column_config.NumberColumn(col, format="%.2f%%")

        st.dataframe(
            df_port,
            use_container_width=True,
            hide_index=True,
            column_config=config_visuale
        )

# ==========================================
# 2. SEZIONE AGGIUNGI TITOLO (Mancava questa riga!)
# ==========================================
elif menu == "Aggiungi Titolo":
    st.header("🆕 Inserimento Nuova Posizione")
    df_analisi = load_analisi_data()
    
    t_in = st.text_input("Ticker (es. AAPL):").upper().strip()
    
    if t_in:
        # --- 1. RECUPERO PREZZO LIVE DA YFINANCE ---
        current_market_price = 0.01
        try:
            ticker_yf = yf.Ticker(t_in)
            hist = ticker_yf.history(period="1d")
            if not hist.empty:
                current_market_price = float(hist['Close'].iloc[-1])
                st.caption(f"📈 Ultimo prezzo di mercato rilevato: **{current_market_price:.2f} $**")
            else:
                st.info("ℹ️ Impossibile recuperare prezzo live. Inserimento manuale richiesto.")
        except Exception as e:
            st.error(f"Errore yfinance: {e}")

        # --- 2. CONTROLLO DATI ANALISI V8 ---
        if 'Ticker' in df_analisi.columns:
            match = df_analisi[df_analisi['Ticker'] == t_in]
        else:
            st.error("⚠️ La colonna 'Ticker' non è stata trovata.")
            match = pd.DataFrame()
        
        default_max, default_min, default_conf = 0.0, 0.0, "N/D"
        
        if not match.empty:
            st.success(f"✅ Titolo trovato nell'Analisi V8!")
            default_max = float(match['P_MAX'].values[0])
            default_min = float(match['P_MIN'].values[0])
            default_conf = str(match['CONF'].values[0])
        else:
            st.warning("⚠️ Titolo non presente nell'Analisi V8.")

        # --- 3. FORM DI INSERIMENTO ---
        with st.form("form_aggiunta"):
            c1, c2 = st.columns(2)
            entry_price = c1.number_input("Prezzo di Carico ($)", min_value=0.0, value=current_market_price, format="%.2f")
            
            st.divider()
            st.subheader("Target di Analisi")
            col_a, col_b, col_c = st.columns(3)
            est_max = col_a.number_input("Estimated Max ($)", value=default_max)
            est_min = col_b.number_input("Estimated Min ($)", value=default_min)
            conf = col_c.text_input("Confidence Score", value=default_conf)
            
            if st.form_submit_button("Conferma Acquisto"):
                df_p = load_portfolio() 
                new_row = {
                    'Ticker': t_in,
                    'Data_Acquisto': datetime.now().strftime("%Y-%m-%d"),
                    'Prezzo_Carico': entry_price,
                    'Max_Raggiunto': entry_price, 
                    'Max_Raggiunto%': 0.0,
                    'Min_Raggiunto': entry_price,
                    'Min_Raggiunto%': 0.0,
                    'Stato': 'OPEN',
                    'Est_Max': est_max, 'Est_Min': est_min, 'Confidence': conf
                }
                df_p = pd.concat([df_p, pd.DataFrame([new_row])], ignore_index=True)
                save_portfolio(df_p)
                st.balloons()
                st.success(f"Posizione su {t_in} salvata!")

# ==========================================
# 3. SEZIONE ANALISI V8
# ==========================================
elif menu == "Analisi V8":
    st.header("🎯 Analisi Predittiva V8")
    df_analisi = load_analisi_data()
    analisi_presente = not df_analisi.empty and len(df_analisi) > 0

    if analisi_presente:
        st.success("✅ Risultati dell'ultima analisi caricati")
        df_display = df_analisi.sort_values('EVI', ascending=False)
        st.dataframe(df_display, use_container_width=True)
    else:
        st.warning("⚠️ Nessuna analisi recente trovata.")

    label_pulsante = "🔄 RICALCOLA ANALISI S&P 500" if analisi_presente else "🚀 AVVIA ANALISI S&P 500"
    
    if st.button(label_pulsante):
        model = load_v8_model()
        if not os.path.exists("tickers_SP500_2026.csv"):
            st.error("❌ File ticker mancante.")
        elif model is None:
            st.error("❌ Modello non caricato.")
        else:
            t_list = pd.read_csv("tickers_SP500_2026.csv")['Ticker'].tolist()
            res = fetch_and_predict(t_list, model, 10)
            
            if not res.empty:
                res_sorted = res.sort_values('EVI', ascending=False)
                try:
                    conn = get_gsheet_connection()
                    conn.update(worksheet="candidati", data=res_sorted)
                    st.cache_data.clear()
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Errore GSheet: {e}")
