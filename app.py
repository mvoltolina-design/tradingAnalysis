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
    path = "transformer_v8_epoch09.pth"
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

if menu == "Dashboard Portafoglio":
    st.header("📊 Il Tuo Portafoglio")
    
   # 1. Caricamento dati
    df_port_raw = load_portfolio()
    
    if df_port_raw.empty:
        st.info("📭 Il portafoglio è attualmente vuoto.")
    else:
        # 2. TRASFORMAZIONE MATEMATICA (* 100)
        # Creiamo una copia per la visualizzazione per non sporcare il file originale
        df_display = df_port_raw.copy()
        
        cols_to_perc = ["max_Raggiunto%", "Min_raggiunto%", "Est_Max", "Est_Min", "Confidence"]
        
        for col in cols_to_perc:
            if col in df_display.columns:
                # Moltiplichiamo per 100 solo se i valori sono decimali (es. < 1 o < 2)
                # Questo evita di moltiplicare all'infinito se ricarichi la pagina
                df_display[col] = pd.to_numeric(df_display[col], errors='coerce') * 100

        # 3. VISUALIZZAZIONE CON COLUMN_CONFIG
        st.dataframe(
            df_display,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Ticker": st.column_config.TextColumn("Ticker", width="small"),
                "Prezzo_Ingresso": st.column_config.NumberColumn("Entry $", format="$ %.2f"),
                
                # Formattiamo con 2 cifre decimali e il simbolo %
                "max_Raggiunto%": st.column_config.NumberColumn("Max Raggiunto", format="%.2f%%"),
                "Min_raggiunto%": st.column_config.NumberColumn("Min Raggiunto", format="%.2f%%"),
                "Est_Max": st.column_config.NumberColumn("Est. Max", format="%.2f%%"),
                "Est_Min": st.column_config.NumberColumn("Est. Min", format="%.2f%%"),
                "Confidence": st.column_config.NumberColumn("Confidenza", format="%.2f%%"),
                
                "Stop_Loss": st.column_config.NumberColumn("S.L.", format="$ %.2f"),
                "Target": st.column_config.NumberColumn("T.P.", format="$ %.2f"),
                "Data": st.column_config.DatetimeColumn("Data", format="DD/MM/YY"),
            }
        )

        st.divider()
        
        # Area Export (usa i dati ORIGINALI, non quelli moltiplicati per 100)
        csv_data = df_port_raw.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Scarica Portafoglio (Dati Reali)", data=csv_data, file_name="portafoglio_v8.csv")
if menu == "Aggiungi Titolo":
    st.header("🆕 Inserimento Nuova Posizione")
    df_analisi = load_analisi_data()
    
    t_in = st.text_input("Ticker (es. AAPL):").upper().strip()
    
    if t_in:
        # CONTROLLO DI SICUREZZA:
        if 'Ticker' in df_analisi.columns:
            match = df_analisi[df_analisi['Ticker'] == t_in]
        else:
            # Se la colonna manca, creiamo un match vuoto per non crashare
            st.error("⚠️ La colonna 'Ticker' non è stata trovata nel foglio 'candidati'.")
            match = pd.DataFrame()
        
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
    
    # 1. Caricamento dati (sfrutta la cache di 1 ora definita in load_analisi_data)
    df_analisi = load_analisi_data()
    
    # Verifichiamo se il DataFrame contiene dati reali (almeno una riga)
    analisi_presente = not df_analisi.empty and len(df_analisi) > 0

    if analisi_presente:
        st.success("✅ Risultati dell'ultima analisi caricati (Validità 1h)")
        
        # Mostriamo i risultati ordinati per EVI (il nostro indice di opportunità)
        df_display = df_analisi.sort_values('EVI', ascending=False)
        st.dataframe(df_display, use_container_width=True)
        
        st.divider()
        st.subheader("Aggiornamento Dati")
        st.write("Vuoi ignorare i dati in memoria e lanciare una nuova scansione di mercato?")
    else:
        st.warning("⚠️ Nessuna analisi recente trovata nel database 'candidati' o dati scaduti.")
        st.info("L'analisi dello S&P 500 richiede circa 10-15 minuti. Assicurati di non chiudere la pagina.")

    # 2. Pulsante di avvio analisi
    # Se i dati esistono, il pulsante serve per il "Ricalcolo", altrimenti per il "Primo Avvio"
    label_pulsante = "🔄 RICALCOLA ANALISI S&P 500" if analisi_presente else "🚀 AVVIA ANALISI S&P 500"
    
    if st.button(label_pulsante):
        model = load_v8_model()
        
        # Controllo presenza file ticker
        if not os.path.exists("tickers_SP500_2026.csv"):
            st.error("❌ File 'tickers_SP500_2026.csv' non trovato. Impossibile procedere.")
        elif model is None:
            st.error("❌ Modello V8 (.pth) non caricato correttamente.")
        else:
            # Lettura lista ticker
            t_list = pd.read_csv("tickers_SP500_2026.csv")['Ticker'].tolist()
            
            # Esecuzione Analisi (usiamo 10 cicli Monte Carlo per bilanciare precisione e velocità)
            res = fetch_and_predict(t_list, model, 10)
            
            if not res.empty:
                st.subheader("Analisi Completata!")
                res_sorted = res.sort_values('EVI', ascending=False)
                st.dataframe(res_sorted, use_container_width=True)
                
                # --- SALVATAGGIO E PULIZIA CACHE ---
                try:
                    conn = get_gsheet_connection()
                    # Sovrascrive il tab 'candidati' con i nuovi dati
                    conn.update(worksheet="candidati", data=res_sorted)
                    
                    # CRITICO: Puliamo la cache di Streamlit. 
                    # Senza questo, load_analisi_data() continuerebbe a leggere i vecchi dati per 1 ora.
                    st.cache_data.clear()
                    
                    st.success("✅ Nuovi risultati salvati nel database 'candidati'!")
                    st.balloons()
                    
                    # Ricarichiamo l'app per aggiornare lo stato della dashboard
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Errore durante il salvataggio su Google Sheets: {e}")
            else:
                st.error("❌ L'analisi non ha prodotto risultati. Controlla il log di debug.")

    # 3. Footer informativo
    if analisi_presente:
        st.caption(f"Ultimo aggiornamento rilevato: {len(df_analisi)} titoli analizzati.")
