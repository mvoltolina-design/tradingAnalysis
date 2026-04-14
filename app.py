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
    def __init__(self, input_dim=16, d_model=256, nhead=8, num_layers=4, dropout=0.1):
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
        return self.fc_out(x[:, -1, :]) # ELIMINA il dropout qui!

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
    path = "transformer_v8.1_refine_epoch8.pth"
    if os.path.exists(path):
        m = IrisTransformer(num_layers=4)
        m.load_state_dict(torch.load(path, map_location="cpu"))
        m.train() # Attivo Dropout per MC
        return m
    return None

#funzioni ereditate da daily predictor v08.08.py che da risultati più ottimistici
def get_market_data(symbol, vix_data):
    try:
        df = yf.download(symbol.replace('.', '-'), period="1y", progress=False, auto_adjust=True)
        df = clean_columns(df)
        if len(df) < 250: return None

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
        
        # Unione con VIX
        df = df.join(vix_data.rename("VIX_Index"), how='left').ffill()
        
        # Preparazione ultime 10 righe versione vecchia
        #df_final = df.dropna().tail(10).copy()
        #df_final['Lookback_Day'] = np.arange(1, 11).astype(float)
        
        # 1. Pulizia e salvataggio in variabile temporanea
        df_clean = df.dropna()
        # 2. Controllo dimensionale: se non ho almeno 10 giorni, inutile procedere
        if len(df_clean) < 10:
            return None
        # 3. Estrazione finale e aggiunta colonna temporale
        df_in = df_clean.tail(10).copy()
        df_in['Lookback_Day'] = np.arange(1, 11).astype(float)
        
        return df_in[COLS_ORDER].values # Restituisce matrice (10, 16)
    except Exception as e:
        print(f"Errore su {symbol}: {e}")
        return None
#funzioni ereditate da daily predictor v08.08.py che da risultati più ottimistici
def task_predict(model_path, ticker_list):
    # Caricamento VIX una sola volta Messo 1y anziché iniziale 1mo che dava risultati più ottimistici
    try:
        # Consiglio: '1y' è più sicuro di '1mo' per coprire i buchi nei weekend/festivi
        vix = yf.download("^VIX", period="1y", progress=False, auto_adjust=True)
        if isinstance(vix.columns, pd.MultiIndex):
            vix.columns = vix.columns.get_level_values(0)
        vix = clean_columns(vix)
        #new
        vix_close = vix['Close'].rename("VIX_Index") # Rinomino qui per sicurezza
        st.write("✅ VIX scaricato con successo.")
    except Exception as e:
        st.error(f"❌ Errore critico download VIX: {e}")
        vix_close = pd.Series(20.0, index=pd.date_range(end=datetime.now(), periods=500), name="VIX_Index")

    model = IrisTransformer()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.train() # MC Dropout attivo
    
    results = []
    print(f"Analisi di {len(ticker_list)} titoli in corso...")
    prog_bar = st.progress(0, text="Inizializzazione...")
    idx=0
    for symbol in ticker_list:
        idx=idx+1
        prog_bar.progress(idx / len(ticker_list), text=f"Analisi in corso: {symbol}...")
        features = get_market_data(symbol, vix_close)
        if features is not None:
            # Verifica che il blocco sia (10, 16)
            input_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
            
            mc_preds = []
            with torch.no_grad():
                for _ in range(30):
                    mc_preds.append(model(input_tensor).numpy())
            
            mc_preds = np.array(mc_preds)
            p_max = np.mean(mc_preds[:,:,3])
            p_min = np.mean(mc_preds[:,:,2])
            std_max = np.std(mc_preds[:,:,3])
            conf = 1 / (1 + std_max)
            
            results.append({
                'Ticker': symbol, 
                'EVI': conf * p_max, 
                'P_MAX': p_max, 
                'P_MIN': p_min, 
                'CONF': conf
            })
    
    return pd.DataFrame(results).sort_values('EVI', ascending=False)

# --- 3. ENGINE DI PREDIZIONE ---
def fetch_and_predict(ticker_list, model, cycles):
    st.write("🔍 **DEBUG START**: Inizio scaricamento dati VIX...")
    
    # 1. Recupero VIX
    try:
        # Consiglio: '1y' è più sicuro di '1mo' per coprire i buchi nei weekend/festivi
        vix = yf.download("^VIX", period="1y", progress=False, auto_adjust=True)
        if isinstance(vix.columns, pd.MultiIndex):
            vix.columns = vix.columns.get_level_values(0)
        vix = clean_columns(vix)
        #new
        vix_close = vix['Close'].rename("VIX_Index") # Rinomino qui per sicurezza
        st.write("✅ VIX scaricato con successo.")
    except Exception as e:
        st.error(f"❌ Errore critico download VIX: {e}")
        vix_close = pd.Series(20.0, index=pd.date_range(end=datetime.now(), periods=500), name="VIX_Index")

    results = []
    prog_bar = st.progress(0, text="Inizializzazione...")
    
    for idx, t in enumerate(ticker_list):
        prog_bar.progress((idx + 1) / len(ticker_list), text=f"Analisi in corso: {t}...")
        
        try:
            # 2. Download Dati Titolo
            tk_fixed = str(t).replace('.', '-')
            df = yf.download(tk_fixed, period="1y", progress=False, auto_adjust=True)
            
            if df.empty: continue
            
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = clean_columns(df)

            if len(df) < 250: continue

            # 3. Feature Engineering
            df['MA21'] = ta.sma(df['Close'], length=21)
            df['MA50'] = ta.sma(df['Close'], length=50)
            df['MA200'] = ta.sma(df['Close'], length=200)
            df['RSI'] = ta.rsi(df['Close'], length=14)
            
            macd_df = ta.macd(df['Close'])
            if macd_df is not None and not macd_df.empty:
                df['MACD'] = macd_df.iloc[:, 0]
                df['MACD_Signal'] = macd_df.iloc[:, 1]
                df['MACD_Hist'] = macd_df.iloc[:, 2] # Allineato a COLS_ORDER
            else:
                continue

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
            
            # Allineamento VIX: uso la serie rinominata
            df = df.join(vix_close, how='left')
            df['VIX_Index'] = df['VIX_Index'].ffill()

            # 4. Preparazione Tensor
            df_clean = df.dropna().copy()
            if len(df_clean) < 10: continue

            df_in = df_clean.tail(10).copy()
            df_in['Lookback_Day'] = np.arange(1, 11).astype(float)
            
            # Qui COLS_ORDER userà 'MACD_Hist' che abbiamo appena creato
            input_data = df_in[COLS_ORDER].values
            input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0)

            # 5. Inferenza Monte Carlo
            mc_preds = []
            with torch.no_grad():
                for _ in range(cycles):
                    # Essenziale: model.train() deve essere attivo per il dropout interno
                    pred = model(input_tensor).numpy()
                    mc_preds.append(pred)
            
            mc_preds = np.array(mc_preds)
            
            # Calcolo medie sui 30 cicli
            p_max = np.mean(mc_preds[:, :, 3])
            p_min = np.mean(mc_preds[:, :, 2])
            std_max = np.std(mc_preds[:, :, 3])
            conf = 1 / (1 + std_max)
            
            results.append({
                'Ticker': t, 
                'EVI': conf * p_max, 
                'P_MAX': p_max, 
                'P_MIN': p_min, 
                'CONF': conf
            })

        except Exception as e:
            st.warning(f"⚠️ Errore su {t}: {e}")
            continue
            
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
    
    # 1. Caricamento dati
    df_port_raw = load_portfolio().copy()
    
    if df_port_raw.empty:
        st.info("📭 Il portafoglio è attualmente vuoto.")
    else:
        # --- FILTRO STATO ---
        # Inseriamo un selettore per filtrare i titoli
        filtro_stato = st.radio("Filtra per stato:", ["Solo OPEN", "Solo CLOSE", "Tutti"], horizontal=True)
        
        if filtro_stato == "Solo OPEN":
            df_port = df_port_raw[df_port_raw['Stato'] == 'OPEN'].copy()
        elif filtro_stato == "Solo CLOSE":
            df_port = df_port_raw[df_port_raw['Stato'] == 'CLOSE'].copy()
        else:
            df_port = df_port_raw.copy()

        # --- LOGICA DI AGGIORNAMENTO DINAMICO ---
        # Aggiorniamo i prezzi solo se ci sono titoli OPEN nella vista attuale
        titoli_open = df_port[df_port['Stato'] == 'OPEN']
        
        if not titoli_open.empty:
            st.write("🔄 Aggiornamento massimi/minimi in corso per i titoli OPEN...")
            
            def extract_date(val):
                if hasattr(val, 'iloc'): val = val.iloc[0]
                if isinstance(val, tuple): val = val[1]
                return val
             
            for index, row in df_port.iterrows():
                if row['Stato'] == 'OPEN':
                    try:
                        ticker = str(row['Ticker']).strip()
                        
                        # --- MODIFICA CRUCIALE: Start Date + 1 giorno ---
                        # Trasformiamo la stringa in oggetto datetime e aggiungiamo un giorno
                        acquisto_dt = pd.to_datetime(row['Data_Acquisto'])
                        start_date_monitor = (acquisto_dt + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
                        
                        # Scarichiamo i dati dal giorno successivo all'acquisto in poi
                        data_yf = yf.download(ticker, start=start_date_monitor, progress=False)
                        
                        # Se è stato acquistato OGGI o ieri sera e non ci sono ancora candele "nuove"
                        # usiamo il prezzo di carico come base neutra
                        if data_yf.empty:
                            prezzo_carico = float(row['Prezzo_Carico'])
                            df_port.at[index, 'Max_Raggiunto'] = prezzo_carico
                            df_port.at[index, 'Min_Raggiunto'] = prezzo_carico
                            df_port.at[index, 'Max_Raggiunto%'] = 0.0
                            df_port.at[index, 'Min_Raggiunto%'] = 0.0
                            continue # Passa al prossimo titolo
                        
                        # --- Se abbiamo dati (siamo al giorno 2 o oltre) ---
                        raw_max = data_yf['High'].max()
                        raw_min = data_yf['Low'].min()
                        
                        val_max = float(raw_max.iloc[0]) if hasattr(raw_max, 'iloc') else float(raw_max)
                        val_min = float(raw_min.iloc[0]) if hasattr(raw_min, 'iloc') else float(raw_min)
                        
                        # Aggiornamento valori nel DataFrame
                        df_port.at[index, 'Max_Raggiunto'] = val_max
                        df_port.at[index, 'Min_Raggiunto'] = val_min
                        
                        # Calcolo percentuali rispetto al carico
                        prezzo_carico = float(row['Prezzo_Carico'])
                        if prezzo_carico > 0:
                            df_port.at[index, 'Max_Raggiunto%'] = (val_max - prezzo_carico) / prezzo_carico
                            df_port.at[index, 'Min_Raggiunto%'] = (val_min - prezzo_carico) / prezzo_carico

                        # Aggiornamento Date Massimi/Minimi
                        idx_max_raw = data_yf['High'].idxmax()
                        idx_min_raw = data_yf['Low'].idxmin()
                        date_max = extract_date(idx_max_raw)
                        date_min = extract_date(idx_min_raw)
                        
                        df_port.at[index, 'Data_Max'] = date_max.strftime('%Y-%m-%d')
                        df_port.at[index, 'Data_Min'] = date_min.strftime('%Y-%m-%d')

                    except Exception as e:
                        st.warning(f"⚠️ Errore aggiornamento {row.get('Ticker', 'N/D')}: {str(e)}")

        # 2. IDENTIFICAZIONE COLONNE PERCENTUALI
        target_perc = ["Est_Max", "Est_Min", "Confidence", "Max_Raggiunto%", "Min_Raggiunto%"]
        found_to_multiply = []

        for col in df_port.columns:
            nome_pulito = col.strip()
            if nome_pulito.endswith('%') or nome_pulito in target_perc:
                try:
                    df_port[col] = pd.to_numeric(df_port[col], errors='coerce') * 100
                    found_to_multiply.append(col)
                except: continue

        # 3. CONFIGURAZIONE VISIVA
        config_visuale = {
            "Ticker": st.column_config.TextColumn("Ticker", width="small"),
            "Prezzo_Carico": st.column_config.NumberColumn("Carico $", format="$ %.2f"),
            "Max_Raggiunto": st.column_config.NumberColumn("Max $", format="$ %.2f"),
            "Min_Raggiunto": st.column_config.NumberColumn("Min $", format="$ %.2f"),
            "Data_Max": st.column_config.TextColumn("Data Max"),
            "Data_Min": st.column_config.TextColumn("Data Min"),
            "Stato": st.column_config.TextColumn("Stato", width="small"),
        }
        
        for col in found_to_multiply:
            config_visuale[col] = st.column_config.NumberColumn(col, format="%.2f%%")

        st.dataframe(
            df_port, 
            use_container_width=True, 
            hide_index=True, 
            column_config=config_visuale
        )
        
        # 4. SALVATAGGIO
        if not titoli_open.empty:
            st.divider()
            if st.button("💾 Salva aggiornamenti su Database", key="save_portfolio_changes"):
                try:
                    # Carichiamo il portafoglio completo dal DB per non perdere i titoli CLOSE non visualizzati
                    df_full_db = load_portfolio().copy()
                    
                    # Aggiorniamo solo le righe dei titoli OPEN che abbiamo ricalcolato
                    for index, row in df_port.iterrows():
                        if row['Stato'] == 'OPEN':
                            # Troviamo la corrispondenza nel DF originale (non moltiplicato per 100)
                            # Riportiamo i valori in decimali
                            for col in found_to_multiply:
                                df_port.at[index, col] = df_port.at[index, col] / 100
                            
                            # Aggiorniamo il database completo
                            mask = (df_full_db['Ticker'] == row['Ticker']) & (df_full_db['Stato'] == 'OPEN')
                            for c in df_port.columns:
                                if c in df_full_db.columns:
                                    df_full_db.loc[mask, c] = df_port.at[index, c]

                    save_portfolio(df_full_db)
                    st.success("✅ Database sincronizzato correttamente!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Errore durante il salvataggio: {e}")
# ==========================================
# 2. SEZIONE AGGIUNGI TITOLO
# ==========================================
elif menu == "Aggiungi Titolo":
    st.header("🆕 Inserimento Nuova Posizione")
    df_analisi = load_analisi_data()
    
    t_in = st.text_input("Ticker (es. AAPL):").upper().strip()
    
    if t_in:
        # --- 1. GESTIONE PREZZO CON SESSION STATE ---
        # Se cambiamo ticker o il prezzo non esiste, lo inizializziamo
        if "last_ticker" not in st.session_state or st.session_state.last_ticker != t_in:
            st.session_state.last_ticker = t_in
            try:
                ticker_yf = yf.Ticker(t_in)
                hist = ticker_yf.history(period="1d")
                if not hist.empty:
                    st.session_state.market_price = float(hist['Close'].iloc[-1])
                else:
                    st.session_state.market_price = 0.00
            except:
                st.session_state.market_price = 0.00

        st.caption(f"📈 Ultimo prezzo di mercato rilevato: **{st.session_state.market_price:.2f} $**")

        # --- 2. CONTROLLO DATI ANALISI V8 ---
        match = df_analisi[df_analisi['Ticker'] == t_in] if 'Ticker' in df_analisi.columns else pd.DataFrame()
        default_max, default_min, default_conf = 0.0, 0.0, "N/D"
        
        if not match.empty:
            st.success(f"✅ Titolo trovato nell'Analisi V8!")
            default_max = float(match['P_MAX'].values[0])
            default_min = float(match['P_MIN'].values[0])
            default_conf = str(match['CONF'].values[0])

        # --- 3. FORM DI INSERIMENTO ---
        with st.form("form_aggiunta"):
            c1, c2 = st.columns(2)
            # USIAMO il valore salvato nel session_state, così non cambia durante il submit
            entry_price = c1.number_input("Prezzo di Carico ($)", min_value=0.0, value=st.session_state.market_price, format="%.2f")
            
            st.divider()
            st.subheader("Target di Analisi")
            col_a, col_b, col_c = st.columns(3)
            est_max = col_a.number_input("Estimated Max ($)", value=default_max)
            est_min = col_b.number_input("Estimated Min ($)", value=default_min)
            conf = col_c.text_input("Confidence Score", value=default_conf)
            
            if st.form_submit_button("Conferma Acquisto"):
                # ... (resto del tuo codice di salvataggio identico)
                df_p = load_portfolio()
                new_row = {
                    'Ticker': t_in,
                    'Data_Acquisto': datetime.now().strftime("%Y-%m-%d"),
                    'Prezzo_Carico': entry_price, # Ora prenderà il valore corretto del widget
                    'Max_Raggiunto': entry_price, 
                    'Max_Raggiunto%': 0.0,
                    'Min_Raggiunto': entry_price,
                    'Min_Raggiunto%': 0.0,
                    'Stato': 'OPEN',
                    'Est_Max': est_max / 100, 
                    'Est_Min': est_min / 100, 
                    'Confidence': float(conf) if conf.replace('.','',1).isdigit() else 0.0
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
    
    # Caricamento dati esistenti per visualizzazione immediata
    df_analisi = load_analisi_data()
    
    if not df_analisi.empty:
        st.success(f"✅ Ultima analisi caricata ({len(df_analisi)} titoli)")
        # Ordiniamo per EVI per vedere i migliori candidati
        df_display = df_analisi.sort_values('EVI', ascending=False)
        st.dataframe(df_display, use_container_width=True, hide_index=True)
    else:
        st.warning("⚠️ Nessuna analisi recente trovata. Avvia un ricalcolo.")

    # Pulsante di attivazione
    if st.button("🚀 AVVIA ANALISI S&P 500", key="run_analysis_v8"):
        with st.spinner("Caricamento modello e scaricamento dati S&P 500..."):
            
            #modello tratto da analisi dailyPredictor V08.08.py
            tickers = pd.read_csv("tickers_SP500_2026.csv")['Ticker'].tolist()
            st.info(f"Analisi avviata su {len(tickers)} titoli... Attendere circa 2-3 minuti.")
            res = task_predict("transformer_v8.1_refine_epoch8.pth", tickers)
            #
            
            ## 1. Caricamento Modello
            #model = load_v8_model()
            
            
            #if model is None:
            #    st.error("❌ Impossibile caricare il modello Transformer.")
            #elif not os.path.exists("tickers_SP500_2026.csv"):
            ##elif not os.path.exists("tickers_EU.csv"):
            #    st.error("❌ File 'tickers_SP500_2026.csv' non trovato.")
            #else:
            #    try:
            #        # 2. Lettura lista ticker dal tuo CSV
            #        t_list = pd.read_csv("tickers_SP500_2026.csv")['Ticker'].tolist()
            #        #t_list = pd.read_csv("tickers_EU.csv")['Ticker'].tolist()
            #        st.info(f"Analisi avviata su {len(t_list)} titoli... Attendere circa 2-3 minuti.")
            #        
            #        # 3. Esecuzione Predizione (Usiamo 'cycles' come definito nella tua funzione)
            #        res = fetch_and_predict(t_list, model, cycles=30)
                    
            if not res.empty:
                # Ordiniamo i risultati
                res_sorted = res.sort_values('EVI', ascending=False)
                
                # 4. Salvataggio su Google Sheets nel worksheet 'candidati'
                try:
                    conn = get_gsheet_connection()
                    conn.update(worksheet="candidati", data=res_sorted)
                    
                    st.success("✅ Analisi completata e salvata su Google Sheets!")
                    st.cache_data.clear() # Fondamentale per vedere i nuovi dati
                    st.rerun() 
                except Exception as ge:
                    st.error(f"❌ Errore durante il salvataggio su GSheet: {ge}")
            else:
                st.error("❌ L'analisi non ha prodotto risultati. Verifica i log o la connessione yfinance.")
                        
                #except Exception as e:
                #    st.error(f"❌ Errore critico durante l'analisi: {str(e)}")
              
