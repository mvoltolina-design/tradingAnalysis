import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
import torch
import torch.nn as nn
import os
from datetime import datetime, time as dtime
from streamlit_gsheets import GSheetsConnection

# ==============================================================================
# CONFIGURAZIONE GLOBALE
# ==============================================================================
MC_CYCLES = 20          # Cicli Monte Carlo (ridotti da 30 per velocità)
DROPOUT_RATE = 0.1      # Dropout per MC Uncertainty
COMMISSION_DEFAULT = 0.0019   # 0.19% default commissione broker
TAX_RATE_IT = 0.26      # 26% tassa sul capital gain (Italia)

COLONNE_PORTAFOGLIO = [
    'Ticker', 'Data_Acquisto', 'Ora_Acquisto', 'Prezzo_Carico',
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

# ==============================================================================
# 1. ARCHITETTURA MODELLO
# ==============================================================================
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
        return self.fc_out(x[:, -1, :]) # ELIMINA il dropout qui!

# ==============================================================================
# 2. FUNZIONI DI UTILITÀ GENERALI
# ==============================================================================
def clean_columns(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def safe_float(val, default=0.0):
    """Converte in float in modo sicuro, gestendo Series e tuple."""
    try:
        if hasattr(val, 'iloc'):
            val = val.iloc[0]
        if isinstance(val, tuple):
            val = val[1]
        return float(val)
    except Exception:
        return default


def format_pct(val, decimals=2):
    """Formatta un valore decimale come stringa percentuale."""
    try:
        return f"{float(val)*100:.{decimals}f}%"
    except Exception:
        return "N/D"


# ==============================================================================
# 3. CONNESSIONE GOOGLE SHEETS
# ==============================================================================
def get_gsheet_connection():
    return st.connection("gsheets", type=GSheetsConnection)


def load_portfolio():
    try:
        conn = get_gsheet_connection()
        df = conn.read(worksheet="Sheet1", ttl=0)
        if df is None or df.empty:
            return pd.DataFrame(columns=COLONNE_PORTAFOGLIO)
        df['Stato'] = df['Stato'].astype(str).str.strip()
        df['Ticker'] = df['Ticker'].astype(str).str.strip()
        # Aggiungi colonne mancanti per retrocompatibilità
        for col in COLONNE_PORTAFOGLIO:
            if col not in df.columns:
                df[col] = None
        # Ora_Acquisto: default a '09:30' se mancante o NaN
        if 'Ora_Acquisto' not in df.columns or df['Ora_Acquisto'].isnull().all():
            df['Ora_Acquisto'] = '09:30'
        df['Ora_Acquisto'] = df['Ora_Acquisto'].fillna('09:30').astype(str).str.strip()
        return df
    except Exception as e:
        st.error(f"Errore caricamento portafoglio: {e}")
        return pd.DataFrame(columns=COLONNE_PORTAFOGLIO)


def save_portfolio(df):
    try:
        conn = get_gsheet_connection()
        df_to_save = df[COLONNE_PORTAFOGLIO].copy()
        conn.update(worksheet="Sheet1", data=df_to_save)
        st.cache_data.clear()
    except Exception as e:
        st.error(f"Errore salvataggio portafoglio: {e}")


@st.cache_data(ttl=3600)
def load_analisi_data():
    try:
        conn = get_gsheet_connection()
        df = conn.read(worksheet="candidati", ttl=3600)
        if df is not None and not df.empty:
            df.columns = [str(c).strip() for c in df.columns]
            return df
        return pd.DataFrame(columns=['Ticker', 'P_MAX', 'P_MIN', 'CONF', 'EVI'])
    except Exception:
        return pd.DataFrame(columns=['Ticker', 'P_MAX', 'P_MIN', 'CONF', 'EVI'])


# ==============================================================================
# 4. FEATURE ENGINEERING & MARKET DATA
# ==============================================================================
@st.cache_data(ttl=3600)
def get_vix_data():
    """Scarica e cachea il VIX per 1h."""
    try:
        vix = yf.download("^VIX", period="1y", progress=False, auto_adjust=True)
        vix = clean_columns(vix)
        return vix['Close'].rename("VIX_Index")
    except Exception:
        return pd.Series(
            20.0,
            index=pd.date_range(end=datetime.now(), periods=500, freq='B'),
            name="VIX_Index"
        )


def get_market_data(symbol, vix_data):
    """
    Scarica e prepara i dati di mercato per un ticker.
    Restituisce la matrice (10, 16) delle ultime 10 sessioni.
    """
    try:
        df = yf.download(
            symbol.replace('.', '-'), period="1y",
            progress=False, auto_adjust=True
        )
        df = clean_columns(df)
        if len(df) < 250:
            return None

        df['MA21']  = ta.sma(df['Close'], length=21)
        df['MA50']  = ta.sma(df['Close'], length=50)
        df['MA200'] = ta.sma(df['Close'], length=200)
        df['RSI']   = ta.rsi(df['Close'], length=14)

        macd = ta.macd(df['Close'])
        if macd is not None and not macd.empty:
            df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = (
                macd.iloc[:, 0], macd.iloc[:, 1], macd.iloc[:, 2]
            )
        else:
            df['MACD'] = df['MACD_Signal'] = df['MACD_Hist'] = 0.0

        df['T_open']  = df['Open'].pct_change() * 100
        df['T_close'] = df['Close'].pct_change() * 100
        df['T_min']   = df['Low'].pct_change() * 100
        df['T_max']   = df['High'].pct_change() * 100

        df['Ratio_MA21']  = df['Close'] / df['MA21']
        df['Ratio_MA50']  = df['Close'] / df['MA50']
        df['Ratio_MA200'] = df['Close'] / df['MA200']
        df['Vol_Ratio']   = df['Volume'] / df['Volume'].rolling(20).mean()
        df['Is_Quarter_End'] = df.index.is_quarter_end.astype(float)
        df['Recent_Div'] = 0.0

        df = df.join(vix_data.rename("VIX_Index"), how='left').ffill()
        df_clean = df.dropna()

        if len(df_clean) < 10:
            return None

        df_in = df_clean.tail(10).copy()
        df_in['Lookback_Day'] = np.arange(1, 11).astype(float)
        return df_in[COLS_ORDER].values

    except Exception as e:
        print(f"Errore su {symbol}: {e}")
        return None


# ==============================================================================
# 5. MODELLO & INFERENZA
# ==============================================================================
@st.cache_resource
def load_v8_model():
    path = "transformer_v8.1_refine_epoch8.pth"
    if os.path.exists(path):
        m = IrisTransformer(num_layers=4, dropout=DROPOUT_RATE)
        m.load_state_dict(torch.load(path, map_location="cpu"))
        m.train()  # Mantiene Dropout attivo per MC
        return m
    return None


def mc_predict(model, input_tensor, cycles=MC_CYCLES):
    """
    Esegue l'inferenza Monte Carlo e restituisce statistiche.
    """
    mc_preds = []
    with torch.no_grad():
        for _ in range(cycles):
            mc_preds.append(model(input_tensor).numpy())
    mc_preds = np.array(mc_preds)  # (cycles, 1, 4)
    p_max  = float(np.mean(mc_preds[:, :, 3]))
    p_min  = float(np.mean(mc_preds[:, :, 2]))
    std_max = float(np.std(mc_preds[:, :, 3]))
    conf   = 1.0 / (1.0 + std_max)
    evi    = conf * p_max
    return p_max, p_min, conf, evi


def task_predict(ticker_list): # Rimosso model_path perché usiamo load_v8_model
    """Analisi batch di tutti i ticker S&P500."""
    vix_data = get_vix_data()
    st.write("✅ VIX caricato.")

    # Usiamo la funzione cacheata per evitare errori di dimensione pesi
    model = load_v8_model()
    
    if model is None:
        st.error("❌ Impossibile caricare il modello transformer_v8.1_refine_epoch8.pth")
        return pd.DataFrame()

    results = []
    prog_bar = st.progress(0, text="Inizializzazione...")

    for idx, symbol in enumerate(ticker_list):
        prog_bar.progress((idx + 1) / len(ticker_list), text=f"Analisi: {symbol}...")
        features = get_market_data(symbol, vix_data)
        
        if features is not None:
            # Assicurati che il tensore sia (Batch, Seq_Len, Features) -> (1, 10, 16)
            tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
            
            # Esegui MC Dropout
            p_max, p_min, conf, evi = mc_predict(model, tensor)
            
            results.append({
                'Ticker': symbol,
                'EVI': evi,
                'P_MAX': p_max,
                'P_MIN': p_min,
                'CONF': conf
            })

    return pd.DataFrame(results).sort_values('EVI', ascending=False)

# ==============================================================================
# 6. LOGICA PORTAFOGLIO — AGGIORNAMENTO METRICHE
# ==============================================================================
def get_entry_datetime(row):
    """
    Combina Data_Acquisto e Ora_Acquisto in un datetime timezone-naive.
    """
    try:
        data_str = str(row['Data_Acquisto']).strip()
        ora_str  = str(row.get('Ora_Acquisto', '09:30')).strip()
        if not ora_str or ora_str in ('nan', 'None', ''):
            ora_str = '09:30'
        return pd.to_datetime(f"{data_str} {ora_str}")
    except Exception:
        return pd.to_datetime(str(row['Data_Acquisto']))


def compute_portfolio_metrics(df_port):
    """
    Aggiorna Max/Min, rendimento giornaliero e di periodo per ogni OPEN.
    Tutti i calcoli partono dall'orario esatto di inserimento.
    """
    updated_rows = []

    for index, row in df_port.iterrows():
        if str(row.get('Stato', '')).strip() != 'OPEN':
            updated_rows.append(row)
            continue

        ticker      = str(row['Ticker']).strip()
        entry_dt    = get_entry_datetime(row)
        entry_price = safe_float(row['Prezzo_Carico'])
        ticker_yf   = ticker.replace('.', '-')

        try:
            # --- Scaricamento dati intraday/giornalieri ---
            # Usiamo interval='1h' per avere granularità oraria
            raw = yf.download(
                ticker_yf,
                start=entry_dt.strftime('%Y-%m-%d'),
                progress=False,
                auto_adjust=True,
                interval='1h'
            )
            raw = clean_columns(raw)

            if raw.empty:
                updated_rows.append(row)
                continue

            # Normalizza indice: rimuovi timezone per confronto sicuro
            if raw.index.tz is not None:
                raw.index = raw.index.tz_localize(None)

            # Filtra SOLO candele successive all'orario di acquisto
            raw_after = raw[raw.index > entry_dt]

            if raw_after.empty:
                # Acquistato oggi, nessuna candela successiva ancora
                row_copy = row.copy()
                row_copy['Max_Raggiunto']   = entry_price
                row_copy['Max_Raggiunto%']  = 0.0
                row_copy['Min_Raggiunto']   = entry_price
                row_copy['Min_Raggiunto%']  = 0.0
                row_copy['Data_Max']        = entry_dt.strftime('%Y-%m-%d %H:%M')
                row_copy['Data_Min']        = entry_dt.strftime('%Y-%m-%d %H:%M')
                updated_rows.append(row_copy)
                continue

            # --- Max / Min dal momento dell'acquisto ---
            real_max = safe_float(raw_after['High'].max())
            real_min = safe_float(raw_after['Low'].min())

            # Protezione: max non può scendere sotto il carico, min non può superarlo
            real_max = max(real_max, entry_price)
            real_min = min(real_min, entry_price)

            idx_max = raw_after['High'].idxmax()
            idx_min = raw_after['Low'].idxmin()
            date_max_str = pd.Timestamp(idx_max).strftime('%Y-%m-%d %H:%M')
            date_min_str = pd.Timestamp(idx_min).strftime('%Y-%m-%d %H:%M')

            row_copy = row.copy()
            row_copy['Max_Raggiunto']  = real_max
            row_copy['Max_Raggiunto%'] = (real_max - entry_price) / entry_price if entry_price else 0.0
            row_copy['Data_Max']       = date_max_str
            row_copy['Min_Raggiunto']  = real_min
            row_copy['Min_Raggiunto%'] = (real_min - entry_price) / entry_price if entry_price else 0.0
            row_copy['Data_Min']       = date_min_str
            updated_rows.append(row_copy)

        except Exception as e:
            st.warning(f"⚠️ Errore aggiornamento {ticker}: {e}")
            updated_rows.append(row)

    return pd.DataFrame(updated_rows)


def get_current_prices(tickers):
    """
    Recupera i prezzi correnti e la variazione intraday per una lista di ticker.
    Restituisce un dict {ticker: {price, change_day_pct, open_price}}
    """
    result = {}
    for ticker in tickers:
        try:
            tk = yf.Ticker(ticker.replace('.', '-'))
            info = tk.fast_info
            price = float(info.last_price) if hasattr(info, 'last_price') else None
            prev_close = float(info.previous_close) if hasattr(info, 'previous_close') else None
            if price and prev_close and prev_close > 0:
                change_day = (price - prev_close) / prev_close
            else:
                change_day = None
            result[ticker] = {
                'price': price,
                'change_day_pct': change_day,
                'prev_close': prev_close
            }
        except Exception:
            result[ticker] = {'price': None, 'change_day_pct': None, 'prev_close': None}
    return result


# ==============================================================================
# 7. CALCOLO RENDIMENTO NETTO COMMISSIONI & TASSE
# ==============================================================================
def compute_net_return(entry_price, current_price, shares, commission_pct, tax_rate):
    """
    Calcola il rendimento netto considerando:
    - Commissioni di acquisto e vendita
    - Tassa sul capital gain (solo su profitti)

    Restituisce un dict con tutte le componenti.
    """
    if not entry_price or not current_price or not shares:
        return None

    cost_buy   = entry_price * shares
    cost_sell  = current_price * shares
    comm_buy   = cost_buy * commission_pct
    comm_sell  = cost_sell * commission_pct

    gross_profit = cost_sell - cost_buy - comm_buy - comm_sell

    # Le tasse si applicano SOLO se c'è un profitto
    tax = max(0.0, gross_profit * tax_rate)
    net_profit = gross_profit - tax

    net_return_pct = net_profit / (cost_buy + comm_buy) if (cost_buy + comm_buy) > 0 else 0.0

    return {
        'cost_buy':       cost_buy,
        'comm_buy':       comm_buy,
        'comm_sell':      comm_sell,
        'gross_profit':   gross_profit,
        'tax':            tax,
        'net_profit':     net_profit,
        'net_return_pct': net_return_pct,
        'breakeven_price': (cost_buy + comm_buy) * (1 + commission_pct) / shares
    }


# ==============================================================================
# 8. INTERFACCIA STREAMLIT
# ==============================================================================
st.set_page_config(
    page_title="V8 Tracker",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Stile CSS minimalista professionale
st.markdown("""
<style>
    .metric-card {
        background: #1a1a2e;
        border: 1px solid #16213e;
        border-radius: 8px;
        padding: 16px;
        margin: 4px 0;
    }
    .positive { color: #00d4aa; }
    .negative { color: #ff4757; }
    .neutral  { color: #a0a0b0; }
    .section-header {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: #6c6c8a;
        margin-bottom: 8px;
    }
    div[data-testid="stMetricValue"] { font-size: 1.1rem; }
</style>
""", unsafe_allow_html=True)

menu = st.sidebar.selectbox(
    "🗂 Navigazione",
    ["📊 Dashboard Portafoglio", "➕ Aggiungi Posizione", "🎯 Analisi V8", "🧮 Simulatore Rendimento"]
)

# ==============================================================================
# SEZIONE 1 — DASHBOARD PORTAFOGLIO
# ==============================================================================
if menu == "📊 Dashboard Portafoglio":
    st.header("📊 Dashboard Portafoglio")

    df_raw = load_portfolio()

    if df_raw.empty:
        st.info("📭 Il portafoglio è vuoto. Aggiungi la prima posizione.")
        st.stop()

    # --- Filtro stato ---
    col_f1, col_f2 = st.columns([3, 1])
    with col_f1:
        filtro = st.radio(
            "Filtra:", ["Solo OPEN", "Solo CLOSE", "Tutti"],
            horizontal=True
        )
    with col_f2:
        aggiorna = st.button("🔄 Aggiorna Dati", type="primary")

    if filtro == "Solo OPEN":
        df_port = df_raw[df_raw['Stato'] == 'OPEN'].copy()
    elif filtro == "Solo CLOSE":
        df_port = df_raw[df_raw['Stato'] == 'CLOSE'].copy()
    else:
        df_port = df_raw.copy()

    open_mask = df_port['Stato'] == 'OPEN'
    open_tickers = df_port.loc[open_mask, 'Ticker'].tolist()

    # --- Aggiornamento metriche ---
    if aggiorna and open_tickers:
        with st.spinner("Aggiornamento metriche in corso..."):
            df_port = compute_portfolio_metrics(df_port)

    # --- Recupero prezzi correnti per indicatori intraday ---
    current_prices = {}
    if open_tickers:
        with st.spinner("Recupero prezzi correnti..."):
            current_prices = get_current_prices(open_tickers)

    # --- KPI GLOBALI ---
    st.divider()
    st.markdown('<p class="section-header">Riepilogo Portafoglio</p>', unsafe_allow_html=True)
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

    total_invested  = 0.0
    total_current   = 0.0
    profitable      = 0
    losing          = 0
    daily_gains     = []

    for _, row in df_port[open_mask].iterrows():
        ticker = str(row['Ticker']).strip()
        ep = safe_float(row['Prezzo_Carico'])
        cp = current_prices.get(ticker, {}).get('price')
        if ep and cp:
            total_invested += ep
            total_current  += cp
            change = (cp - ep) / ep
            if change >= 0: profitable += 1
            else:           losing    += 1
        cd = current_prices.get(ticker, {}).get('change_day_pct')
        if cd is not None:
            daily_gains.append(cd)

    total_return_pct = (total_current - total_invested) / total_invested if total_invested else 0
    avg_daily = np.mean(daily_gains) if daily_gains else 0

    kpi1.metric("Posizioni Aperte",    len(df_port[open_mask]))
    kpi2.metric("In Profitto",         profitable, delta=None)
    kpi3.metric("In Perdita",          losing,     delta=None)
    kpi4.metric("Rendimento Medio Portafoglio",
                f"{total_return_pct*100:.2f}%",
                delta=f"{total_return_pct*100:.2f}%")
    kpi5.metric("Rendimento Medio Giornaliero",
                f"{avg_daily*100:.2f}%",
                delta=f"{avg_daily*100:.2f}%")

    st.divider()

    # --- TABELLA DETTAGLIO ---
    st.markdown('<p class="section-header">Dettaglio Posizioni</p>', unsafe_allow_html=True)

    rows_display = []
    for _, row in df_port.iterrows():
        ticker = str(row['Ticker']).strip()
        ep     = safe_float(row['Prezzo_Carico'])
        stato  = str(row.get('Stato', '')).strip()

        cp_info = current_prices.get(ticker, {})
        cp      = cp_info.get('price')
        cd      = cp_info.get('change_day_pct')

        rend_totale = (cp - ep) / ep if (cp and ep) else None
        rend_oggi   = cd

        entry_dt_str = str(row.get('Data_Acquisto', ''))
        ora_str      = str(row.get('Ora_Acquisto', '09:30'))

        display_row = {
            'Ticker':          ticker,
            'Data Acquisto':   f"{entry_dt_str} {ora_str}",
            'Carico $':        ep,
            'Prezzo Attuale':  cp if cp else 'N/D',
            '∆ Oggi %':        f"{rend_oggi*100:+.2f}%" if rend_oggi is not None else 'N/D',
            '∆ Totale %':      f"{rend_totale*100:+.2f}%" if rend_totale is not None else 'N/D',
            'Max Raggiunto $': safe_float(row.get('Max_Raggiunto')),
            'Max %':           safe_float(row.get('Max_Raggiunto%')),
            'Min Raggiunto $': safe_float(row.get('Min_Raggiunto')),
            'Min %':           safe_float(row.get('Min_Raggiunto%')),
            'Data Max':        str(row.get('Data_Max', '')),
            'Data Min':        str(row.get('Data_Min', '')),
            'Stato':           stato,
            'Est Max %':       safe_float(row.get('Est_Max')),
            'Est Min %':       safe_float(row.get('Est_Min')),
            'Confidence':      safe_float(row.get('Confidence')),
        }
        rows_display.append(display_row)

    df_display = pd.DataFrame(rows_display)

    # Formattazione colonne percentuali
    pct_cols = ['Max %', 'Min %', 'Est Max %', 'Est Min %']
    for c in pct_cols:
        df_display[c] = pd.to_numeric(df_display[c], errors='coerce') * 100

    col_cfg = {
        'Ticker':          st.column_config.TextColumn("Ticker", width="small"),
        'Data Acquisto':   st.column_config.TextColumn("Data/Ora Acquisto"),
        'Carico $':        st.column_config.NumberColumn("Carico $",  format="$ %.2f"),
        'Prezzo Attuale':  st.column_config.TextColumn("Prezzo Live"),
        '∆ Oggi %':        st.column_config.TextColumn("∆ Oggi"),
        '∆ Totale %':      st.column_config.TextColumn("∆ Totale"),
        'Max Raggiunto $': st.column_config.NumberColumn("Max $",  format="$ %.2f"),
        'Max %':           st.column_config.NumberColumn("Max %",  format="%.2f%%"),
        'Min Raggiunto $': st.column_config.NumberColumn("Min $",  format="$ %.2f"),
        'Min %':           st.column_config.NumberColumn("Min %",  format="%.2f%%"),
        'Data Max':        st.column_config.TextColumn("Data Max"),
        'Data Min':        st.column_config.TextColumn("Data Min"),
        'Stato':           st.column_config.TextColumn("Stato", width="small"),
        'Est Max %':       st.column_config.NumberColumn("Est Max %", format="%.2f%%"),
        'Est Min %':       st.column_config.NumberColumn("Est Min %", format="%.2f%%"),
        'Confidence':      st.column_config.NumberColumn("Conf", format="%.3f"),
    }

    st.dataframe(df_display, use_container_width=True, hide_index=True, column_config=col_cfg)

    # --- SALVA ---
    if aggiorna and open_tickers:
        st.divider()
        if st.button("💾 Salva aggiornamenti su Google Sheets"):
            try:
                df_full = load_portfolio().copy()
                for _, row in df_port[open_mask].iterrows():
                    mask = (df_full['Ticker'] == row['Ticker']) & (df_full['Stato'] == 'OPEN')
                    for c in ['Max_Raggiunto', 'Max_Raggiunto%', 'Min_Raggiunto',
                              'Min_Raggiunto%', 'Data_Max', 'Data_Min']:
                        if c in df_port.columns and c in df_full.columns:
                            df_full.loc[mask, c] = row.get(c)
                save_portfolio(df_full)
                st.success("✅ Aggiornamento salvato!")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Errore salvataggio: {e}")


# ==============================================================================
# SEZIONE 2 — AGGIUNGI POSIZIONE
# ==============================================================================
elif menu == "➕ Aggiungi Posizione":
    st.header("➕ Inserimento Nuova Posizione")

    df_analisi = load_analisi_data()
    t_in = st.text_input("Ticker (es. AAPL):").upper().strip()

    if t_in:
        # Session state per prezzo e ora
        if st.session_state.get("last_ticker") != t_in:
            st.session_state.last_ticker = t_in
            try:
                info = yf.Ticker(t_in.replace('.', '-')).fast_info
                st.session_state.market_price = float(info.last_price)
            except Exception:
                st.session_state.market_price = 0.0

        market_price = st.session_state.get("market_price", 0.0)
        st.caption(f"📈 Ultimo prezzo rilevato: **${market_price:.2f}**")

        # Dati da analisi V8 se disponibili
        match = df_analisi[df_analisi['Ticker'] == t_in] if 'Ticker' in df_analisi.columns else pd.DataFrame()
        default_max, default_min, default_conf = 0.0, 0.0, 0.0
        if not match.empty:
            st.success("✅ Ticker trovato nell'ultima analisi V8!")
            default_max  = float(match['P_MAX'].values[0])
            default_min  = float(match['P_MIN'].values[0])
            default_conf = float(match['CONF'].values[0])

        with st.form("form_aggiunta"):
            c1, c2, c3 = st.columns(3)
            entry_price = c1.number_input(
                "Prezzo di Carico ($)", min_value=0.0,
                value=market_price, format="%.2f"
            )
            entry_date = c2.date_input("Data Acquisto", value=datetime.now().date())
            entry_time = c3.time_input("Ora Acquisto (mercato locale)",
                                       value=dtime(9, 30))

            st.subheader("🎯 Target V8")
            ca, cb, cc = st.columns(3)
            est_max = ca.number_input("Est Max (%)", value=default_max, format="%.4f")
            est_min = cb.number_input("Est Min (%)", value=default_min, format="%.4f")
            conf_in = cc.number_input("Confidence", value=default_conf, format="%.4f", min_value=0.0, max_value=1.0)

            submitted = st.form_submit_button("✅ Conferma Acquisto", type="primary")

        if submitted:
            df_p = load_portfolio()
            new_row = {
                'Ticker':          t_in,
                'Data_Acquisto':   entry_date.strftime("%Y-%m-%d"),
                'Ora_Acquisto':    entry_time.strftime("%H:%M"),
                'Prezzo_Carico':   entry_price,
                'Max_Raggiunto':   entry_price,
                'Max_Raggiunto%':  0.0,
                'Data_Max':        entry_date.strftime("%Y-%m-%d"),
                'Min_Raggiunto':   entry_price,
                'Min_Raggiunto%':  0.0,
                'Data_Min':        entry_date.strftime("%Y-%m-%d"),
                'Stato':           'OPEN',
                'Est_Max':         est_max / 100,
                'Est_Min':         est_min / 100,
                'Confidence':      conf_in,
            }
            df_p = pd.concat([df_p, pd.DataFrame([new_row])], ignore_index=True)
            save_portfolio(df_p)
            st.balloons()
            st.success(f"✅ Posizione su **{t_in}** inserita a ${entry_price:.2f} — {entry_date} {entry_time.strftime('%H:%M')}")


# ==============================================================================
# SEZIONE 3 — ANALISI V8
# ==============================================================================
elif menu == "🎯 Analisi V8":
    st.header("🎯 Analisi Predittiva V8")
    st.caption(f"Modello: IrisTransformer v8.1 | MC Cycles: {MC_CYCLES} | Dropout: {DROPOUT_RATE}")

    df_analisi = load_analisi_data()

    if not df_analisi.empty:
        st.success(f"✅ Ultima analisi disponibile — {len(df_analisi)} titoli")
        st.dataframe(
            df_analisi.sort_values('EVI', ascending=False),
            use_container_width=True, hide_index=True
        )
    else:
        st.warning("⚠️ Nessuna analisi recente. Avvia il ricalcolo.")

    st.divider()

    if st.button("🚀 AVVIA ANALISI S&P 500", type="primary"):
        model_path = "transformer_v8.1_refine_epoch8.pth"
        if not os.path.exists(model_path):
            st.error(f"❌ Modello non trovato: {model_path}")
        elif not os.path.exists("tickers_SP500_2026.csv"):
            st.error("❌ File tickers_SP500_2026.csv non trovato.")
        else:
            tickers = pd.read_csv("tickers_SP500_2026.csv")['Ticker'].tolist()
            st.info(f"Analisi su {len(tickers)} titoli — {MC_CYCLES} cicli MC — attendi ~2 min.")

            with st.spinner("Elaborazione..."):
                res = task_predict(model_path, tickers)

            if not res.empty:
                res_sorted = res.sort_values('EVI', ascending=False)
                try:
                    conn = get_gsheet_connection()
                    conn.update(worksheet="candidati", data=res_sorted)
                    st.success("✅ Analisi completata e salvata su Google Sheets!")
                    st.cache_data.clear()
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Errore salvataggio GSheet: {e}")
                    st.dataframe(res_sorted)
            else:
                st.error("❌ Nessun risultato. Verifica connessione yfinance.")


# ==============================================================================
# SEZIONE 4 — SIMULATORE RENDIMENTO NETTO
# ==============================================================================
elif menu == "🧮 Simulatore Rendimento":
    st.header("🧮 Simulatore Rendimento Netto")
    st.caption("Calcola il rendimento al netto di commissioni e tasse sul capital gain.")

    # --- Input ---
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("💼 Posizione")
        sim_ticker = st.text_input("Ticker (opzionale, per recupero prezzo)").upper().strip()

        if sim_ticker:
            try:
                cp_live = float(yf.Ticker(sim_ticker.replace('.', '-')).fast_info.last_price)
                st.caption(f"Prezzo live: ${cp_live:.2f}")
            except Exception:
                cp_live = 0.0
        else:
            cp_live = 0.0

        sim_entry  = st.number_input("Prezzo di Acquisto ($)", min_value=0.01, value=max(cp_live, 1.0), format="%.4f")
        sim_exit   = st.number_input("Prezzo di Vendita / Attuale ($)", min_value=0.01, value=max(cp_live * 1.05, 1.05), format="%.4f")
        sim_shares = st.number_input("Numero di Azioni", min_value=1, value=10, step=1)

    with col2:
        st.subheader("⚙️ Parametri")
        sim_comm = st.slider(
            "Commissione Broker (%)",
            min_value=0.0, max_value=1.0,
            value=COMMISSION_DEFAULT * 100,
            step=0.01, format="%.2f"
        ) / 100

        sim_tax = st.slider(
            "Aliquota Capital Gain (%)",
            min_value=0.0, max_value=50.0,
            value=TAX_RATE_IT * 100,
            step=0.5, format="%.1f"
        ) / 100

        st.info(f"🇮🇹 Aliquota italiana standard: **26%**\n\nModificabile per altri paesi o regimi fiscali.")

    st.divider()

    # --- Calcolo ---
    result = compute_net_return(sim_entry, sim_exit, sim_shares, sim_comm, sim_tax)

    if result:
        gross_pct = (sim_exit - sim_entry) / sim_entry
        net_pct   = result['net_return_pct']

        st.subheader("📊 Risultati")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Rendimento Lordo",      f"{gross_pct*100:+.2f}%")
        m2.metric("Rendimento Netto",      f"{net_pct*100:+.2f}%",
                  delta=f"{(net_pct - gross_pct)*100:.2f}% effetto costi")
        m3.metric("Utile/Perdita Netto",   f"${result['net_profit']:+.2f}")
        m4.metric("Breakeven Price",       f"${result['breakeven_price']:.4f}")

        st.divider()
        st.subheader("📋 Dettaglio Costi")

        detail_df = pd.DataFrame([
            {"Voce": "Costo Acquisto (lordo)",   "Importo $": result['cost_buy']},
            {"Voce": "Commissione Acquisto",      "Importo $": result['comm_buy']},
            {"Voce": "Ricavo Vendita (lordo)",    "Importo $": result['cost_buy'] + result['gross_profit'] + result['comm_sell']},
            {"Voce": "Commissione Vendita",       "Importo $": result['comm_sell']},
            {"Voce": "Utile Lordo (pre-tasse)",   "Importo $": result['gross_profit']},
            {"Voce": f"Tasse Capital Gain ({sim_tax*100:.1f}%)", "Importo $": result['tax']},
            {"Voce": "✅ Utile Netto",             "Importo $": result['net_profit']},
        ])
        st.dataframe(
            detail_df,
            hide_index=True,
            use_container_width=True,
            column_config={
                "Importo $": st.column_config.NumberColumn("Importo $", format="$ %.4f")
            }
        )

        # --- Analisi di sensitività ---
        st.divider()
        st.subheader("📉 Sensitività al Prezzo di Uscita")

        price_range = np.linspace(sim_entry * 0.85, sim_entry * 1.30, 40)
        sens_rows = []
        for p in price_range:
            r = compute_net_return(sim_entry, p, sim_shares, sim_comm, sim_tax)
            if r:
                sens_rows.append({
                    'Prezzo Uscita': round(p, 2),
                    'Netto %':       round(r['net_return_pct'] * 100, 2),
                    'Lordo %':       round((p - sim_entry) / sim_entry * 100, 2),
                })

        df_sens = pd.DataFrame(sens_rows)
        st.line_chart(df_sens.set_index('Prezzo Uscita')[['Lordo %', 'Netto %']])

        # Evidenzia breakeven
        st.caption(f"🎯 Il breakeven netto (incluse commissioni) è a **${result['breakeven_price']:.4f}**")

    st.divider()
    st.markdown("""
    **Note:**
    - La tassa sul capital gain in Italia è **26%** e si applica solo sulle plusvalenze.
    - Le minusvalenze possono compensare plusvalenze future (regime amministrato).
    - Le commissioni inserite si intendono applicate sia all'acquisto che alla vendita.
    - Questo simulatore è puramente indicativo — consulta un commercialista per la tua situazione specifica.
    """)
