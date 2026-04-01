import streamlit as st
import yfinance as yf
import pandas as pd
import os
from datetime import datetime

# --- 1. ARCHITETTURA SPECIFICA PER EPOCH 09 -
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

# --- 2. LOGICA DI CALCOLO ---
def clean_columns(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [str(c).strip() for c in df.columns]
    return df

@st.cache_data(ttl=3600,show_spinner=False)
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


# --- CONFIGURAZIONE FILE PERSISTENTE ---
PORTFOLIO_FILE = "portfolio_v8.csv"

def load_portfolio():
    if os.path.exists(PORTFOLIO_FILE):
        return pd.read_csv(PORTFOLIO_FILE)
    else:
        return pd.DataFrame(columns=[
            'Ticker', 'Data_Acquisto', 'Prezzo_Carico', 
            'Max_Raggiunto', 'Data_Max', 'Min_Raggiunto', 'Data_Min', 'Stato'
        ])

def save_portfolio(df):
    df.to_csv(PORTFOLIO_FILE, index=False)

# --- FUNZIONE DI AGGIORNAMENTO PREZZI ---
def update_portfolio_metrics():
    df = load_portfolio()
    if df.empty: return df
    
    active_indices = df[df['Stato'] == 'OPEN'].index
    if len(active_indices) == 0: return df

    # Creiamo una lista di ticker per fare un download unico (più veloce)
    tickers_to_update = df.loc[active_indices, 'Ticker'].unique().tolist()
    
    try:
        # Download bulk di tutti i ticker attivi
        data_bulk = yf.download([t.replace('.', '-') for t in tickers_to_update], 
                                period="1d", progress=False, auto_adjust=True)
        
        for idx in active_indices:
            ticker = df.at[idx, 'Ticker']
            t_search = ticker.replace('.', '-')
            
            # Gestione dati se MultiIndex (più ticker) o SingleIndex (un solo ticker)
            if len(tickers_to_update) > 1:
                current_p = float(data_bulk['Close'][t_search].iloc[-1])
            else:
                current_p = float(data_bulk['Close'].iloc[-1])

            if np.isnan(current_p): continue

            # --- LOGICA DI AGGIORNAMENTO MASSIMI E MINIMI ---
            
            # Aggiorna il Massimo se il prezzo attuale è il più alto mai visto dall'acquisto
            if current_p > df.at[idx, 'Max_Raggiunto']:
                df.at[idx, 'Max_Raggiunto'] = current_p
                df.at[idx, 'Data_Max'] = datetime.now().strftime("%d/%m %H:%M")
            
            # Aggiorna il Minimo se il prezzo attuale è il più basso mai visto dall'acquisto
            # IMPORTANTE: Se è la prima volta, il min è uguale al prezzo di carico
            if current_p < df.at[idx, 'Min_Raggiunto']:
                df.at[idx, 'Min_Raggiunto'] = current_p
                df.at[idx, 'Data_Min'] = datetime.now().strftime("%d/%m %H:%M")
                
    except Exception as e:
        st.warning(f"Errore durante l'aggiornamento prezzi: {e}")
                
    save_portfolio(df)
    return df

# --- INTERFACCIA ---
st.title("📂 V8 Multi-Day Portfolio Tracker")

menu = st.sidebar.selectbox("Menu", ["Dashboard Portafoglio", "Aggiungi Titolo", "Analisi V8"])

if menu == "Aggiungi Titolo":
    st.header("🛒 Nuovo Acquisto")
    t_input = st.text_input("Ticker:").upper()
    if t_input:
        data_yf = yf.download(t_input.replace('.', '-'), period="1d", progress=False, auto_adjust=True)
        if not data_yf.empty:
            # Gestione sicura delle colonne (rimuove il MultiIndex se presente)
            if isinstance(data_yf.columns, pd.MultiIndex):
                data_yf.columns = data_yf.columns.get_level_values(0)
            
            # Estrazione sicura dell'ultimo prezzo di chiusura
            current_p = float(data_yf['Close'].iloc[-1])
            st.metric("Prezzo Attuale", f"${current_p:.2f}")
            
            entry_p = st.number_input("Conferma Prezzo d'Acquisto:", value=current_p)
            
            if st.button("Inserisci in Portafoglio"):
                df = load_portfolio()
                new_row = {
                    'Ticker': t_input,
                    'Data_Acquisto': datetime.now().strftime("%Y-%m-%d"),
                    'Prezzo_Carico': entry_p,
                    'Max_Raggiunto': current_p,
                    'Data_Max': datetime.now().strftime("%d/%m %H:%M"),
                    'Min_Raggiunto': current_p,
                    'Data_Min': datetime.now().strftime("%d/%m %H:%M"),
                    'Stato': 'OPEN'
                }
                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                save_portfolio(df)
                st.success(f"{t_input} aggiunto al portafoglio di oggi!")
        else:
            st.error(f"Nessun dato trovato per il ticker {t_input}. Verifica il simbolo.")

elif menu == "Dashboard Portafoglio":
    st.header("📈 Monitoraggio Titoli Attivi")
    df = update_portfolio_metrics()
    
    if df.empty:
        st.info("Nessun titolo in portafoglio.")
    else:
        # Raggruppiamo per data di acquisto
        date_groups = df[df['Stato'] == 'OPEN']['Data_Acquisto'].unique()
        
        for d in sorted(date_groups, reverse=True):
            with st.expander(f"📅 Portfolio del {d}", expanded=True):
                sub_df = df[(df['Data_Acquisto'] == d) & (df['Stato'] == 'OPEN')]
                
                for _, row in sub_df.iterrows():
                    # Calcolo percentuali basate sul prezzo d'acquisto utente
                    def perc(val): return ((val - row['Prezzo_Carico']) / row['Prezzo_Carico']) * 100
                    # Funzione interna per calcolare la percentuale rispetto al carico
                    def calc_perc(val): 
                        return ((val - row['Prezzo_Carico']) / row['Prezzo_Carico']) * 100
                    # Layout a 3 colonne: Info Titolo | Massimo | Minimo
                    c1, c2, c3 = st.columns([1.2, 2, 2])
                    with c1:
                        st.subheader(f" {row['Ticker']}")
                        st.caption(f"💰 Carico: **${row['Prezzo_Carico']:.2f}**")
                        st.caption(f"📅 Inizio: {row['Data_Acquisto']}")
                    
                    with c2:
                        st.markdown("##### 🚀 Massimo")
                        diff_max = calc_perc(row['Max_Raggiunto'])
                        # Mostriamo Valore Assoluto + Percentuale
                        st.write(f"**${row['Max_Raggiunto']:.2f}** ({diff_max:+.2f}%)")
                        st.caption(f"🕒 {row['Data_Max']}")
                    
                    with c3:
                        st.markdown("##### ⚠️ Minimo")
                        diff_min = calc_perc(row['Min_Raggiunto'])
                        # Mostriamo Valore Assoluto + Percentuale
                        st.write(f"**${row['Min_Raggiunto']:.2f}** ({diff_min:+.2f}%)")
                        st.caption(f"🕒 {row['Data_Min']}")
                    
                    # Bottone per chiudere l'operazione
                    if st.button(f"Chiudi {row['Ticker']}", key=f"btn_close_{row['Ticker']}_{row['Data_Acquisto']}"):
                        df.loc[(df['Ticker'] == row['Ticker']) & (df['Data_Acquisto'] == row['Data_Acquisto']), 'Stato'] = 'CLOSED'
                        save_portfolio(df)
                        st.success(f"Posizione su {row['Ticker']} spostata nello storico.")
                        st.rerun()
                    
                    st.divider()
elif menu == "Analisi V8":
    st.header("🎯 Deep Analysis - Epoch 11")
    
    # Caricamento Modello (Assicurati che num_layers sia 4 come abbiamo appurato)
    @st.cache_resource
    def load_v8_model():
        model_path = "transformer_v8_epoch09.pth"
        if os.path.exists(model_path):
            m = IrisTransformer(num_layers=4) # Architettura corretta a 4 layer
            m.load_state_dict(torch.load(model_path, map_location="cpu"))
            m.train() # Importante per Monte Carlo Dropout
            return m
        return None

    model_v8 = load_v8_model()

    if model_v8 is None:
        st.error("File modello .pth non trovato nel repository!")
    else:
        if os.path.exists("tickers_SP500_2026.csv"):
            df_tickers = pd.read_csv("tickers_SP500_2026.csv")
            ticker_list = df_tickers['Ticker'].tolist()
            
            st.write(f"Pronto ad analizzare **{len(ticker_list)}** titoli dello S&P 500.")
            cycles = st.slider("Cicli Monte Carlo (Precisione)", 10, 100, 30)

            if st.button("🚀 AVVIA ANALISI COMPLETA", use_container_width=True):
                # Chiamata alla funzione di predizione (senza cache per vedere la barra)
                with st.container():
                    results_df = fetch_and_predict(ticker_list, model_v8, cycles)
                
                if not results_df.empty:
                    st.success("✅ Analisi completata con successo!")
                    
                    # Ordiniamo per EVI (Expected Value Index)
                    results_df = results_df.sort_values('EVI', ascending=False)
                    
                    # Visualizzazione Tabella Risultati
                    st.dataframe(
                        results_df[['Ticker', 'EVI', 'P_MAX', 'P_MIN', 'CONF']].style.format({
                            'EVI': '{:.2f}', 
                            'P_MAX': '{:.2f}%', 
                            'P_MIN': '{:.2f}%', 
                            'CONF': '{:.2f}'
                        }), 
                        use_container_width=True
                    )
                else:
                    st.error("L'analisi non ha prodotto risultati. Verifica i log.")
        else:
            st.error("File tickers_SP500_2026.csv non trovato.")
