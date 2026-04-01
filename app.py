import streamlit as st
import yfinance as yf
import pandas as pd
import os
from datetime import datetime

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

    for idx in active_indices:
        ticker = df.at[idx, 'Ticker']
        # Usiamo auto_adjust=True per prezzi puliti da dividendi
        data = yf.download(ticker.replace('.', '-'), period="1d", progress=False, auto_adjust=True)
        
        if not data.empty:
            # --- FIX CRUCIALE: Gestione MultiIndex ---
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            
            # Ora 'Close' è accessibile in modo sicuro
            current_p = float(data['Close'].iloc[-1])
            
            # Aggiorna Massimo Reale
            if current_p > df.at[idx, 'Max_Raggiunto']:
                df.at[idx, 'Max_Raggiunto'] = current_p
                df.at[idx, 'Data_Max'] = datetime.now().strftime("%d/%m %H:%M")
            
            # Aggiorna Minimo Reale
            if current_p < df.at[idx, 'Min_Raggiunto']:
                df.at[idx, 'Min_Raggiunto'] = current_p
                df.at[idx, 'Data_Min'] = datetime.now().strftime("%d/%m %H:%M")
        else:
            st.warning(f"Impossibile aggiornare i dati per {ticker} al momento.")
                
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
                    
                    c1, c2, c3 = st.columns([1, 2, 2])
                    with c1:
                        st.subheader(row['Ticker'])
                        st.caption(f"Carico: ${row['Prezzo_Carico']:.2f}")
                    
                    with c2:
                        st.write(f"🚀 **Max:** {perc(row['Max_Raggiunto']):+.2f}%")
                        st.caption(f"Raggiunto il: {row['Data_Max']}")
                    
                    with c3:
                        st.write(f"⚠️ **Min:** {perc(row['Min_Raggiunto']):+.2f}%")
                        st.caption(f"Raggiunto il: {row['Data_Min']}")
                    
                    if st.button(f"Chiudi Posizione {row['Ticker']}", key=f"close_{row['Ticker']}_{d}"):
                        df.loc[(df['Ticker'] == row['Ticker']) & (df['Data_Acquisto'] == d), 'Stato'] = 'CLOSED'
                        save_portfolio(df)
                        st.rerun()
                    st.divider()
