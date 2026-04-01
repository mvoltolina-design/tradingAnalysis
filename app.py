import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import os

# --- INIZIALIZZAZIONE PORTAFOGLIO ---
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = {} # Formato: { 'AAPL': {'entry_price': 150.0, 'max_seen': 150.0, 'min_seen': 150.0} }

# --- FUNZIONI DI SUPPORTO ---
def get_live_price(ticker):
    try:
        data = yf.download(ticker.replace('.', '-'), period="1d", progress=False)
        return float(data['Close'].iloc[-1])
    except:
        return None

# --- UI ---
st.title("🚀 V8 Portfolio & Analysis")

tab1, tab2, tab3 = st.tabs(["🔍 Analisi", "📈 Portafoglio", "⚙️ Dettaglio Ticker"])

# --- TAB 1: ANALISI (Il tuo codice precedente va qui) ---
with tab1:
    st.info("Esegui l'analisi V8 per trovare nuove opportunità.")
    # (Inserisci qui il pulsante "AVVIA ANALISI COMPLETA" già sviluppato)

# --- TAB 2: PORTAFOGLIO ---
with tab2:
    st.header("I Tuoi Titoli")
    if not st.session_state.portfolio:
        st.write("Il portafoglio è vuoto.")
    else:
        for t, info in list(st.session_state.portfolio.items()):
            current_p = get_live_price(t)
            if current_p:
                # Aggiornamento massimi e minimi reali raggiunti
                if current_p > info['max_seen']: st.session_state.portfolio[t]['max_seen'] = current_p
                if current_p < info['min_seen']: st.session_state.portfolio[t]['min_seen'] = current_p
                
                perf_reale = ((current_p - info['entry_price']) / info['entry_price']) * 100
                max_reale = ((info['max_seen'] - info['entry_price']) / info['entry_price']) * 100
                min_reale = ((info['min_seen'] - info['entry_price']) / info['entry_price']) * 100
                
                with st.expander(f"{t} | {perf_reale:.2f}%"):
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Prezzo Attuale", f"{current_p:.2f}")
                    c2.metric("Max Raggiunto", f"{max_reale:.2f}%", delta_color="normal")
                    c3.metric("Min Raggiunto", f"{min_reale:.2f}%", delta_color="inverse")
                    
                    if st.button(f"Rimuovi {t}", key=f"del_{t}"):
                        del st.session_state.portfolio[t]
                        st.rerun()

# --- TAB 3: DETTAGLIO & ACQUISTO ---
with tab3:
    st.header("Gestione Titolo")
    ticker_to_search = st.text_input("Inserisci Ticker (es. NVDA):").upper()
    
    if ticker_to_search:
        current_price = get_live_price(ticker_to_search)
        
        if current_price:
            st.metric(f"Prezzo attuale di {ticker_to_search}", f"${current_price:.2f}")
            
            entry_input = st.number_input("Prezzo di acquisto:", value=current_price)
            
            if st.button(f"Aggiungi {ticker_to_search} al Portafoglio"):
                st.session_state.portfolio[ticker_to_search] = {
                    'entry_price': entry_input,
                    'max_seen': current_price,
                    'min_seen': current_price
                }
                st.success(f"{ticker_to_search} aggiunto con successo!")
        else:
            st.error("Impossibile recuperare il prezzo. Verifica il ticker.")
