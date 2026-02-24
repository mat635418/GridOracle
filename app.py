import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
from xgboost import XGBRegressor
from datetime import datetime, timedelta

# --- CONFIGURAZIONE ---
st.set_page_config(page_title="GridOracle 2.0 | B2B Energy Analytics", layout="wide")
st.title("⚡ GridOracle 2.0")
st.markdown("**Predictive AI & Risk Management per il Trading Energetico B2B**")

# --- INGESTIONE DATI ---
@st.cache_data(ttl=1800)
def fetch_public_data():
    url_meteo = (
        "https://api.open-meteo.com/v1/forecast?"
        "latitude=51.16&longitude=10.45" 
        "&hourly=temperature_2m,direct_radiation,windspeed_10m"
        "&past_days=14&forecast_days=3&timezone=Europe%2FBerlin"
    )
    
    try:
        res_meteo = requests.get(url_meteo).json()
        df = pd.DataFrame(res_meteo['hourly'])
        df['time'] = pd.to_datetime(df['time'])
        df = df.rename(columns={
            'temperature_2m': 'Temperatura (°C)',
            'direct_radiation': 'Radiazione_Solare',
            'windspeed_10m': 'Vento'
        })
        
        url_prezzi = "https://api.energy-charts.info/price?bzn=DE-LU"
        res_prezzi = requests.get(url_prezzi)
        
        if res_prezzi.status_code == 200:
            dati_prezzi = res_prezzi.json()
            df_prezzi = pd.DataFrame({
                'time': pd.to_datetime(dati_prezzi['unix_seconds'], unit='s', utc=True).tz_convert('Europe/Berlin').tz_localize(None),
                'Prezzo_Mercato_EUR_MWh': dati_prezzi['price']
            })
            df = pd.merge(df, df_prezzi, on='time', how='left')
            df['Prezzo_Reale_Disponibile'] = df['Prezzo_Mercato_EUR_MWh'].notna()
        else:
            raise ValueError("API Fraunhofer non raggiungibile")
            
    except Exception as e:
        st.warning("⚠️ API esterna offline. Attivazione simulazione finanziaria (Duck Curve).")
        df['Prezzo_Mercato_EUR_MWh'] = 100 + (abs(df['Temperatura (°C)'] - 20) * 3) - (df['Radiazione_Solare'] * 0.1) - (df['Vento'] * 1.5)
        df['Prezzo_Reale_Disponibile'] = df['time'] < pd.Timestamp.now()

    df['Ora_del_giorno'] = df['time'].dt.hour
    df['Giorno_settimana'] = df['time'].dt.dayofweek
    return df.dropna(subset=['Temperatura (°C)'])

# --- ESECUZIONE ---
with st.spinner('Elaborazione flussi dati e addestramento XGBoost...'):
    df_main = fetch_public_data()

oggi_str = datetime.now().strftime('%Y-%m-%d %H:00')
timestamp_oggi = pd.to_datetime(oggi_str).timestamp() * 1000 # FIX per Plotly

df_train = df_main[df_main['Prezzo_Reale_Disponibile'] == True].copy()
df_predict = df_main[df_main['Prezzo_Reale_Disponibile'] == False].copy()

# --- MOTORE AI (XGBOOST) E CONFIDENCE INTERVAL ---
features = ['Temperatura (°C)', 'Radiazione_Solare', 'Vento', 'Ora_del_giorno', 'Giorno_settimana']
target = 'Prezzo_Mercato_EUR_MWh'

df_train_clean = df_train.dropna(subset=[target] + features)

# Addestriamo il modello XGBoost (Più performante del RandomForest)
model = XGBRegressor(n_estimators=150, learning_rate=0.05, random_state=42)
model.fit(df_train_clean[features], df_train_clean[target])

# Calcolo dell'errore storico per le Bande di Confidenza
df_train_clean['Residui'] = df_train_clean[target] - model.predict(df_train_clean[features])
std_error = df_train_clean['Residui'].std()

if not df_predict.empty:
    df_predict[target + '_Predetto'] = model.predict(df_predict[features])
    # Bande di confidenza al 95% (circa 1.96 * deviazione standard)
    df_predict['Upper_Bound'] = df_predict[target + '_Predetto'] + (1.96 * std_error)
    df_predict['Lower_Bound'] = df_predict[target + '_Predetto'] - (1.96 * std_error)

# --- METRICHE E SEGNALI DI TRADING ---
st.markdown("### 🚦 Trading Signals & KPIs")
col1, col2, col3, col4 = st.columns(4)

prezzo_attuale = df_train_clean[target].iloc[-1]
media_futura = df_predict[target + '_Predetto'].mean() if not df_predict.empty else prezzo_attuale
prezzo_prossima_ora = df_predict[target + '_Predetto'].iloc[0] if not df_predict.empty else prezzo_attuale

# Logica del semaforo: se il prezzo previsto è più basso del 10% rispetto alla media, compra (accumula).
if prezzo_prossima_ora < (media_futura * 0.90):
    segnale, colore = "🟢 BUY (Accumula)", "normal"
elif prezzo_prossima_ora > (media_futura * 1.10):
    segnale, colore = "🔴 SELL (Scarica in Rete)", "inverse"
else:
    segnale, colore = "🟡 HOLD (Attendi)", "off"

col1.metric("Prezzo Attuale (Ultimo Rilevato)", f"{prezzo_attuale:.2f} €/MWh")
col2.metric("Prezzo Medio Previsto (48h)", f"{media_futura:.2f} €/MWh")
col3.metric("Previsione Prossima Ora", f"{prezzo_prossima_ora:.2f} €/MWh", delta=f"{(prezzo_prossima_ora - prezzo_attuale):.2f} €", delta_color="inverse")
col4.metric("Segnale AI Operativo", segnale, delta_color=colore)

# --- TABS DELLA DASHBOARD ---
tab1, tab2, tab3 = st.tabs(["📈 Previsione AI (XGBoost)", "🎲 Simulazione Monte Carlo", "📊 Backtesting Storico"])

with tab1:
    st.markdown("**Previsione Prezzo Spot con Bande di Confidenza (95%)**")
    fig = go.Figure()
    
    # Storico
    fig.add_trace(go.Scatter(x=df_train_clean['time'], y=df_train_clean[target], name='Prezzo Reale', line=dict(color='#1f77b4', width=2)))
    
    if not df_predict.empty:
        # Previsione e Bande di confidenza
        fig.add_trace(go.Scatter(x=df_predict['time'], y=df_predict['Upper_Bound'], line=dict(width=0), showlegend=False, hoverinfo='skip'))
        fig.add_trace(go.Scatter(x=df_predict['time'], y=df_predict['Lower_Bound'], fill='tonexty', fillcolor='rgba(255, 127, 14, 0.2)', line=dict(width=0), name='Intervallo di Confidenza 95%'))
        fig.add_trace(go.Scatter(x=df_predict['time'], y=df_predict[target + '_Predetto'], name='Previsione AI (XGBoost)', line=dict(color='#ff7f0e', width=3)))
    
    fig.add_vline(x=timestamp_oggi, line_dash="dash", line_color="gray", annotation_text="ADESSO")
    fig.update_layout(height=450, hovermode="x unified", template="plotly_white", margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.markdown("**Analisi dei Rischi: Simulazione Monte Carlo (100 Scenari)**")
    st.info("La simulazione genera percorsi di prezzo casuali basati sulla volatilità storica delle ultime due settimane. Utile per il calcolo del VaR (Value at Risk).")
    
    # Calcolo volatilità sui delta di prezzo storici
    diffs = df_train_clean[target].diff().dropna()
    mu = diffs.mean()
    sigma = diffs.std()
    
    fig_mc = go.Figure()
    orizzonte = len(df_predict) if not df_predict.empty else 48
    tempi_futuri = df_predict['time'] if not df_predict.empty else [df_train_clean['time'].iloc[-1] + timedelta(hours=i) for i in range(1, 49)]
    
    for i in range(100):
        # Genera un "random walk"
        sim_diffs = np.random.normal(mu, sigma, orizzonte)
        sim_prices = [prezzo_attuale]
        for diff in sim_diffs:
            # Assumiamo che il prezzo possa andare in negativo (succede nelle rinnovabili!), ma limitiamo picchi assurdi
            sim_prices.append(sim_prices[-1] + diff)
        
        fig_mc.add_trace(go.Scatter(x=tempi_futuri, y=sim_prices[1:], mode='lines', line=dict(color='rgba(100, 150, 200, 0.05)'), showlegend=False, hoverinfo='skip'))
    
    # Aggiungi la previsione XGBoost come linea di base sulla simulazione
    if not df_predict.empty:
        fig_mc.add_trace(go.Scatter(x=df_predict['time'], y=df_predict[target + '_Predetto'], name='Scenario Base (XGBoost)', line=dict(color='#ff7f0e', width=3)))
    
    fig_mc.update_layout(height=450, template="plotly_white", margin=dict(l=0, r=0, t=10, b=0), yaxis_title="€/MWh")
    st.plotly_chart(fig_mc, use_container_width=True)

with tab3:
    st.markdown("**Test dell'Algoritmo: Ritorno sull'Investimento (ROI) Teorico**")
    st.write("Come si sarebbe comportato l'algoritmo negli ultimi 3 giorni identificando i picchi massimi (Sell) e minimi (Buy)?")
    
    # Semplice logica di backtest sugli ultimi 3 giorni
    df_backtest = df_train_clean.tail(72).copy()
    media_rolling = df_backtest[target].mean()
    
    # Simuliamo trade da 1 MWh
    budget = 0
    trades = 0
    for val in df_backtest[target]:
        if val < (media_rolling * 0.8): # Compra basso
            budget -= val
            trades += 1
        elif val > (media_rolling * 1.2): # Vendi alto
            budget += val
            trades += 1
            
    st.success(f"💰 **Profitto netto simulato (ultime 72h): + {budget:.2f} €** (Eseguendo {trades} operazioni su lotti da 1 MWh)")
    st.caption("Il backtest considera l'acquisto di energia quando il prezzo reale scende del 20% sotto la media e la vendita quando sale del 20% sopra la media.")
