import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import requests
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta

# --- CONFIGURAZIONE ---
st.set_page_config(page_title="GridOracle | B2B Energy Analytics", layout="wide")
st.title("⚡ GridOracle")
st.markdown("**Predictive AI per il Trading Energetico B2B** (Alimentato da dati 100% Pubblici)")

# --- INGESTIONE DATI (ZERO API KEYS) ---
@st.cache_data(ttl=1800)
def fetch_public_data():
    # 1. Scarichiamo il Meteo (Open-Meteo: Vento, Sole, Temperatura)
    url_meteo = (
        "https://api.open-meteo.com/v1/forecast?"
        "latitude=51.16&longitude=10.45" # Germania (Mercato energetico più liquido d'Europa)
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
        
        # 2. Scarichiamo i Prezzi Reali (Fraunhofer Energy Charts: Mercato DE-LU)
        # NESSUNA API KEY RICHIESTA
        url_prezzi = "https://api.energy-charts.info/price?bzn=DE-LU"
        res_prezzi = requests.get(url_prezzi)
        
        if res_prezzi.status_code == 200:
            dati_prezzi = res_prezzi.json()
            # L'API restituisce timestamp unix e prezzi
            df_prezzi = pd.DataFrame({
                'time': pd.to_datetime(dati_prezzi['unix_seconds'], unit='s', utc=True).tz_convert('Europe/Berlin').tz_localize(None),
                'Prezzo_Mercato_EUR_MWh': dati_prezzi['price']
            })
            # Uniamo i due dataset in base all'orario
            df = pd.merge(df, df_prezzi, on='time', how='left')
            
            # Riempiamo i buchi futuri (dove non c'è ancora il prezzo) per poter fare la previsione
            df['Prezzo_Reale_Disponibile'] = df['Prezzo_Mercato_EUR_MWh'].notna()
        else:
            raise ValueError("API Fraunhofer non raggiungibile")
            
    except Exception as e:
        st.warning("⚠️ API dei prezzi reali temporaneamente offline. Attivazione simulazione finanziaria realistica (Duck Curve).")
        # Simulazione fallback se l'API esterna è giù, per garantire che l'app funzioni sempre
        df['Prezzo_Mercato_EUR_MWh'] = 100 + (abs(df['Temperatura (°C)'] - 20) * 3) - (df['Radiazione_Solare'] * 0.1) - (df['Vento'] * 1.5)
        df['Prezzo_Reale_Disponibile'] = df['time'] < pd.Timestamp.now()

    # Feature Engineering (Aggiungiamo l'ora del giorno, fondamentale per l'energia)
    df['Ora_del_giorno'] = df['time'].dt.hour
    
    return df.dropna(subset=['Temperatura (°C)']) # Rimuove eventuali righe corrotte

# --- ESECUZIONE ---
with st.spinner('Connessione alle banche dati pubbliche in corso...'):
    df_main = fetch_public_data()

# Separiamo storico (su cui abbiamo o calcoliamo il prezzo) e futuro (da prevedere)
oggi_str = datetime.now().strftime('%Y-%m-%d %H:%00')
df_train = df_main[df_main['Prezzo_Reale_Disponibile'] == True].copy()
df_predict = df_main[df_main['Prezzo_Reale_Disponibile'] == False].copy()

# --- MOTORE AI (MACHINE LEARNING) ---
features = ['Temperatura (°C)', 'Radiazione_Solare', 'Vento', 'Ora_del_giorno']
target = 'Prezzo_Mercato_EUR_MWh'

# Addestramento
model = RandomForestRegressor(n_estimators=100, random_state=42)
# Rimuoviamo i NaN per il training in caso di disallineamenti API
df_train_clean = df_train.dropna(subset=[target] + features) 
model.fit(df_train_clean[features], df_train_clean[target])

# Se abbiamo dati futuri da prevedere, applichiamo il modello
if not df_predict.empty:
    df_predict[target + '_Predetto'] = model.predict(df_predict[features])
else:
    # Se per qualche motivo l'API ha restituito tutto, prevediamo sul target stesso per mostrare la linea
    df_predict = df_main[df_main['time'] >= pd.Timestamp.now() - timedelta(hours=12)].copy()
    df_predict[target + '_Predetto'] = model.predict(df_predict[features])

# --- DASHBOARD UI ---
st.subheader("📈 Previsione Prezzo Spot dell'Energia (Day-Ahead)")
st.markdown("Incrocio tra dati meteorologici e prezzi di mercato. L'area rossa indica la **previsione dell'Intelligenza Artificiale** per le prossime ore.")

fig = go.Figure()

# Linea dello storico (Prezzo Reale)
fig.add_trace(go.Scatter(
    x=df_train_clean['time'], 
    y=df_train_clean[target], 
    name='Prezzo Reale (€/MWh)', 
    line=dict(color='white', width=2)
))

# Linea del futuro (Previsione AI)
fig.add_trace(go.Scatter(
    x=df_predict['time'], 
    y=df_predict[target + '_Predetto'], 
    name='Previsione AI (€/MWh)', 
    line=dict(color='#00ffcc', width=3, dash='solid')
))

# Abbellimenti del grafico
fig.add_vline(x=pd.to_datetime(oggi_str).timestamp() * 1000, line_dash="dash", line_color="gray", annotation_text="ADESSO")
fig.update_layout(
    height=450, 
    hovermode="x unified",
    plot_bgcolor='rgba(17, 17, 17, 1)', # Tema scuro "Trading"
    paper_bgcolor='rgba(17, 17, 17, 1)',
    font=dict(color='white'),
    margin=dict(l=0, r=0, t=30, b=0)
)
st.plotly_chart(fig, use_container_width=True)

# --- ANALISI FATTORI (TABS) ---
st.markdown("### 🔍 Analisi Fattori di Rete")
tab1, tab2 = st.tabs(["Clima & Rinnovabili (Feature)", "Esplora Dati Grezzi"])

with tab1:
    fig_clima = go.Figure()
    fig_clima.add_trace(go.Scatter(x=df_main['time'], y=df_main['Radiazione_Solare'], name='Solare (W/m²)', fill='tozeroy', line=dict(color='orange')))
    fig_clima.add_trace(go.Scatter(x=df_main['time'], y=df_main['Vento']*10, name='Vento (Scalato)', line=dict(color='cyan')))
    fig_clima.update_layout(height=300, hovermode="x unified", margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig_clima, use_container_width=True)

with tab2:
    st.dataframe(df_main.set_index('time').sort_index(ascending=False), use_container_width=True)
