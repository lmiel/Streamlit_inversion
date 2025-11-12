# streamlit_app.py
import streamlit as st
import pandas as pd
from datetime import datetime
from inference import predict_one_day_bbva, predict_one_day_san
import altair as alt
from datetime import datetime, timedelta
import json
import os

st.set_page_config(page_title="Cartera BBVA & Santander", page_icon="📈", layout="wide")

# ------------------------------
# Configuración
# ------------------------------
TICKERS = {
    "BBVA": {
        "pred_file": "data/BBVA_model_dataset_2.csv",  # esperado: Date, Close, Pred_Close, Pred_Low, Pred_High
    },
    "SAN": {
        "pred_file": "data/SAN_model_dataset_2.csv",
    },
}
DEFAULT_CASH = 10_000.0
DEFAULT_THRESH = 0.0
BROKER_FEE = 0.0  # puedes poner 0.001 para 0.1%
STATE_FILE = "portfolio_state.json"


# ------------------------------
# File-based state management
# ------------------------------
def load_state():
    """Load state from file or return default state"""
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, 'r') as f:
                state = json.load(f)
                # Convert history timestamps back to datetime objects for display
                for record in state.get("history", []):
                    if "time" in record and isinstance(record["time"], str):
                        record["time"] = datetime.fromisoformat(record["time"])
                return state
        except (json.JSONDecodeError, FileNotFoundError):
            pass
    
    # Return default state
    return {
        "cash": DEFAULT_CASH,
        "positions": {tic: 0 for tic in TICKERS.keys()},
        "avg_price": {tic: 0.0 for tic in TICKERS.keys()},
        "history": []
    }

def save_state(state):
    """Save state to file"""
    # Make a copy to avoid modifying the original
    state_to_save = state.copy()
    # Convert datetime objects to ISO format strings
    state_to_save["history"] = []
    for record in state["history"]:
        record_copy = record.copy()
        if "time" in record_copy and isinstance(record_copy["time"], datetime):
            record_copy["time"] = record_copy["time"].isoformat()
        state_to_save["history"].append(record_copy)
    
    with open(STATE_FILE, 'w') as f:
        json.dump(state_to_save, f, indent=2)

def reset_state():
    """Reset state to default values"""
    if os.path.exists(STATE_FILE):
        os.remove(STATE_FILE)

# Load initial state
portfolio_state = load_state()


# ------------------------------
# Datos: lector robusto
# ------------------------------
def load_historic(path: str, ticker: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Normaliza nombres esperados
    cols = {c.lower(): c for c in df.columns}
    for needed in ["Date", f"{ticker}_Close", f"{ticker}_Low", f"{ticker}_High"]:
        if needed not in [c for c in df.columns]:
            raise ValueError(f"Falta columna {needed} en {path} las columnas son {df.columns.tolist()}")
    # Enforce types
    df["Date"] = pd.to_datetime(df[cols.get("date")])
    df = df.sort_values("Date").reset_index(drop=True)
    # Renombra consistente
    df = df.rename(
        columns={
            cols.get(f"{ticker}_Close"): "Close",
            cols.get(f"{ticker}_Low"): "Low",
            cols.get(f"{ticker}_High"): "High",
        }
    )
    return df


DATA = {tic: load_historic(info["pred_file"], tic) for tic, info in TICKERS.items()}


# ------------------------------
# Lógica de recomendación
# ------------------------------
def recommend(ticker, row):
    # regla: pred vs close
    if row[f"{ticker}_Pred_Close"] >= row[f"{ticker}_Close"] * 1.01:
        return "BUY"
    elif row[f"{ticker}_Pred_Close"] <= row[f"{ticker}_Close"] * 0.99:
        return "SELL"
    else:
        return "HOLD"


# ------------------------------
# Sidebar
# ------------------------------
st.sidebar.header("⚙️ Parámetros")
new_cash = st.sidebar.number_input(
    "💰 Saldo disponible (€)", min_value=0.0, step=100.0, value=portfolio_state["cash"]
)
# Update cash if changed
if new_cash != portfolio_state["cash"]:
    portfolio_state["cash"] = new_cash
    save_state(portfolio_state)

if st.sidebar.button("🔄 Reset app", type="secondary"):
    reset_state()
    portfolio_state = load_state()
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.subheader("📦 Cartera")
for tic in TICKERS:
    st.sidebar.write(
        f"**{tic}**: {portfolio_state['positions'][tic]} acciones @ {portfolio_state['avg_price'][tic]:.2f} €"
    )
st.sidebar.write(f"**Efectivo:** {portfolio_state['cash']:,.2f} €")

st.title("📈 Predicción y Cartera — BBVA & Santander")

# ------------------------------
# UI: Tabs por ticker
# ------------------------------
tabs = st.tabs([f"🏦 {tic}" for tic in TICKERS])


def trade(ticker, side, qty, price):
    qty = int(qty)
    if qty <= 0:
        st.warning("Cantidad debe ser positiva.")
        return
    cost = qty * price
    if side == "BUY":
        total = cost
        if total > portfolio_state["cash"]:
            st.error("Saldo insuficiente.")
            return
        # promedio ponderado
        pos = portfolio_state["positions"][ticker]
        avg = portfolio_state["avg_price"][ticker]
        new_pos = pos + qty
        new_avg = ((pos * avg) + cost) / new_pos if new_pos > 0 else 0.0
        portfolio_state["positions"][ticker] = new_pos
        portfolio_state["avg_price"][ticker] = new_avg
        portfolio_state["cash"] -= total
        st.success(f"Compradas {qty} {ticker} @ {price:.2f} €")
        portfolio_state["history"].append(
            {
                "time": datetime.now(),
                "ticker": ticker,
                "side": "BUY",
                "qty": qty,
                "price": price,
            }
        )
    else:  # SELL
        pos = portfolio_state["positions"][ticker]
        if qty > pos:
            st.error("No tienes suficientes acciones para vender.")
            return
        revenue = cost
        portfolio_state["positions"][ticker] = pos - qty
        portfolio_state["cash"] += revenue
        st.success(f"Vendidas {qty} {ticker} @ {price:.2f} €")
        portfolio_state["history"].append(
            {
                "time": datetime.now(),
                "ticker": ticker,
                "side": "SELL",
                "qty": qty,
                "price": price,
            }
        )
        # si posición queda en 0, resetea avg
        if portfolio_state["positions"][ticker] == 0:
            portfolio_state["avg_price"][ticker] = 0.0
    
    # Save state after any trade
    save_state(portfolio_state)

def get_whole_prediction(tic):
    if tic == "BBVA":
        predict = predict_one_day_bbva
    else:
        predict = predict_one_day_san
    
    dates = [datetime(2025, 11, 3), datetime(2025, 11, 4), datetime(2025, 11, 5)]

    rows = []
    for d in dates:
        y_pred, y_real, mae, rmse = predict(target_date=d)
        rows.append({
            "Date": pd.Timestamp(d),
            f"{tic}_Pred_Close": float(y_pred[0]),
            f"{tic}_Pred_Low": float(y_pred[1]),
            f"{tic}_Pred_High": float(y_pred[2]),
        })

    results = pd.DataFrame(rows).sort_values("Date").reset_index(drop=True)
    return results


for tab, tic in zip(tabs, TICKERS.keys()):
    with tab:
        df = DATA[tic].copy().iloc[:-1]
        predict = get_whole_prediction(tic)
        # Merge historical data with predictions on Date
        df_with_pred = df.merge(predict, on="Date", how="outer", suffixes=("", "_pred"))
        last = df.iloc[-1]
        rec = recommend(tic, df_with_pred.iloc[-1])
        print(df_with_pred.columns)
        colA, colB, colC = st.columns([2.5, 1.2, 1.2])
        with colA:
            st.subheader(f"{tic} — Precio vs Predicción (últimos 21 días)")
            sub = df_with_pred.iloc[-21:].copy()
            # Gráfico simple con bandas

            chart = alt.Chart(sub).encode(x="Date:T")
            line_real = chart.mark_line().encode(
                y=alt.Y(f"{tic}_Close:Q", title="€"), tooltip=["Date", f"{tic}_Close"]
            )
            band = chart.mark_area(opacity=0.15).encode(
                y=f"{tic}_Low:Q", y2=f"{tic}_High:Q"
            )
            line_pred = chart.mark_line(strokeDash=[4, 3], color='red').encode(
                y=alt.Y(f"{tic}_Pred_Close:Q"), tooltip=["Date", f"{tic}_Pred_Close"]
            )
            st.altair_chart(
                (line_real + band + line_pred).properties(height=300),
                use_container_width=True,
            )

        with colB:
            st.subheader("📌 Estado")
            st.metric("Precio actual (Close)", f"{last[f'{tic}_Close']:.2f} €")
            st.metric("Predicción próxima", f"{df_with_pred.iloc[-1][f'{tic}_Pred_Close']:.2f} €")
            st.metric("Recomendación", rec)

        with colC:
            st.subheader("🛒 Operar")
            qty = st.number_input("Cantidad", min_value=0, step=1, key=f"qty_{tic}")
            side = st.radio(
                "Acción", ["BUY", "SELL"], horizontal=True, key=f"side_{tic}"
            )
            exec_price = float(
                last[f"{tic}_Close"]
            )  # puedes cambiar a Pred_Close si prefieres
            if st.button("Ejecutar", key=f"exec_{tic}"):
                trade(tic, side, qty, exec_price)

        # KPIs de posición
        st.markdown("---")
        c1, c2, c3, c4 = st.columns(4)
        pos = portfolio_state["positions"][tic]
        avg = portfolio_state["avg_price"][tic]
        mtm = pos * last[f"{tic}_Close"]
        pnl = (last[f"{tic}_Close"] - avg) * pos if pos > 0 else 0.0
        with c1:
            st.metric("Acciones", pos)
        with c2:
            st.metric("Precio medio", f"{avg:.2f} €")
        with c3:
            st.metric("Valor posición", f"{mtm:,.2f} €")
        with c4:
            st.metric("P&L no realizado", f"{pnl:,.2f} €")

# ------------------------------
# Resumen global
# ------------------------------
st.markdown("## 📦 Resumen de cartera")
tot_val = portfolio_state["cash"]
for tic in TICKERS:
    tot_val += portfolio_state["positions"][tic] * DATA[tic].iloc[-1][f"{tic}_Close"]
c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Efectivo", f"{portfolio_state['cash']:,.2f} €")
with c2:
    st.metric("Valor total cartera", f"{tot_val:,.2f} €")
with c3:
    invested = tot_val - portfolio_state["cash"]
    pct_inv = 0.0 if tot_val == 0 else invested / tot_val * 100
    st.metric("% Invertido", f"{pct_inv:.1f} %")

# Historial
if portfolio_state["history"]:
    st.markdown("### 🧾 Historial de operaciones")
    hist = pd.DataFrame(portfolio_state["history"])
    hist = hist.sort_values("time", ascending=False)
    st.dataframe(hist, use_container_width=True)
else:
    st.info("Sin operaciones aún. Usa los botones BUY/SELL para empezar.")
