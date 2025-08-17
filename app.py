
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("Quotex Trading Helper")

uploaded_file = st.file_uploader("📂 ارفع ملف السوق (CSV)", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### أول 10 صفوف من البيانات")
    st.dataframe(df.head(10))

    st.write("### الرسم البياني")
    fig, ax = plt.subplots()
    ax.plot(df['time'], df['close'], label="Close Price")
    ax.set_xlabel("Time")
    ax.set_ylabel("Price")
    ax.legend()
    st.pyplot(fig)
else:
    st.info("⬆️ الرجاء رفع ملف CSV للعرض")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="Quotex Signals (Candles + RSI + MACD)", layout="wide")
st.title("📊 Quotex – شموع + فريمات + RSI + MACD + إشارات دخول")

# --------- إعدادات واجهة المستخدم ---------
pair = st.selectbox("🔹 اختر الزوج:", ["EUR/USD", "EUR/JPY", "GBP/USD"])
timeframe = st.selectbox("⏱️ اختر الفريم:", ["1m", "5m", "15m", "30m", "1h"])

uploaded_file = st.file_uploader("⬆️ ارفع ملف CSV (أعمدة: time/timestamp, open, high, low, close)", type=["csv"])

st.caption("💡 يقبل العمود الزمني باسم time أو timestamp. يجب أن تكون القيم رقمية.")

# --------- دوال المؤشرات ---------
def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    close = close.astype(float)
    delta = close.diff()
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    gain = pd.Series(gain, index=close.index)
    loss = pd.Series(loss, index=close.index)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    # لتفادي القسمة على صفر
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(method="bfill")

def compute_ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def compute_macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = compute_ema(close, fast)
    ema_slow = compute_ema(close, slow)
    macd = ema_fast - ema_slow
    macd_signal = compute_ema(macd, signal)
    macd_hist = macd - macd_signal
    return macd, macd_signal, macd_hist

# --------- تحميل/تحضير البيانات ---------
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    # توحيد أسماء الأعمدة
    df.columns = [c.strip().lower() for c in df.columns]

    # قبول time أو timestamp
    if "time" not in df.columns and "timestamp" in df.columns:
        df.rename(columns={"timestamp": "time"}, inplace=True)

    required = {"time", "open", "high", "low", "close"}
    if not required.issubset(df.columns):
        st.error(f"❌ الملف يجب أن يحتوي على الأعمدة: {required}")
        st.stop()

    # تحويل الأنواع
    df["time"] = pd.to_datetime(df["time"])
    for c in ["open", "high", "low", "close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close"]).sort_values("time")

    # إعادة تجميع حسب الفريم
    if timeframe != "1m":
        rule = {"5m": "5T", "15m": "15T", "30m": "30T", "1h": "1H"}[timeframe]
        df = df.set_index("time").resample(rule).agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last"
        }).dropna().reset_index()

    # --------- مؤشرات RSI و MACD ---------
    df["RSI"] = compute_rsi(df["close"], period=14)
    df["MACD"], df["MACD_signal"], df["MACD_hist"] = compute_macd(df["close"])

    # --------- توليد إشارات دخول ---------
    # قاعدة بسيطة:
    # BUY  حين يتقاطع MACD فوق Signal و RSI >= 50
    # SELL حين يتقاطع MACD تحت Signal و RSI <= 50
    signals = []
    for i in range(1, len(df)):
        prev_macd = df["MACD"].iat[i-1]
        prev_sig = df["MACD_signal"].iat[i-1]
        cur_macd = df["MACD"].iat[i]
        cur_sig = df["MACD_signal"].iat[i]
        rsi_val = df["RSI"].iat[i]
        t = df["time"].iat[i]
        price = df["close"].iat[i]

        # تقاطع صاعد
        if (prev_macd <= prev_sig) and (cur_macd > cur_sig) and (rsi_val >= 50):
            signals.append({"time": t, "type": "BUY", "price": price, "reason": "MACD cross↑ + RSI ≥ 50"})
        # تقاطع هابط
        if (prev_macd >= prev_sig) and (cur_macd < cur_sig) and (rsi_val <= 50):
            signals.append({"time": t, "type": "SELL", "price": price, "reason": "MACD cross↓ + RSI ≤ 50"})

        # فلتر إضافي اختياري: مناطق التشبع
        if rsi_val < 30 and cur_macd > cur_sig:
            signals.append({"time": t, "type": "BUY", "price": price, "reason": "RSI < 30 + MACD دعم"})
        if rsi_val > 70 and cur_macd < cur_sig:
            signals.append({"time": t, "type": "SELL", "price": price, "reason": "RSI > 70 + MACD ضغط"})

    sig_df = pd.DataFrame(signals)

    # --------- رسم الشموع + العلامات ---------
    st.subheader("📈 الشموع اليابانية")
    fig = go.Figure(data=[go.Candlestick(
        x=df["time"],
        open=df["open"],
        high=df["high"],
        low=df["low"],
        close=df["close"],
        name="Candles"
    )])

    # نقاط الإشارات على الشارت
    if not sig_df.empty:
        buys = sig_df[sig_df["type"] == "BUY"]
        sells = sig_df[sig_df["type"] == "SELL"]
        if not buys.empty:
            fig.add_trace(go.Scatter(
                x=buys["time"], y=buys["price"],
                mode="markers",
                marker_symbol="triangle-up",
                marker_size=12,
                name="BUY",
                hovertext=buys["reason"]
            ))
        if not sells.empty:
            fig.add_trace(go.Scatter(
                x=sells["time"], y=sells["price"],
                mode="markers",
                marker_symbol="triangle-down",
                marker_size=12,
                name="SELL",
                hovertext=sells["reason"]
            ))

    fig.update_layout(xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

    # --------- RSI ---------
    st.subheader("📉 RSI (14)")
    fig_rsi = go.Figure()
    fig_rsi.add_trace(go.Scatter(x=df["time"], y=df["RSI"], mode="lines", name="RSI"))
    fig_rsi.add_hline(y=70, line_dash="dot")
    fig_rsi.add_hline(y=30, line_dash="dot")
    fig_rsi.update_yaxes(range=[0, 100])
    st.plotly_chart(fig_rsi, use_container_width=True)

    # --------- MACD ---------
    st.subheader("📊 MACD")
    fig_macd = go.Figure()
    fig_macd.add_trace(go.Scatter(x=df["time"], y=df["MACD"], mode="lines", name="MACD"))
    fig_macd.add_trace(go.Scatter(x=df["time"], y=df["MACD_signal"], mode="lines", name="Signal"))
    fig_macd.add_trace(go.Bar(x=df["time"], y=df["MACD_hist"], name="Hist"))
    st.plotly_chart(fig_macd, use_container_width=True)

    # --------- جدول الإشارات ---------
    st.subheader("📌 أوقات دخول الصفقات (آخر 30 إشارة)")
    if not sig_df.empty:
        st.dataframe(sig_df.tail(30).sort_values("time"))
        st.download_button(
            "⬇️ تحميل الإشارات CSV",
            data=sig_df.to_csv(index=False).encode("utf-8"),
            file_name=f"signals_{pair.replace('/','')}_{timeframe}.csv",
            mime="text/csv"
        )
    else:
        st.info("لا توجد إشارات حسب القواعد الحالية.")

else:
    st.info("⬆️ ارفع ملف CSV للمتابعة.")

