
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
import matplotlib.pyplot as plt

st.title("📊 تحليل أزواج العملات OTC")

# قراءة الملف
df = pd.read_csv("market_data_otc.csv")

# اختيار الزوج
pair = st.selectbox("اختر زوج العملات", df["pair"].unique())

# فلترة البيانات
data = df[df["pair"] == pair]

# عرض جدول صغير
st.write("📅 بيانات السوق:", data.head())

# رسم بياني
fig, ax = plt.subplots()
ax.plot(data["time"], data["close"], label="سعر الإغلاق")
ax.set_xlabel("الوقت")
ax.set_ylabel("السعر")
ax.legend()
st.pyplot(fig)
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import ta  # مكتبة المؤشرات الفنية

st.set_page_config(page_title="منصة التداول", layout="wide")

# --- واجهة ---
st.title("📊 منصة تحليل السوق - Quotex")

# رفع ملف البيانات
uploaded_file = st.file_uploader("📥 حمّل ملف البيانات (CSV)", type=["csv"])

if uploaded_file is not None:
    # قراءة الملف
    df = pd.read_csv(uploaded_file)

    # تحويل العمود الزمني
    df["time"] = pd.to_datetime(df["time"])

    # اختيار الفريم
    timeframe = st.selectbox("⏱ اختر الفريم الزمني", ["1 دقيقة", "5 دقائق", "15 دقيقة", "ساعة", "4 ساعات"])

    # حساب المؤشرات الفنية
    df["rsi"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi()
    df["macd"] = ta.trend.MACD(df["close"]).macd()
    df["macd_signal"] = ta.trend.MACD(df["close"]).macd_signal()

    # --- الرسم البياني ---
    st.subheader("📈 الرسم البياني (Candlesticks)")
    fig = go.Figure(data=[go.Candlestick(
        x=df["time"],
        open=df["open"],
        high=df["high"],
        low=df["low"],
        close=df["close"],
        name="Candlesticks"
    )])

    st.plotly_chart(fig, use_container_width=True)

    # --- مؤشر RSI ---
    st.subheader("📉 مؤشر RSI")
    fig_rsi = go.Figure()
    fig_rsi.add_trace(go.Scatter(x=df["time"], y=df["rsi"], mode="lines", name="RSI"))
    fig_rsi.add_hline(y=70, line=dict(color="red", dash="dash"))
    fig_rsi.add_hline(y=30, line=dict(color="green", dash="dash"))
    st.plotly_chart(fig_rsi, use_container_width=True)

    # --- مؤشر MACD ---
    st.subheader("📉 مؤشر MACD")
    fig_macd = go.Figure()
    fig_macd.add_trace(go.Scatter(x=df["time"], y=df["macd"], mode="lines", name="MACD"))
    fig_macd.add_trace(go.Scatter(x=df["time"], y=df["macd_signal"], mode="lines", name="Signal"))
    st.plotly_chart(fig_macd, use_container_width=True)

else:
    st.warning("⚠️ رجاءً ارفع ملف CSV يحتوي بيانات السوق (time, open, high, low, close).")
