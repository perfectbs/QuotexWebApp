
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
