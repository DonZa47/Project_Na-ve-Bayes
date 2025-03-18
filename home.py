import streamlit as st
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_iris

# โหลดชุดข้อมูล Iris
iris = load_iris()
X, y = iris.data, iris.target
model = GaussianNB()
model.fit(X, y)  # ฝึกโมเดลล่วงหน้า

# ตั้งค่าหน้าเว็บ Streamlit
st.title("Naïve Bayes Classifier - Iris Dataset")
st.write("ป้อนคุณสมบัติของดอกไม้เพื่อทำนายประเภท")

# รับค่าจากผู้ใช้ผ่าน slider
sepal_length = st.slider("Sepal Length (cm)", float(X[:,0].min()), float(X[:,0].max()), float(X[:,0].mean()))
sepal_width = st.slider("Sepal Width (cm)", float(X[:,1].min()), float(X[:,1].max()), float(X[:,1].mean()))
petal_length = st.slider("Petal Length (cm)", float(X[:,2].min()), float(X[:,2].max()), float(X[:,2].mean()))
petal_width = st.slider("Petal Width (cm)", float(X[:,3].min()), float(X[:,3].max()), float(X[:,3].mean()))

# สร้างอาร์เรย์ข้อมูลจากค่าที่ป้อน
input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

# ทำนายผลลัพธ์
prediction = model.predict(input_data)
prediction_proba = model.predict_proba(input_data)

# แสดงผลลัพธ์
st.subheader("ผลลัพธ์ที่ได้:")
st.write(f"ชนิดของดอกไม้ที่คาดการณ์: *{iris.target_names[prediction[0]]}*")

# แสดงความน่าจะเป็นของแต่ละคลาส
st.subheader("ความน่าจะเป็นของแต่ละประเภท:")
df_proba = pd.DataFrame(prediction_proba, columns=iris.target_names)
st.dataframe(df_proba.style.format("{:.2%}"))