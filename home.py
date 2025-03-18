import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

# 1️⃣ โหลดชุดข้อมูลตัวอย่าง (Spam/Ham Dataset)
data = {
    "text": ["Free money now!!!", "Call me tomorrow", "Win a lottery!", 
             "Meeting at 10 AM", "Congratulations! You won a prize!", 
             "Let's go to lunch", "Get free rewards now!", "How are you?",
             "Click here to claim your free gift!", "Hey, are you free tonight?"],
    "label": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  # 1 = Spam, 0 = Ham
}
df = pd.DataFrame(data)

# 2️⃣ แปลงข้อความเป็นเวกเตอร์
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df["text"])
y = df["label"]

# 3️⃣ แบ่งข้อมูล Train/Test และฝึกโมเดล Naïve Bayes
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = MultinomialNB()
model.fit(X_train, y_train)

# 🎈 Streamlit UI
st.title("📩 Spam Detection with Naïve Bayes")
st.write("🔍 **พิมพ์ข้อความที่ต้องการตรวจสอบ** ว่าเป็น **Spam หรือไม่**")

# รับข้อความจากผู้ใช้
user_input = st.text_area("พิมพ์ข้อความที่นี่ (พิมพ์ได้หลายบรรทัด)", "")

# ถ้าผู้ใช้ป้อนข้อความ
if st.button("🔎 ทำนายผล"):
    if user_input.strip():  # ตรวจสอบว่าผู้ใช้ไม่ได้ส่งข้อความว่าง
        # แปลงข้อความจากผู้ใช้ให้เป็นเวกเตอร์
        user_texts = user_input.split("\n")  # รองรับหลายบรรทัด
        user_vectors = vectorizer.transform(user_texts)
        
        # ทำนายผลลัพธ์
        predictions = model.predict(user_vectors)
        
        # แสดงผลลัพธ์เป็นตาราง
        result_df = pd.DataFrame({"ข้อความ": user_texts, "ผลการทำนาย": ["🚨 Spam" if pred == 1 else "✅ Ham" for pred in predictions]})
        st.write("### 📊 ผลการทำนาย")
        st.dataframe(result_df, use_container_width=True)
    else:
        st.warning("⚠️ กรุณาป้อนข้อความก่อนทำการทำนาย")
else:st.write("ไม่ทำนาย")
