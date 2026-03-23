import streamlit as st
import joblib
import pandas as pd
import numpy as np

# --- 1. ตั้งค่าหน้าเว็บ (Page Config) ---
st.set_page_config(
    page_title="Airline Satisfaction Predictor",
    page_icon="✈️",
    layout="centered"
)

# --- 2. โหลดโมเดล Pipeline ---
@st.cache_resource
def load_model():
    # โหลดไฟล์ .pkl ที่คุณอัปโหลดขึ้น GitHub
    return joblib.load('airline_satisfaction_pipeline.pkl')

model = load_model()

# --- 3. ส่วนหัวของเว็บไซต์ ---
st.title("✈️ Airline Passenger Satisfaction")
st.markdown("""
ระบบทำนายความพึงพอใจของผู้โดยสารสายการบิน โดยใช้เทคโนโลยี **Machine Learning**
""")

# --- 4. ส่วนรับข้อมูลจากผู้ใช้งาน ---
st.subheader("📋 ข้อมูลการเดินทาง")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("เพศ (Gender)", ["Male", "Female"])
    customer_type = st.selectbox("ประเภทลูกค้า", ["Loyal Customer", "disloyal Customer"])
    age = st.number_input("อายุ (ปี)", min_value=1, max_value=100, value=30)
    type_of_travel = st.selectbox("วัตถุประสงค์การเดินทาง", ["Personal Travel", "Business travel"])

with col2:
    customer_class = st.selectbox("ชั้นที่นั่ง (Class)", ["Eco", "Eco Plus", "Business"])
    flight_distance = st.number_input("ระยะทางบิน (Flight Distance)", min_value=1, max_value=10000, value=500)
    departure_delay = st.number_input("ดีเลย์ขาออก (นาที)", min_value=0, max_value=1500, value=0)
    arrival_delay = st.number_input("ดีเลย์ขาเข้า (นาที)", min_value=0, max_value=1500, value=0)

st.subheader("⭐ ให้คะแนนการบริการ (1-5 คะแนน)")
wifi_service = st.slider("ความพึงพอใจต่อ Wifi บนเครื่อง", 1, 5, 3)
booking_ease = st.slider("ความง่ายในการจองตั๋วออนไลน์", 1, 5, 3)
seat_comfort = st.slider("ความสบายของที่นั่ง", 1, 5, 3)
entertainment = st.slider("ความบันเทิงบนเครื่อง", 1, 5, 3)
onboard_service = st.slider("การบริการบนเครื่อง", 1, 5, 3)
leg_room = st.slider("พื้นที่วางขา (Leg room)", 1, 5, 3)

# --- 5. การประมวลผลและการแสดงผล ---
if st.button("ประเมินความพึงพอใจ", type="primary"):
    # สร้าง DataFrame เบื้องต้นจาก Input
    input_data = {
        'id': [0], # ใส่ค่าหลอกไว้เผื่อโมเดลถามหา
        'Gender': [gender],
        'Customer Type': [customer_type],
        'Age': [age],
        'Type of Travel': [type_of_travel],
        'Class': [customer_class],
        'Flight Distance': [flight_distance],
        'Inflight wifi service': [wifi_service],
        'Departure/Arrival convenience': [3],
        'Ease of Online booking': [booking_ease],
        'Gate location': [3],
        'Food and drink': [3],
        'Online boarding': [3],
        'Seat comfort': [seat_comfort],
        'Inflight entertainment': [entertainment],
        'On-board service': [onboard_service],
        'Leg room service': [leg_room],
        'Baggage handling': [3],
        'Checkin service': [3],
        'Inflight service': [3],
        'Cleanliness': [3],
        'Departure Delay in Minutes': [departure_delay],
        'Arrival Delay in Minutes': [arrival_delay]
    }
    
    df = pd.DataFrame(input_data)

    # --- ฟังก์ชันพิเศษ: ตรวจสอบและเรียงคอลัมน์ให้ตรงตามที่โมเดลต้องการ ---
    try:
        # ตรวจสอบว่าโมเดลต้องการคอลัมน์ชื่ออะไรบ้าง
        if hasattr(model, 'feature_names_in_'):
            expected_features = model.feature_names_in_
            
            # ถ้าคอลัมน์ไหนขาดไป ให้เติมด้วยเลข 0
            for col in expected_features:
                if col not in df.columns:
                    df[col] = 0
            
            # เรียงลำดับคอลัมน์ให้ตรงเป๊ะตามที่โมเดลเคยเห็นตอนฝึกสอน
            df = df[expected_features]
        
        # ทำนายผล
        prediction = model.predict(df)[0]
        
        st.divider()
        
        # ตรวจสอบค่าที่ส่งออกมา (บางโมเดลส่งเป็นเลข 0/1 บางโมเดลส่งเป็นข้อความ)
        if str(prediction) == '1' or str(prediction).lower() == 'satisfied':
            st.success(f"🎉 **ผลการทำนาย: ผู้โดยสารพึงพอใจ (Satisfied)**")
        else:
            st.warning(f"⚠️ **ผลการทำนาย: ไม่พึงพอใจ (Neutral or Dissatisfied)**")
            
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการทำนาย: {e}")
        st.info("คำแนะนำ: ตรวจสอบว่าไฟล์โมเดลที่อัปโหลดตรงกับเวอร์ชันที่ใช้ใน Colab หรือไม่")

# --- ส่วนท้าย ---
st.caption("จัดทำโดย: 67160363 PHOKIN KETMAYOON | Final Project Machine Learning")
