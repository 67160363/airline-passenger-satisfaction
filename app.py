import streamlit as st
import joblib
import pandas as pd
import numpy as np

# --- 1. การตั้งค่าหน้าเว็บ (Page Config) ---
st.set_page_config(
    page_title="Airline Satisfaction Predictor Pro",
    page_icon="✈️",
    layout="wide"
)

# --- 2. โหลดโมเดล Pipeline ---
@st.cache_resource
def load_model():
    return joblib.load('airline_satisfaction_pipeline.pkl')

model = load_model()

# --- 3. ส่วนหัวของเว็บไซต์ ---
st.title("✈️ Airline Passenger Satisfaction Predictor")
st.markdown("### ระบบวิเคราะห์ความพึงพอใจแบบแม่นยำสูง (High Accuracy Version)")
st.info("กรุณากรอกข้อมูลให้ครบถ้วนเพื่อให้ AI คำนวณผลได้แม่นยำที่สุด")

# --- 4. ส่วนรับข้อมูลจากผู้ใช้งาน ---
with st.container():
    st.subheader("📋 ข้อมูลพื้นฐานและการเดินทาง")
    c1, c2, c3 = st.columns(3)
    with c1:
        gender = st.selectbox("เพศ (Gender)", ["Male", "Female"])
        customer_type = st.selectbox("ประเภทลูกค้า", ["Loyal Customer", "disloyal Customer"])
    with c2:
        age = st.number_input("อายุ (Age)", min_value=1, max_value=100, value=30)
        type_of_travel = st.selectbox("วัตถุประสงค์การเดินทาง", ["Business travel", "Personal Travel"])
    with c3:
        customer_class = st.selectbox("ชั้นที่นั่ง (Class)", ["Business", "Eco", "Eco Plus"])
        flight_distance = st.number_input("ระยะทางบิน (Flight Distance)", min_value=1, max_value=10000, value=1500)

st.divider()

# เพิ่มส่วนการให้คะแนนแบบละเอียดเพื่อความแม่นยำ
st.subheader("⭐ การประเมินคุณภาพการบริการ (1-5 คะแนน)")
col_a, col_b, col_c = st.columns(3)

with col_a:
    st.markdown("**ความสะดวกในการรับบริการ**")
    online_boarding = st.slider("การเช็คอินออนไลน์ (สำคัญมาก)", 1, 5, 4)
    booking_ease = st.slider("ความง่ายในการจองตั๋วออนไลน์", 1, 5, 4)
    checkin_service = st.slider("การบริการที่เคาน์เตอร์เช็คอิน", 1, 5, 4)
    gate_location = st.slider("ตำแหน่งประตูขึ้นเครื่อง", 1, 5, 3)

with col_b:
    st.markdown("**ประสบการณ์บนเครื่อง**")
    wifi_service = st.slider("บริการ Wifi บนเครื่อง", 1, 5, 4)
    seat_comfort = st.slider("ความสบายของที่นั่ง", 1, 5, 4)
    leg_room = st.slider("พื้นที่วางขา (Leg room)", 1, 5, 4)
    food_drink = st.slider("อาหารและเครื่องดื่ม", 1, 5, 3)

with col_c:
    st.markdown("**บริการเสริมและความสะอาด**")
    onboard_service = st.slider("การดูแลของพนักงานบนเครื่อง", 1, 5, 4)
    entertainment = st.slider("ความบันเทิงบนเครื่อง", 1, 5, 4)
    cleanliness = st.slider("ความสะอาดภายในเครื่อง", 1, 5, 4)
    baggage_handling = st.slider("การจัดการสัมภาระ", 1, 5, 4)

st.divider()
col_d1, col_d2 = st.columns(2)
with col_d1:
    dep_delay = st.number_input("ดีเลย์ขาออก (นาที)", 0, 1500, 0)
with col_d2:
    arr_delay = st.number_input("ดีเลย์ขาเข้า (นาที)", 0, 1500, 0)

# --- 5. การประมวลผลและการแสดงผล ---
if st.button("วิเคราะห์ความพึงพอใจ", type="primary", use_container_width=True):
    # เตรียมข้อมูลให้ตรงตามโครงสร้างที่โมเดลต้องการ
    input_data = {
        'id': [0],
        'Gender': [gender],
        'Customer Type': [customer_type],
        'Age': [age],
        'Type of Travel': [type_of_travel],
        'Class': [customer_class],
        'Flight Distance': [flight_distance],
        'Inflight wifi service': [wifi_service],
        'Departure/Arrival convenience': [3], # ค่ากลาง
        'Ease of Online booking': [booking_ease],
        'Gate location': [gate_location],
        'Food and drink': [food_drink],
        'Online boarding': [online_boarding],
        'Seat comfort': [seat_comfort],
        'Inflight entertainment': [entertainment],
        'On-board service': [onboard_service],
        'Leg room service': [leg_room],
        'Baggage handling': [baggage_handling],
        'Checkin service': [checkin_service],
        'Inflight service': [onboard_service],
        'Cleanliness': [cleanliness],
        'Departure Delay in Minutes': [dep_delay],
        'Arrival Delay in Minutes': [arr_delay]
    }
    
    df = pd.DataFrame(input_data)

    try:
        # จัดเรียงคอลัมน์ให้ตรงตามโมเดล
        if hasattr(model, 'feature_names_in_'):
            expected_features = model.feature_names_in_
            for col in expected_features:
                if col not in df.columns:
                    df[col] = 0
            df = df[expected_features]
        
        # ทำนายผล
        prediction = model.predict(df)[0]
        
        st.markdown("### 📊 ผลการวิเคราะห์จาก AI")
        if str(prediction) == '1' or str(prediction).lower() == 'satisfied':
            st.success("🎉 **ผู้โดยสารมีแนวโน้ม: พึงพอใจ (Satisfied)**")
            st.balloons()
        else:
            st.warning("⚠️ **ผู้โดยสารมีแนวโน้ม: ไม่พึงพอใจ (Neutral or Dissatisfied)**")
            st.info("💡 ข้อแนะนำ: ปัจจัยอย่าง 'Online Boarding' และ 'Class' มีผลอย่างมากต่อการทำนาย")
            
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาด: {e}")

# --- ส่วนท้าย ---
st.divider()
st.caption(f"พัฒนาโดย: 67160363 PHOKIN KETMAYOON | Machine Learning Final Project")
