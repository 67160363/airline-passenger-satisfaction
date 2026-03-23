✈️ Airline Passenger Satisfaction Prediction

Machine Learning Deployment Project - Final Assignment

📌 1. การนิยามปัญหา (Problem Definition)

หมวดที่ 1: การนิยามปัญหาและการเลือก Dataset (3 คะแนนดิบ)

ในอุตสาหกรรมการบินที่มีการแข่งขันสูง การรักษาความพึงพอใจของผู้โดยสารเป็นปัจจัยสำคัญที่สุดในการสร้างความภักดีต่อแบรนด์ (Brand Loyalty) โปรเจกต์นี้มีวัตถุประสงค์เพื่อพัฒนาแบบจำลอง Machine Learning ที่สามารถทำนายความพึงพอใจของผู้โดยสาร (Satisfied หรือ Neutral/Dissatisfied) จากข้อมูลการสำรวจจริง เพื่อช่วยให้สายการบินสามารถระบุจุดที่ควรปรับปรุงบริการได้อย่างแม่นยำ

ทำไมต้องใช้ Machine Learning?
ความพึงพอใจของผู้โดยสารไม่ได้ขึ้นอยู่กับปัจจัยเดียว แต่เกิดจากความสัมพันธ์ของหลายตัวแปร (เช่น ระยะทางบิน, ประเภทที่นั่ง, การบริการบนเครื่อง) การใช้ ML ช่วยให้เราตรวจจับรูปแบบพฤติกรรมที่ซับซ้อนและระบุปัจจัยวิกฤต (Key Drivers) ที่ส่งผลต่อความพอใจได้ดีกว่าการวิเคราะห์สถิติแบบดั้งเดิม

📊 2. ข้อมูลที่ใช้ (Dataset & Features)

หมวดที่ 1: ข้อมูลและ Features

Source: ชุดข้อมูลจาก Kaggle [Airline Passenger Satisfaction]

ความน่าเชื่อถือ: ข้อมูลจากการสำรวจผู้โดยสารจริง ครอบคลุมหลายแง่มุมของการบริการ

ตัวแปรสำคัญ (Features):

Gender, Age, Customer Type, Type of Travel, Class

Flight Distance: ระยะทางของเที่ยวบิน

Inflight services: คะแนนความพึงพอใจ 1-5 (เช่น Wifi, Online booking, Seat comfort, Inflight entertainment)

Delays: Departure/Arrival delay in minutes (ระยะเวลาที่ล่าช้า)

⚙️ 3. กระบวนการเตรียมข้อมูลและโมเดล (Methodology)

หมวดที่ 2 & 3: การเตรียมข้อมูลและ Model Development (10 คะแนนดิบ)

เราออกแบบระบบการทำงานที่โปร่งใสและตรวจสอบได้โดยใช้ Scikit-learn Pipeline:

EDA (Exploratory Data Analysis): วิเคราะห์การกระจายตัวและจัดการ Missing Values ในคอลัมน์ Arrival Delay ด้วยค่า Median เนื่องจากข้อมูลมี Outliers สูง (จัดการอย่างมีเหตุผลตามเกณฑ์หมวด 2)

Preprocessing Pipeline:

SimpleImputer: จัดการค่าที่หายไปอย่างเป็นระบบ

StandardScaler: ปรับสเกลข้อมูลตัวเลขเพื่อให้โมเดลทำงานได้เสถียร

OneHotEncoder: แปลงข้อมูลหมวดหมู่ (Categorical) ให้เป็นตัวเลขที่คอมพิวเตอร์เข้าใจ

Model Development: เปรียบเทียบประสิทธิภาพระหว่างโมเดลหลายตัว เช่น Random Forest และ XGBoost (เพื่อคะแนนโบนัส)

Evaluation: ใช้ 5-Fold Cross-validation และวัดผลด้วย F1-Score/Recall เพื่อให้แน่ใจว่าเราไม่ได้มองแค่ความแม่นยำรวม (Accuracy) แต่ให้ความสำคัญกับการระบุผู้โดยสารที่ไม่พอใจด้วย

🚀 Deployed Web Application

หมวดที่ 4: Deployed Web Application (5 คะแนนดิบ)

แอปพลิเคชันถูก Deploy บน Streamlit Cloud พร้อมระบบที่ใช้งานง่ายและปลอดภัย:
👉 [ลิงก์แอปพลิเคชันจะถูกอัปเดตที่นี่เมื่อ Deploy เสร็จสมบูรณ์]

คุณสมบัติเด่น:

มี Input Validation ป้องกันการกรอกข้อมูลที่ผิดพลาด

แสดงผลการทำนายพร้อมค่าความมั่นใจ (Confidence Probability)

UI ออกแบบมาให้ผู้ใช้ทั่วไป (Non-technical user) เข้าใจได้ทันที

จัดทำโดย: 67160363 PHOKIN KETMAYOON
