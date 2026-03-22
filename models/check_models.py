import os
import urllib.request
import json
from dotenv import load_dotenv

# 1. โหลดค่าจากไฟล์ .env
load_dotenv()

# 2. ดึง API Key มาจากตัวแปรสภาพแวดล้อม
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("❌ ไม่พบ GEMINI_API_KEY ในไฟล์ .env กรุณาตรวจสอบอีกครั้ง")
    exit()

# 3. เตรียม URL สำหรับเช็คลิสต์โมเดล
url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"

print(f"🔍 กำลังตรวจสอบโมเดลที่ใช้งานได้สำหรับ Key: {api_key[:5]}...{api_key[-5:]}")

try:
    with urllib.request.urlopen(url) as response:
        data = json.loads(response.read().decode())
        print("\n✅ รายชื่อโมเดลที่คุณสามารถใช้ได้จริง:")
        print("-" * 50)
        
        found_flash = False
        for m in data.get('models', []):
            # กรองเฉพาะโมเดlที่รองรับการสร้างเนื้อหา (generateContent)
            if 'generateContent' in m.get('supportedGenerationMethods', []):
                model_name = m['name'].replace('models/', '')
                print(f"⭐ {model_name}")
                if "gemini-1.5-flash" in model_name:
                    found_flash = True
        
        print("-" * 50)
        if not found_flash:
            print("⚠️ คำเตือน: ไม่พบชื่อ gemini-1.5-flash ในรายการด้านบน")
            print("แนะนำให้ใช้ชื่อรุ่นตามที่ปรากฏในรายการข้างต้นไปใส่ใน .env แทนครับ")

except Exception as e:
    print(f"❌ เกิดข้อผิดพลาดในการเรียก API: {e}")
    print("ตรวจสอบว่า API Key ถูกต้องและอินเทอร์เน็ตใช้งานได้ปกติ")