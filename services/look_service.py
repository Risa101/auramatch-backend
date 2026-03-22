# เพิ่มบรรทัดนี้ที่ด้านบนสุดของไฟล์
from models.look_model import get_looks_from_db

def get_looks_by_color(personal_color):
    # ตอนนี้จะสามารถเรียกใช้ฟังก์ชันนี้ได้แล้วโดยไม่เกิด Error "not defined"
    raw_looks = get_looks_from_db(personal_color)
    
    if not raw_looks:
        return []

    formatted_looks = []
    for look in raw_looks:
        formatted_looks.append({
            "id": look.get('look_id'),
            "name": look.get('look_name'),
            "personal_color": look.get('personal_color'),
            "image_url": look.get('image_url'), 
            "status": look.get('status')
        })
        
    return formatted_looks