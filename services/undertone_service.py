"""
undertone_service.py

วิเคราะห์ Skin Undertone (cool/warm/neutral) จากรูปภาพ แล้วดึงสินค้าบลัชออนจริงจาก
ฐานข้อมูล (products) ที่เหมาะกับ undertone นั้นมาแนะนำ

ส่วนตรวจ undertone พอร์ตมาจาก detection_skin_for_cream(2).ipynb — โน้ตบุ๊กต้นฉบับ
"ไม่มี" การเทรน ML model ใดๆ (ไม่มี dataset, ไม่มี .h5/.pkl, ไม่มี neural network)
เป็น classical computer-vision ล้วนๆ ใช้ CIELAB (Lab) color space เพียงอย่างเดียว
ตลอดทั้ง pipeline (ตามคำขอ — ไม่ผสม YCrCb):
  1. simple_skin_mask()   — BGR→Lab→threshold ช่อง a*/b*→morphology open/close→largest component
  2. estimate_undertone() — cell 7 เดิม: BGR→Lab เฉลี่ยบนพิกเซลผิว→คำนวณมุม ITA→classify

หมายเหตุสำคัญ: notebook ต้นฉบับ (cell 5) ใช้ YCrCb สำหรับหา skin mask ไม่ใช่ CIELAB —
ไม่มีขั้นตอน Lab-based masking อยู่ใน notebook ให้พอร์ตตรงๆ จึงต้อง "คิดค่า threshold
a*/b* ขึ้นใหม่" โดย calibrate จากตัวอย่างสีผิวจริง 20 เฉด (ขาวมากถึงเข้มมาก) แล้วเผื่อ
margin สมเหตุสมผล — ไม่ใช่ค่าที่ตัดตรงจาก notebook เหมือนส่วนอื่น (แจ้งไว้ให้ชัดเจน)
ยังคงกลยุทธ์เดิมของ notebook ไว้คือ "ไม่จำกัดช่องความสว่าง" (Y ใน YCrCb เดิมไม่ถูกจำกัด
เช่นกัน) จำกัดเฉพาะช่อง chrominance (a*/b* แทน Cr/Cb)

ellipse kernel, morphology และสูตร ITA ใน estimate_undertone() คัดลอกจาก notebook ตรงๆ

ส่วนคำแนะนำสินค้า: notebook เดิม (cell 9, blush_suggestions()) ใช้ตาราง hex สีตายตัว
(hardcode) — เปลี่ยนมาดึงสินค้าบลัชออนจริงจาก DB ผ่าน
product_service.get_blush_products_by_undertone() แทน ไม่มีค่าตายตัวเหลืออยู่แล้ว

เพิ่มเติม: estimate_season()/analyze_season() — แทนที่ RandomForest (ml_service.py,
เทรนจาก features RGB ล้วน ไม่มี dataset เดิมเหลือให้เทรนใหม่แบบ Lab ได้) ด้วยกฎ
classical ที่อยู่ใน Lab เหมือนกันทั้งไฟล์ ใช้ ITA (แกน warm/cool อันเดียวกับ
estimate_undertone) ไขว้กับ L* (แกน light/deep) ตามทฤษฎี 4-season มาตรฐาน:
warm+light=Spring, warm+deep=Autumn, cool+light=Summer, cool+deep=Winter
เกณฑ์ ITA=19 (จุดกึ่งกลางของช่วง neutral 10-28 ที่ใช้อยู่แล้ว) และ L*=55 เป็นค่า
ประมาณทางวิศวกรรม ไม่ได้ calibrate จาก labeled dataset (ไม่มีให้ใช้) เหมือนที่เกณฑ์
a*/b* ของ simple_skin_mask() ก็ระบุไว้ชัดเจนว่ามาจากไหน — แจ้งไว้ตรงนี้เพื่อความโปร่งใส
เท่ากัน ยังคงเป็นแค่ fallback ชั้นกลาง (Color Engine ชนะอยู่ดีถ้าทำงานได้)
"""

import math

import numpy as np
import cv2 as cv

from services.product_service import get_blush_products_by_undertone

# เกณฑ์แบ่งกลุ่มสำหรับ estimate_season() — ดู docstring ด้านบนสำหรับที่มา
_SEASON_ITA_SPLIT = 19.0   # >= = cool-leaning, < = warm-leaning
_SEASON_L_SPLIT = 55.0     # >= = light, < = deep


def _skin_lab_mean(bgr: np.ndarray, skin_mask: np.ndarray):
    """
    ค่าเฉลี่ย L*/a*/b* (สเกลจริง ไม่ใช่ uint8 ของ OpenCV) เฉพาะพิกเซลผิวตาม mask
    ใช้ร่วมกันทั้ง estimate_undertone() และ estimate_season() — คำนวณครั้งเดียว
    ไม่ต้องคัดลอกสูตรแปลง Lab ซ้ำสองที่

    คืนค่า None ถ้าพิกเซลผิวน้อยเกินไปจะเชื่อถือค่าเฉลี่ยไม่ได้ (< 500 พิกเซล)
    """
    skin = cv.bitwise_and(bgr, bgr, mask=skin_mask)
    lab = cv.cvtColor(skin, cv.COLOR_BGR2Lab)

    m = skin_mask > 0
    if m.sum() < 500:
        return None

    L = lab[..., 0][m].mean() * (100 / 255.0)
    a = (lab[..., 1][m].mean() - 128)
    b = (lab[..., 2][m].mean() - 128)
    return L, a, b


def simple_skin_mask(bgr: np.ndarray) -> np.ndarray:
    """
    สร้าง mask ของบริเวณผิว (skin mask) ด้วย CIELAB (Lab) เพียงอย่างเดียว — แทนที่
    YCrCb ของ notebook ต้นฉบับ (cell 5) ตามคำขอ ให้ทั้ง pipeline อยู่ใน color space
    เดียว (Lab) ตลอด

    ช่วง threshold ของช่อง a*/b* (OpenCV Lab เป็น uint8, a/b มี offset +128 จาก
    ค่าจริง — a*=b*=128 คือจุด "เทากลาง/ไม่มีสี" พอดี สีเทา/ขาว/ดำล้วนตกที่จุดนี้
    ไม่ว่าจะสว่างแค่ไหน) ได้จากการแปลงตัวอย่างสีผิวอ้างอิง 20 เฉด BGR→Lab จริงด้วย
    cv2 แล้วดูช่วงที่ได้ (a: 134-157, b: 136-174) — ไม่ใช่ค่าที่ port มาจาก notebook
    เพราะ notebook ไม่มีขั้นตอนนี้ (ดู docstring ท้ายไฟล์)

    ขอบล่างต้อง "สูงกว่า 128 อย่างมีนัยสำคัญ" เสมอ (ห้ามแตะ/ต่ำกว่า 128) ไม่งั้นจะกิน
    จุดเทากลางเข้ามาด้วย ทำให้พื้นหลังสีเทา/ขาว/ดำถูกจัดเป็น "ผิว" ผิดๆ — ทดสอบแล้วว่า
    เผื่อ margin กว้างเกิน (ขอบล่าง=128) ทำให้ขาว/ดำ/เทาล้วนหลุดเข้ามาเป็นผิวจริง จึงใช้
    ขอบล่าง a=132, b=130 (สูงกว่าจุดเทากลาง แต่ยังต่ำกว่าค่าต่ำสุดที่ calibrate ได้)

    ไม่จำกัดช่อง L* (ความสว่าง) เหมือนที่ notebook เดิมก็ไม่จำกัดช่อง Y เช่นกัน —
    รักษากลยุทธ์ "กรองด้วย chrominance เท่านั้น" ของ notebook ไว้

    คืนค่า: mask (0 = ไม่ใช่ผิว, 255 = ผิว)
    """
    lab = cv.cvtColor(bgr, cv.COLOR_BGR2Lab)

    lower = np.array([0, 132, 130], dtype=np.uint8)
    upper = np.array([255, 180, 200], dtype=np.uint8)
    m = cv.inRange(lab, lower, upper)

    k = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    m = cv.morphologyEx(m, cv.MORPH_OPEN, k, iterations=2)
    m = cv.morphologyEx(m, cv.MORPH_CLOSE, k, iterations=2)

    num, labels, stats, _ = cv.connectedComponentsWithStats(m, connectivity=8)
    if num > 1:
        areas = stats[1:, cv.CC_STAT_AREA]
        largest = 1 + np.argmax(areas)
        m = np.where(labels == largest, 255, 0).astype(np.uint8)

    return m


def estimate_undertone(bgr: np.ndarray, skin_mask: np.ndarray):
    """
    ประเมิน undertone (cool/warm/neutral) จากภาพ — เหมือน notebook cell 7 ทุกขั้นตอน
    คืนค่า: (undertone, ITA, L*)
    """
    stats = _skin_lab_mean(bgr, skin_mask)
    if stats is None:
        return "unknown", 0.0, 0.0
    L, a, b = stats

    ITA = math.degrees(math.atan2((L - 50), b if abs(b) > 1e-3 else 1e-3))

    if ITA >= 28:
        tone = "cool"
    elif ITA <= 10:
        tone = "warm"
    else:
        tone = "neutral"

    return tone, ITA, L


def estimate_season(bgr: np.ndarray, skin_mask: np.ndarray):
    """
    ประเมิน personal color season (Spring/Summer/Autumn/Winter) จาก Lab ล้วนๆ —
    แทนที่ RandomForest (RGB features) เดิม ดู docstring ท้ายไฟล์สำหรับที่มาของเกณฑ์

    ไขว้ 2 แกน: warm/cool (มุม ITA เดียวกับ estimate_undertone) กับ light/deep (L*)
    คืนค่า: (season, ITA, L*)
    """
    stats = _skin_lab_mean(bgr, skin_mask)
    if stats is None:
        return "unknown", 0.0, 0.0
    L, a, b = stats

    ITA = math.degrees(math.atan2((L - 50), b if abs(b) > 1e-3 else 1e-3))
    is_cool = ITA >= _SEASON_ITA_SPLIT
    is_light = L >= _SEASON_L_SPLIT

    if is_cool:
        season = "Summer" if is_light else "Winter"
    else:
        season = "Spring" if is_light else "Autumn"

    return season, ITA, L


def analyze_undertone(image_bytes: bytes) -> dict:
    """
    Entry point สำหรับ controller — เทียบเท่า notebook cell 11 (ตัดส่วน Colab upload
    dialog / matplotlib display ออก) บวกกับดึงสินค้าบลัชออนจริงจาก DB แทน hardcode

    Args:
        image_bytes: รูปภาพในรูปแบบ bytes

    Returns:
        dict {
            undertone: "cool" | "warm" | "neutral" | "unknown",
            ita: float (องศา),
            l_star: float (0-100, ความสว่างผิว),
            skin_pixel_count: int,
            recommended_products: [ { product_id, name, image_url, price,
                                       suitable_for_color, personal_color_tags,
                                       shades, shop_url }, ... ]  # จาก DB จริง
        }
    """
    file_bytes = np.frombuffer(image_bytes, np.uint8)
    bgr = cv.imdecode(file_bytes, cv.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("ไม่สามารถอ่านไฟล์รูปภาพได้ (รูปแบบไฟล์ไม่ถูกต้องหรือเสียหาย)")

    mask = simple_skin_mask(bgr)
    undertone, ita, l_star = estimate_undertone(bgr, mask)
    products = get_blush_products_by_undertone(undertone)

    return {
        "undertone": undertone,
        "ita": round(float(ita), 2),
        "l_star": round(float(l_star), 2),
        "skin_pixel_count": int((mask > 0).sum()),
        "recommended_products": products,
    }


def analyze_season(image_bytes: bytes) -> dict:
    """
    Entry point สำหรับ controller — วิเคราะห์ personal color season จาก Lab ล้วนๆ
    แทนที่ RandomForest (services/ml_service.py, ถูกลบแล้ว) ดู estimate_season()

    Args:
        image_bytes: รูปภาพในรูปแบบ bytes

    Returns:
        dict {
            season: "Spring" | "Summer" | "Autumn" | "Winter" | "unknown",
            ita: float (องศา),
            l_star: float (0-100, ความสว่างผิว),
            skin_pixel_count: int,
        }
    """
    file_bytes = np.frombuffer(image_bytes, np.uint8)
    bgr = cv.imdecode(file_bytes, cv.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("ไม่สามารถอ่านไฟล์รูปภาพได้ (รูปแบบไฟล์ไม่ถูกต้องหรือเสียหาย)")

    mask = simple_skin_mask(bgr)
    season, ita, l_star = estimate_season(bgr, mask)

    return {
        "season": season,
        "ita": round(float(ita), 2),
        "l_star": round(float(l_star), 2),
        "skin_pixel_count": int((mask > 0).sum()),
    }
