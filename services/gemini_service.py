import base64
import json
import os
import re
import ssl
import urllib.error
import urllib.request

try:
    import certifi
    HAS_CERTIFI = True
except ImportError:
    HAS_CERTIFI = False

try:
    import google.auth
    import google.auth.transport.requests
    from google.oauth2 import service_account
    HAS_GOOGLE_AUTH = True
except ImportError:
    HAS_GOOGLE_AUTH = False

# cache โมเดลที่ค้นหาแล้ว เพื่อไม่ต้อง call API ซ้ำทุก request
_cached_model = None


def _extract_text(payload):
    try:
        candidates = payload.get('candidates', [])
        if not candidates:
            return "ไม่พบคำตอบจาก AI"
        content = candidates[0].get('content', {})
        parts = content.get('parts', [])
        if parts:
            full_text = "".join([p.get('text', '') for p in parts if 'text' in p])
            return full_text if full_text else "AI ประมวลผลสำเร็จแต่ไม่มีข้อความตอบกลับ"
        return "ไม่สามารถดึงข้อมูลจากโครงสร้างคำตอบได้"
    except Exception as e:
        return f"เกิดข้อผิดพลาดในการดึงข้อความ: {str(e)}"


def _get_available_models(api_key):
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
    try:
        context = ssl.create_default_context(cafile=certifi.where()) if HAS_CERTIFI else ssl.create_default_context()
        with urllib.request.urlopen(urllib.request.Request(url), timeout=10, context=context) as resp:
            data = json.loads(resp.read().decode())
            return [m['name'].split('/')[-1] for m in data.get('models', [])
                    if 'generateContent' in m.get('supportedGenerationMethods', [])]
    except Exception:
        return []


def _get_model(api_key):
    """ค้นหาโมเดลที่ใช้งานได้จาก .env (ไม่ cache เพื่อให้อัปเดตได้)"""
    global _cached_model

    # อ่านจาก .env ทุกครั้ง ไม่ใช้ cache เพื่อให้เปลี่ยนโมเดลได้โดยไม่ต้อง restart
    env_model = os.getenv("GEMINI_IMAGE_MODEL", "").strip()
    if env_model:
        if _cached_model != env_model:
            _cached_model = env_model
            print(f"✅ ใช้โมเดลจาก .env: {env_model}")
        return _cached_model

    available = _get_available_models(api_key)
    priority_list = [
        "gemini-1.5-flash",
        "gemini-flash-latest",
        "gemini-2.0-flash",
        "gemini-3-flash-preview",
    ]
    target = next((p for p in priority_list if p in available), None)
    if not target and available:
        target = available[0]
    if not target:
        raise RuntimeError("บัญชีนี้ไม่มีโมเดลที่รองรับการใช้งานผ่าน API")

    _cached_model = target
    print(f"✅ ค้นพบและ cache โมเดล: {target}")
    return _cached_model


def _get_access_token():
    """ดึง access token จาก service account JSON"""
    print(f"🔍 HAS_GOOGLE_AUTH={HAS_GOOGLE_AUTH}")
    if not HAS_GOOGLE_AUTH:
        print("⚠️ google-auth ไม่ได้ติดตั้ง")
        return None
    sa_path = os.getenv("GEMINI_SERVICE_ACCOUNT", "").strip()
    print(f"🔍 GEMINI_SERVICE_ACCOUNT={sa_path!r}")
    if not sa_path:
        return None
    if not os.path.isabs(sa_path):
        sa_path = os.path.join(os.path.dirname(__file__), "..", sa_path)
    sa_path = os.path.abspath(sa_path)
    print(f"🔍 resolved path={sa_path}, exists={os.path.exists(sa_path)}")
    if not os.path.exists(sa_path):
        return None
    try:
        creds = service_account.Credentials.from_service_account_file(
            sa_path,
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        creds.refresh(google.auth.transport.requests.Request())
        print(f"✅ Service account token ได้รับสำเร็จ")
        return creds.token
    except Exception as e:
        print(f"⚠️ Service account auth failed: {e}")
        return None


def _call_gemini(api_key, model, body):
    """ส่ง request ไปยัง Gemini API (ใช้ service account token ถ้ามี)"""
    token = _get_access_token()
    if token:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}"
        }
        print(f"🔑 ใช้ Service Account token")
    else:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
        headers = {"Content-Type": "application/json"}
        print(f"🔑 ใช้ API Key")

    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers=headers,
        method="POST"
    )
    context = ssl.create_default_context(cafile=certifi.where()) if HAS_CERTIFI else ssl.create_default_context()
    with urllib.request.urlopen(req, timeout=90, context=context) as resp:
        return json.loads(resp.read().decode("utf-8"))


def analyze_face_with_gemini(image_bytes, image_mime=None):
    """วิเคราะห์โครงหน้าและ personal color season จากรูปภาพ"""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("ไม่พบ GEMINI_API_KEY ในไฟล์ .env")

    # ใช้ GEMINI_ANALYZE_MODEL สำหรับ analyze (text output) ถ้าไม่มีให้ใช้ gemini-2.0-flash
    model = os.getenv("GEMINI_ANALYZE_MODEL", "gemini-2.0-flash").strip() or _get_model(api_key)
    print(f"🔍 Analyzing face with model: {model}")

    prompt = (
        "Analyze this face photo carefully. Determine:\n"
        "1. Personal color season based on skin undertone, hair color, and overall coloring. "
        "Choose EXACTLY ONE from: Spring, Summer, Autumn, Winter\n"
        "2. Face shape. Choose EXACTLY ONE from: Oval, Round, Square, Heart, Diamond, Rectangle\n"
        "3. Recommended makeup based on the face features:\n"
        "   - brows: choose ONE from [softArch, straight, arched]\n"
        "   - eyes: choose ONE from [natural, cat, dolly]\n"
        "   - nose: choose ONE from [softContour, definedContour, natural]\n"
        "   - lips: choose ONE from [gradient, full, soft]\n\n"
        "Return ONLY a valid JSON object with NO other text, NO markdown, NO explanation:\n"
        '{"season":"...","faceShape":"...","face":{"brows":"...","eyes":"...","nose":"...","lips":"..."}}'
    )

    body = {
        "contents": [{
            "parts": [
                {"text": prompt},
                {
                    "inline_data": {
                        "mime_type": image_mime or "image/jpeg",
                        "data": base64.b64encode(image_bytes).decode("ascii")
                    }
                }
            ]
        }],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 256
        }
    }

    try:
        result = _call_gemini(api_key, model, body)

        if 'promptFeedback' in result and result['promptFeedback'].get('blockReason'):
            raise RuntimeError(f"รูปภาพถูกปฏิเสธ: {result['promptFeedback']['blockReason']}")

        text = _extract_text(result).strip()

        # ดึง JSON จาก response (Gemini บางครั้งใส่ markdown มาด้วย)
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if not json_match:
            raise RuntimeError(f"ไม่พบ JSON ใน AI response: {text[:200]}")

        data = json.loads(json_match.group())

        # validate + fallback ค่าที่ได้
        valid_seasons = {"Spring", "Summer", "Autumn", "Winter"}
        valid_shapes = {"Oval", "Round", "Square", "Heart", "Diamond", "Rectangle"}
        valid_brows = {"softArch", "straight", "arched"}
        valid_eyes = {"natural", "cat", "dolly"}
        valid_nose = {"softContour", "definedContour", "natural"}
        valid_lips = {"gradient", "full", "soft"}

        season = data.get("season", "Spring")
        if season not in valid_seasons:
            season = "Spring"

        face_shape = data.get("faceShape", "Oval")
        if face_shape not in valid_shapes:
            face_shape = "Oval"

        face = data.get("face", {})
        brows = face.get("brows", "softArch")
        if brows not in valid_brows:
            brows = "softArch"
        eyes = face.get("eyes", "natural")
        if eyes not in valid_eyes:
            eyes = "natural"
        nose = face.get("nose", "natural")
        if nose not in valid_nose:
            nose = "natural"
        lips = face.get("lips", "gradient")
        if lips not in valid_lips:
            lips = "gradient"

        return {
            "success": True,
            "season": season,
            "faceShape": face_shape,
            "face": {"brows": brows, "eyes": eyes, "nose": nose, "lips": lips}
        }

    except urllib.error.HTTPError as e:
        error_msg = e.read().decode("utf-8", errors="ignore")
        print(f"❌ API Error: {error_msg}")
        raise RuntimeError(f"API Error {e.code}: {error_msg}")
    except json.JSONDecodeError as e:
        raise RuntimeError(f"ไม่สามารถ parse JSON จาก AI response: {str(e)}")
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        raise


def generate_image_with_gemini(image_bytes=None, image_mime=None, prompt=None):
    """Generate a makeup-applied image using Gemini image-generation model."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("ไม่พบ GEMINI_API_KEY ในไฟล์ .env")

    # ใช้ GEMINI_IMAGE_MODEL สำหรับ generate image (รองรับ image output)
    image_model = os.getenv("GEMINI_IMAGE_MODEL", "gemini-2.0-flash-exp-image-generation").strip()
    print(f"🚀 Image generation model: {image_model}")

    final_prompt = (
        prompt or
        "Apply natural everyday makeup to this exact face photo. Do NOT change the person's face shape, "
        "bone structure, skin tone, eye shape, nose, lips shape, or any facial features — the person's identity "
        "must remain 100% identical. Only add: light foundation to even skin tone, soft blush on cheeks, "
        "subtle neutral eyeshadow, defined brows following their natural arch, thin eyeliner, mascara, and a "
        "natural lip tint. The result must look like the same real person wearing light makeup. "
        "Keep the same lighting, angle, background, and photo style. Photo-realistic output only."
    ).strip()

    parts = [{"text": final_prompt}]
    if image_bytes:
        parts.append({
            "inline_data": {
                "mime_type": image_mime or "image/jpeg",
                "data": base64.b64encode(image_bytes).decode("ascii")
            }
        })

    body = {
        "contents": [{"parts": parts}],
        "generationConfig": {
            "responseModalities": ["TEXT", "IMAGE"],
            "temperature": 1.0,
            "maxOutputTokens": 8192
        }
    }

    try:
        result = _call_gemini(api_key, image_model, body)

        if 'promptFeedback' in result and result['promptFeedback'].get('blockReason'):
            return {"success": False, "text": f"Image blocked: {result['promptFeedback']['blockReason']}"}

        # Extract generated image from response parts
        candidates = result.get('candidates', [])
        if candidates:
            parts_out = candidates[0].get('content', {}).get('parts', [])
            for part in parts_out:
                if 'inlineData' in part or 'inline_data' in part:
                    img_data = part.get('inlineData') or part.get('inline_data')
                    img_mime = img_data.get('mimeType') or img_data.get('mime_type', 'image/png')
                    img_b64 = img_data.get('data', '')
                    print(f"✅ Got generated image ({img_mime}, {len(img_b64)} chars b64)")
                    return {
                        "success": True,
                        "text": "Makeup applied successfully",
                        "data_url": f"data:{img_mime};base64,{img_b64}"
                    }

        # No image in response — fall back to original
        text_response = _extract_text(result)
        print(f"⚠️ No image in response, returning original. Text: {text_response[:100]}")
        fallback_url = (
            f"data:{image_mime or 'image/jpeg'};base64,{base64.b64encode(image_bytes).decode('ascii')}"
            if image_bytes else None
        )
        return {"success": True, "text": text_response, "data_url": fallback_url}

    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8", errors="ignore")
        print(f"❌ API Error {e.code}: {error_body}")
        if e.code == 429:
            raise RuntimeError("QUOTA_EXCEEDED: Gemini API rate limit reached. Try again later.")
        raise RuntimeError(f"API Error {e.code}: {error_body}")
    except Exception as e:
        print(f"❌ Connection Error: {str(e)}")
        raise RuntimeError(f"Connection error: {str(e)}")
