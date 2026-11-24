import streamlit as st
import ollama
from ollama import Client
from groq import Groq
from PIL import Image, ImageOps
import base64
import io
import os

client = Client(
    host="https://ollama.com",
    headers={'Authorization': 'Bearer ' + os.environ.get('856c5b698272428ca5b21a6ef75232a7.lPsM-d_aW-lKFG7u_Ox5iTgH', '')}
)

def encode_image(image: Image.Image) -> str:
    """將圖片轉換為 Base64（保證為RGB，避免JPEG儲存失敗）"""
    # 修正 EXIF 方向（避免旋轉問題）
    image = ImageOps.exif_transpose(image)
    # 若非RGB，轉成RGB 以利JPEG編碼
    if image.mode not in ("RGB",):
        image = image.convert("RGB")
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def ollama_cloud_analyze_image(image, prompt):
    """使用 Ollama API 分析圖片"""
    try: 
        base64_image = encode_image(image)        
        response = client.chat(
            model='gemma4:12b',  # 指定您要使用的雲端模型
            messages=[
                {
                    'role': 'user',
                    'content': prompt,
                    'images': [base64_image]  # Ollama 支援直接傳入檔案路徑
                }
            ],
        )
        return response["message"]["content"].strip()
    except Exception as e:
        return f"Ollama API 錯誤: {str(e)}"

def ollama_analyze_image(image, prompt):
    """使用 Ollama API 分析圖片"""
    try:
        base64_image = encode_image(image)
        response = ollama.chat(
            model="gemma4:12b",  # llava:13b
            messages=[{
                "role": "user",
                "content": prompt,
                "images": [base64_image],
            }]
        )
        # 依 Ollama Python SDK，回傳物件結構如下
        return response["message"]["content"].strip()
    except Exception as e:
        return f"Ollama API 錯誤: {str(e)}"

def groq_analyze_image(image, prompt):
    """使用 Groq API 分析圖片"""
    try:
        #if not GROQ_API_KEY or GROQ_API_KEY == "<API-KEY>":
        #    return "Groq API 錯誤: 尚未設定 API Key。"
        base64_image = encode_image(image)
        image_content = {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
        }

        client = Groq()
        completion = client.chat.completions.create(
            model="qwen/qwen3.8-27b",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    image_content
                ]
            }],
            temperature=0,
            max_completion_tokens=1024,
            top_p=0.5,
            stream=False,
            stop=None
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Groq API 錯誤: {str(e)}"

st.title("Ollama-Groq視覺分析助手")

api_choice = st.radio("選擇 API:", ["Ollama", "Ollama-Cloud", "Groq"], horizontal=True)

uploaded_file = st.file_uploader(
    "上傳圖檔",
    type=["jpg", "jpeg", "png"],
    help="請上傳您想分析的圖檔"
)

default_prompt = "請針對此圖片，使用繁體中文說明你看到了什麼？如果你是一位觀光客，請說明你看到的特點。"
prompt = st.text_area("請輸入提示詞", value=default_prompt, height=100)

if st.button("分析影像"):
    if uploaded_file is None:
        st.warning("請先上傳圖檔。")
    else:
        col1, col2 = st.columns(2)
        image = Image.open(uploaded_file)

        # 修正：這裡用 width (不是 use_container_width)
        col1.image(image, caption="已上傳的圖檔")

        with st.spinner(f"使用 {api_choice} API 分析中..."):
            if api_choice == "Ollama":
                result = ollama_analyze_image(image, prompt)  
            elif api_choice == "Ollama-Cloud":
                result = ollama_cloud_analyze_image(image, prompt)
            else:
                result = groq_analyze_image(image, prompt)

        st.success("影像分析完成!")
        col2.write(result)
