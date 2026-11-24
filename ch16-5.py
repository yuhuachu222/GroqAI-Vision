import streamlit as st
import ollama
from groq import Groq
from PIL import Image
import base64
import io
import re
import os

# 請在下方填入您的 Groq API Key
GROQ_API_KEY=os.getenv("Groq_API_KEY")


# 車牌辨識的提示詞
prompt = """
請仔細檢查這張圖片，尋找車牌。
如果找到車牌，請使用繁體中文來提供以下的資訊：
1. 車牌號碼（格式應為2-3個大寫英文字母 + 連字號 + 4-5個數字，如 AB-1234）
2. 車牌所在車輛的顏色
3. 車牌所在車輛的大致類型（轎車、休旅車、卡車等）

若無法清楚辨識，請說明原因（如圖片模糊、角度不佳等）。
"""

def encode_image(image):
    """將圖片轉換為 Base64 編碼"""
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def extract_license_plate(text):
    """
    從文本中提取車牌號碼
    支援台灣車牌格式: 
    - 2-3個英文字母 + 4-5個數字
    - 例如: AB-1234, ABC-12, 99-9999
    """
    # 台灣車牌正則表達式, 可以偵測ABC-8888、ABC8888，甚至像 ABC 8888格式。
    taiwan_plate_pattern = r'[A-Z]{2,3}[-\s]?\d{4,5}'   
    # 尋找符合台灣車牌格式的字串
    matches = re.findall(taiwan_plate_pattern, text)
    return matches[0] if matches else "未偵測到車牌"

def ollama_analyze_image(image):
    """使用 Ollama Cloud API 分析圖片中的車牌"""
    try: 
        base64_image = encode_image(image)        
        response = ollama.chat(
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


def groq_analyze_image(image):
    """使用 Groq API 分析圖片中的車牌"""
    try:
        base64_image = encode_image(image)
        image_content = {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
        }
         
        client = Groq(api_key=GROQ_API_KEY)
        
        completion = client.chat.completions.create(
            model="qwen/qwen3.8-27b",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    image_content
                ]
            }],
            temperature=1,
            max_completion_tokens=512,
            top_p=1,
            stream=False,
            stop=None
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Groq API 錯誤: {str(e)}"

# Streamlit 應用程式
st.title("🚗 車牌辨識助手")    
# API 選擇
api_choice = st.radio(
    "選擇 API:", 
    ["Ollama", "Groq"], 
    horizontal=True
)
# 圖檔上傳
uploaded_file = st.file_uploader(
    "上傳車輛圖檔", 
    type=["jpg", "jpeg", "png"], 
    help="請上傳包含車牌的清晰圖檔"
)
# 分析按鈕
if st.button("辨識車牌") and uploaded_file is not None:
    column1, column2 = st.columns(2)    
    # 讀取上傳的圖檔
    image = Image.open(uploaded_file)
    column1.image(image, caption="已上傳的車輛圖檔")
    # 顯示分析中的載入動畫
    with st.spinner(f"使用 {api_choice} API 分析中, 請稍等一下..."):
        # 依據選擇的 API 進行分析
        if api_choice == "Ollama":
            result = ollama_analyze_image(image) #uploaded_file
        else:
            result = groq_analyze_image(image)    
    # 取出車牌
    license_plate = extract_license_plate(result)
    # 顯示分析結果
    st.success("車牌辨識完成!")
    column2.markdown(f"### 辨識結果\n{result}")
    # 顯示車牌
    st.markdown(f"## 🚘 車牌號碼: `{license_plate}`")