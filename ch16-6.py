
import streamlit as st
import cv2
import numpy as np
import ollama
from groq import Groq
import base64
import io

# 請在下方填入即時影像的網址
traffic_video_url = "https://trafficvideo.tainan.gov.tw/6a0feb9b"

# 路況分析提示詞
traffic_prompt = """
請仔細分析這張即時路況畫面，提供以下詳細資訊：
1. 交通流量狀態
- 道路擁擠程度（輕微、中度、嚴重）
- 車輛數量和密度
- 是否有明顯的壅塞或停滯
2. 道路環境觀察
- 車道數量
- 道路類型（市區道路、快速道路、交叉路口）
- 天氣和光線條件
3. 異常狀況偵測
- 是否有事故
- 是否有施工或道路障礙
- 是否有緊急車輛（救護車、警車、消防車）
4. 可能的交通風險
- 可能造成延遲的因素
- 建議駕駛人注意的特殊情況
請使用繁體中文且盡可能的提供具體、客觀和詳細的描述，
協助判斷目前的路況。
"""


def encode_image(image):
    """將圖片轉換為 Base64 編碼"""

    # 確認影像是否存在
    if image is None:
        raise ValueError("尚未取得有效影像")

    success, buffer = cv2.imencode('.jpg', image)

    if not success:
        raise ValueError("影像編碼失敗")

    return base64.b64encode(buffer).decode("utf-8")


def ollama_analyze_traffic(image):
    """使用 Ollama API 分析路況"""

    try:
        base64_image = encode_image(image)

        response = ollama.chat(
            model="gemma4:12b",
            messages=[{
                "role": "user",
                "content": traffic_prompt,
                "images": [base64_image]
            }]
        )

        return response["message"]["content"].strip()

    except Exception as e:
        return f"Ollama API 錯誤: {str(e)}"


def groq_analyze_traffic(image):
    """使用 Groq API 分析路況"""

    try:
        base64_image = encode_image(image)

        image_content = {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
            }
        }

        # 自動讀取 GROQ_API_KEY 環境變數
        client = Groq()

        completion = client.chat.completions.create(
            model="qwen/qwen3.8-27b",
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": traffic_prompt
                    },
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


# ==========================================
# Streamlit 應用程式主體
# ==========================================

st.title("🚦 即時路況分析系統")

# API 選擇
api_choice = st.radio(
    "選擇 API:",
    ["Ollama", "Groq"],
    horizontal=True
)

# 使用 session_state 持久化儲存影格
if "current_frame" not in st.session_state:
    st.session_state.current_frame = None

# 即時影像
if traffic_video_url:

    try:
        # 建立即時影像顯示區域
        st_frame = st.empty()

        # 分析按鈕
        if st.button("分析路況"):

            current_frame = st.session_state.current_frame

            # 確認是否已取得有效影像
            if current_frame is None:

                st.warning("尚未取得路況影像，請稍後再試。")

            else:

                column1, column2 = st.columns(2)

                # 顯示目前取得的路況影像
                column1.image(
                    current_frame,
                    caption="目前的路況",
                    channels="BGR"
                )

                # 顯示分析中的載入動畫
                with st.spinner(
                    f"使用 {api_choice} API 分析中，請稍等一下..."
                ):

                    # 依據選擇的 API 進行分析
                    if api_choice == "Ollama":

                        result = ollama_analyze_traffic(
                            current_frame
                        )

                    else:

                        result = groq_analyze_traffic(
                            current_frame
                        )

                # 顯示分析結果
                if result and "API 錯誤:" not in result:

                    st.success("路況分析完成!")

                    column2.markdown(
                        f"### 路況分析結果\n{result}"
                    )

                else:

                    column2.error(
                        result or "未取得分析結果"
                    )

        # ==========================================
        # 連接網路攝影機
        # 注意：這段不在分析按鈕的 if 裡面
        # ==========================================

        vid_cap = cv2.VideoCapture(traffic_video_url)

        try:

            # 檢查攝影機是否成功連線
            if not vid_cap.isOpened():

                st.error(
                    "無法連接網路攝影機，請檢查串流網址。"
                )

                st.stop()

            # 持續讀取即時影像
            while True:

                success, image = vid_cap.read()

                # 檢查影像是否成功取得
                if not success or image is None:

                    st.warning("無法取得攝影機影像")

                    break

                # 深拷貝影像到 session_state
                st.session_state.current_frame = np.copy(
                    image
                )

                # 顯示即時影像
                st_frame.image(
                    image,
                    channels="BGR"
                )

        finally:

            # 釋放攝影機資源
            vid_cap.release()

    except Exception as e:

        st.error(f"載入網路攝影機錯誤: {str(e)}")

else:

    st.error(
        "錯誤! 沒有監控攝影機 IP Camera 的 URL 網址..."
    )
