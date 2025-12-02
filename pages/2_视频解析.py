import cv2
import numpy as np
import base64
import requests
import tqdm
from PIL import Image
import io
import streamlit as st

# ---------------------- 1. 共享配置（API密钥+颜色同步）----------------------
API_KEY = "ms-9f99616d-d3cf-4783-922a-1ed9599fec3a"  # 已预设你的魔搭API密钥，无需修改
COLOR_SCHEMES = [
    {"bg": "#121212", "card": "#1E1E1E", "btn": "#8B5CF6", "accent": "#8B5CF6"},
    {"bg": "#1E1E2E", "card": "#2D2D44", "btn": "#6366F1", "accent": "#6366F1"},
    {"bg": "#1A1E3B", "card": "#2A2F55", "btn": "#3B82F6", "accent": "#3B82F6"},
    {"bg": "#2A1B3D", "card": "#3D2B5C", "btn": "#A855F7", "accent": "#A855F7"},
    {"bg": "#1B3B2A", "card": "#2B5C45", "btn": "#22C55E", "accent": "#22C55E"}
]
current_color = COLOR_SCHEMES[st.session_state.get("color_idx", 0)]

# ---------------------- 2. 界面样式（与图片页同步）----------------------
st.markdown(f"""
    <style>
        .stApp {{background-color: {current_color["bg"]}; color: #E0E0E0;}}
        .func-card {{
            background-color: {current_color["card"]};
            border-radius: 20px;
            padding: 20px;
            margin: 10px 0;
            border: 1px solid #333;
        }}
        .stButton > button {{
            background-color: {current_color["btn"]};
            color: white;
            border-radius: 10px;
            padding: 8px 16px;
            border: none;
        }}
        .stButton > button:hover {{background-color: {current_color["accent"]};}}
        .stTextArea > div > textarea {{
            background-color: {current_color["card"]};
            color: #E0E0E0;
            border-radius: 10px;
            border: 1px solid #444;
        }}
        .stFileUploader > div > div {{
            background-color: {current_color["card"]};
            border-radius: 10px;
            border: 1px dashed #444;
        }}
    </style>
""", unsafe_allow_html=True)

# ---------------------- 3. 核心工具函数（视频分析）----------------------
def video_to_keyframes(video_file):
    temp_video_path = "temp_video.mp4"
    with open(temp_video_path, "wb") as f:
        f.write(video_file.getbuffer())
    
    cap = cv2.VideoCapture(temp_video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    keyframes = []
    frame_interval = fps  # 每秒1帧
    
    with st.spinner(f"提取关键帧（共{total_frames}帧）..."):
        progress_bar = st.progress(0)
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % frame_interval == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_pil = Image.fromarray(frame_rgb)
                frame_pil.thumbnail((640, 360))
                keyframes.append(frame_pil)
            frame_idx += 1
            progress_bar.progress(min(frame_idx / total_frames, 1.0))
    
    cap.release()
    return keyframes, fps

def image_to_base64(image):
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format="JPEG")
    return base64.b64encode(img_byte_arr.getvalue()).decode("utf-8")

def analyze_video(video_file):
    # 提取关键帧
    keyframes, fps = video_to_keyframes(video_file)
    if len(keyframes) == 0:
        return "视频帧提取失败，请更换视频文件"
    
    # 关键帧转Base64
    base64_frames = [image_to_base64(frame) for frame in keyframes]
    
    # 调用API分析
    url = "https://api-inference.modelscope.cn/v1/chat/completions"
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": "Qwen/Qwen2.5-VL-72B-Instruct",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"""分析视频{len(base64_frames)}个关键帧，输出：
1. 核心主体：贯穿始终的人物/物体
2. 画面风格：艺术风格+色彩基调
3. 运镜手法：类型+速度
4. 分镜头：切换点+时长
5. 场景转换：方式
分点清晰，可直接参考"""}
                ] + [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}} for b64 in base64_frames]
            }
        ],
        "max_tokens": 800,
        "temperature": 0.6
    }
    response = requests.post(url, headers=headers, json=payload, timeout=90)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]

# ---------------------- 4. 核心功能布局（仅上传+分析+结果）----------------------
st.title("🎬 视频全维度分析")

# 1. 视频上传+分析按钮（功能卡片）
with st.container():
    st.markdown('<div class="func-card">', unsafe_allow_html=True)
    uploaded_video = st.file_uploader("上传视频（MP4/AVI/MKV，≤200MB）", type=["mp4", "avi", "mkv"])
    if uploaded_video:
        video_size = round(uploaded_video.size / 1024 / 1024, 2)
        st.markdown(f"📊 视频信息：{uploaded_video.name}（大小：{video_size}MB）")
    analyze_btn = st.button("🎯 开始视频分析", type="primary")
    st.markdown('</div>', unsafe_allow_html=True)

# 2. 结果展示框（功能卡片）
with st.container():
    st.markdown('<div class="func-card">', unsafe_allow_html=True)
    st.subheader("📝 分析结果")
    result_box = st.text_area("分析结果将显示在这里（可直接复制）", height=300, disabled=True, key="video_result")
    
    if analyze_btn and uploaded_video:
        try:
            with st.spinner("分析中...（约10-20秒）"):
                result = analyze_video(uploaded_video)
                st.text_area("✅ 分析完成", value=result, height=300, key="video_result_active")
        except Exception as e:
            st.error(f"分析失败：{str(e)}")
    st.markdown('</div>', unsafe_allow_html=True)
