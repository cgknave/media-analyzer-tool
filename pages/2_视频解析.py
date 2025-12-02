import cv2
import numpy as np
import base64
import requests
import tqdm
from PIL import Image
import io
import streamlit as st

# ---------------------- 1. 共享配置（API密钥+颜色同步）----------------------
API_KEY = "ms-9f99616d-d3cf-4783-922a-1ed9599fec3a"
COLOR_SCHEMES = [
    {"bg": "#121212", "card": "#1E1E1E", "btn": "#8B5CF6", "accent": "#A78BFA"},
    {"bg": "#1E1E2E", "card": "#2D2D44", "btn": "#6366F1", "accent": "#818CF8"},
    {"bg": "#1A1E3B", "card": "#2A2F55", "btn": "#3B82F6", "accent": "#60A5FA"},
    {"bg": "#2A1B3D", "card": "#3D2B5C", "btn": "#A855F7", "accent": "#C084FC"},
    {"bg": "#1B3B2A", "card": "#2B5C45", "btn": "#22C55E", "accent": "#4ADE80"}
]
current_color = COLOR_SCHEMES[st.session_state.get("color_idx", 0)]

# ---------------------- 2. 界面样式（增强视觉层次）----------------------
st.markdown(f"""
    <style>
        .stApp {{
            background-color: {current_color["bg"]};
            color: #E0E0E0;
            font-family: 'Segoe UI', Roboto, sans-serif;
        }}
        /* 功能卡片 - 阴影+圆角优化 */
        .func-card {{
            background-color: {current_color["card"]};
            border-radius: 16px;
            padding: 24px;
            margin: 16px 0;
            border: 1px solid #333;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
            transition: box-shadow 0.3s ease;
        }}
        .func-card:hover {{
            box-shadow: 0 6px 16px rgba(0,0,0,0.4);
        }}
        /* 按钮样式 */
        .stButton > button {{
            background-color: {current_color["btn"]};
            color: white;
            border-radius: 10px;
            padding: 10px 20px;
            border: none;
            font-weight: 500;
            transition: all 0.3s ease;
            box-shadow: 0 2px 8px rgba(139, 92, 246, 0.3);
        }}
        .stButton > button:hover {{
            background-color: {current_color["accent"]};
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(139, 92, 246, 0.5);
        }}
        /* 输入框样式 */
        .stTextArea > div > textarea {{
            background-color: {current_color["card"]};
            color: #E0E0E0;
            border-radius: 10px;
            border: 1px solid #444;
            padding: 12px;
            transition: border-color 0.3s ease;
            width: 100% !important;
        }}
        .stTextArea > div > textarea:focus {{
            border-color: {current_color["accent"]};
            outline: none;
            box-shadow: 0 0 0 2px rgba(168, 85, 247, 0.2);
        }}
        /* 文件上传器 */
        .stFileUploader > div > div {{
            background-color: {current_color["card"]};
            border-radius: 10px;
            border: 1px dashed #555;
            padding: 32px;
            transition: border-color 0.3s ease;
        }}
        .stFileUploader > div > div:hover {{
            border-color: {current_color["accent"]};
        }}
        /* 标题样式 */
        .page-title {{
            color: {current_color["accent"]};
            font-weight: 600;
            margin-bottom: 8px;
        }}
        /* 提示文字 */
        .hint-text {{
            color: #999;
            font-size: 14px;
            margin-top: 8px;
        }}
        /* 视频预览容器 */
        .video-container {{
            border-radius: 10px;
            overflow: hidden;
            border: 1px solid #444;
        }}
    </style>
""", unsafe_allow_html=True)

# ---------------------- 3. 核心工具函数 ----------------------
def video_to_keyframes(video_file):
    # 保存临时视频
    temp_video_path = "temp_video.mp4"
    with open(temp_video_path, "wb") as f:
        f.write(video_file.getbuffer())
    
    cap = cv2.VideoCapture(temp_video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = round(total_frames / fps, 1)  # 视频时长（秒）
    keyframes = []
    frame_interval = fps  # 每秒1帧
    
    # 提取关键帧
    with st.spinner(f"📹 提取关键帧（共{total_frames}帧，时长{duration}秒）..."):
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
    return keyframes, fps, duration

def image_to_base64(image):
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format="JPEG")
    return base64.b64encode(img_byte_arr.getvalue()).decode("utf-8")

def analyze_video(video_file):
    # 提取关键帧
    keyframes, fps, duration = video_to_keyframes(video_file)
    if len(keyframes) == 0:
        return "❌ 视频帧提取失败，请更换视频文件重试（建议MP4格式，时长≤30秒）"
    
    # 关键帧转Base64（最多取10帧）
    base64_frames = [image_to_base64(frame) for frame in keyframes[:10]]
    
    # 调用API分析
    url = "https://api-inference.modelscope.cn/v1/chat/completions"
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"""以下是视频的{len(base64_frames)}个关键帧（每秒1帧，总时长{duration}秒），分析后输出结构化结果：
1. 核心主体：贯穿始终的人物/物体
2. 画面风格：艺术风格+色彩基调
3. 运镜手法：运镜类型+移动速度
4. 分镜头：镜头切换点+每个镜头时长
5. 场景转换：场景类型+转换方式
分点清晰呈现，简洁明了"""}
            ]
        }
    ]
    # 添加所有关键帧图片
    for frame in base64_frames:
        messages[0]["content"].append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frame}"}})
    
    payload = {
        "model": "Qwen/Qwen2.5-VL-72B-Instruct",
        "messages": messages,
        "max_tokens": 800,
        "temperature": 0.6
    }
    response = requests.post(url, headers=headers, json=payload, timeout=90)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]

# ---------------------- 4. 页面核心逻辑（修复text_area参数）----------------------
def main():
    # 页面标题
    st.markdown(f"<h1 class='page-title'>🎬 视频全维度分析</h1>", unsafe_allow_html=True)
    st.markdown("<p class='hint-text'>支持MP4/AVI/MKV格式，单文件≤200MB，建议时长≤30秒（分析约10-20秒）</p>", unsafe_allow_html=True)

    # 1. 视频上传+预览区域
    with st.container():
        st.markdown('<div class="func-card">', unsafe_allow_html=True)
        col1, col2 = st.columns([2, 1])
        
        with col1:
            uploaded_video = st.file_uploader(
                "上传视频",
                type=["mp4", "avi", "mkv"],
                key="video_upload",
                label_visibility="collapsed"
            )
            analyze_btn = st.button("🎯 开始视频分析", type="primary", use_container_width=True)
        
        # 视频信息+预览（右侧）
        with col2:
            if uploaded_video:
                video_size = round(uploaded_video.size / 1024 / 1024, 2)
                st.markdown(f"📊 视频信息：\n- 文件名：{uploaded_video.name}\n- 大小：{video_size}MB")
                # 视频预览
                st.markdown('<div class="video-container">', unsafe_allow_html=True)
                st.video(uploaded_video, format="video/mp4")
                st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # 2. 结果展示区域（移除use_container_width参数）
    with st.container():
        st.markdown('<div class="func-card">', unsafe_allow_html=True)
        st.subheader("📝 分析结果")
        
        # 初始化结果文本框
        result_placeholder = st.empty()
        with result_placeholder.container():
            st.text_area(
                "分析结果将显示在这里（可直接复制）",
                height=350,
                key="video_result",
                placeholder="点击上方按钮开始分析..."
            )

        # 分析逻辑执行
        if analyze_btn and uploaded_video:
            try:
                with st.spinner("🔍 正在分析视频内容...（关键帧提取+AI分析）"):
                    result = analyze_video(uploaded_video)
                    # 更新结果文本框（移除use_container_width参数）
                    with result_placeholder.container():
                        st.text_area(
                            "✅ 分析完成",
                            value=result,
                            height=350,
                            key="video_result_active"
                        )
            except Exception as e:
                st.error(f"❌ 分析失败：{str(e)}", icon="⚠️")
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
