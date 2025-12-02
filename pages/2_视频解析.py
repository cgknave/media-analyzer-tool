import cv2
import numpy as np
import base64
import requests
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
        /* 关键帧预览 */
        .keyframes-container {{
            display: flex;
            gap: 8px;
            overflow-x: auto;
            padding: 8px 0;
            margin: 16px 0;
        }}
        .keyframe-item {{
            min-width: 120px;
            border-radius: 8px;
            overflow: hidden;
            border: 2px solid transparent;
            transition: all 0.3s ease;
        }}
        .keyframe-item:hover {{
            border-color: {current_color["accent"]};
            transform: scale(1.05);
        }}
    </style>
""", unsafe_allow_html=True)

# ---------------------- 3. 核心工具函数（优化设计师参考价值）----------------------
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
    frame_interval = max(1, fps // 2)  # 每0.5秒1帧（更密集，便于设计参考）
    
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
                frame_pil.thumbnail((320, 180))  # 缩小预览图
                keyframes.append(frame_pil)
            frame_idx += 1
            progress_bar.progress(min(frame_idx / total_frames, 1.0))
    
    cap.release()
    return keyframes, fps, duration

def image_to_base64(image):
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format="JPEG")
    return base64.b64encode(img_byte_arr.getvalue()).decode("utf-8")

def analyze_video_design(video_file):
    # 提取关键帧
    keyframes, fps, duration = video_to_keyframes(video_file)
    if len(keyframes) == 0:
        return "❌ 视频帧提取失败，请更换视频文件重试（建议MP4格式，时长≤30秒）"
    
    # 关键帧转Base64（最多取15帧，更全面）
    base64_frames = [image_to_base64(frame) for frame in keyframes[:15]]
    
    # 调用API分析（突出设计参考价值）
    url = "https://api-inference.modelscope.cn/v1/chat/completions"
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"""作为平面设计师的视频参考工具，分析以下{len(base64_frames)}个关键帧（每秒2帧，总时长{duration}秒），输出结构化设计参考：
1. 视觉风格：整体设计风格（如极简/复古/国潮）+ 风格统一逻辑
2. 色彩系统：主色调+色彩变化规律（便于动态设计参考）
3. 构图技巧：镜头构图规则+视角变化（分镜头设计参考）
4. 元素设计：核心视觉元素+元素运动规律（动态元素参考）
5. 光影运用：布光方式+光影变化（动态光影参考）
6. 设计借鉴：适合应用的设计场景+可复用的设计技巧
分点清晰，突出设计参考价值，简洁实用"""}
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

# 导出关键帧为图片包
def export_keyframes(keyframes):
    # 创建ZIP文件
    import zipfile
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for idx, frame in enumerate(keyframes):
            img_byte_arr = io.BytesIO()
            frame.save(img_byte_arr, format="PNG")
            zip_file.writestr(f"关键帧_{idx+1}.png", img_byte_arr.getvalue())
    zip_buffer.seek(0)
    return zip_buffer

# ---------------------- 4. 页面核心逻辑（新增设计师友好功能）----------------------
def main():
    # 页面标题
    st.markdown(f"<h1 class='page-title'>🎬 视频设计参考工具</h1>", unsafe_allow_html=True)
    st.markdown("<p class='hint-text'>支持MP4/AVI/MKV格式，单文件≤200MB，提取动态设计参考（适合短视频/动态海报设计）</p>", unsafe_allow_html=True)

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
            st.markdown("<div class='btn-group'>", unsafe_allow_html=True)
            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                analyze_btn = st.button("📊 设计分析", type="primary", use_container_width=True)
            with col_btn2:
                export_frames_btn = st.button("📥 导出关键帧", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
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

    # 2. 关键帧预览区域
    if uploaded_video:
        with st.container():
            st.markdown('<div class="func-card">', unsafe_allow_html=True)
            st.subheader("🎞️ 关键帧预览（设计参考用）")
            
            # 提取关键帧（缓存避免重复计算）
            if "keyframes" not in st.session_state or st.session_state.get("video_name") != uploaded_video.name:
                keyframes, fps, duration = video_to_keyframes(uploaded_video)
                st.session_state.keyframes = keyframes
                st.session_state.fps = fps
                st.session_state.duration = duration
                st.session_state.video_name = uploaded_video.name
            else:
                keyframes = st.session_state.keyframes
                fps = st.session_state.fps
                duration = st.session_state.duration
            
            # 横向滚动显示关键帧
            st.markdown('<div class="keyframes-container">', unsafe_allow_html=True)
            for idx, frame in enumerate(keyframes[:20]):  # 最多显示20帧
                st.markdown(f'<div class="keyframe-item">', unsafe_allow_html=True)
                st.image(frame, caption=f"帧{idx+1}", use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown(f"📝 关键帧信息：共{len(keyframes)}帧 | 帧率：{fps}fps | 时长：{duration}秒")
            
            # 导出关键帧按钮功能
            if export_frames_btn:
                try:
                    with st.spinner("📥 正在打包关键帧..."):
                        zip_data = export_keyframes(keyframes)
                        st.download_button(
                            label="✅ 下载关键帧包（ZIP）",
                            data=zip_data,
                            file_name="视频关键帧.zip",
                            mime="application/zip",
                            use_container_width=True
                        )
                except Exception as e:
                    st.error(f"❌ 导出失败：{str(e)}", icon="⚠️")
            st.markdown('</div>', unsafe_allow_html=True)

    # 3. 结果展示区域
    with st.container():
        st.markdown('<div class="func-card">', unsafe_allow_html=True)
        st.subheader("📝 设计参考分析结果")
        
        # 初始化结果文本框
        result_placeholder = st.empty()
        with result_placeholder.container():
            st.text_area(
                "分析结果将显示在这里（包含风格/色彩/构图/元素参考）",
                height=350,
                key="video_result",
                placeholder="点击「设计分析」按钮开始..."
            )

        # 分析逻辑执行
        if analyze_btn and uploaded_video:
            try:
                with st.spinner("🔍 正在分析视频设计元素...（关键帧提取+AI分析）"):
                    result = analyze_video_design(uploaded_video)
                    # 更新结果文本框
                    with result_placeholder.container():
                        st.text_area(
                            "✅ 设计分析完成（可直接复制参考）",
                            height=350,
                            key="video_result_active",
                            value=result
                        )
            except Exception as e:
                st.error(f"❌ 分析失败：{str(e)}", icon="⚠️")
        
        # 导出分析报告
        if st.session_state.get("video_result_active"):
            st.download_button(
                label="📥 导出分析报告（TXT）",
                data=st.session_state.get("video_result_active", ""),
                file_name="视频设计分析报告.txt",
                mime="text/plain",
                use_container_width=True
            )
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
