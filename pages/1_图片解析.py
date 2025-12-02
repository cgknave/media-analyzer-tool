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
    </style>
""", unsafe_allow_html=True)

# ---------------------- 3. 核心工具函数 ----------------------
def image_to_base64(image):
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format="JPEG")
    return base64.b64encode(img_byte_arr.getvalue()).decode("utf-8")

def analyze_image(image):
    img_base64 = image_to_base64(image)
    url = "https://api-inference.modelscope.cn/v1/chat/completions"
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": "Qwen/Qwen2.5-VL-72B-Instruct",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": """详细分析图片，输出结构化结果：
1. 核心主体：人物/物体/动作
2. 纹理材质：表面质感+材质类型
3. 光影细节：光影类型+光源方向+明暗对比
4. 色彩氛围：主色调+色彩数值+色调类型
5. 场景背景：场景类型+背景层级
6. 构图视角：构图规则+视角
分点呈现，简洁明了"""},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}}
                ]
            }
        ],
        "max_tokens": 600,
        "temperature": 0.6
    }
    response = requests.post(url, headers=headers, json=payload, timeout=60)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]

# ---------------------- 4. 页面核心逻辑（修复text_area参数）----------------------
def main():
    # 页面标题
    st.markdown(f"<h1 class='page-title'>📷 图片细化分析</h1>", unsafe_allow_html=True)
    st.markdown("<p class='hint-text'>支持JPG/PNG/WebP格式，单文件≤200MB，分析约3-5秒</p>", unsafe_allow_html=True)

    # 1. 图片上传区域
    with st.container():
        st.markdown('<div class="func-card">', unsafe_allow_html=True)
        col1, col2 = st.columns([2, 1])
        
        with col1:
            uploaded_img = st.file_uploader(
                "上传图片",
                type=["jpg", "jpeg", "png", "webp"],
                key="img_upload",
                label_visibility="collapsed"
            )
            analyze_btn = st.button("🚀 开始图片分析", type="primary", use_container_width=True)
        
        # 图片预览（右侧）
        with col2:
            if uploaded_img:
                img = Image.open(uploaded_img).convert("RGB")
                st.image(
                    img, 
                    caption="预览图", 
                    use_container_width=True, 
                    clamp=True
                )
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
                key="img_result",
                placeholder="点击上方按钮开始分析..."
            )

        # 分析逻辑执行
        if analyze_btn and uploaded_img:
            try:
                with st.spinner("🔍 正在分析图片细节..."):
                    img = Image.open(uploaded_img).convert("RGB")
                    result = analyze_image(img)
                    # 更新结果文本框（移除use_container_width参数）
                    with result_placeholder.container():
                        st.text_area(
                            "✅ 分析完成",
                            value=result,
                            height=350,
                            key="img_result_active"
                        )
            except Exception as e:
                st.error(f"❌ 分析失败：{str(e)}", icon="⚠️")
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
