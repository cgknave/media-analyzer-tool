import cv2
import numpy as np
import base64
import requests
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

# ---------------------- 2. 界面样式（仅核心功能框，同步颜色）----------------------
st.markdown(f"""
    <style>
        .stApp {{background-color: {current_color["bg"]}; color: #E0E0E0;}}
        /* 功能卡片（仅保留核心功能框） */
        .func-card {{
            background-color: {current_color["card"]};
            border-radius: 20px;
            padding: 20px;
            margin: 10px 0;
            border: 1px solid #333;
        }}
        /* 按钮样式 */
        .stButton > button {{
            background-color: {current_color["btn"]};
            color: white;
            border-radius: 10px;
            padding: 8px 16px;
            border: none;
        }}
        .stButton > button:hover {{background-color: {current_color["accent"]};}}
        /* 结果文本框 */
        .stTextArea > div > textarea {{
            background-color: {current_color["card"]};
            color: #E0E0E0;
            border-radius: 10px;
            border: 1px solid #444;
        }}
        /* 上传组件 */
        .stFileUploader > div > div {{
            background-color: {current_color["card"]};
            border-radius: 10px;
            border: 1px dashed #444;
        }}
    </style>
""", unsafe_allow_html=True)

# ---------------------- 3. 核心工具函数（图片分析）----------------------
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

# ---------------------- 4. 核心功能布局（仅保留上传+分析+结果）----------------------
st.title("📷 图片细化分析")

# 1. 图片上传+分析按钮（功能卡片）
with st.container():
    st.markdown('<div class="func-card">', unsafe_allow_html=True)
    uploaded_img = st.file_uploader("上传图片（JPG/PNG/WebP，≤200MB）", type=["jpg", "jpeg", "png", "webp"])
    if uploaded_img:
        img = Image.open(uploaded_img).convert("RGB")
        st.image(img, caption="图片预览", use_container_width=True, clamp=True, width=300)
    analyze_btn = st.button("🚀 开始图片分析", type="primary")
    st.markdown('</div>', unsafe_allow_html=True)

# 2. 结果展示框（功能卡片）
with st.container():
    st.markdown('<div class="func-card">', unsafe_allow_html=True)
    st.subheader("📝 分析结果")
    result_box = st.text_area("分析结果将显示在这里（可直接复制）", height=300, disabled=True, key="img_result")
    
    if analyze_btn and uploaded_img:
        try:
            with st.spinner("分析中...（约3-5秒）"):
                result = analyze_image(img)
                st.text_area("✅ 分析完成", value=result, height=300, key="img_result_active")
        except Exception as e:
            st.error(f"分析失败：{str(e)}")
    st.markdown('</div>', unsafe_allow_html=True)
