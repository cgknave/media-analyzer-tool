import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import base64
import requests
from zipfile import ZipFile
import tempfile

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

# ---------------------- 2. 界面样式 ----------------------
st.markdown(f"""
    <style>
        .stApp {{
            background-color: {current_color["bg"]};
            color: #E0E0E0;
            font-family: 'Segoe UI', Roboto, sans-serif;
        }}
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
        .tool-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 16px;
            margin: 24px 0;
        }}
        .tool-card {{
            background-color: rgba(255,255,255,0.05);
            border-radius: 12px;
            padding: 20px;
            border: 1px solid #444;
            transition: all 0.3s ease;
        }}
        .tool-card:hover {{
            border-color: {current_color["accent"]};
            transform: translateY(-5px);
            box-shadow: 0 8px 16px rgba(0,0,0,0.3);
        }}
        .tool-icon {{
            font-size: 24px;
            margin-bottom: 12px;
            color: {current_color["accent"]};
        }}
        .page-title {{
            color: {current_color["accent"]};
            font-weight: 600;
            margin-bottom: 8px;
        }}
        .hint-text {{
            color: #999;
            font-size: 14px;
            margin-top: 8px;
        }}
        .color-block {{
            width: 100%;
            height: 40px;
            border-radius: 6px;
            margin: 8px 0;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: 500;
            font-size: 12px;
            text-shadow: 0 1px 2px rgba(0,0,0,0.3);
        }}
    </style>
""", unsafe_allow_html=True)

# ---------------------- 3. 设计师专属工具函数 ----------------------
# 3.1 智能抠图（主体分离）
def remove_background(image):
    img_base64 = image_to_base64(image)
    url = "https://api-inference.modelscope.cn/v1/image/segmentation"
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    payload = {
        "image": img_base64,
        "parameters": {"model": "vitmatte-image"}
    }
    response = requests.post(url, headers=headers, json=payload, timeout=60)
    response.raise_for_status()
    
    # 解析抠图结果
    result_data = base64.b64decode(response.json()["output"]["mask"])
    mask = Image.open(io.BytesIO(result_data)).convert("L")
    
    # 应用遮罩
    img_rgba = image.convert("RGBA")
    mask_array = np.array(mask)
    alpha_channel = np.where(mask_array > 128, 255, 0).astype(np.uint8)
    img_rgba.putalpha(Image.fromarray(alpha_channel))
    
    return img_rgba

# 3.2 图片转Base64
def image_to_base64(image):
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format="PNG")
    return base64.b64encode(img_byte_arr.getvalue()).decode("utf-8")

# 3.3 批量配色提取
def batch_extract_colors(images):
    color_results = []
    for idx, img in enumerate(images):
        # 简化配色提取（批量优化速度）
        img_small = img.resize((50, 50))
        img_array = np.array(img_small).reshape(-1, 3)
        unique_colors = np.unique(img_array, axis=0)
        
        # 取前5个主要颜色
        main_colors = unique_colors[:5] if len(unique_colors)>=5 else unique_colors
        
        # 转换为HEX
        def rgb_to_hex(rgb):
            return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"
        
        color_results.append({
            "image_idx": idx+1,
            "colors": [{"rgb": c, "hex": rgb_to_hex(c)} for c in main_colors]
        })
    return color_results

# 3.4 设计规范生成
def generate_design_spec(colors, font_info, layout_info):
    spec = f"""# 设计规范文档
## 1. 配色方案
"""
    if colors:
        spec += "### 主色\n"
        spec += f"- HEX: {colors['main']['hex']}\n"
        spec += f"- RGB: {colors['main']['rgb']}\n"
        spec += f"- CMYK: {colors['main']['cmyk'][0]}%,{colors['main']['cmyk'][1]}%,{colors['main']['cmyk'][2]}%,{colors['main']['cmyk'][3]}%\n\n"
        
        spec += "### 辅助色\n"
        for i, color in enumerate(colors['secondary']):
            spec += f"- 辅助色{i+1}：HEX {color['hex']} | RGB {color['rgb']}\n"
        
        if colors['neutral']:
            spec += "\n### 中性色\n"
            for i, color in enumerate(colors['neutral']):
                spec += f"- 中性色{i+1}：HEX {color['hex']} | RGB {color['rgb']}\n"
    
    spec += f"""
## 2. 字体规范
{font_info if font_info else '未识别到字体信息'}

## 3. 布局规范
{layout_info if layout_info else '未获取到布局信息'}

## 4. 应用建议
- 主色用于品牌标识、重点按钮等核心元素
- 辅助色用于强调信息、区分模块
- 中性色用于背景、文字等次要元素
- 字体建议保持统一，字重变化控制在2-3种以内
- 布局遵循视觉层级，重要信息优先展示
"""
    return spec

# 3.5 相似风格推荐
def recommend_similar_style(image):
    img_base64 = image_to_base64(image)
    url = "https://api-inference.modelscope.cn/v1/chat/completions"
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": "Qwen/Qwen2.5-VL-72B-Instruct",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": """作为平面设计师，根据图片风格，推荐：
1. 3个相似风格的设计参考方向
2. 2个可获取同类素材的网站（国内可用）
3. 1个设计软件中的相关预设/插件
简洁明了，突出实用性"""},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}}
                ]
            }
        ],
        "max_tokens": 300,
        "temperature": 0.6
    }
    response = requests.post(url, headers=headers, json=payload, timeout=60)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]

# 3.6 配色提取（复用图片解析中的函数）
def extract_colors(image, n_colors=5):
    img = image.resize((100, 100))
    img_array = np.array(img).reshape(-1, 3)
    
    from sklearn.cluster import KMeans
    from collections import Counter
    kmeans = KMeans(n_clusters=n_colors, random_state=42)
    kmeans.fit(img_array)
    colors = kmeans.cluster_centers_.astype(int)
    labels = kmeans.labels_
    
    color_counts = Counter(labels)
    sorted_colors = [colors[i] for i in color_counts.most_common(n_colors)]
    
    main_color = sorted_colors[0]
    secondary_colors = sorted_colors[1:4] if n_colors >=4 else sorted_colors[1:]
    
    neutral_colors = []
    for color in sorted_colors:
        brightness = (color[0] * 0.299 + color[1] * 0.587 + color[2] * 0.114)
        if brightness < 50 or brightness > 200:
            neutral_colors.append(color)
    
    def rgb_to_hex(rgb):
        return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"
    
    def rgb_to_cmyk(rgb):
        r, g, b = rgb[0]/255, rgb[1]/255, rgb[2]/255
        k = 1 - max(r, g, b)
        if k == 1:
            return (0, 0, 0, 100)
        c = (1 - r - k) / (1 - k)
        m = (1 - g - k) / (1 - k)
        y = (1 - b - k) / (1 - k)
        return (round(c*100), round(m*100), round(y*100), round(k*100))
    
    return {
        "main": {"rgb": main_color, "hex": rgb_to_hex(main_color), "cmyk": rgb_to_cmyk(main_color)},
        "secondary": [{"rgb": c, "hex": rgb_to_hex(c), "cmyk": rgb_to_cmyk(c)} for c in secondary_colors],
        "neutral": [{"rgb": c, "hex": rgb_to_hex(c), "cmyk": rgb_to_cmyk(c)} for c in neutral_colors]
    }

# ---------------------- 4. 页面核心逻辑 ----------------------
def main():
    # 页面标题
    st.markdown(f"<h1 class='page-title'>🎨 设计师专属工具集</h1>", unsafe_allow_html=True)
    st.markdown("<p class='hint-text'>平面设计师高效工作工具箱，包含抠图、配色、规范生成等核心功能</p>", unsafe_allow_html=True)

    # 工具分类标签页
    tab1, tab2, tab3, tab4 = st.tabs([
        "🖼️ 图片处理", 
        "🎨 配色工具", 
        "📋 设计规范", 
        "🔍 风格参考"
    ])

    # 标签页1：图片处理（抠图+批量处理）
    with tab1:
        st.markdown('<div class="func-card">', unsafe_allow_html=True)
        st.subheader("智能抠图
