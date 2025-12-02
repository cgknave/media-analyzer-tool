import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageDraw
import io
import base64
import requests
from zipfile import ZipFile
from collections import Counter
from sklearn.cluster import KMeans
import webcolors
import colorsys  # 标准库，直接导入无需安装
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.units import inch

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
        .skill-card {{
            background-color: rgba(255,255,255,0.03);
            border-radius: 8px;
            padding: 16px;
            margin: 8px 0;
            border-left: 3px solid {current_color["accent"]};
        }}
    </style>
""", unsafe_allow_html=True)

# ---------------------- 3. 核心工具函数 ----------------------
# 3.1 图片转Base64
def image_to_base64(image):
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format="PNG")
    return base64.b64encode(img_byte_arr.getvalue()).decode("utf-8")

# 3.2 智能抠图（主体分离）
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

# 3.3 配色提取（基础版）
def extract_colors(image, n_colors=5):
    img = image.resize((100, 100))  # 缩小图片提高效率
    img_array = np.array(img).reshape(-1, 3)
    
    # K-means聚类提取主色
    kmeans = KMeans(n_clusters=n_colors, random_state=42)
    kmeans.fit(img_array)
    colors = kmeans.cluster_centers_.astype(int)
    labels = kmeans.labels_
    
    # 计算颜色占比
    color_counts = Counter(labels)
    sorted_colors = [colors[i] for i in color_counts.most_common(n_colors)]
    
    # 分类：主色（占比最高）、辅助色（中间3个）、中性色（最暗/最亮）
    main_color = sorted_colors[0]
    secondary_colors = sorted_colors[1:4] if n_colors >=4 else sorted_colors[1:]
    
    # 判断中性色（亮度接近0或255）
    neutral_colors = []
    for color in sorted_colors:
        brightness = (color[0] * 0.299 + color[1] * 0.587 + color[2] * 0.114)
        if brightness < 50 or brightness > 200:
            neutral_colors.append(color)
    
    # 转换为HEX/RGB/CMYK
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
    
    result = {
        "main": {"rgb": main_color, "hex": rgb_to_hex(main_color), "cmyk": rgb_to_cmyk(main_color)},
        "secondary": [{"rgb": c, "hex": rgb_to_hex(c), "cmyk": rgb_to_cmyk(c)} for c in secondary_colors],
        "neutral": [{"rgb": c, "hex": rgb_to_hex(c), "cmyk": rgb_to_cmyk(c)} for c in neutral_colors]
    }
    return result

# 3.4 批量配色提取
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

# 3.5 生成配色方案（基于配色模式）
def generate_color_scheme(main_rgb, mode):
    r, g, b = main_rgb
    
    # 互补色（色轮对面）
    if mode == "互补色配色":
        comp_r = 255 - r
        comp_g = 255 - g
        comp_b = 255 - b
        return {
            "main": {"rgb": (r, g, b), "hex": f"#{r:02x}{g:02x}{b:02x}"},
            "secondary": [{"rgb": (comp_r, comp_g, comp_b), "hex": f"#{comp_r:02x}{comp_g:02x}{comp_b:02x}"}],
            "neutral": []
        }
    
    # 相似色（色轮相邻）
    elif mode == "相似色配色":
        def adjust_hue(rgb, offset):
            h, s, v = colorsys.rgb_to_hsv(rgb[0]/255, rgb[1]/255, rgb[2]/255)
            h = (h + offset) % 1.0
            nr, ng, nb = colorsys.hsv_to_rgb(h, s, v)
            return (round(nr*255), round(ng*255), round(nb*255))
        
        sim1 = adjust_hue(main_rgb, 0.1)
        sim2 = adjust_hue(main_rgb, -0.1)
        return {
            "main": {"rgb": (r, g, b), "hex": f"#{r:02x}{g:02x}{b:02x}"},
            "secondary": [
                {"rgb": sim1, "hex": f"#{sim1[0]:02x}{sim1[1]:02x}{sim1[2]:02x}"},
                {"rgb": sim2, "hex": f"#{sim2[0]:02x}{sim2[1]:02x}{sim2[2]:02x}"}
            ],
            "neutral": []
        }
    
    # 分割互补色
    elif mode == "分割互补色":
        def adjust_hue(rgb, offset):
            h, s, v = colorsys.rgb_to_hsv(rgb[0]/255, rgb[1]/255, rgb[2]/255)
            h = (h + offset) % 1.0
            nr, ng, nb = colorsys.hsv_to_rgb(h, s, v)
            return (round(nr*255), round(ng*255), round(nb*255))
        
        comp = (255 - r, 255 - g, 255 - b)
        split1 = adjust_hue(comp, 0.08)
        split2 = adjust_hue(comp, -0.08)
        return {
            "main": {"rgb": (r, g, b), "hex": f"#{r:02x}{g:02x}{b:02x}"},
            "secondary": [
                {"rgb": split1, "hex": f"#{split1[0]:02x}{split1[1]:02x}{split1[2]:02x}"},
                {"rgb": split2, "hex": f"#{split2[0]:02x}{split2[1]:02x}{split2[2]:02x}"}
            ],
            "neutral": []
        }
    
    # 三色配色（色轮均匀分布）
    elif mode == "三色配色":
        def adjust_hue(rgb, offset):
            h, s, v = colorsys.rgb_to_hsv(rgb[0]/255, rgb[1]/255, rgb[2]/255)
            h = (h + offset) % 1.0
            nr, ng, nb = colorsys.hsv_to_rgb(h, s, v)
            return (round(nr*255), round(ng*255), round(nb*255))
        
        color1 = adjust_hue(main_rgb, 1/3)
        color2 = adjust_hue(main_rgb, 2/3)
        return {
            "main": {"rgb": (r, g, b), "hex": f"#{r:02x}{g:02x}{b:02x}"},
            "secondary": [
                {"rgb": color1, "hex": f"#{color1[0]:02x}{color1[1]:02x}{color1[2]:02x}"},
                {"rgb": color2, "hex": f"#{color2[0]:02x}{color2[1]:02x}{color2[2]:02x}"}
            ],
            "neutral": []
        }
    
    # 自动识别（默认）
    else:
        base_colors = extract_colors(Image.fromarray(np.uint8([[main_rgb]])))
        return base_colors

# 3.6 生成配色色卡
def generate_color_card(scheme):
    card_width = 800
    card_height = 500
    card = Image.new("RGB", (card_width, card_height), color="#f5f5f5")
    draw = ImageDraw.Draw(card)
    
    # 绘制主色区域
    main_height = 200
    draw.rectangle([0, 0, card_width, main_height], fill=scheme["main"]["hex"])
    draw.text((20, 20), f"主色: {scheme['main']['hex']}", fill="white", font_size=24)
    draw.text((20, 60), f"RGB: {scheme['main']['rgb']}", fill="white", font_size=18)
    
    # 绘制辅助色区域
    sec_height = 150
    sec_width = card_width // len(scheme["secondary"])
    for i, color in enumerate(scheme["secondary"]):
        x1 = i * sec_width
        x2 = (i+1) * sec_width
        draw.rectangle([x1, main_height, x2, main_height + sec_height], fill=color["hex"])
        draw.text((x1 + 10, main_height + 20), f"辅助色{i+1}: {color['hex']}", fill="white", font_size=16)
    
    # 绘制中性色区域
    neu_height = 150
    if scheme["neutral"]:
        neu_width = card_width // len(scheme["neutral"])
        for i, color in enumerate(scheme["neutral"]):
            x1 = i * neu_width
            x2 = (i+1) * neu_width
            text_color = "white" if sum(color["rgb"])
