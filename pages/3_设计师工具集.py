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
import colorsys
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
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
            text_color = "white" if sum(color["rgb"]) < 382 else "black"  # 根据亮度选择文字颜色
            draw.rectangle([x1, main_height + sec_height, x2, main_height + sec_height + neu_height], fill=color["hex"])
            draw.text((x1 + 10, main_height + sec_height + 20), f"中性色{i+1}: {color['hex']}", fill=text_color, font_size=16)
    
    img_byte_arr = io.BytesIO()
    card.save(img_byte_arr, format="PNG")
    return img_byte_arr.getvalue()

# 3.7 设计规范生成
def generate_design_spec(colors, font_info, layout_info):
    spec = f"""# 设计规范文档
## 1. 配色方案
"""
    if colors:
        spec += "### 主色\n"
        spec += f"- HEX: {colors['main']['hex']}\n"
        spec += f"- RGB: {colors['main']['rgb']}\n"
        if "cmyk" in colors["main"]:
            spec += f"- CMYK: {colors['main']['cmyk'][0]}%,{colors['main']['cmyk'][1]}%,{colors['main']['cmyk'][2]}%,{colors['main']['cmyk'][3]}%\n"
        spec += "- 应用场景：品牌标识、重点按钮、核心视觉元素\n\n"
        
        spec += "### 辅助色\n"
        for i, color in enumerate(colors['secondary']):
            spec += f"#### 辅助色{i+1}\n"
            spec += f"- HEX: {color['hex']}\n"
            spec += f"- RGB: {color['rgb']}\n"
            if "cmyk" in color:
                spec += f"- CMYK: {color['cmyk'][0]}%,{color['cmyk'][1]}%,{color['cmyk'][2]}%,{color['cmyk'][3]}%\n"
            spec += "- 应用场景：强调信息、区分模块、辅助图形\n\n"
        
        if colors['neutral']:
            spec += "### 中性色\n"
            for i, color in enumerate(colors['neutral']):
                spec += f"#### 中性色{i+1}\n"
                spec += f"- HEX: {color['hex']}\n"
                spec += f"- RGB: {color['rgb']}\n"
                if "cmyk" in color:
                    spec += f"- CMYK: {color['cmyk'][0]}%,{color['cmyk'][1]}%,{color['cmyk'][2]}%,{color['cmyk'][3]}%\n"
                spec += "- 应用场景：背景、文字、次要元素\n\n"
    
    spec += f"""## 2. 字体规范
{font_info if font_info else '### 字体建议'}
- 中文主字体：思源黑体、微软雅黑、苹方（通用性强）
- 英文主字体：Roboto、Montserrat、Open Sans（无衬线字体）
- 字体总数：不超过2-3种（避免杂乱）
- 正文字号：印刷品≥9pt，网页≥14px，移动端≥16px
- 标题字号：正文的1.5-2倍（保持层级）

## 3. 布局规范
{layout_info if layout_info else '### 布局基本原则'}
- 对齐方式：统一左对齐/居中对齐/右对齐（避免混合对齐）
- 间距规范：行间距1.5-1.8倍，模块间距统一
- 视觉层级：重要信息放大/加粗/高饱和，次要信息缩小/常规/低饱和
- 留白原则：适当留白（画面呼吸感，突出主体）

## 4. 应用建议
- 保持设计一致性：配色、字体、间距在整套设计中统一
- 适配不同场景：印刷品用CMYK，电子屏用RGB
- 考虑无障碍：文字与背景明度差≥3:1（提高可读性）
- 导出规范：印刷品300dpi，电子屏72dpi，透明背景用PNG格式
"""
    return spec

# 3.8 生成PDF格式设计规范
def generate_spec_pdf(spec_content):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []
    
    # 标题
    title_style = styles["Heading1"]
    title_style.alignment = 1  # 居中
    title_style.textColor = (0.1, 0.1, 0.1)
    story.append(Paragraph("设计规范文档", title_style))
    story.append(Spacer(1, 0.2*inch))
    
    # 正文样式
    body_style = styles["BodyText"]
    body_style.fontSize = 11
    body_style.textColor = (0.2, 0.2, 0.2)
    body_style.spaceAfter = 12
    
    # 解析内容并添加到PDF
    for line in spec_content.split('\n'):
        line = line.strip()
        if not line:
            story.append(Spacer(1, 0.1*inch))
            continue
        
        if line.startswith('# '):
            # 一级标题
            h1_style = styles["Heading2"]
            h1_style.textColor = (0.1, 0.1, 0.1)
            story.append(Paragraph(line.lstrip('# '), h1_style))
        elif line.startswith('## '):
            # 二级标题
            h2_style = styles["Heading3"]
            h2_style.textColor = (0.2, 0.2, 0.2)
            story.append(Paragraph(line.lstrip('## '), h2_style))
        elif line.startswith('### '):
            # 三级标题
            h3_style = styles["Heading4"]
            h3_style.textColor = (0.3, 0.3, 0.3)
            story.append(Paragraph(line.lstrip('### '), h3_style))
        elif line.startswith('- '):
            # 列表项
            story.append(Paragraph(f"• {line.lstrip('- ')}", body_style))
        else:
            # 普通文本
            story.append(Paragraph(line, body_style))
    
    doc.build(story)
    buffer.seek(0)
    return buffer

# 3.9 相似风格推荐
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
                    {"type": "text", "text": """作为专业平面设计师，根据图片风格，推荐以下内容（简洁实用）：
1. 3个相似风格的设计参考方向（具体可落地）
2. 2个国内可用的同类素材网站（注明网站特点）
3. 1个设计软件中的相关预设/插件推荐
格式清晰，分点列出，突出实用性"""},
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

# 3.10 识别字体规范
def recognize_font_style(image):
    img_base64 = image_to_base64(image)
    url = "https://api-inference.modelscope.cn/v1/chat/completions"
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": "Qwen/Qwen2.5-VL-72B-Instruct",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "识别图片中文字的以下信息，输出简洁规范：\n1. 字体名称（中文+英文，如适用）\n2. 字重（常规/粗体/黑体等）\n3. 字号估算（px）\n4. 字间距/行间距特点\n5. 字体搭配建议"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}}
                ]
            }
        ],
        "max_tokens": 200,
        "temperature": 0.5
    }
    response = requests.post(url, headers=headers, json=payload, timeout=60)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]

# 3.11 识别布局规范
def recognize_layout_style(image):
    img_base64 = image_to_base64(image)
    url = "https://api-inference.modelscope.cn/v1/chat/completions"
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": "Qwen/Qwen2.5-VL-72B-Instruct",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "分析图片的布局规范，输出简洁实用的参考：\n1. 对齐方式（左对齐/居中/右对齐/混合）\n2. 间距比例（行间距/模块间距特点）\n3. 视觉层级（信息优先级划分）\n4. 网格系统（是否使用网格，网格特点）\n5. 布局适用场景"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}}
                ]
            }
        ],
        "max_tokens": 300,
        "temperature": 0.5
    }
    response = requests.post(url, headers=headers, json=payload, timeout=60)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]

# ---------------------- 4. 页面核心逻辑 ----------------------
def main():
    # 页面标题
    st.markdown(f"<h1 class='page-title'>🎨 设计师专属工具集</h1>", unsafe_allow_html=True)
    st.markdown("<p class='hint-text'>平面设计师高效工作工具箱，覆盖抠图、配色、规范生成、风格参考全流程</p>", unsafe_allow_html=True)

    # 工具分类标签页
    tab1, tab2, tab3, tab4 = st.tabs([
        "🖼️ 图片处理", 
        "🎨 配色工具", 
        "📋 设计规范", 
        "🔍 风格参考"
    ])

    # ---------------------- 标签页1：图片处理（抠图+批量处理）----------------------
    with tab1:
        # 智能抠图功能
        st.markdown('<div class="func-card">', unsafe_allow_html=True)
        st.subheader("✂️ 智能抠图（主体分离）")
        st.markdown("<p class='hint-text'>自动分离图片主体与背景，导出透明底PNG，适合LOGO/产品图处理</p>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 上传图片")
            upload_img = st.file_uploader(
                "选择需要抠图的图片（支持JPG/PNG/WebP）",
                type=["jpg", "jpeg", "png", "webp"],
                key="remove_bg_upload",
                label_visibility="collapsed"
            )
            remove_bg_btn = st.button("🚀 开始智能抠图", use_container_width=True, type="primary")
        
        with col2:
            st.markdown("#### 抠图结果预览")
            result_placeholder = st.empty()
            
            if upload_img and remove_bg_btn:
                try:
                    with st.spinner("✂️ 正在处理...（保留主体，去除背景）"):
                        img = Image.open(upload_img).convert("RGB")
                        result_img = remove_background(img)
                        
                        # 显示结果
                        with result_placeholder.container():
                            st.image(result_img, caption="抠图完成（透明底）", use_container_width=True)
                            
                            # 保存为PNG并提供下载
                            img_byte_arr = io.BytesIO()
                            result_img.save(img_byte_arr, format="PNG")
                            img_byte_arr.seek(0)
                            
                            st.download_button(
                                label="📥 下载透明底图片",
                                data=img_byte_arr,
                                file_name="抠图结果_透明底.png",
                                mime="image/png",
                                use_container_width=True
                            )
                except Exception as e:
                    st.error(f"❌ 抠图失败：{str(e)}", icon="⚠️")
            else:
                with result_placeholder.container():
                    st.info("上传图片后点击「开始智能抠图」，支持人物、产品、LOGO等主体分离", icon="ℹ️")
        
        st.markdown('</div>', unsafe_allow_html=True)

        # 批量图片处理
        st.markdown('<div class="func-card">', unsafe_allow_html=True)
        st.subheader("📦 批量图片处理")
        st.markdown("<p class='hint-text'>支持批量上传图片，统一执行配色提取/格式转换/压缩，提高工作效率</p>", unsafe_allow_html=True)
        
        col1, col2 = st.columns([3, 1])
        with col1:
            batch_upload = st.file_uploader(
                "批量上传图片（最多10张）",
                type=["jpg", "jpeg", "png", "webp"],
                key="batch_upload",
                label_visibility="collapsed",
                accept_multiple_files=True
            )
            
            # 显示已上传图片预览
            if batch_upload:
                st.markdown(f"✅ 已上传 {len(batch_upload)} 张图片")
                cols = st.columns(min(5, len(batch_upload)))
                for idx, file in enumerate(batch_upload[:5]):
                    with cols[idx]:
                        img_prev = Image.open(file).convert("RGB").thumbnail((100, 100))
                        st.image(file, caption=f"图片{idx+1}", use_container_width=True)
                if len(batch_upload) > 5:
                    st.markdown(f"... 还有 {len(batch_upload)-5} 张图片")
        
        with col2:
            st.markdown("#### 选择操作类型")
            batch_action = st.selectbox(
                "批量操作",
                options=["批量提取配色", "批量转换为PNG", "批量压缩图片"],
                key="batch_action"
            )
            batch_process_btn = st.button("🚀 开始批量处理", use_container_width=True, type="primary")
        
        # 批量处理结果展示
        if batch_upload and batch_process_btn:
            if len(batch_upload) > 10:
                st.warning("⚠️ 最多支持10张图片批量处理，请减少上传数量后重试", icon="⚠️")
            else:
                try:
                    with st.spinner(f"🚀 正在{batch_action}..."):
                        images = [Image.open(file).convert("RGB") for file in batch_upload]
                        
                        if batch_action == "批量提取配色":
                            # 批量提取配色
                            color_results = batch_extract_colors(images)
                            
                            # 显示结果
                            st.markdown("### 🎨 批量配色提取结果", unsafe_allow_html=True)
                            for result in color_results:
                                file_name = batch_upload[result["image_idx"]-1].name
                                st.markdown(f"#### 图片{result['image_idx']}：{file_name}", unsafe_allow_html=True)
                                
                                cols = st.columns(len(result['colors']))
                                for idx, color in enumerate(result['colors']):
                                    with cols[idx]:
                                        st.markdown(
                                            f'<div class="color-block" style="background-color: {color["hex"]};">{color["hex"]}</div>',
                                            unsafe_allow_html=True
                                        )
                                        st.markdown(f"RGB: {color['rgb']}")
                                st.markdown("---", unsafe_allow_html=True)
                            
                            # 导出所有配色方案
                            def export_batch_colors(color_results):
                                content = "批量配色方案汇总\n"
                                content += "="*50 + "\n"
                                for result in color_results:
                                    file_name = batch_upload[result["image_idx"]-1].name
                                    content += f"\n【图片{result['image_idx']}：{file_name}】\n"
                                    for color in result['colors']:
                                        content += f"- HEX: {color['hex']} | RGB: {color['rgb']}\n"
                                return content.encode("utf-8")
                            
                            color_content = export_batch_colors(color_results)
                            st.download_button(
                                label="📥 导出所有配色方案（TXT）",
                                data=color_content,
                                file_name="批量配色方案汇总.txt",
                                mime="text/plain",
                                use_container_width=True
                            )
                        
                        elif batch_action == "批量转换为PNG":
                            # 批量转换为PNG
                            zip_buffer = io.BytesIO()
                            with ZipFile(zip_buffer, 'w') as zip_file:
                                for idx, (img, file) in enumerate(zip(images, batch_upload)):
                                    img_byte_arr = io.BytesIO()
                                    img.save(img_byte_arr, format="PNG")
                                    filename = f"转换后的图片_{idx+1}_{file.name.split('.')[0]}.png"
                                    zip_file.writestr(filename, img_byte_arr.getvalue())
                            zip_buffer.seek(0)
                            
                            st.success(f"✅ 成功转换{len(images)}张图片为PNG格式", icon="✅")
                            st.download_button(
                                label="📥 下载PNG图片包（ZIP）",
                                data=zip_buffer,
                                file_name="批量PNG转换结果.zip",
                                mime="application/zip",
                                use_container_width=True
                            )
                        
                        elif batch_action == "批量压缩图片":
                            # 批量压缩图片（控制尺寸和质量）
                            zip_buffer = io.BytesIO()
                            with ZipFile(zip_buffer, 'w') as zip_file:
                                for idx, (img, file) in enumerate(zip(images, batch_upload)):
                                    # 限制最大尺寸（1920x1080）
                                    img.thumbnail((1920, 1080))
                                    # 保存为JPG，质量80（平衡质量和体积）
                                    img_byte_arr = io.BytesIO()
                                    img.save(img_byte_arr, format="JPEG", quality=80)
                                    filename = f"压缩后的图片_{idx+1}_{file.name.split('.')[0]}.jpg"
                                    zip_file.writestr(filename, img_byte_arr.getvalue())
                            zip_buffer.seek(0)
                            
                            st.success(f"✅ 成功压缩{len(images)}张图片（保持清晰度，减小体积）", icon="✅")
                            st.download_button(
                                label="📥 下载压缩图片包（ZIP）",
                                data=zip_buffer,
                                file_name="批量压缩图片.zip",
                                mime="application/zip",
                                use_container_width=True
                            )
                except Exception as e:
                    st.error(f"❌ 批量处理失败：{str(e)}", icon="⚠️")
        st.markdown('</div>', unsafe_allow_html=True)

    # ---------------------- 标签页2：配色工具（高级配色+色卡导出）----------------------
    with tab2:
        st.markdown('<div class="func-card">', unsafe_allow_html=True)
        st.subheader("🎨 高级配色方案生成")
        st.markdown("<p class='hint-text'>基于参考图生成专业配色方案，支持多种配色模式，直接导出可用色卡</p>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 上传参考图")
            color_ref_img = st.file_uploader(
                "选择配色参考图片（JPG/PNG/WebP）",
                type=["jpg", "jpeg", "png", "webp"],
                key="color_ref_upload",
                label_visibility="collapsed"
            )
            
            st.markdown("#### 配色模式选择")
            color_mode = st.selectbox(
                "选择专业配色规则",
                options=["自动识别", "互补色配色", "相似色配色", "分割互补色", "三色配色"],
                key="color_mode"
            )
            
            st.markdown("#### 配色参数设置")
            with st.expander("展开参数设置", expanded=False):
                neutral_count = st.slider("中性色数量", min_value=0, max_value=3, value=2, step=1)
                saturation = st.slider("整体饱和度调整", min_value=0.5, max_value=1.5, value=1.0, step=0.1)
            
            generate_color_btn = st.button("🎨 生成配色方案", use_container_width=True, type="primary")
        
        with col2:
            st.markdown("#### 生成的配色方案")
            color_result_placeholder = st.empty()
            
            if color_ref_img and generate_color_btn:
                try:
                    with st.spinner(f"🎨 正在生成{color_mode}方案..."):
                        img = Image.open(color_ref_img).convert("RGB")
                        
                        # 提取参考图主色
                        base_colors = extract_colors(img)
                        main_rgb = base_colors["main"]["rgb"]
                        
                        # 调整饱和度
                        def adjust_saturation(rgb, saturation):
                            h, s, v = colorsys.rgb_to_hsv(rgb[0]/255, rgb[1]/255, rgb[2]/255)
                            s = max(0, min(1, s * saturation))
                            nr, ng, nb = colorsys.hsv_to_rgb(h, s, v)
                            return (round(nr*255), round(ng*255), round(nb*255))
                        
                        main_rgb = adjust_saturation(main_rgb, saturation)
                        
                        # 生成配色方案
                        color_scheme = generate_color_scheme(main_rgb, color_mode)
                        
                        # 添加中性色（从参考图提取）
                        if neutral_count > 0:
                            neutral_colors = base_colors["neutral"][:neutral_count]
                            color_scheme["neutral"] = neutral_colors
                        
                        # 显示配色方案
                        with color_result_placeholder.container():
                            # 主色展示
                            st.markdown("### 主色（核心色）", unsafe_allow_html=True)
                            st.markdown(
                                f'<div class="color-block" style="background-color: {color_scheme["main"]["hex"]}; height: 80px; font-size: 16px;">{color_scheme["main"]["hex"]}</div>',
                                unsafe_allow_html=True
                            )
                            st.markdown(f"RGB: {color_scheme['main']['rgb']}")
                            if "cmyk" in color_scheme["main"]:
                                st.markdown(f"CMYK: {color_scheme['main']['cmyk'][0]}%,{color_scheme['main']['cmyk'][1]}%,{color_scheme['main']['cmyk'][2]}%,{color_scheme['main']['cmyk'][3]}%")
                            
                            # 辅助色展示
                            st.markdown("### 辅助色（搭配色）", unsafe_allow_html=True)
                            if color_scheme["secondary"]:
                                cols = st.columns(len(color_scheme["secondary"]))
                                for idx, color in enumerate(color_scheme["secondary"]):
                                    with cols[idx]:
                                        st.markdown(
                                            f'<div class="color-block" style="background-color: {color["hex"]};">{color["hex"]}</div>',
                                            unsafe_allow_html=True
                                        )
                                        st.markdown(f"RGB: {color['rgb']}")
                            else:
                                st.info("无辅助色，可增加配色模式复杂度", icon="ℹ️")
                            
                            # 中性色展示
                            if color_scheme["neutral"]:
                                st.markdown("### 中性色（背景/文字用）", unsafe_allow_html=True)
                                cols = st.columns(len(color_scheme["neutral"]))
                                for idx, color in enumerate(color_scheme["neutral"]):
                                    with cols[idx]:
                                        st.markdown(
                                            f'<div class="color-block" style="background-color: {color["hex"]};">{color["hex"]}</div>',
                                            unsafe_allow_html=True
                                        )
                                        st.markdown(f"RGB: {color['rgb']}")
                            
                            # 配色应用建议
                            st.markdown("### 📌 配色应用建议", unsafe_allow_html=True)
                            if color_mode == "互补色配色":
                                st.markdown("- 适合需要强对比的设计（海报标题、重点按钮）")
                                st.markdown("- 建议主色占比70%，辅助色30%（避免刺眼）")
                            elif color_mode == "相似色配色":
                                st.markdown("- 适合需要柔和过渡的设计（Banner、插画背景）")
                                st.markdown("- 可通过明度差异增加层次（主色深，辅助色浅）")
                            elif color_mode == "分割互补色":
                                st.markdown("- 适合需要平衡对比的设计（产品详情页、画册）")
                                st.markdown("- 主色+两个辅助色按6:2:2比例分配")
                            elif color_mode == "三色配色":
                                st.markdown("- 适合需要丰富色彩的设计（活动页、儿童产品）")
                                st.markdown("- 保持一种颜色为主，其他两种为辅助")
                            
                            # 生成色卡和规范
                            color_card_data = generate_color_card(color_scheme)
                            
                            # 下载按钮组
                            col_dl1, col_dl2 = st.columns(2)
                            with col_dl1:
                                st.download_button(
                                    label="📥 导出色卡（PNG）",
                                    data=color_card_data,
                                    file_name=f"{color_mode}_配色方案色卡.png",
                                    mime="image/png",
                                    use_container_width=True
                                )
                            
                            # 生成配色规范文本
                            color_spec = f"""# {color_mode}配色方案规范
## 基础信息
- 参考图：{color_ref_img.name}
- 饱和度调整：{saturation}
- 生成时间：{st.session_state.get("current_time", "未知")}

## 主色
- HEX: {color_scheme['main']['hex']}
- RGB: {color_scheme['main']['rgb']}
{'- CMYK: ' + str(color_scheme['main']['cmyk']) + '（百分比）' if 'cmyk' in color_scheme['main'] else ''}
- 应用场景：品牌标识、重点按钮、核心视觉元素
- 占比建议：60-70%

## 辅助色
"""
                            for i, color in enumerate(color_scheme["secondary"]):
                                color_spec += f"""### 辅助色{i+1}
- HEX: {color['hex']}
- RGB: {color['rgb']}
{'- CMYK: ' + str(color['cmyk']) + '（百分比）' if 'cmyk' in color else ''}
- 应用场景：强调信息、区分模块、辅助图形
- 占比建议：15-20%
"""
                            
                            if color_scheme["neutral"]:
                                color_spec += "\n## 中性色\n"
                                for i, color in enumerate(color_scheme["neutral"]):
                                    color_spec += f"""### 中性色{i+1}
- HEX: {color['hex']}
- RGB: {color['rgb']}
{'- CMYK: ' + str(color['cmyk']) + '（百分比）' if 'cmyk' in color else ''}
- 应用场景：背景、文字、次要元素
- 占比建议：10-15%
"""
                            
                            color_spec += f"""
## 配色原则
- 避免高饱和色大面积叠加（视觉疲劳）
- 确保文字与背景明度差≥3:1（提高可读性）
- 整套设计中保持配色一致性
- 印刷品使用CMYK色值，电子屏使用RGB色值
"""
                            
                            with col_dl2:
                                st.download_button(
                                    label="📥 导出配色规范（TXT）",
                                    data=color_spec.encode("utf-8"),
                                    file_name=f"{color_mode}_配色规范.txt",
                                    mime="text/plain",
                                    use_container_width=True
                                )
                except Exception as e:
                    st.error(f"❌ 配色方案生成失败：{str(e)}", icon="⚠️")
            else:
                with color_result_placeholder.container():
                    st.info("上传参考图片，选择配色模式后点击「生成配色方案」", icon="ℹ️")
                    st.markdown("### 配色模式说明", unsafe_allow_html=True)
                    st.markdown("- 自动识别：智能提取参考图原有配色", unsafe_allow_html=True)
                    st.markdown("- 互补色配色：色轮对面颜色（强对比）", unsafe_allow_html=True)
                    st.markdown("- 相似色配色：色轮相邻颜色（柔和过渡）", unsafe_allow_html=True)
                    st.markdown("- 分割互补色：主色+两个相邻补色（平衡对比）", unsafe_allow_html=True)
                    st.markdown("- 三色配色：色轮均匀分布三色（丰富和谐）", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ---------------------- 标签页3：设计规范（自动生成+PDF导出）----------------------
    with tab3:
        st.markdown('<div class="func-card">', unsafe_allow_html=True)
        st.subheader("📋 自动生成设计规范文档")
        st.markdown("<p class='hint-text'>上传参考图，智能生成配色、字体、布局规范，支持PDF/TXT导出，直接用于工作交付</p>", unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown("#### 上传参考图")
            spec_ref_img = st.file_uploader(
                "选择设计参考图片（JPG/PNG/WebP）",
                type=["jpg", "jpeg", "png", "webp"],
                key="spec_ref_upload",
                label_visibility="collapsed"
            )
            
            st.markdown("#### 规范包含模块")
            spec_options = st.multiselect(
                "选择需要生成的规范模块",
                options=["配色规范", "字体规范", "布局规范", "应用建议"],
                default=["配色规范", "字体规范", "布局规范", "应用建议"],
                key="spec_options"
            )
            
            st.markdown("#### 导出格式")
            export_formats = st.multiselect(
                "选择需要导出的格式",
                options=["TXT格式", "PDF格式"],
                default=["TXT格式", "PDF格式"],
                key="export_formats"
            )
            
            generate_spec_btn = st.button("📋 生成设计规范", use_container_width=True, type="primary")
        
        with col2:
            st.markdown("#### 设计规范文档预览")
            spec_placeholder = st.empty()
            
            if spec_ref_img and generate_spec_btn:
                try:
                    with st.spinner("📋 正在分析图片并生成规范...（约10秒）"):
                        img = Image.open(spec_ref_img).convert("RGB")
                        
                        # 1. 提取配色规范
                        color_spec = extract_colors(img) if "配色规范" in spec_options else None
                        
                        # 2. 提取字体规范（调用API）
                        font_info = ""
                        if "字体规范" in spec_options:
                            font_info = recognize_font_style(img)
                        
                        # 3. 提取布局规范（调用API）
                        layout_info = ""
                        if "布局规范" in spec_options:
                            layout_info = recognize_layout_style(img)
                        
                        # 4. 生成完整规范文档
                        full_spec = generate_design_spec(color_spec, font_info, layout_info)
                        
                        # 显示规范文档预览
                        with spec_placeholder.container():
                            st.text_area(
                                "设计规范文档（可直接复制使用）",
                                value=full_spec,
                                height=450,
                                key="spec_preview",
                                disabled=False
                            )
                        
                        # 导出文件
                        st.markdown("### 📤 导出规范文档", unsafe_allow_html=True)
                        cols_dl = st.columns(len(export_formats))
                        
                        for idx, fmt in enumerate(export_formats):
                            with cols_dl[idx]:
                                if fmt == "TXT格式":
                                    st.download_button(
                                        label="📥 下载TXT格式",
                                        data=full_spec.encode("utf-8"),
                                        file_name="设计规范文档.txt",
                                        mime="text/plain",
                                        use_container_width=True
                                    )
                                elif fmt == "PDF格式":
                                    pdf_buffer = generate_spec_pdf(full_spec)
                                    st.download_button(
                                        label="📥 下载PDF格式",
                                        data=pdf_buffer,
                                        file_name="设计规范文档.pdf",
                                        mime="application/pdf",
                                        use_container_width=True
                                    )
                except Exception as e:
                    st.error(f"❌ 生成设计规范失败：{str(e)}", icon="⚠️")
            else:
                with spec_placeholder.container():
                    st.info("上传参考图片，选择需要的规范模块后点击「生成设计规范」", icon="ℹ️")
                    st.markdown("### 规范文档包含内容", unsafe_allow_html=True)
                    st.markdown('<div class="skill-card">', unsafe_allow_html=True)
                    st.markdown("#### 配色规范", unsafe_allow_html=True)
                    st.markdown("- 主色/辅助色/中性色的HEX/RGB/CMYK色值", unsafe_allow_html=True)
                    st.markdown("- 各颜色应用场景和占比建议", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    st.markdown('<div class="skill-card">', unsafe_allow_html=True)
                    st.markdown("#### 字体规范", unsafe_allow_html=True)
                    st.markdown("- 字体名称、字重、字号估算", unsafe_allow_html=True)
                    st.markdown("- 字间距/行间距特点和搭配建议", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    st.markdown('<div class="skill-card">', unsafe_allow_html=True)
                    st.markdown("#### 布局规范", unsafe_allow_html=True)
                    st.markdown("- 对齐方式、间距比例、视觉层级", unsafe_allow_html=True)
                    st.markdown("- 网格系统和适用场景", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    st.markdown('<div class="skill-card">', unsafe_allow_html=True)
                    st.markdown("#### 应用建议", unsafe_allow_html=True)
                    st.markdown("- 设计一致性要求和无障碍适配", unsafe_allow_html=True)
                    st.markdown("- 不同场景导出规范（印刷/电子屏）", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ---------------------- 标签页4：风格参考（相似风格+素材推荐+技巧库）----------------------
    with tab4:
        # 风格参考与素材推荐
        st.markdown('<div class="func-card">', unsafe_allow_html=True)
        st.subheader("🔍 设计风格参考与素材推荐")
        st.markdown("<p class='hint-text'>上传参考图，获取相似风格推荐、国内可用素材网站和设计软件插件建议</p>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 上传参考图")
            style_ref_img = st.file_uploader(
                "选择风格参考图片（JPG/PNG/WebP）",
                type=["jpg", "jpeg", "png", "webp"],
                key="style_ref_upload",
                label_visibility="collapsed"
            )
            recommend_btn = st.button("🔍 获取风格推荐", use_container_width=True, type="primary")
        
        with col2:
            st.markdown("#### 风格参考与素材推荐")
            style_placeholder = st.empty()
            
            if style_ref_img and recommend_btn:
                try:
                    with st.spinner("🔍 正在分析风格并推荐参考..."):
                        img = Image.open(style_ref_img).convert("RGB")
                        # 获取风格推荐
                        style_recommend = recommend_similar_style(img)
                        
                        # 显示推荐结果
                        with style_placeholder.container():
                            st.text_area(
                                "相似风格与素材推荐（可直接复制参考）",
                                value=style_recommend,
                                height=300,
                                key="style_recommend_preview"
                            )
                        
                        # 导出推荐结果
                        st.download_button(
                            label="📥 下载风格推荐（TXT）",
                            data=style_recommend.encode("utf-8"),
                            file_name="设计风格推荐.txt",
                            mime="text/plain",
                            use_container_width=True
                        )
                        
                        # 额外推荐常用设计素材网站（国内可用）
                        st.markdown("### 📚 国内常用设计素材网站汇总", unsafe_allow_html=True)
                        material_sites = """
1. 站酷（ZCOOL）- https://www.zcool.com.cn/
   - 特点：设计师社区+高质量原创素材，支持商用授权
   - 适用：品牌设计、海报设计、UI设计素材
   
2. 千图网 - https://www.58pic.com/
   - 特点：免费+付费素材齐全，模板丰富
   - 适用：PPT模板、海报模板、电商素材
   
3. 包图网 - https://ibaotu.com/
   - 特点：设计模板+视频素材+音效素材，一站式服务
   - 适用：短视频封面、电商详情页、活动海报
   
4. 摄图网 - https://699pic.com/
   - 特点：正版图片+视频素材，授权清晰
   - 适用：商业广告、宣传册、公众号配图
   
5. 视觉中国 - https://www.vcg.com/
   - 特点：高端正版素材，适合大型商业项目
   - 适用：品牌广告、户外海报、产品包装
   
6. 花瓣网 - https://huaban.com/
   - 特点：设计灵感收集，类似Pinterest
   - 适用：灵感收集、风格参考、排版借鉴
   
7. 创客贴 - https://www.chuangkit.com/
   - 特点：在线设计工具+模板，新手友好
   - 适用：快速制作海报、公众号封面、PPT
"""
                        st.text_area(
                            "国内常用设计素材网站（可复制保存）",
                            value=material_sites,
                            height=250,
                            key="material_sites",
                            disabled=False
                        )
                except Exception as e:
                    st.error(f"❌ 获取风格推荐失败：{str(e)}", icon="⚠️")
            else:
                with style_placeholder.container():
                    st.info("上传参考图片后点击「获取风格推荐」，获取相似风格和素材网站", icon="ℹ️")
        
        st.markdown('</div>', unsafe_allow_html=True)

        # 设计技巧知识库
        st.markdown('<div class="func-card">', unsafe_allow_html=True)
        st.subheader("📖 设计师常用技巧知识库")
        st.markdown("<p class='hint-text'>平面设计高频使用的技巧和规范，随时查阅，避免踩坑</p>", unsafe_allow_html=True)
        
        # 技巧分类选择
        skill_category = st.selectbox(
            "选择技巧分类",
            options=["配色技巧", "排版技巧", "字体搭配", "构图规则", "导出规范", "无障碍设计"],
            key="skill_category"
        )
        
        # 技巧内容展示
        skills_content = {
            "配色技巧": """
# 配色技巧知识库
## 一、核心配色原则
1. 三色原则：主色1种 + 辅助色2-3种 + 中性色不限
2. 对比原则：色相对比（互补/相似）、明度对比（≥3:1）、饱和度对比（主高辅低）
3. 占比原则：主色60-70% + 辅助色20-30% + 中性色10-15%
4. 一致性原则：整套设计保持配色系统统一

## 二、行业配色参考
- 科技类：蓝色（#165DFF）+ 深灰（#333333）+ 白色（#FFFFFF）
  - 特点：科技感、专业、冷静
- 母婴类：浅粉（#FFE6EF）+ 浅蓝（#E6F7FF）+ 米白（#FFF8E6）
  - 特点：柔和、温馨、安全
- 餐饮类：橙色（#FF7D00）+ 棕色（#8C6138）+ 米色（#F5F0E6）
  - 特点：刺激食欲、温暖、亲切
- 金融类：深蓝色（#0F3460）+ 金色（#D4AF37）+ 浅灰（#F5F5F5）
  - 特点：稳重、高端、可信
- 教育类：绿色（#36B37E）+ 蓝色（#007AFF）+ 白色（#FFFFFF）
  - 特点：成长、专业、清新

## 三、配色避坑指南
1. 避免高饱和三色同时大面积使用（视觉疲劳）
2. 避免红绿色搭配（色盲用户不可见，约8%男性色盲）
3. 避免深色背景+深色文字（明度差不足，可读性差）
4. 避免过多相似色叠加（层次不清晰，区分困难）
5. 避免印刷品使用RGB色值（颜色偏差大）

## 四、实用配色工具推荐
1. 在线工具：Adobe Color、Coolors、中国色
2. 插件工具：PS/AI的Color Harmony、Figma的Colorful
3. 参考工具：Pinterest、站酷、Behance的配色合集
""",
            "排版技巧": """
# 排版技巧知识库
## 一、核心排版原则
1. 对齐原则：所有元素保持统一对齐（左对齐优先，中文阅读更舒适）
2. 亲密性原则：相关元素靠近，无关元素远离（建立视觉分组）
3. 重复原则：重复使用颜色、字体、间距（增强一致性）
4. 对比原则：重要信息放大/加粗/高饱和（建立视觉层级）

## 二、间距规范（通用标准）
1. 行间距：
   - 正文：1.5-1.8倍字号
   - 标题：1.2倍字号
   - 多行长文本：1.8-2.0倍字号
2. 字间距：
   - 正文：0-50（默认）
   - 标题：50-200（根据字体调整）
   - 英文大写：100-300（提高可读性）
3. 模块间距：
   - 同级模块：统一间距（建议为正文行高的1-1.5倍）
   - 父子模块：子模块间距为父模块的1/2
4. 页边距：
   - A4印刷品：左右≥2cm，上下≥2.5cm
   - 网页设计：左右≥15px（移动端）/ ≥30px（桌面端）

## 三、视觉层级建立技巧
1. 字号层级：标题（18-24px）→ 副标题（16px）→ 正文（14px）→ 辅助文字（12px）
2. 字重层级：黑体（900）→ 粗体（700）→ 常规（400）→ 轻量（300）
3. 颜色层级：主色（高饱和）→ 辅助色（中饱和）→ 中性色（低饱和）
4. 位置层级：视觉中心（上方/左侧）→ 次要位置（下方/右侧）

## 四、排版避坑指南
1. 避免混合对齐方式（左对齐+居中对齐同时使用）
2. 避免文本两端对齐（中文会出现大量空白，可读性差）
3. 避免模块间距不一致（视觉混乱，无秩序感）
4. 避免过多装饰元素（分散注意力，弱化核心信息）
5. 避免文本行过长（单行≤50字符，移动端≤35字符）
""",
            "字体搭配": """
# 字体搭配技巧知识库
## 一、字体选择原则
1. 通用性原则：优先选择系统自带字体（避免字体缺失）
   - 中文系统字体：思源黑体、微软雅黑、苹方、宋体
   - 英文系统字体：Roboto、Montserrat、Open Sans、Arial
2. 风格统一原则：字体风格与设计主题一致（复古设计用衬线字体，现代设计用无衬线字体）
3. 数量控制原则：整套设计不超过2-3种字体（1种主字体+1-2种辅助字体）
4. 可读性原则：正文字体优先选择无衬线字体（中文阅读更舒适）

## 二、经典字体搭配组合
1. 商务正式风：
   - 中文：思源黑体 Bold（标题）+ 思源黑体 Regular（正文）
   - 英文：Montserrat Bold（标题）+ Open Sans Regular（正文）
2. 复古文艺风：
   - 中文：宋体（标题）+ 思源宋体 Light（正文）
   - 英文：Playfair Display（标题）+ Lora（正文）
3. 现代简约风：
   - 中文：苹方 Light（标题）+ 苹方 Regular（正文）
   - 英文：Roboto Light（标题）+ Roboto Regular（正文）
4. 活泼创意风：
   - 中文：站酷快乐体（标题）+ 思源黑体 Regular（正文）
   - 英文：Poppins Bold（标题）+ Inter Regular（正文）

## 三、字体使用规范
1. 字号规范：
   - 印刷品正文：≥9pt（12px）
   - 网页正文：≥14px
   - 移动端正文：≥16px
   - 老年群体使用：≥18px
2. 字重规范：
   - 避免使用斜体中文（可读性差，不美观）
   - 字重变化不超过3种（黑体+常规+轻量足够）
   - 标题使用粗体/黑体，正文使用常规字重
3. 特殊场景规范：
   - 电商设计：突出价格用粗体+大字号
   - 海报设计：标题可用艺术字体，正文必须用易读字体
   - 长文本设计：正文用轻量/常规字重，行间距1.8倍

## 四、字体避坑指南
1. 避免中文字体+西文字体随意搭配（风格冲突）
2. 避免艺术字体用于正文（可读性差，阅读疲劳）
3. 避免过小字号+过细字重（印刷不清晰，屏幕显示模糊）
4. 避免商用未授权字体（版权风险，可能面临赔偿）
5. 避免同一页面使用过多字重（视觉杂乱，层级混乱）
""",
            "构图规则": """
# 构图规则知识库
## 一、基础构图方法
1. 三分法构图：
   - 操作：将画面分为9宫格，主体放在交叉点或分割线上
   - 适用：风景、人物、产品图（最常用，稳定和谐）
2. 对称构图：
   - 操作：左右/上下对称，主体居中
   - 适用：建筑、LOGO、正式海报（稳定、庄重、平衡）
3. 对角线构图：
   - 操作：主体沿对角线分布
   - 适用：动态场景、产品展示（动感、延伸感、活力）
4. 框架构图：
   - 操作：用前景元素（门窗、树枝、阴影）形成框架
   - 适用：风景、人像（聚焦主体、增加层次、引导视线）

## 二、进阶构图方法
1. 引导线构图：
   - 操作：利用画面中的线条（道路、河流、阴影）引导视线到主体
   - 适用：风景、街拍、建筑（自然引导，突出主体）
2. 三角形构图：
   - 操作：元素形成三角形（正三角/倒三角）
   - 适用：产品组合、人物群像（正三角稳定，倒三角有张力）
3. 留白构图：
   - 操作：主体占比小，大量留白
   - 适用：极简设计、高端产品（突出主体、呼吸感、高级感）
4. 黄金比例构图：
   - 操作：按1:1.618比例划分画面，主体放在黄金螺旋中心
   - 适用：人像、产品特写（自然和谐，视觉舒适）

## 三、不同场景构图技巧
1. 产品设计构图：
   - 中心构图+轻微留白（突出产品，简洁大气）
   - 对角线构图（展示产品全貌，有动感）
   - 前后景搭配（增加层次，突出产品）
2. 海报设计构图：
   - 上中下构图（标题+主体+信息）
   - 左对齐构图（文字+图像，阅读流畅）
   - 非对称构图（活泼有创意，吸引注意力）
3. 电商详情页构图：
   - 产品居中+纯白背景（突出产品细节）
   - 场景化构图（产品使用场景，增强代入感）
   - 对比构图（产品前后对比，突出优势）

## 四、构图避坑指南
1. 避免主体居中+无留白（压抑，无呼吸感）
2. 避免元素过多+无焦点（视觉混乱，不知道看哪里）
3. 避免地平线居中（分割画面，视觉不平衡）
4. 避免视线冲出画面（主体视线方向应留有空间）
5. 避免构图过于呆板（适当打破规则，增加创意）
""",
            "导出规范": """
# 导出规范知识库
## 一、图片格式选择规范
1. JPG格式：
