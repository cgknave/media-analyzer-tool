import cv2
import numpy as np
import base64
import requests
from PIL import Image
import io
import streamlit as st
from collections import Counter
from sklearn.cluster import KMeans
import webcolors
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
        /* 颜色块样式 */
        .color-block {{
            width: 100%;
            height: 60px;
            border-radius: 8px;
            margin: 8px 0;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: 500;
            text-shadow: 0 1px 2px rgba(0,0,0,0.3);
        }}
    </style>
""", unsafe_allow_html=True)
# ---------------------- 3. 设计师专属工具函数 ----------------------
# 3.1 配色提取（主色+辅助色+中性色）
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
# 3.2 图片转Base64
def image_to_base64(image):
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format="JPEG")
    return base64.b64encode(img_byte_arr.getvalue()).decode("utf-8")
# 3.3 OCR文字提取
def extract_text(image):
    img_base64 = image_to_base64(image)
    url = "https://api-inference.modelscope.cn/v1/ocr/text-recognition"
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    payload = {
        "image": img_base64,
        "parameters": {"detect_direction": True, "language": "ch"}
    }
    response = requests.post(url, headers=headers, json=payload, timeout=30)
    response.raise_for_status()
    result = response.json()
    return "\n".join([item["text"] for item in result["items"]]) if "items" in result else "未识别到文字"
# 3.4 设计风格识别
def recognize_design_style(image):
    img_base64 = image_to_base64(image)
    url = "https://api-inference.modelscope.cn/v1/chat/completions"
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": "Qwen/Qwen2.5-VL-72B-Instruct",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": """作为专业平面设计师，分析图片的设计风格，输出：
1. 风格名称（如极简主义、复古风、国潮风、赛博朋克等）
2. 核心特点（色彩、排版、元素、质感）
3. 适用场景
4. 设计技巧借鉴
分点清晰，简洁实用"""},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}}
                ]
            }
        ],
        "max_tokens": 500,
        "temperature": 0.5
    }
    response = requests.post(url, headers=headers, json=payload, timeout=60)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]
# 3.5 核心分析函数
def analyze_image_comprehensive(image):
    img_base64 = image_to_base64(image)
    url = "https://api-inference.modelscope.cn/v1/chat/completions"
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": "Qwen/Qwen2.5-VL-72B-Instruct",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": """作为平面设计师的参考工具，详细分析图片，输出结构化结果：
1. 核心主体：人物/物体/动作（设计焦点）
2. 纹理材质：表面质感+材质类型（便于材质复用）
3. 光影细节：光影类型+光源方向+明暗对比（布光参考）
4. 色彩氛围：主色调+色彩搭配逻辑（配色参考）
5. 场景背景：场景类型+背景层级（构图参考）
6. 构图视角：构图规则+视角+视觉层级（排版参考）
分点呈现，简洁明了，突出设计参考价值"""},
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
# ---------------------- 4. 页面核心逻辑（新增设计师功能标签页）----------------------
def main():
    # 页面标题
    st.markdown(f"<h1 class='page-title'>📷 图片设计分析工具</h1>", unsafe_allow_html=True)
    st.markdown("<p class='hint-text'>支持JPG/PNG/WebP格式，单文件≤200MB，专为平面设计师优化</p>", unsafe_allow_html=True)
    # 1. 图片上传区域
    with st.container():
        st.markdown('<div class="func-card">', unsafe_allow_html=True)
        col1, col2 = st.columns([2, 1])
        
        with col1:
            uploaded_img = st.file_uploader(
                "上传图片（支持单张/批量）",
                type=["jpg", "jpeg", "png", "webp"],
                key="img_upload",
                label_visibility="collapsed",
                accept_multiple_files=False  # 批量功能在工具集单独实现
            )
            st.markdown("<div class='btn-group'>", unsafe_allow_html=True)
            col_btn1, col_btn2, col_btn3 = st.columns(3)
            with col_btn1:
                analyze_btn = st.button("📊 全面分析", type="primary", use_container_width=True)
            with col_btn2:
                color_btn = st.button("🎨 提取配色", use_container_width=True)
            with col_btn3:
                text_btn = st.button("📝 提取文字", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
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
                # 图片基本信息
                width, height = img.size
                st.markdown(f"📏 尺寸：{width}×{height}px")
                file_size = round(uploaded_img.size / 1024 / 1024, 2)
                st.markdown(f"📁 大小：{file_size}MB")
        st.markdown('</div>', unsafe_allow_html=True)
    # 2. 功能标签页（删除材质分析）
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 全面分析", 
        "🎨 配色提取", 
        "📝 文字识别", 
        "🎯 风格识别"
    ])
    # 初始化结果存储
    if "analysis_result" not in st.session_state:
        st.session_state.analysis_result = ""
    if "color_result" not in st.session_state:
        st.session_state.color_result = None
    if "text_result" not in st.session_state:
        st.session_state.text_result = ""
    if "style_result" not in st.session_state:
        st.session_state.style_result = ""
    # 标签页1：全面分析
    with tab1:
        st.markdown('<div class="func-card">', unsafe_allow_html=True)
        st.subheader("设计参考分析")
        result_placeholder = st.empty()
        
        with result_placeholder.container():
            st.text_area(
                "点击「全面分析」按钮获取结果（包含构图/光影/色彩/材质参考）",
                height=350,
                key="img_analysis_result",
                value=st.session_state.analysis_result,
                placeholder="点击上方按钮开始分析..."
            )
        
        if analyze_btn and uploaded_img:
            try:
                with st.spinner("🔍 正在分析设计细节..."):
                    img = Image.open(uploaded_img).convert("RGB")
                    result = analyze_image_comprehensive(img)
                    st.session_state.analysis_result = result
                    # 刷新结果
                    with result_placeholder.container():
                        st.text_area(
                            "✅ 分析完成（可直接复制参考）",
                            height=350,
                            key="img_analysis_result_active",
                            value=result
                        )
            except Exception as e:
                st.error(f"❌ 分析失败：{str(e)}", icon="⚠️")
        st.markdown('</div>', unsafe_allow_html=True)
    # 标签页2：配色提取
    with tab2:
        st.markdown('<div class="func-card">', unsafe_allow_html=True)
        st.subheader("配色方案提取（支持HEX/RGB/CMYK）")
        
        if color_btn and uploaded_img:
            try:
                with st.spinner("🎨 正在提取配色方案..."):
                    img = Image.open(uploaded_img).convert("RGB")
                    colors = extract_colors(img)
                    st.session_state.color_result = colors
                
                # 显示配色结果
                st.markdown("### 主色（占比最高）")
                main_color = colors["main"]
                st.markdown(f'<div class="color-block" style="background-color: {main_color["hex"]};">{main_color["hex"]}</div>', unsafe_allow_html=True)
                st.markdown(f"RGB: {main_color['rgb']} | CMYK: {main_color['cmyk'][0]}%,{main_color['cmyk'][1]}%,{main_color['cmyk'][2]}%,{main_color['cmyk'][3]}%")
                
                st.markdown("### 辅助色（搭配参考）")
                cols = st.columns(len(colors["secondary"]))
                for idx, color in enumerate(colors["secondary"]):
                    with cols[idx]:
                        st.markdown(f'<div class="color-block" style="background-color: {color["hex"]};">{color["hex"]}</div>', unsafe_allow_html=True)
                        st.markdown(f"RGB: {color['rgb']}")
                
                if colors["neutral"]:
                    st.markdown("### 中性色（背景/文字用）")
                    cols = st.columns(len(colors["neutral"]))
                    for idx, color in enumerate(colors["neutral"]):
                        with cols[idx]:
                            st.markdown(f'<div class="color-block" style="background-color: {color["hex"]};">{color["hex"]}</div>', unsafe_allow_html=True)
                            st.markdown(f"RGB: {color['rgb']}")
                
                # 导出色卡按钮
                def export_color_card(colors):
                    # 创建色卡图片
                    from PIL import ImageDraw
                    card_width = 800
                    card_height = 400
                    card = Image.new("RGB", (card_width, card_height), color="#ffffff")
                    draw = ImageDraw.Draw(card)
                    
                    # 绘制主色
                    main_width = card_width // 2
                    draw.rectangle([0, 0, main_width, card_height], fill=colors["main"]["hex"])
                    
                    # 绘制辅助色
                    sec_width = card_width // (2 * len(colors["secondary"]))
                    for i, color in enumerate(colors["secondary"]):
                        x1 = main_width + i * sec_width
                        x2 = x1 + sec_width
                        draw.rectangle([x1, 0, x2, card_height//2], fill=color["hex"])
                    
                    # 绘制中性色
                    if colors["neutral"]:
                        neu_width = card_width // (2 * len(colors["neutral"]))
                        for i, color in enumerate(colors["neutral"]):
                            x1 = main_width + i * neu_width
                            x2 = x1 + neu_width
                            draw.rectangle([x1, card_height//2, x2, card_height], fill=color["hex"])
                    
                    # 保存为BytesIO
                    img_byte_arr = io.BytesIO()
                    card.save(img_byte_arr, format="PNG")
                    return img_byte_arr.getvalue()
                
                color_card_data = export_color_card(colors)
                st.download_button(
                    label="📥 导出色卡（PNG）",
                    data=color_card_data,
                    file_name="配色方案色卡.png",
                    mime="image/png",
                    use_container_width=True
                )
                
            except Exception as e:
                st.error(f"❌ 配色提取失败：{str(e)}", icon="⚠️")
        else:
            st.info("点击「提取配色」按钮，自动生成可复用的配色方案", icon="ℹ️")
        st.markdown('</div>', unsafe_allow_html=True)
    # 标签页3：文字识别
    with tab3:
        st.markdown('<div class="func-card">', unsafe_allow_html=True)
        st.subheader("OCR文字提取（支持中文/英文）")
        result_placeholder = st.empty()
        
        with result_placeholder.container():
            st.text_area(
                "点击「提取文字」按钮获取结果",
                height=350,
                key="img_text_result",
                value=st.session_state.text_result,
                placeholder="点击上方按钮开始提取..."
            )
        
        if text_btn and uploaded_img:
            try:
                with st.spinner("📝 正在识别文字..."):
                    img = Image.open(uploaded_img).convert("RGB")
                    result = extract_text(img)
                    st.session_state.text_result = result
                    # 刷新结果
                    with result_placeholder.container():
                        st.text_area(
                            "✅ 文字提取完成（可直接复制）",
                            height=350,
                            key="img_text_result_active",
                            value=result
                        )
            except Exception as e:
                st.error(f"❌ 文字提取失败：{str(e)}", icon="⚠️")
        
        # 导出文字按钮
        if st.session_state.text_result:
            st.download_button(
                label="📥 导出文字（TXT）",
                data=st.session_state.text_result,
                file_name="提取的文字.txt",
                mime="text/plain",
                use_container_width=True
            )
        st.markdown('</div>', unsafe_allow_html=True)
    # 标签页4：风格识别
    with tab4:
        st.markdown('<div class="func-card">', unsafe_allow_html=True)
        st.subheader("设计风格识别（学习参考）")
        
        if uploaded_img:
            style_btn = st.button("🎯 识别风格", use_container_width=True)
            result_placeholder = st.empty()
            
            with result_placeholder.container():
                st.text_area(
                    "点击「识别风格」按钮获取结果",
                    height=350,
                    key="img_style_result",
                    value=st.session_state.style_result,
                    placeholder="点击按钮识别设计风格..."
                )
            
            if style_btn:
                try:
                    with st.spinner("🎨 正在识别设计风格..."):
                        img = Image.open(uploaded_img).convert("RGB")
                        result = recognize_design_style(img)
                        st.session_state.style_result = result
                        # 刷新结果
                        with result_placeholder.container():
                            st.text_area(
                                "✅ 风格识别完成（设计参考）",
                                height=350,
                                key="img_style_result_active",
                                value=result
                            )
                except Exception as e:
                    st.error(f"❌ 风格识别失败：{str(e)}", icon="⚠️")
        else:
            st.info("请先上传图片，再点击「识别风格」按钮", icon="ℹ️")
        st.markdown('</div>', unsafe_allow_html=True)
if __name__ == "__main__":
    main()
