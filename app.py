import streamlit as st

# ---------------------- 1. 初始化颜色会话状态（5种配色方案）----------------------
COLOR_SCHEMES = [
    # 1. 黑曜石色（默认）
    {"bg": "#121212", "card": "#1E1E1E", "btn": "#8B5CF6", "accent": "#A78BFA"},
    # 2. 深灰色
    {"bg": "#1E1E2E", "card": "#2D2D44", "btn": "#6366F1", "accent": "#818CF8"},
    # 3. 深蓝色
    {"bg": "#1A1E3B", "card": "#2A2F55", "btn": "#3B82F6", "accent": "#60A5FA"},
    # 4. 深紫色
    {"bg": "#2A1B3D", "card": "#3D2B5C", "btn": "#A855F7", "accent": "#C084FC"},
    # 5. 深绿色
    {"bg": "#1B3B2A", "card": "#2B5C45", "btn": "#22C55E", "accent": "#4ADE80"}
]

# 初始化Session State：当前颜色索引（默认0）
if "color_idx" not in st.session_state:
    st.session_state.color_idx = 0

# 切换颜色函数（点击轮盘时调用）
def switch_color():
    st.session_state.color_idx = (st.session_state.color_idx + 1) % 5

# ---------------------- 2. 右上角彩色轮盘（放大+视觉优化）----------------------
current_color = COLOR_SCHEMES[st.session_state.color_idx]
st.markdown(f"""
    <style>
        /* 颜色切换轮盘 - 放大+阴影+提示 */
        .color-wheel {{
            position: fixed;
            top: 15px;
            right: 15px;
            width: 15px;
            height: 15px;
            border-radius: 50%;
            background: linear-gradient(45deg, #8B5CF6, #3B82F6, #22C55E, #F59E0B, #EF4444);
            cursor: pointer;
            z-index: 9999;
            box-shadow: 0 0 10px rgba(139, 92, 246, 0.8);
            transition: transform 0.3s ease;
        }}
        .color-wheel:hover {{
            transform: scale(1.2);
        }}
        /* 页面基础样式 */
        .stApp {{
            background-color: {current_color["bg"]};
            color: #E0E0E0;
            font-family: 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
        }}
        /* 导航栏样式优化 */
        .stNavigation {{
            background-color: {current_color["card"]} !important;
            border-bottom: 1px solid #333 !important;
        }}
        .stNavigationItem {{
            border-radius: 8px !important;
            margin: 0 4px !important;
        }}
        .stNavigationItem:hover {{
            background-color: rgba(255,255,255,0.05) !important;
        }}
    </style>
    <!-- 颜色轮盘 + 点击事件 -->
    <div class="color-wheel" onclick="window.parent.streamlitCommandQueue.push({{'type':'setSessionState','args':{{'color_idx':{(st.session_state.color_idx + 1) % 5}}}}})"></div>
    <!-- 颜色切换提示 -->
    <div style="position: fixed; top: 40px; right: 15px; background: {current_color["card"]}; padding: 4px 8px; border-radius: 4px; font-size: 11px; z-index: 9998;">
        点击切换主题色
    </div>
""", unsafe_allow_html=True)

# ---------------------- 3. 顶部功能导航（新增设计工具集）----------------------
st.set_page_config(
    page_title="设计师媒体解析工具",
    page_icon="🎨",
    layout="wide"
)

# 定义三个功能页面
image_page = st.Page(
    page="pages/1_图片解析.py",
    title="图片解析",
    icon="📷"
)
video_page = st.Page(
    page="pages/2_视频解析.py",
    title="视频解析",
    icon="🎬"
)
design_tool_page = st.Page(
    page="pages/3_设计师工具集.py",
    title="设计师工具集",
    icon="🎨"
)

# 顶部导航
pg = st.navigation(
    pages=[image_page, video_page, design_tool_page],
    position="top"
)

# 运行当前选中的页面
pg.run()
