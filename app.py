import streamlit as st

# ---------------------- 1. 初始化颜色会话状态（5种配色方案）----------------------
# 定义5种页面配色（背景色+卡片色+按钮色+强调色）
COLOR_SCHEMES = [
    # 1. 黑曜石色（默认）
    {"bg": "#121212", "card": "#1E1E1E", "btn": "#8B5CF6", "accent": "#8B5CF6"},
    # 2. 深灰色
    {"bg": "#1E1E2E", "card": "#2D2D44", "btn": "#6366F1", "accent": "#6366F1"},
    # 3. 深蓝色
    {"bg": "#1A1E3B", "card": "#2A2F55", "btn": "#3B82F6", "accent": "#3B82F6"},
    # 4. 深紫色
    {"bg": "#2A1B3D", "card": "#3D2B5C", "btn": "#A855F7", "accent": "#A855F7"},
    # 5. 深绿色
    {"bg": "#1B3B2A", "card": "#2B5C45", "btn": "#22C55E", "accent": "#22C55E"}
]

# 初始化Session State：当前颜色索引（默认0）
if "color_idx" not in st.session_state:
    st.session_state.color_idx = 0

# 切换颜色函数（点击轮盘时调用）
def switch_color():
    st.session_state.color_idx = (st.session_state.color_idx + 1) % 5  # 循环切换5种颜色

# ---------------------- 2. 右上角3px*3px彩色轮盘（点击切换颜色）----------------------
current_color = COLOR_SCHEMES[st.session_state.color_idx]
st.markdown(f"""
    <style>
        /* 固定彩色轮盘在右上角 */
        .color-wheel {{
            position: fixed;
            top: 15px;
            right: 15px;
            width: 3px;
            height: 3px;
            border-radius: 50%; /* 圆形 */
            background: linear-gradient(45deg, #8B5CF6, #3B82F6, #22C55E, #F59E0B, #EF4444); /* 彩色轮盘样式 */
            cursor: pointer;
            z-index: 9999; /* 置于顶层，不被遮挡 */
        }}
        /* 页面基础样式（关联颜色变量） */
        .stApp {{
            background-color: {current_color["bg"]};
            color: #E0E0E0;
        }}
    </style>
    <!-- 彩色轮盘按钮（点击触发切换函数） -->
    <div class="color-wheel" onclick="window.parent.streamlitCommandQueue.push({{'type':'setSessionState','args':{{'color_idx':{st.session_state.color_idx + 1 % 5}}}}})"></div>
""", unsafe_allow_html=True)

# ---------------------- 3. 顶部功能跳转（图片解析/视频解析）----------------------
st.set_page_config(
    page_title="媒体解析工具",
    page_icon="📽️",
    layout="wide"
)

# 定义两个功能页面（路径必须正确，对应pages文件夹下的文件）
image_page = st.Page(
    page="pages/1_图片解析.py",  # 明确写全路径，避免识别失败
    title="图片解析",
    icon="📷"
)
video_page = st.Page(
    page="pages/2_视频解析.py",
    title="视频解析",
    icon="🎬"
)

# 顶部水平导航（参数格式严格匹配Streamlit 1.32.0+要求）
pg = st.navigation(
    pages=[image_page, video_page],  # 用列表包裹，避免语法错误
    position="top",  # 固定顶部
)

# 运行当前选中的页面（必须加这一行，否则页面不显示）
pg.run()
