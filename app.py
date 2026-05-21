import os
import tempfile
from pathlib import Path
from datetime import datetime

# 如果你的环境偶尔出现 OpenMP 冲突，可以保留这一行
# 必须放在 import torch 相关模块之前
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st

from read_stl_size import read_stl_size
from predict_structure import predict_structure
from body_panels import generate_carton_dxf
from dxf_preview import dxf_to_png
from chat_control import parse_chat_command, apply_overrides


# =========================================================
# 1. 页面基础设置
# =========================================================

st.set_page_config(
    page_title="基于深度学习的包装结构生成系统",
    layout="wide"
)

st.title("基于深度学习的包装结构 CAD 图生成系统")
st.write("上传 STL 三维模型，通过对话方式设置或修改包装结构，系统自动生成多个包装展开图方案。")


# =========================================================
# 2. 路径与输出目录
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


# =========================================================
# 3. session_state 初始化
# =========================================================

if "chat_items" not in st.session_state:
    # 这里存聊天记录和每一次生成结果
    # 每个元素可以是：
    # {"type": "message", "role": "user"/"assistant", "content": "..."}
    # {"type": "result", "result_id": 1, "structure": {...}, "dxf_path": "...", "png_path": "...", "summary": "..."}
    st.session_state.chat_items = []

if "pending_overrides" not in st.session_state:
    # 生成前用户通过对话提出的要求，先存在这里
    st.session_state.pending_overrides = {}

if "current_structure" not in st.session_state:
    # 当前结构参数。第一次生成时来自 AI 预测，之后可以被对话持续修改
    st.session_state.current_structure = None

if "result_counter" not in st.session_state:
    # 生成结果编号，每次生成 +1
    st.session_state.result_counter = 0

if "last_uploaded_filename" not in st.session_state:
    st.session_state.last_uploaded_filename = None

if "last_stl_info" not in st.session_state:
    st.session_state.last_stl_info = None


# =========================================================
# 4. 侧边栏输入
# =========================================================

st.sidebar.header("产品附加信息")

weight = st.sidebar.number_input(
    "产品重量 / kg",
    min_value=0.01,
    max_value=10.0,
    value=0.3,
    step=0.05
)

fragility = st.sidebar.selectbox(
    "易碎等级",
    options=[1, 2, 3],
    index=0,
    help="1 = 不易碎，2 = 中等，3 = 易碎"
)

category_name = st.sidebar.selectbox(
    "产品类别",
    options=[
        "牙膏类",
        "瓶装类",
        "普通日用品",
        "化妆品",
        "药品"
    ],
    index=0
)

CATEGORY_MAP = {
    "牙膏类": 0,
    "瓶装类": 1,
    "普通日用品": 2,
    "化妆品": 3,
    "药品": 4
}

category_id = CATEGORY_MAP[category_name]

clearance = st.sidebar.number_input(
    "装配间隙 / mm",
    min_value=0.0,
    max_value=20.0,
    value=3.0,
    step=0.5
)

st.sidebar.divider()

if st.sidebar.button("清空对话和所有生成结果"):
    st.session_state.chat_items = []
    st.session_state.pending_overrides = {}
    st.session_state.current_structure = None
    st.session_state.result_counter = 0
    st.rerun()

if st.sidebar.button("重新从 AI 预测开始"):
    st.session_state.current_structure = None
    st.session_state.pending_overrides = {}
    st.session_state.chat_items.append(
        {
            "type": "message",
            "role": "assistant",
            "content": "已清空当前结构，下一次生成会重新使用深度学习模型预测。"
        }
    )
    st.rerun()


# =========================================================
# 5. 上传 STL 文件
# =========================================================

uploaded_file = st.file_uploader(
    "请上传产品 STL 文件",
    type=["stl"]
)

if uploaded_file is None:
    st.info("请先上传一个 STL 文件。")
    st.stop()

# 如果换了 STL 文件，清空当前结构和历史生成结果，避免旧模型套到新 STL 上
if st.session_state.last_uploaded_filename != uploaded_file.name:
    st.session_state.last_uploaded_filename = uploaded_file.name
    st.session_state.chat_items = []
    st.session_state.pending_overrides = {}
    st.session_state.current_structure = None
    st.session_state.result_counter = 0


# =========================================================
# 6. 保存临时 STL
# =========================================================

with tempfile.NamedTemporaryFile(delete=False, suffix=".stl") as tmp:
    tmp.write(uploaded_file.getvalue())
    stl_path = tmp.name


# =========================================================
# 7. 读取 STL 尺寸
# =========================================================

try:
    product_L, product_W, product_H = read_stl_size(stl_path)
except Exception as e:
    st.error(f"STL 文件读取失败：{e}")
    st.stop()


# =========================================================
# 8. 显示产品尺寸
# =========================================================

st.subheader("1. STL 模型尺寸识别结果")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("产品长度 L / mm", f"{product_L:.2f}")

with col2:
    st.metric("产品宽度 W / mm", f"{product_W:.2f}")

with col3:
    st.metric("产品高度 H / mm", f"{product_H:.2f}")


# =========================================================
# 9. 允许用户手动修正尺寸
# =========================================================

st.subheader("2. 产品尺寸确认")
st.write("如果 STL 模型方向不标准，可以在这里手动修正长、宽、高。")

c1, c2, c3 = st.columns(3)

with c1:
    product_L = st.number_input(
        "确认产品长度 L / mm",
        min_value=1.0,
        value=float(round(product_L, 2)),
        step=1.0
    )

with c2:
    product_W = st.number_input(
        "确认产品宽度 W / mm",
        min_value=1.0,
        value=float(round(product_W, 2)),
        step=1.0
    )

with c3:
    product_H = st.number_input(
        "确认产品高度 H / mm",
        min_value=1.0,
        value=float(round(product_H, 2)),
        step=1.0
    )


# =========================================================
# 10. 工具函数：生成一次包装展开图
# =========================================================

def generate_one_result(structure, product_L, product_W, product_H, clearance):
    """
    根据当前结构生成一次 DXF 和 PNG，并把结果加入聊天流。
    """

    st.session_state.result_counter += 1
    result_id = st.session_state.result_counter

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sid = structure.get("structure_id", "custom")

    dxf_filename = f"result_{result_id:03d}_structure_{sid}_{timestamp}.dxf"
    png_filename = f"result_{result_id:03d}_structure_{sid}_{timestamp}.png"

    dxf_path = OUTPUT_DIR / dxf_filename
    png_path = OUTPUT_DIR / png_filename

    # 这些参数需要你的 body_panels.generate_carton_dxf 支持
    # 如果你还没有加 length 参数，先按我下面给你的 body_panels 修改方式加上
    generate_carton_dxf(
        filename=str(dxf_path),

        product_L=product_L,
        product_W=product_W,
        product_H=product_H,
        clearance=clearance,

        top_tuck_type=structure["top_tuck_type"],
        top_dust_type=structure["top_dust_type"],

        bottom_type=structure["bottom_type"],
        bottom_tuck_type=structure["bottom_tuck_type"],
        bottom_dust_type=structure["bottom_dust_type"],

        top_tuck_length=structure.get("top_tuck_length"),
        bottom_tuck_length=structure.get("bottom_tuck_length"),
        dust_flap_length=structure.get("dust_flap_length"),
        glue_tab_width=structure.get("glue_tab_width")
    )

    png_ok = False
    try:
        dxf_to_png(str(dxf_path), str(png_path))
        png_ok = True
    except Exception as e:
        st.session_state.chat_items.append(
            {
                "type": "message",
                "role": "assistant",
                "content": f"DXF 已生成，但第 {result_id} 个结果的 PNG 预览生成失败：{e}"
            }
        )

    summary = (
        f"已生成第 {result_id} 个包装结构方案："
        f"{structure.get('structure_name', '自定义结构')}。"
        f"上插舌={structure['top_tuck_type']}，"
        f"上防尘翼={structure['top_dust_type']}，"
        f"底部={structure['bottom_type']}。"
    )

    st.session_state.chat_items.append(
        {
            "type": "result",
            "result_id": result_id,
            "summary": summary,
            "structure": structure.copy(),
            "dxf_path": str(dxf_path),
            "png_path": str(png_path) if png_ok else None,
            "dxf_filename": dxf_filename,
            "png_filename": png_filename,
        }
    )

    st.session_state.chat_items.append(
        {
            "type": "message",
            "role": "assistant",
            "content": summary + "你可以继续说：上插舌长度改成30，底部改成123锁底，或者再生成一个。"
        }
    )


# =========================================================
# 11. 聊天式生成区域
# =========================================================

st.subheader("3. 对话式包装结构生成")

st.write(
    "你可以像聊天一样控制系统。比如："
    "底部改成123锁底；上插舌改成普通插舌；上插舌长度改成30；"
    "防尘翼长度改成60；我要更牢固一点；开始生成。"
)

# 先显示所有历史消息和所有生成结果
for item in st.session_state.chat_items:
    if item["type"] == "message":
        with st.chat_message(item["role"]):
            st.write(item["content"])

    elif item["type"] == "result":
        with st.chat_message("assistant"):
            st.markdown(f"### 生成结果 {item['result_id']}")
            st.write(item["summary"])

            if item.get("png_path") and Path(item["png_path"]).exists():

                preview_col, _ = st.columns([1, 2])

                with preview_col:
                    st.image(
                        item["png_path"],
                        caption=f"生成结果 {item['result_id']} - 小图预览",
                        width=500
                    )

                with st.expander(f"查看生成结果 {item['result_id']} 的高清大图"):
                    st.image(
                        item["png_path"],
                        caption=f"生成结果 {item['result_id']} - 高清展开图",
                        width=1200
                    )

            else:
                st.warning("该结果没有 PNG 预览，但 DXF 文件仍然可以下载。")

            dxf_path = Path(item["dxf_path"])
            if dxf_path.exists():
                with open(dxf_path, "rb") as f:
                    st.download_button(
                        label=f"下载结果 {item['result_id']} 的 DXF 文件",
                        data=f,
                        file_name=item["dxf_filename"],
                        mime="application/dxf",
                        key=f"download_dxf_{item['result_id']}"
                    )

            if item.get("png_path"):
                png_path = Path(item["png_path"])
                if png_path.exists():
                    with open(png_path, "rb") as f:
                        st.download_button(
                            label=f"下载结果 {item['result_id']} 的 PNG 图片",
                            data=f,
                            file_name=item["png_filename"],
                            mime="image/png",
                            key=f"download_png_{item['result_id']}"
                        )


# 显示当前未生成前缓存的要求
if st.session_state.current_structure is None and st.session_state.pending_overrides:
    st.info("当前已记录但还没有生成的要求：")
    st.json(st.session_state.pending_overrides)

# 显示当前结构参数
if st.session_state.current_structure is not None:
    st.info("当前结构参数。你后续发消息会在这个基础上继续修改。")
    st.json(st.session_state.current_structure)


# =========================================================
# 12. 处理用户输入
# =========================================================

user_prompt = st.chat_input("请输入你的包装结构要求，例如：上插舌长度改成30，底部用123锁底，开始生成")

if user_prompt:
    st.session_state.chat_items.append(
        {
            "type": "message",
            "role": "user",
            "content": user_prompt
        }
    )

    overrides, reply, should_generate = parse_chat_command(user_prompt)

    # 用户如果只说“生成一个”、“开始生成”、“再生成一个”，也要触发生成
    if any(key in user_prompt for key in ["生成", "再来一个", "再生成", "出图", "画出来", "开始"]):
        should_generate = True

    # 还没有当前结构：先缓存用户要求
    if st.session_state.current_structure is None:
        if overrides:
            st.session_state.pending_overrides.update(overrides)
            reply = reply + "我已经记录这些要求。"

    # 已经有当前结构：直接修改当前结构，并自动生成一个新结果
    else:
        if overrides:
            st.session_state.current_structure = apply_overrides(
                st.session_state.current_structure,
                overrides
            )
            reply = reply + "我已经把这些修改应用到当前结构上，并为你生成一个新的方案。"

            # 关键：只要用户修改了当前结构，就自动触发生成
            should_generate = True

    st.session_state.chat_items.append(
        {
            "type": "message",
            "role": "assistant",
            "content": reply
        }
    )

    # 如果用户要求生成，则生成一个新结果，并追加到聊天流中
    if should_generate:
        try:
            # 第一次生成：先用 AI 预测，再叠加用户之前说过的 pending_overrides
            if st.session_state.current_structure is None:
                structure = predict_structure(
                    product_L=product_L,
                    product_W=product_W,
                    product_H=product_H,
                    weight=weight,
                    fragility=fragility,
                    category_id=category_id
                )

                structure = apply_overrides(
                    structure,
                    st.session_state.pending_overrides
                )

                st.session_state.current_structure = structure
                st.session_state.pending_overrides = {}

            # 后续生成：在当前结构基础上直接生成一个新结果
            structure = st.session_state.current_structure

            generate_one_result(
                structure=structure,
                product_L=product_L,
                product_W=product_W,
                product_H=product_H,
                clearance=clearance
            )

        except Exception as e:
            st.session_state.chat_items.append(
                {
                    "type": "message",
                    "role": "assistant",
                    "content": f"生成失败：{e}"
                }
            )

    st.rerun()


# =========================================================
# 13. 备用按钮：不输入文字也可以点击生成
# =========================================================

if st.button("AI预测并生成一个新方案", type="primary"):
    try:
        if st.session_state.current_structure is None:
            structure = predict_structure(
                product_L=product_L,
                product_W=product_W,
                product_H=product_H,
                weight=weight,
                fragility=fragility,
                category_id=category_id
            )

            structure = apply_overrides(
                structure,
                st.session_state.pending_overrides
            )

            st.session_state.current_structure = structure
            st.session_state.pending_overrides = {}

        structure = st.session_state.current_structure

        generate_one_result(
            structure=structure,
            product_L=product_L,
            product_W=product_W,
            product_H=product_H,
            clearance=clearance
        )

    except Exception as e:
        st.session_state.chat_items.append(
            {
                "type": "message",
                "role": "assistant",
                "content": f"生成失败：{e}"
            }
        )

    st.rerun()
