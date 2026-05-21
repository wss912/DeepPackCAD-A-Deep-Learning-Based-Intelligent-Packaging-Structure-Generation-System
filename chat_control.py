import re


# =========================================================
# 1. 基础工具：提取数字
# =========================================================

def extract_number(text):
    """
    从文字里提取第一个数字。

    例如：
        上插舌长度改成30 -> 30
        防尘翼长度改成60 -> 60
        粘贴翼宽度18.5 -> 18.5
    """
    match = re.search(r"\d+(\.\d+)?", text)
    if match:
        return float(match.group())
    return None


# =========================================================
# 2. 插舌与防尘翼匹配规则
# =========================================================

def match_dust_type_by_tuck_type(tuck_type):
    """
    根据插舌类型匹配防尘翼类型。

    普通插舌 normal        -> 普通斜边防尘翼 slant
    切缝锁定插舌 slit_lock -> 切缝锁定配套防尘翼 slit_lock_dust
    """
    if tuck_type == "slit_lock":
        return "slit_lock_dust"
    return "slant"


# =========================================================
# 3. 对话解析
# =========================================================

def parse_chat_command(user_text):
    """
    把用户自然语言转换成包装结构参数修改。

    返回：
        overrides: 要修改的参数字典
        reply: 系统回复
        should_generate: 是否需要生成新图
    """

    text = user_text.strip().replace(" ", "")

    overrides = {}
    changed = []
    should_generate = False

    if any(key in text for key in [
        "生成", "开始生成", "生产", "出图", "画出来",
        "生成盒型", "生成展开图", "再生成", "再来一个"
    ]):
        should_generate = True

    number = extract_number(text)

    # =====================================================
    # 1. 长度参数修改
    # =====================================================

    if any(key in text for key in ["上插舌长度", "顶部插舌长度"]):
        if number is not None:
            overrides["top_tuck_length"] = number
            changed.append(f"上插舌长度设置为 {number:g} mm")
        else:
            changed.append("我识别到了你想修改上插舌长度，但没有识别到具体数值")

    elif any(key in text for key in ["下插舌长度", "底部插舌长度"]):
        if number is not None:
            overrides["bottom_tuck_length"] = number
            overrides["bottom_type"] = "tuck"
            changed.append(f"下插舌长度设置为 {number:g} mm")
        else:
            changed.append("我识别到了你想修改下插舌长度，但没有识别到具体数值")

    elif any(key in text for key in ["防尘翼长度", "上防尘翼长度", "下防尘翼长度"]):
        if number is not None:
            overrides["dust_flap_length"] = number
            changed.append(f"防尘翼长度设置为 {number:g} mm")
        else:
            changed.append("我识别到了你想修改防尘翼长度，但没有识别到具体数值")

    elif any(key in text for key in ["粘贴翼宽度", "胶舌宽度", "胶边宽度"]):
        if number is not None:
            overrides["glue_tab_width"] = number
            changed.append(f"粘贴翼宽度设置为 {number:g} mm")
        else:
            changed.append("我识别到了你想修改粘贴翼宽度，但没有识别到具体数值")

    elif any(key in text for key in ["间隙", "装配间隙"]):
        if number is not None:
            overrides["clearance"] = number
            changed.append(f"装配间隙设置为 {number:g} mm")
        else:
            changed.append("我识别到了你想修改装配间隙，但没有识别到具体数值")

    # =====================================================
    # 2. 上插舌类型
    # =====================================================

    if any(key in text for key in ["上插舌", "上面插舌", "顶部插舌"]):

        if any(key in text for key in ["普通", "普通插舌", "简单"]):
            overrides["top_tuck_type"] = "normal"
            overrides["top_dust_type"] = "slant"
            changed.append("上插舌设置为普通插舌，并自动匹配普通防尘翼")

        elif any(key in text for key in ["切缝", "锁定", "切缝锁定"]):
            overrides["top_tuck_type"] = "slit_lock"
            overrides["top_dust_type"] = "slit_lock_dust"
            changed.append("上插舌设置为切缝锁定插舌，并自动匹配切缝锁定防尘翼")

    # =====================================================
    # 3. 上防尘翼类型
    # =====================================================

    if any(key in text for key in ["上防尘翼", "上面防尘翼", "顶部防尘翼"]):

        if any(key in text for key in ["斜边", "普通防尘翼", "普通"]):
            overrides["top_dust_type"] = "slant"
            changed.append("上防尘翼设置为斜边防尘翼")

        elif any(key in text for key in ["切缝", "配合切缝", "锁定"]):
            overrides["top_dust_type"] = "slit_lock_dust"
            changed.append("上防尘翼设置为切缝锁定配套防尘翼")

    # =====================================================
    # 4. 底部结构
    # =====================================================

    if any(key in text for key in ["底部", "盒底", "下面", "下部", "锁底"]):

        if any(key in text for key in ["123", "锁底", "123锁底"]):
            overrides["bottom_type"] = "123_lock"
            overrides["bottom_tuck_type"] = "normal"
            overrides["bottom_dust_type"] = "slant"
            changed.append("底部设置为123锁底")

        elif any(key in text for key in ["普通底", "插舌底", "普通插舌底", "普通"]):
            overrides["bottom_type"] = "tuck"
            overrides["bottom_tuck_type"] = "normal"
            overrides["bottom_dust_type"] = "slant"
            changed.append("底部设置为普通插舌底，并自动匹配普通防尘翼")

    # =====================================================
    # 5. 下插舌类型
    # =====================================================

    if any(key in text for key in ["下插舌", "下面插舌", "底部插舌"]):

        if any(key in text for key in ["普通", "普通插舌"]):
            overrides["bottom_type"] = "tuck"
            overrides["bottom_tuck_type"] = "normal"
            overrides["bottom_dust_type"] = "slant"
            changed.append("下插舌设置为普通插舌，并自动匹配普通防尘翼")

        elif any(key in text for key in ["切缝", "锁定", "切缝锁定"]):
            overrides["bottom_type"] = "tuck"
            overrides["bottom_tuck_type"] = "slit_lock"
            overrides["bottom_dust_type"] = "slit_lock_dust"
            changed.append("下插舌设置为切缝锁定插舌，并自动匹配切缝锁定防尘翼")

    # =====================================================
    # 6. 下防尘翼类型
    # =====================================================

    if any(key in text for key in ["下防尘翼", "下面防尘翼", "底部防尘翼"]):

        if any(key in text for key in ["斜边", "普通"]):
            overrides["bottom_type"] = "tuck"
            overrides["bottom_dust_type"] = "slant"
            changed.append("下防尘翼设置为斜边防尘翼")

        elif any(key in text for key in ["切缝", "配合切缝", "锁定"]):
            overrides["bottom_type"] = "tuck"
            overrides["bottom_dust_type"] = "slit_lock_dust"
            # 如果用户明确要切缝锁定防尘翼，底部插舌也要配套成切缝锁定插舌
            overrides["bottom_tuck_type"] = "slit_lock"
            changed.append("下防尘翼设置为切缝锁定配套防尘翼，并自动匹配切缝锁定下插舌")

    # =====================================================
    # 7. 模糊语义
    # =====================================================

    if any(key in text for key in ["更牢固", "结实", "承重", "重物", "稳一点"]):
        overrides["top_tuck_type"] = "slit_lock"
        overrides["top_dust_type"] = "slit_lock_dust"
        overrides["bottom_type"] = "123_lock"
        overrides["bottom_tuck_type"] = "normal"
        overrides["bottom_dust_type"] = "slant"
        changed.append("已按更牢固结构设置：切缝锁定插舌 + 123锁底")

    if any(key in text for key in ["简单一点", "普通一点", "便宜一点", "简单结构"]):
        overrides["top_tuck_type"] = "normal"
        overrides["top_dust_type"] = "slant"
        overrides["bottom_type"] = "tuck"
        overrides["bottom_tuck_type"] = "normal"
        overrides["bottom_dust_type"] = "slant"
        changed.append("已按简单结构设置：普通插舌盒")

    # =====================================================
    # 8. 生成回复
    # =====================================================

    if not changed and not should_generate:
        reply = (
            "我还没有识别到明确的结构修改。你可以这样说：\n"
            "上插舌改成普通插舌；\n"
            "上插舌改成切缝锁定插舌；\n"
            "下插舌改成普通插舌；\n"
            "下插舌改成切缝锁定插舌；\n"
            "防尘翼长度改成60；\n"
            "底部改成123锁底；\n"
            "我要更牢固一点；\n"
            "开始生成。"
        )

    elif changed:
        reply = "，".join(changed) + "。"

    else:
        reply = "好的，我开始生成一个新的包装结构方案。"

    return overrides, reply, should_generate


# =========================================================
# 4. 应用修改并自动匹配
# =========================================================

def apply_overrides(structure, overrides):
    """
    把用户对话解析出来的参数应用到当前结构。
    同时自动匹配插舌和防尘翼。
    """

    new_structure = structure.copy()

    for key, value in overrides.items():
        new_structure[key] = value

    # =====================================================
    # 上部插舌和防尘翼自动匹配
    # =====================================================
    # 只有用户没有明确指定 top_dust_type 时，才根据 top_tuck_type 自动匹配

    if "top_tuck_type" in overrides and "top_dust_type" not in overrides:
        new_structure["top_dust_type"] = match_dust_type_by_tuck_type(
            new_structure.get("top_tuck_type", "normal")
        )

    # =====================================================
    # 下部插舌和防尘翼自动匹配
    # =====================================================

    if new_structure.get("bottom_type") == "tuck":
        if "bottom_tuck_type" in overrides and "bottom_dust_type" not in overrides:
            new_structure["bottom_dust_type"] = match_dust_type_by_tuck_type(
                new_structure.get("bottom_tuck_type", "normal")
            )

        if "bottom_dust_type" in overrides and "bottom_tuck_type" not in overrides:
            if new_structure.get("bottom_dust_type") == "slit_lock_dust":
                new_structure["bottom_tuck_type"] = "slit_lock"
            else:
                new_structure["bottom_tuck_type"] = "normal"

    # =====================================================
    # 123 锁底不再使用普通下插舌和普通下防尘翼
    # 参数保留只是为了界面显示，不参与绘图
    # 为了避免界面误导，这里统一显示成 normal + slant
    # =====================================================

    if new_structure.get("bottom_type") == "123_lock":
        new_structure["bottom_tuck_type"] = "normal"
        new_structure["bottom_dust_type"] = "slant"

    return new_structure
