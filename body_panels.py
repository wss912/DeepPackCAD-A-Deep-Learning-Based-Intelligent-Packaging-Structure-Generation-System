from dataclasses import dataclass
import ezdxf
import math
from openpyxl.descriptors import Length


# =========================================================
# 1. 参数定义
# =========================================================

@dataclass
class ProductParams:
    """
    产品参数
    """
    product_L: float          # 产品长度
    product_W: float          # 产品宽度
    product_H: float          # 产品高度
    clearance: float          # 装配间隙


@dataclass
class CartonParams:
    """
    纸盒结构参数
    """
    box_L: float              # 纸盒长度
    box_W: float              # 纸盒宽度
    box_H: float              # 纸盒高度

    glue_tab_width: float     # 粘贴翼宽度
    top_tuck_length: float    # 上插舌长度
    bottom_tuck_length: float # 下插舌长度
    dust_flap_length: float   # 防尘翼长度


@dataclass
class Panel:
    """
    面板对象
    CAD 坐标系：
    x 向右
    y 向上

    这里主体面板顶部 y = 0
    主体面板向下展开，所以底部 y = -height
    """
    name: str
    x: float
    y: float
    width: float
    height: float

    def top_edge(self):
        return (self.x, self.y), (self.x + self.width, self.y)

    def bottom_edge(self):
        return (self.x, self.y - self.height), (self.x + self.width, self.y - self.height)

    def left_edge(self):
        return (self.x, self.y), (self.x, self.y - self.height)

    def right_edge(self):
        return (self.x + self.width, self.y), (self.x + self.width, self.y - self.height)

    def center(self):
        return self.x + self.width / 2, self.y - self.height / 2


# =========================================================
# 2. 参数计算
# =========================================================

def calculate_carton_params(product: ProductParams) -> CartonParams:
    """
    由产品尺寸计算纸盒参数。
    以后深度学习模型就是替代/优化这里的参数预测。
    """

    box_L = product.product_L + 2 * product.clearance
    box_W = product.product_W + 2 * product.clearance
    box_H = product.product_H + 2 * product.clearance

    glue_tab_width = 15
    top_tuck_length = box_W
    bottom_tuck_length = box_W
    dust_flap_length = 0.5 * box_L

    return CartonParams(
        box_L=box_L,
        box_W=box_W,
        box_H=box_H,
        glue_tab_width=glue_tab_width,
        top_tuck_length=top_tuck_length,
        bottom_tuck_length=bottom_tuck_length,
        dust_flap_length=dust_flap_length
    )


# =========================================================
# 3. 构建主体面板
# =========================================================

def build_body_panels(params: CartonParams):
    """
    管型折叠纸盒主体结构：

    glue_tab | back | side1 | front | side2
    """

    L = params.box_L
    W = params.box_W
    H = params.box_H
    G = params.glue_tab_width

    x_glue = 0
    x_back = x_glue + G
    x_side1 = x_back + L
    x_front = x_side1 + W
    x_side2 = x_front + L

    y_top = 0

    panels = {
        "glue_tab": Panel("glue_tab", x_glue, y_top, G, H),
        "back": Panel("back", x_back, y_top, L, H),
        "side1": Panel("side1", x_side1, y_top, W, H),
        "front": Panel("front", x_front, y_top, L, H),
        "side2": Panel("side2", x_side2, y_top, W, H),
    }

    return panels


# =========================================================
# 4. DXF 基础工具函数
# =========================================================

def setup_dxf():
    """
    创建 DXF 文档，并设置图层。
    """

    doc = ezdxf.new("R2010")
    msp = doc.modelspace()

    # 设置单位为毫米
    doc.units = ezdxf.units.MM

    # 创建虚线线型
    if not doc.linetypes.has_entry("DASHED"):
        doc.linetypes.new(
            "DASHED",
            dxfattribs={
                "description": "Dashed line",
                "pattern": [6.0, 3.0, -3.0],
            }
        )

    # CUT：刀线 / 外轮廓
    if not doc.layers.has_entry("CUT"):
        doc.layers.new(
            "CUT",
            dxfattribs={
                "color": 1,             # 红色
                "linetype": "CONTINUOUS"
            }
        )

    # CREASE：压痕线 / 折线
    if not doc.layers.has_entry("CREASE"):
        doc.layers.new(
            "CREASE",
            dxfattribs={
                "color": 5,             # 蓝色
                "linetype": "DASHED"
            }
        )

    # TEXT：文字
    if not doc.layers.has_entry("TEXT"):
        doc.layers.new(
            "TEXT",
            dxfattribs={
                "color": 7
            }
        )

    return doc, msp


def add_cut_line(msp, start, end):
    """
    添加刀线，实线。
    """
    msp.add_line(
        start,
        end,
        dxfattribs={
            "layer": "CUT"
        }
    )

def add_cut_arc(msp, center, radius, start_angle, end_angle):
    """
    画刀线圆弧
    """
    msp.add_arc(
        center=center,
        radius=radius,
        start_angle=start_angle,
        end_angle=end_angle,
        dxfattribs={"layer": "CUT"}
    )
def add_crease_line(msp, start, end):
    """
    添加压痕线，虚线。
    """
    msp.add_line(
        start,
        end,
        dxfattribs={
            "layer": "CREASE",
            "linetype": "DASHED"
        }
    )
def add_cut_polyline(msp, points):
    """
    添加不闭合刀线折线。
    适合画插舌、防尘翼外轮廓。
    """
    msp.add_lwpolyline(
        points,
        close=False,
        dxfattribs={
            "layer": "CUT"
        }
    )
def add_angle_line(msp, x, y, length, angle_deg, layer="CUT"):
    """
    从点 (x, y) 开始，按指定角度和长度画一条线。

    angle_deg:
        0度   = 向右
        90度  = 向下或向上取决于你的坐标系
        -90度 = 反方向
    """

    angle_rad = math.radians(angle_deg)

    x2 = x + length * math.cos(angle_rad)
    y2 = y + length * math.sin(angle_rad)

    msp.add_line(
        (x, y),
        (x2, y2),
        dxfattribs={"layer": layer}
    )

    return x2, y2
def add_text(msp, text, position, height=6):
    """
    添加文字标注。
    """
    entity = msp.add_text(
        text,
        dxfattribs={
            "height": height,
            "layer": "TEXT"
        }
    )
    entity.dxf.insert = position
# =========================================================
# 5. 模块绘制函数
# =========================================================
def draw_glue_tab(msp, panel: Panel, slant=8):
    """
    绘制侧边粘贴翼。

    粘贴翼右边是和 back 面板连接的压痕线；
    其他边是刀线。
    """

    x = panel.x
    y = panel.y
    w = panel.width
    h = panel.height

    p1 = (x, y - slant)          # 左上斜点
    p2 = (x + w, y)              # 右上连接点
    p3 = (x + w, y - h)          # 右下连接点
    p4 = (x, y - h + slant)      # 左下斜点

    # 刀线：上斜边、左边、下斜边
    add_cut_line(msp, p1, p2)
    add_cut_line(msp, p1, p4)
    add_cut_line(msp, p4, p3)

    # 压痕线：粘贴翼和后面板连接边
    add_crease_line(msp, p2, p3)
def draw_body_edges(msp, panels, bottom_type="tuck"):
    """
            绘制主体面板的自由刀线和内部折线。

            bottom_type:
                tuck      普通下插舌 + 下防尘翼
                123_lock  1-2-3 锁底
            """

    glue_tab = panels["glue_tab"]
    back = panels["back"]
    side1 = panels["side1"]
    front = panels["front"]
    side2 = panels["side2"]

    y_top = 0
    y_bottom = -back.height

    # 最右侧外边界，刀线
    x_right = side2.x + side2.width
    add_cut_line(msp, (x_right, y_top), (x_right, y_bottom))

    # front 面板上边没有上插舌，画刀线
    add_cut_line(
        msp,
        (front.x, y_top),
        (front.x + front.width, y_top)
    )

    # back 面板下边：
    # 普通底部时，它是自由刀线；
    # 123 锁底时，它要连接锁底片，所以不能画刀线。
    if bottom_type == "tuck":
        add_cut_line(
            msp,
            (back.x, y_bottom),
            (back.x + back.width, y_bottom)
        )

    # 内部竖向折线
    # glue_tab 和 back 之间的折线已经在 draw_glue_tab() 里画了
    # back 面板下边没有插舌，画刀线
    # 内部竖向折线
    # glue_tab 和 back 之间的折线已经在 draw_glue_tab 里画了
    crease_xs = [
        side1.x,  # back 和 side1
        front.x,  # side1 和 front
        side2.x,  # front 和 side2
    ]

    for x in crease_xs:
        add_crease_line(
            msp,
            (x, y_top),
            (x, y_bottom)
        )
def draw_tuck_flap_normal(msp, panel: Panel, edge="top", length=30, cut=0):
    """
    绘制普通插舌。

    edge="top"    表示挂在面板上边
    edge="bottom" 表示挂在面板下边

    插舌外轮廓是刀线；
    插舌和面板连接处是压痕线。
    """

    if edge == "top":
        (x1, y1), (x2, y2) = panel.top_edge()
        add_cut_line(
            msp,
            (x1,y1),
            (x1 + cut, y1 + length)
        )
        add_cut_line(
            msp,
            (x2 - cut, y2 + length),
            (x2, y2)
        )
        add_crease_line(msp, (x1, y1), (x2, y2))
        add_crease_line(msp, (x1, y1+length), (x2, y2+length))
        msp.add_arc(
            center=(x1+(panel.width/6), y1+length),
            radius=panel.width/6,
            start_angle=90,
            end_angle=180,
            dxfattribs={"layer": "CUT"}
        )
        msp.add_arc(
            center=(x2 - (panel.width / 6), y1 + length),
            radius=panel.width / 6,
            start_angle=0,
            end_angle=90,
            dxfattribs={"layer": "CUT"}
        )
        add_cut_line(
            msp,
            (x1+panel.width/6, y1+length+panel.width/6),
            (x2-panel.width/6, y1+length+panel.width/6)
        )
    elif edge == "bottom":
        (x1, y1), (x2, y2) = panel.bottom_edge()
        add_cut_line(
            msp,
            (x1, y1),
            (x1 + cut, y1 - length)
        )
        add_cut_line(
            msp,
            (x2 - cut, y2 - length),
            (x2, y2)
        )
        add_crease_line(msp, (x1, y1), (x2, y2))
        add_crease_line(msp, (x1, y1 - length), (x2, y2 - length))
        msp.add_arc(
            center=(x1 + (panel.width / 6), y1 -length),
            radius=panel.width / 6,
            start_angle=180,
            end_angle=-90,
            dxfattribs={"layer": "CUT"}
        )
        msp.add_arc(
            center=(x2 - (panel.width / 6), y1 -length),
            radius=panel.width / 6,
            start_angle=-90,
            end_angle=-0,
            dxfattribs={"layer": "CUT"}
        )
        add_cut_line(
            msp,
            (x1 + panel.width / 6, y1 - length - panel.width / 6),
            (x2 - panel.width / 6, y1 - length - panel.width / 6)
        )

    else:
        raise ValueError("edge 只能是 'top' 或 'bottom'")
def draw_slit_lock_tuck_flap(msp, panel: Panel, edge="top", length=30, cut=0,slit_offset=12, slit_depth=10):
    """
    绘制切缝锁定插舌。
    它是在普通插舌的基础上，增加两条锁定切缝。
    外轮廓：CUT 刀线
    连接边：CREASE 压痕线
    切缝：CUT 刀线
    """
    if edge == "top":
        (x1, y1), (x2, y2) = panel.top_edge()
        add_cut_line(
            msp,
            (x1, y1),
            (x1 + cut, y1 + length)
        )
        add_cut_line(
            msp,
            (x2 - cut, y2 + length),
            (x2, y2)
        )
        add_cut_line(
            msp,
            (x1, y1 + length),
            (x1 + panel.width/11, y2 + length)
        )
        add_cut_line(
            msp,
            (x1+panel.width/11, y2 + length),
            (x1+panel.width/11, y2+length-length/12)
        )
        add_cut_line(
            msp,
            (x2 - panel.width / 11, y2 + length),
            (x2 - panel.width / 11, y2+length-length/12)
        )
        add_cut_line(
            msp,
            (x1 + panel.width / 11, y2 + length),
            (x1 + panel.width / 11, y2 + length - length / 12)
        )
        add_cut_line(
            msp,
            (x2 - panel.width / 11, y2 + length),
            (x2, y2 + length)
        )
        add_crease_line(msp, (x1, y1), (x2, y2))
        add_crease_line(msp, (x1 + panel.width / 11, y2 + length - length / 12),
                        (x2 - panel.width / 11, y2 + length- length / 12))

        msp.add_arc(
            center=(x1 + (panel.width / 6), y1 + length),
            radius=panel.width / 6,
            start_angle=90,
            end_angle=180,
            dxfattribs={"layer": "CUT"}
        )
        msp.add_arc(
            center=(x2 - (panel.width / 6), y1 + length),
            radius=panel.width / 6,
            start_angle=0,
            end_angle=90,
            dxfattribs={"layer": "CUT"}
        )
        add_cut_line(
            msp,
            (x1 + panel.width / 6, y1 + length + panel.width / 6),
            (x2 - panel.width / 6, y1 + length + panel.width / 6)
        )
    elif edge == "bottom":
        (x1, y1), (x2, y2) = panel.bottom_edge()
        add_cut_line(
            msp,
            (x1, y1),
            (x1 + cut, y1 - length)
        )
        add_cut_line(
            msp,
            (x2 - cut, y2 - length),
            (x2, y2)
        )
        add_cut_line(
            msp,
            (x1, y1 - length),
            (x1 + panel.width / 11, y2 - length)
        )
        add_cut_line(
            msp,
            (x1 + panel.width / 11, y2 - length),
            (x1 + panel.width / 11, y2 - length + length / 12)
        )
        add_cut_line(
            msp,
            (x2 - panel.width / 11, y2 - length),
            (x2 - panel.width / 11, y2 - length + length / 12)
        )
        add_cut_line(
            msp,
            (x2 - panel.width / 11, y2 - length),
            (x2, y2 - length)
        )
        add_crease_line(msp, (x1, y1), (x2, y2))
        add_crease_line(msp, (x1 + panel.width / 11, y2 - length + length / 12),
                        (x2 - panel.width / 11, y2 - length + length / 12))

        msp.add_arc(
            center=(x1 + (panel.width / 6), y1 - length),
            radius=panel.width / 6,
            start_angle=180,
            end_angle=270,
            dxfattribs={"layer": "CUT"}
        )
        msp.add_arc(
            center=(x2 - (panel.width / 6), y1 - length),
            radius=panel.width / 6,
            start_angle=270,
            end_angle=360,
            dxfattribs={"layer": "CUT"}
        )
        add_cut_line(
            msp,
            (x1 + panel.width / 6, y1 - length - panel.width / 6),
            (x2 - panel.width / 6, y1 - length - panel.width / 6)
        )
def draw_tuck_module(msp, panel: Panel, edge="top", length=30, tuck_type="normal"):
    """
    插舌模块统一入口。

    tuck_type:
        normal     普通插舌
        slit_lock  切缝锁定插舌
    """

    if tuck_type == "normal":
        draw_tuck_flap_normal(
            msp,
            panel=panel,
            edge=edge,
            length=length
        )

    elif tuck_type == "slit_lock":
        draw_slit_lock_tuck_flap(
            msp,
            panel=panel,
            edge=edge,
            length=length
        )

    else:
        raise ValueError(f"未知插舌类型: {tuck_type}")
def draw_dust_flap(msp, panel: Panel, edge="top", length=40, cut=6):
    """
    绘制防尘翼。

    防尘翼也是挂在面板边上的附属结构。
    外轮廓为刀线；
    连接边为压痕线。
    """
    if edge == "top":
        (x1, y1), (x2, y2) = panel.top_edge()

        points = [
            (x1, y1),
            (x1 + cut, y1 + length),
            (x2 - cut, y2 + length),
            (x2, y2)
        ]

        add_cut_polyline(msp, points)
        add_crease_line(msp, (x1, y1), (x2, y2))

    elif edge == "bottom":
        (x1, y1), (x2, y2) = panel.bottom_edge()

        points = [
            (x1, y1),
            (x1 + cut, y1 - length),
            (x2 - cut, y2 - length),
            (x2, y2)
        ]

        add_cut_polyline(msp, points)
        add_crease_line(msp, (x1, y1), (x2, y2))

    else:
        raise ValueError("edge 只能是 'top' 或 'bottom'")
def draw_slit_lock_dust_flap(msp, panel: Panel, edge="top", length=40):
    """
    绘制直边防尘翼。

    外轮廓：CUT 刀线
    连接边：CREASE 压痕线
    """
    length=(panel.width/1.8+panel.width)/2
    if edge == "top":
        (x1, y1), (x2, y2) = panel.top_edge()

        points = [
            (x1, y1),
            (x1,y1+length*5/13),
            (x1+2/13*length, y1 + length*7/13),
            (x1+2/13*length+length*1.61/13,y1+ length*7/13+length*6/13),
            (x2-3.61/13*length,y1+length),
            (x2-2/13*length,y2+length*7/13),
            (x2,y2+5/13*length),
            (x2,y2)
        ]

        add_cut_polyline(msp, points)
        add_crease_line(msp, (x1, y1), (x2, y2))

    elif edge == "bottom":
        (x1, y1), (x2, y2) = panel.bottom_edge()
        points = [
            (x1, y1),
            (x1,y1-length*5/13),
            (x1+2/13*length, y1 - length*7/13),
            (x1+2/13*length+length*1.61/13,y1- length*7/13-length*6/13),
            (x2-3.61/13*length,y1-length),
            (x2-2/13*length,y2-length*7/13),
            (x2,y2-5/13*length),
            (x2,y2)
        ]

        add_cut_polyline(msp, points)
        add_crease_line(msp, (x1, y1), (x2, y2))
    else:
        raise ValueError("edge 只能是 'top' 或 'bottom'")
def draw_dust_module(msp, panel: Panel, edge="top", length=40, dust_type="slant"):
    """
    防尘翼模块统一入口。

    dust_type:
        slant     斜边防尘翼
        slit_lock_dust  直边防尘翼
    """
    length=(panel.width/1.8+panel.width)/2
    if dust_type == "slant":
        draw_dust_flap(
            msp,
            panel=panel,
            edge=edge,
            length=length
        )

    elif dust_type == "slit_lock_dust":
        draw_slit_lock_dust_flap(
            msp,
            panel=panel,
            edge=edge,
            length=length
        )

    else:
        raise ValueError(f"未知防尘翼类型: {dust_type}")
def draw_panel_names(msp, panels):
    """
    给每个面板添加文字标注。
    """
    for panel in panels.values():
        cx, cy = panel.center()
        add_text(msp, panel.name, (cx - 8, cy), height=5)
def get_123_lock_angles(box_L, box_W):
    """
    根据书上的 1-2-3 锁底角度设计规律确定角度。

    返回：
    a_angle : ∠a
    b_angle : ∠b
    need_bite : 是否需要增加咬合结构
    """

    ratio = box_L / box_W

    if ratio <= 1.5:
        return 30, 60, False
    elif ratio <= 2.5:
        return 45, 45, False
    else:
        return 45, 45, True


def draw_123_lock_bottom(
    msp,
    panels,
    a_angle=None,
    b_angle=None,
    show_aux_text=True
):
    """
    稳定版 1-2-3 锁底结构。

    关键修正：
    1. 所有几何点统一使用 a_dx / b_dx。
    2. 对 L/W 较小的盒型限制 a_dx，避免左右摇翼交叉。
    3. 对 b_dx 也进行限制，避免侧摇翼过尖。
    4. small_edge 重新基于 a_dx 计算，避免出现负值。
    """

    back = panels["back"]
    side1 = panels["side1"]
    front = panels["front"]
    side2 = panels["side2"]

    box_L = back.width
    box_W = side1.width
    w = box_W

    # 自动选择角度
    if a_angle is None or b_angle is None:
        a_angle, b_angle, need_bite = get_123_lock_angles(box_L, box_W)
    else:
        need_bite = box_L / box_W > 2.5

    flap_depth = box_W / 2

    # =====================================================
    # 1. 安全计算 a_dx / b_dx
    # =====================================================

    raw_a_dx = math.tan(math.radians(90 - a_angle)) * (w / 2)
    raw_b_dx = math.tan(math.radians(b_angle)) * (w / 2)

    # a_dx 不能太大，否则 back/front 的左右折线会交叉
    max_a_dx = box_L * 0.33

    if raw_a_dx > max_a_dx:
        a_dx = max_a_dx
    else:
        a_dx = raw_a_dx

    # b_dx 不能太大，否则侧翼会过尖甚至越界
    max_b_dx = box_W * 0.75

    if raw_b_dx > max_b_dx:
        b_dx = max_b_dx
    else:
        b_dx = raw_b_dx

    # 如果非常短胖，强制不用咬合
    # 因为 L/W 小的时候再加咬合结构会更乱
    if box_L / box_W <= 2.5:
        need_bite = False

    # 咬合小边长度，必须基于 a_dx 计算
    small_edge = (box_L - 2 * a_dx) / 3

    if small_edge < box_L * 0.08:
        small_edge = box_L * 0.08

    # 文字位置用
    side_depth = min(a_dx, box_L / 3)

    # =====================================================
    # 2. side1 底部侧摇翼
    # =====================================================

    (x1, y1), (x2, y2) = side1.bottom_edge()

    p0 = (x1, y1)
    p1 = (x1, y1 - w)
    p2 = (x2 - b_dx, y1 - w)
    p3 = (x2 - b_dx, y1 - w / 2)
    p4 = (x2, y2)

    add_cut_polyline(msp, [p0, p1, p2, p3, p4])
    add_crease_line(msp, (x1, y1), (x2, y2))

    if show_aux_text:
        add_text(msp, "1", (x1 + w / 2 - 2, y1 - side_depth / 2), height=5)

    # =====================================================
    # 3. back 底部摇翼
    # =====================================================

    (x1, y1), (x2, y2) = back.bottom_edge()

    if need_bite:
        p0 = (x1, y1)
        p1 = (x1 + a_dx, y1 - w / 2)
        p2 = (x1 + a_dx, y1 - w)

        p3 = (x1 + a_dx + small_edge, y1 - w)
        p4 = (x1 + a_dx + small_edge, y1 - w / 2)

        p5 = (x1 + a_dx + 2 * small_edge, y1 - w / 2)
        p6 = (x1 + a_dx + 2 * small_edge, y1 - w)

        p7 = (x1 + a_dx + 3 * small_edge, y1 - w)
        p8 = (x1 + a_dx + 3 * small_edge, y1 - w / 2)

        p9 = (x2, y2)

        add_cut_polyline(msp, [p0, p1, p2, p3, p4, p5, p6, p7, p8, p9])

    else:
        p0 = (x1, y1)
        p1 = (x1 + a_dx, y1 - w / 2)
        p2 = (x1 + a_dx, y1 - w)
        p3 = (x2 - a_dx, y1 - w)
        p4 = (x2 - a_dx, y1 - w / 2)
        p5 = (x2, y2)

        add_cut_polyline(msp, [p0, p1, p2, p3, p4, p5])

    add_crease_line(msp, (x1, y1), (x2, y2))

    if show_aux_text:
        add_text(msp, "2", (x1 + box_L / 2 - 2, y1 - flap_depth / 2), height=5)

    # =====================================================
    # 4. front 底部摇翼
    # =====================================================

    (x1, y1), (x2, y2) = front.bottom_edge()

    if need_bite:
        p0 = (x1, y1)
        p1 = (x1, y1 - w)

        p2 = (x1 + a_dx, y1 - w)
        p3 = (x1 + a_dx, y1 - w / 2)

        p4 = (x1 + a_dx + small_edge, y1 - w / 2)
        p5 = (x1 + a_dx + small_edge, y1 - w)

        p6 = (x1 + a_dx + 2 * small_edge, y1 - w)
        p7 = (x1 + a_dx + 2 * small_edge, y1 - w / 2)

        p8 = (x1 + a_dx + 3 * small_edge, y1 - w / 2)
        p9 = (x1 + a_dx + 3 * small_edge, y1 - w)

        p10 = (x2, y1 - w)
        p11 = (x2, y2)

        add_cut_polyline(msp, [p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11])

    else:
        p0 = (x1, y1)
        p1 = (x1, y1 - w)
        p2 = (x1 + a_dx, y1 - w)
        p3 = (x1 + a_dx, y1 - w / 2)
        p4 = (x2 - a_dx, y1 - w / 2)
        p5 = (x2 - a_dx, y1 - w)
        p6 = (x2, y1 - w)
        p7 = (x2, y2)

        add_cut_polyline(msp, [p0, p1, p2, p3, p4, p5, p6, p7])

    add_crease_line(msp, (x1, y1), (x2, y2))

    if show_aux_text:
        add_text(msp, "3", (x1 + box_L / 2 - 2, y1 - flap_depth / 2), height=5)

    # =====================================================
    # 5. side2 底部侧摇翼
    # =====================================================

    (x1, y1), (x2, y2) = side2.bottom_edge()

    p0 = (x2, y2)
    p1 = (x2, y2 - w)
    p2 = (x1 + b_dx, y1 - w)
    p3 = (x1 + b_dx, y1 - w / 2)
    p4 = (x1, y2)

    add_cut_polyline(msp, [p0, p1, p2, p3, p4])
    add_crease_line(msp, (x1, y1), (x2, y2))

    if show_aux_text:
        add_text(msp, "1", (x1 + w / 2 - 2, y1 - side_depth / 2), height=5)

    # =====================================================
    # 6. 角度说明
    # =====================================================

    if show_aux_text:
        add_text(
            msp,
            f"a={a_angle}, b={b_angle}, a_dx={a_dx:.1f}, b_dx={b_dx:.1f}",
            (front.x + box_L / 2 - 30, front.y - front.height - flap_depth - 10),
            height=4
        )
def draw_top_module(
    msp,
    panels,
    params,
    top_tuck_type="slit_lock",
    top_dust_type="slit_lock_dust"
):
    back = panels["back"]
    side1 = panels["side1"]
    side2 = panels["side2"]

    draw_tuck_module(
        msp,
        panel=back,
        edge="top",
        length=params.top_tuck_length,
        tuck_type=top_tuck_type
    )

    draw_dust_module(
        msp,
        panel=side1,
        edge="top",
        length=params.dust_flap_length,
        dust_type=top_dust_type
    )

    draw_dust_module(
        msp,
        panel=side2,
        edge="top",
        length=params.dust_flap_length,
        dust_type=top_dust_type
    )
def draw_bottom_module(
    msp,
    panels,
    params,
    bottom_type="tuck",
    bottom_tuck_type="normal",
    bottom_dust_type="slant"
):
    front = panels["front"]
    side1 = panels["side1"]
    side2 = panels["side2"]

    if bottom_type == "tuck":
        draw_tuck_module(
            msp,
            panel=front,
            edge="bottom",
            length=params.bottom_tuck_length,
            tuck_type=bottom_tuck_type
        )

        draw_dust_module(
            msp,
            panel=side1,
            edge="bottom",
            length=params.dust_flap_length,
            dust_type=bottom_dust_type
        )

        draw_dust_module(
            msp,
            panel=side2,
            edge="bottom",
            length=params.dust_flap_length,
            dust_type=bottom_dust_type
        )

    elif bottom_type == "123_lock":
        draw_123_lock_bottom(
            msp,
            panels,
            a_angle=None,
            b_angle=None,
            show_aux_text=True
        )

    else:
        raise ValueError(f"未知底部结构类型: {bottom_type}")
# =========================================================
# 6. 总生成函数
# =========================================================

def generate_carton_dxf(
    filename="folding_carton.dxf",

    product_L=120,
    product_W=40,
    product_H=180,
    clearance=3,

    top_tuck_type="slit_lock",
    top_dust_type="slit_lock_dust",

    bottom_type="tuck",
    bottom_tuck_type="normal",
    bottom_dust_type="slant",
    top_tuck_length=None,
    bottom_tuck_length=None,
    dust_flap_length=None,
    glue_tab_width=None,
    show_panel_names=True
):
    product = ProductParams(
        product_L=product_L,
        product_W=product_W,
        product_H=product_H,
        clearance=clearance
    )

    params = calculate_carton_params(product)
    if glue_tab_width is not None:
        params.glue_tab_width = glue_tab_width

    if top_tuck_length is not None:
        params.top_tuck_length = top_tuck_length

    if bottom_tuck_length is not None:
        params.bottom_tuck_length = bottom_tuck_length

    if dust_flap_length is not None:
        params.dust_flap_length = dust_flap_length
    panels = build_body_panels(params)

    doc, msp = setup_dxf()

    glue_tab = panels["glue_tab"]

    draw_glue_tab(msp, glue_tab)

    draw_body_edges(
        msp,
        panels,
        bottom_type=bottom_type
    )

    draw_top_module(
        msp,
        panels,
        params,
        top_tuck_type=top_tuck_type,
        top_dust_type=top_dust_type
    )

    draw_bottom_module(
        msp,
        panels,
        params,
        bottom_type=bottom_type,
        bottom_tuck_type=bottom_tuck_type,
        bottom_dust_type=bottom_dust_type
    )

    if show_panel_names:
        draw_panel_names(msp, panels)

    doc.saveas(filename)

    print(f"DXF 文件已生成：{filename}")
    print("纸盒参数：")
    print(params)

    return params, panels

# =========================================================
# 7. 程序入口
# =========================================================
    """  dust_type:
        slant     斜边防尘翼
        slit_lock_dust  直边防尘翼
        tuck_type:
        normal
        slit_lock
    """
if __name__ == "__main__":
    generate_carton_dxf(
        filename="test_tuck_bottom.dxf",
        product_L=60,
        product_W=45,
        product_H=100,
        clearance=3,

        top_tuck_type="slit_lock",
        top_dust_type="slit_lock_dust",

        bottom_type="123_lock",
        bottom_tuck_type="normal",
        bottom_dust_type="slant"
    )