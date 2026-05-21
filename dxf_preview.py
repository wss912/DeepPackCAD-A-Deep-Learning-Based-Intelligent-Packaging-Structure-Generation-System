import ezdxf
import matplotlib.pyplot as plt
from ezdxf.addons.drawing import matplotlib as ezdxf_matplotlib
from ezdxf.addons.drawing.properties import RenderContext
from ezdxf.addons.drawing.frontend import Frontend


def dxf_to_png(dxf_path, png_path, dpi=600):
    """
    把 DXF 文件渲染成高清 PNG 图片，用于网页预览。

    dpi 越大越清楚：
        300 = 一般清晰
        600 = 高清
        800 = 更高清但文件更大
    """

    doc = ezdxf.readfile(dxf_path)
    msp = doc.modelspace()

    # 图纸是横向展开图，所以 figure 尽量设宽一点
    fig = plt.figure(figsize=(18, 10), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])

    ctx = RenderContext(doc)
    backend = ezdxf_matplotlib.MatplotlibBackend(ax)
    frontend = Frontend(ctx, backend)

    frontend.draw_layout(msp, finalize=True)

    ax.set_aspect("equal")
    ax.axis("off")

    plt.savefig(
        png_path,
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0.05
    )

    plt.close(fig)

    return png_path