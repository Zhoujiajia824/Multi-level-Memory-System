"""
六视角环视拼接（surround-view mosaic）
======================================
把 nuScenes 6 个相机图像拼成一张 2x3 网格的 surround-view mosaic 图，
替代单张前视角图进入 VLM 场景理解 / 决策 / DINOv2 特征 / 短期记忆 / 中期记忆。

布局（与 config perception.cameras 行优先顺序一致）：
    上排：CAM_FRONT_LEFT | CAM_FRONT | CAM_FRONT_RIGHT
    下排：CAM_BACK_LEFT  | CAM_BACK  | CAM_BACK_RIGHT

每个子图可叠加相机名标签，便于 VLM 明确识别视角。缺图时该格留黑底并标注 (missing)，
不整体失败。
"""
from __future__ import annotations

import os
from typing import List, Optional

from PIL import Image, ImageDraw, ImageFont

from src.vla_memory.common.logging_utils import get_logger

logger = get_logger("surround_mosaic")

MOSAIC_COLS = 3
MOSAIC_ROWS = 2


def _load_font(size: int) -> ImageFont.ImageFont:
    """加载字体，优先常用字体，失败回退 PIL 默认。"""
    candidates = ["arial.ttf", "DejaVuSans.ttf", "msyh.ttc"]
    for name in candidates:
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return ImageFont.load_default()


def build_surround_mosaic(
    image_paths: List[str],
    cameras: List[str],
    cell_width: int = 480,
    cell_height: int = 270,
    label_subimages: bool = True,
    out_path: str = "",
) -> str:
    """把 6 路相机图拼成 2x3 mosaic 并落盘。

    Args:
        image_paths: 各相机图像绝对路径，顺序与 ``cameras`` 对齐。
        cameras: 相机名列表（行优先），决定各格归属与标签。
        cell_width: 每子图 resize 宽（像素）。
        cell_height: 每子图 resize 高（像素）。
        label_subimages: 是否在每个子图左上角写相机名（白字+黑描边）。
        out_path: mosaic 输出 JPEG 路径。

    Returns:
        落盘后的 mosaic 路径（= out_path）。
    """
    canvas_w, canvas_h = MOSAIC_COLS * cell_width, MOSAIC_ROWS * cell_height
    canvas = Image.new("RGB", (canvas_w, canvas_h), (0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    font = _load_font(max(16, cell_height // 12))

    n_cells = MOSAIC_COLS * MOSAIC_ROWS
    for idx in range(min(len(cameras), n_cells)):
        cam = cameras[idx]
        col = idx % MOSAIC_COLS
        row = idx // MOSAIC_COLS
        x0, y0 = col * cell_width, row * cell_height

        path = image_paths[idx] if idx < len(image_paths) else ""
        cell = None
        if path and os.path.exists(path):
            try:
                img = Image.open(path).convert("RGB").resize((cell_width, cell_height))
                cell = img
            except Exception as e:
                logger.warning("读取相机 %s 图像失败，该格以缺图处理: %s", cam, e)
                cell = None

        if cell is None:
            # 缺图：深灰底 + 提示
            cell = Image.new("RGB", (cell_width, cell_height), (24, 24, 24))
            d = ImageDraw.Draw(cell)
            d.text(
                (12, cell_height // 2),
                f"{cam}\n(missing)",
                fill=(200, 200, 200),
                font=font,
            )

        canvas.paste(cell, (x0, y0))

        if label_subimages:
            # 左上角相机名：黑色描边 + 白色字，保证任意背景可读
            tx, ty = x0 + 8, y0 + 6
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    draw.text((tx + dx, ty + dy), cam, fill=(0, 0, 0), font=font)
            draw.text((tx, ty), cam, fill=(255, 255, 255), font=font)

    # 落盘
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    canvas.save(out_path, "JPEG", quality=90)
    return out_path
