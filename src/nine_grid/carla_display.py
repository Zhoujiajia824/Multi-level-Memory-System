"""CARLA显示副本的相机投影；不改原始RGB、模型输入和规划状态。"""
from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageDraw


def plan_mosaic_overlay(rgb, cameras, visualization):
    """将UE世界坐标折线投到六相机拼图；近裁剪和画布裁剪均逐线段执行。"""
    source = np.asarray(rgb)
    if source.dtype != np.uint8 or source.ndim != 3 or source.shape[2] != 3:
        raise ValueError('CARLA显示画面必须为uint8 RGB')
    image = Image.fromarray(source.copy())
    points = np.asarray(visualization.get('points_carla_xyz', []), dtype=float)
    if points.size == 0:
        return np.asarray(image).copy(), 0
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError('显示轨迹必须是有限CARLA世界三维点')
    order = ('front_left', 'front', 'front_right', 'back_left', 'back', 'back_right')
    indexed = {camera['view']: camera for camera in cameras}
    if len(cameras) != 6 or set(indexed) != set(order):
        raise ValueError('显示投影需要六路标定')
    segments = 0
    for index, view in enumerate(order):
        camera = indexed[view]
        width, height = camera['width'], camera['height']
        if source.shape[:2] != (height * 2, width * 3):
            raise ValueError('显示拼图尺寸与相机不一致')
        inverse = np.asarray(camera['world_to_camera_ue'], dtype=float)
        if inverse.shape != (4, 4) or not np.isfinite(inverse).all():
            raise ValueError('显示相机外参无效')
        local = np.c_[points, np.ones(len(points))] @ inverse.T
        focal = width / (2 * math.tan(math.radians(camera['fov_deg']) / 2))
        x, y = index % 3 * width, index // 3 * height
        tile = image.crop((x, y, x + width, y + height))
        draw = ImageDraw.Draw(tile)
        for start, stop in zip(local, local[1:]):
            a, b = start[:3].copy(), stop[:3].copy()
            # UE相机x为前，y为右，z为上。不能将镜头后的点折回画面。
            if max(a[0], b[0]) <= .1:
                continue
            if min(a[0], b[0]) < .1:
                crossing = a + (.1 - a[0]) / (b[0] - a[0]) * (b - a)
                if a[0] < .1:
                    a = crossing
                else:
                    b = crossing
            pixels = [(width / 2 + focal * p[1] / p[0], height / 2 - focal * p[2] / p[0]) for p in (a, b)]
            # Liang–Barsky裁剪，避免近裁剪后极大像素值交给绘图库。
            p0, p1 = np.asarray(pixels[0]), np.asarray(pixels[1])
            delta = p1 - p0
            lo, hi = 0., 1.
            for sign, distance in ((-delta[0], p0[0]), (delta[0], width - 1 - p0[0]),
                                   (-delta[1], p0[1]), (delta[1], height - 1 - p0[1])):
                if abs(sign) < 1e-12:
                    if distance < 0:
                        hi = -1.
                        break
                elif sign < 0:
                    lo = max(lo, distance / sign)
                else:
                    hi = min(hi, distance / sign)
            if lo <= hi:
                draw.line([tuple(p0 + lo * delta), tuple(p0 + hi * delta)], fill=(0, 230, 255), width=3)
                segments += 1
        image.paste(tile, (x, y))
    if segments:
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, image.height - 22, 270, image.height), fill=(0, 0, 0))
        draw.text((5, image.height - 18), 'MPC PLAN / display overlay', fill=(0, 230, 255))
    return np.asarray(image).copy(), segments
