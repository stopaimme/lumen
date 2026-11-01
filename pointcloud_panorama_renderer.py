#!/usr/bin/env python
# coding=utf-8
"""
Point Cloud to Panorama Rendering
==================================

功能：
1. 从多视角图像(MV-images) + 深度图反投影生成点云
2. 渲染全景图（equirectangular / cubemap）

渲染方法：
1. 纯 PyTorch GPU 实现（高性能，支持百万级点云）
2. PyTorch3D 实现（高质量，可微分渲染）

支持的投影方式：
- Equirectangular (球面投影)
- Cubemap (6面立方体贴图)
"""

import os
import re
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Tuple, Optional, Union, List, Dict
from math import pi
from PIL import Image

try:
    import OpenEXR
    import Imath
    HAS_OPENEXR = True
except ImportError:
    HAS_OPENEXR = False
    print("[ERROR] OpenEXR library not installed!")
    print("        Please install with: pip install OpenEXR")
    print("        Note: This script requires OpenEXR to read .exr depth files.")


# ============================================================================
# 数据加载和反投影功能（从 surfel_render.py 集成）
# ============================================================================

def load_depth_exr(filepath: str) -> np.ndarray:
    """
    从 EXR 文件中读取深度图（直接使用 OpenEXR 库）
    
    Args:
        filepath: EXR 文件路径
        
    Returns:
        depth: numpy array, shape (H, W)
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Depth file not found: {filepath}")
    
    if not HAS_OPENEXR:
        raise ImportError(
            "OpenEXR library is required to read .exr files.\n"
            "Please install it with: pip install OpenEXR\n"
            "Note: OpenCV's EXR support is often disabled by default."
        )
    
    # 使用 OpenEXR 读取
    exr_file = OpenEXR.InputFile(filepath)
    header = exr_file.header()
    
    dw = header['dataWindow']
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1
    
    # 尝试读取不同的通道名称
    channels = header['channels'].keys()
    depth_channel = None
    
    for ch in ['Y', 'Z', 'R', 'V', 'depth', 'Depth']:
        if ch in channels:
            depth_channel = ch
            break
    
    if depth_channel is None:
        raise ValueError(f"No depth channel found in EXR. Available channels: {list(channels)}")
    
    # 读取深度数据
    depth_str = exr_file.channel(depth_channel, Imath.PixelType(Imath.PixelType.FLOAT))
    depth = np.frombuffer(depth_str, dtype=np.float32)
    depth = depth.reshape((height, width))
    
    print(f"[INFO] Loaded depth from {filepath}: shape={depth.shape}, range=[{depth.min():.4f}, {depth.max():.4f}]")
    
    return depth


def load_transform_json(json_path: str) -> Tuple[List[Dict], Dict]:
    """
    从 transform.json 读取相机参数
    
    Args:
        json_path: transform.json 文件路径
        
    Returns:
        frames: list of dict, 每个包含 'frame', 'w2c', 'fx', 'fy' 等
        camera_params: dict, 包含相机参数（如果在全局定义）
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Transform file not found: {json_path}")
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # 支持两种格式：
    # 1. 数组格式：直接是帧列表 [{"w": 512, "h": 512, ...}, ...]
    # 2. 字典格式：{"frames": [...], "camera_angle_x": ...}
    if isinstance(data, list):
        frames = data
        camera_params = {}
    else:
        frames = data.get('frames', [])
        # 提取全局相机参数
        camera_params = {
            'camera_angle_x': data.get('camera_angle_x', None),
            'camera_angle_y': data.get('camera_angle_y', None),
            'fl_x': data.get('fx', data.get('fl_x', None)),
            'fl_y': data.get('fy', data.get('fl_y', None)),
            'cx': data.get('cx', None),
            'cy': data.get('cy', None),
            'w': data.get('w', None),
            'h': data.get('h', None),
            'k1': data.get('k1', 0.0),
            'k2': data.get('k2', 0.0),
            'p1': data.get('p1', 0.0),
            'p2': data.get('p2', 0.0),
        }
    
    print(f"[INFO] Loaded {len(frames)} frames from {json_path}")
    
    return frames, camera_params


def compute_intrinsics_from_params(
    camera_params: Dict, 
    frame_data: Optional[Dict] = None, 
    img_width: Optional[int] = None, 
    img_height: Optional[int] = None
) -> Tuple[np.ndarray, int, int]:
    """
    从相机参数计算内参矩阵，支持全局参数或逐帧参数
    
    Args:
        camera_params: dict with global camera parameters
        frame_data: dict with per-frame camera parameters (优先使用)
        img_width: 图像宽度（如果都没有）
        img_height: 图像高度（如果都没有）
        
    Returns:
        K: 3x3 intrinsics matrix
        width: 图像宽度
        height: 图像高度
    """
    # 优先使用帧级参数，其次使用全局参数
    if frame_data:
        width = frame_data.get('w', camera_params.get('w', img_width))
        height = frame_data.get('h', camera_params.get('h', img_height))
        fx = frame_data.get('fx', camera_params.get('fl_x', None))
        fy = frame_data.get('fy', camera_params.get('fl_y', None))
        cx = frame_data.get('cx', camera_params.get('cx', None))
        cy = frame_data.get('cy', camera_params.get('cy', None))
        fov_x = frame_data.get('fovx', camera_params.get('camera_angle_x', None))
    else:
        width = camera_params.get('w', img_width)
        height = camera_params.get('h', img_height)
        fx = camera_params.get('fl_x', None)
        fy = camera_params.get('fl_y', None)
        cx = camera_params.get('cx', None)
        cy = camera_params.get('cy', None)
        fov_x = camera_params.get('camera_angle_x', None)
    
    if width is None or height is None:
        raise ValueError("Image dimensions not found in camera_params or frame_data")
    
    # 优先使用焦距参数
    if fx is not None and fy is not None:
        fx = float(fx)
        fy = float(fy)
    elif fov_x is not None:
        # 从视场角计算焦距
        fx = float(width) / (2.0 * np.tan(float(fov_x) / 2.0))
        fy = fx  # 假设正方形像素
    else:
        raise ValueError("No focal length or field of view found in camera_params or frame_data")
    
    # 主点（默认在图像中心）
    if cx is not None and cy is not None:
        cx = float(cx)
        cy = float(cy)
    else:
        cx = float(width) / 2.0
        cy = float(height) / 2.0
    
    # 构建内参矩阵
    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=np.float32)
    
    return K, int(width), int(height)


def unproject_depth_to_points(depth: np.ndarray, K: np.ndarray, c2w: np.ndarray, 
                              coordinate_system: str = 'PyTorch3D') -> np.ndarray:
    """
    将深度图反投影到3D点云（世界坐标系）
    
    坐标系说明：
    - Blender: +X右, +Y前, +Z上
    - PyTorch3D: +X左, +Y上, +Z前
    
    Args:
        depth: (H, W) 深度图
        K: (3, 3) 内参矩阵
        w2c: (4, 4) world-to-camera 变换矩阵
        coordinate_system: 'blender' 或 'pytorch3d'，指定输入 w2c 矩阵的坐标系

    Returns:
        point_map: (H, W, 3) PyTorch3D坐标系中的3D点
    """
    H, W = depth.shape

    # 创建像素网格
    u, v = np.meshgrid(np.arange(W), np.arange(H))

    # 反投影到相机坐标系
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    X = (u - cx) * depth / fx
    Y = -(v - cy) * depth / fy  # Y 轴翻转（图像坐标系到相机坐标系）
    Z = -depth  # 深度值（相机前方）

    # 相机坐标系中的点 (H, W, 3)
    points_cam = np.stack([X, Y, Z], axis=-1)


    
    # 如果输入是 Blender 坐标系，需要转换到 PyTorch3D 坐标系
    
    R, t = c2w[:3, :3], c2w[:3, 3]

    # 展平、变换、重塑
    points_cam_flat = points_cam.reshape(-1, 3)
    points_world_flat = (R @ points_cam_flat.T).T + t
    if coordinate_system == 'blender':
        # Blender → PyTorch3D 转换矩阵
        # Blender: +X右, +Y前, +Z上
        # PyTorch3D: +X左, +Y上, +Z前
        # 转换: PyTorch3D_X = -Blender_X, PyTorch3D_Y = Blender_Z, PyTorch3D_Z = Blender_Y
        blender_to_pytorch3d = np.array([
            [-1,  0,  0],
            [ 0,  0,  1],
            [ 0,  1,  0]
        ], dtype=np.float32)
        
        points_world_flat = (blender_to_pytorch3d @ points_world_flat.T).T

    point_map = points_world_flat.reshape(H, W, 3)


    

    return point_map






def unproject_depth_to_points2(depth: np.ndarray, K: np.ndarray, c2w: np.ndarray, 
                              coordinate_system: str = 'pytorch3d') -> np.ndarray:
    """
    将深度图反投影到3D点云（世界坐标系）
    
    坐标系说明：
    - Blender: +X右, +Y前, +Z上
    - PyTorch3D: +X左, +Y上, +Z前
    
    Args:
        depth: (H, W) 深度图
        K: (3, 3) 内参矩阵
        w2c: (4, 4) world-to-camera 变换矩阵
        coordinate_system: 'blender' 或 'pytorch3d'，指定输入 w2c 矩阵的坐标系

    Returns:
        point_map: (H, W, 3) PyTorch3D坐标系中的3D点
    """
    H, W = depth.shape

    # 创建像素网格
    u, v = np.meshgrid(np.arange(W), np.arange(H))

    # 反投影到相机坐标系
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    X = -(u - cx) * depth / fx
    Y = -(v - cy) * depth / fy  # Y 轴翻转（图像坐标系到相机坐标系）
    Z = depth  # 深度值（相机前方）

    # 相机坐标系中的点 (H, W, 3)
    points_cam = np.stack([X, Y, Z], axis=-1)


    
    # 如果输入是 Blender 坐标系，需要转换到 PyTorch3D 坐标系
    
    R, t = c2w[:3, :3], c2w[:3, 3]

    # 展平、变换、重塑
    points_cam_flat = points_cam.reshape(-1, 3)
    points_world_flat = points_cam_flat @ R + t

    point_map = points_world_flat.reshape(H, W, 3)


    

    return point_map



def unproject_depth_to_points3(depth: np.ndarray, K: np.ndarray, c2w: np.ndarray, 
                              coordinate_system: str = 'blender') -> np.ndarray:
    """
    将深度图反投影到3D点云（世界坐标系）
    
    坐标系说明：
    - Blender: +X右, +Y前, +Z上
    - PyTorch3D: +X左, +Y上, +Z前
    
    Args:
        depth: (H, W) 深度图
        K: (3, 3) 内参矩阵
        w2c: (4, 4) world-to-camera 变换矩阵
        coordinate_system: 'blender' 或 'pytorch3d'，指定输入 w2c 矩阵的坐标系

    Returns:
        point_map: (H, W, 3) PyTorch3D坐标系中的3D点
    """
    H, W = depth.shape

    # 创建像素网格
    u, v = np.meshgrid(np.arange(W), np.arange(H))

    # 反投影到相机坐标系
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    X = (u - cx) * depth / fx
    Y = -(v - cy) * depth / fy  # Y 轴翻转（图像坐标系到相机坐标系）
    Z = -depth  # 深度值（相机前方）

    # 相机坐标系中的点 (H, W, 3)
    points_cam = np.stack([X, Y, Z], axis=-1)


    
    # 如果输入是 Blender 坐标系，需要转换到 PyTorch3D 坐标系
    
    R, t = c2w[:3, :3], c2w[:3, 3]

    # 展平、变换、重塑
    points_cam_flat = points_cam.reshape(-1, 3)
    points_world_flat = (R @ points_cam_flat.T).T + t
    if coordinate_system == 'blender':
        # Blender → PyTorch3D 转换矩阵
        # Blender: +X右, +Y前, +Z上
        # PyTorch3D: +X左, +Y上, +Z前
        # 转换: PyTorch3D_X = -Blender_X, PyTorch3D_Y = Blender_Z, PyTorch3D_Z = Blender_Y
        blender_to_pytorch3d = np.array([
            [-1,  0,  0],
            [ 0,  1,  0],
            [ 0,  0,  -1]
        ], dtype=np.float32)
        
        points_world_flat = (blender_to_pytorch3d @ points_world_flat.T).T

    point_map = points_world_flat.reshape(H, W, 3)


    

    return point_map



def unproject_depth_to_points4(depth: np.ndarray, K: np.ndarray, c2w: np.ndarray, 
                              coordinate_system: str = 'blender') -> np.ndarray:
    """
    将深度图反投影到3D点云（世界坐标系）
    
    坐标系说明：
    - Blender: +X右, +Y前, +Z上
    - PyTorch3D: +X左, +Y上, +Z前
    
    Args:
        depth: (H, W) 深度图
        K: (3, 3) 内参矩阵
        w2c: (4, 4) world-to-camera 变换矩阵
        coordinate_system: 'blender' 或 'pytorch3d'，指定输入 w2c 矩阵的坐标系

    Returns:
        point_map: (H, W, 3) PyTorch3D坐标系中的3D点
    """
    H, W = depth.shape

    # 创建像素网格
    u, v = np.meshgrid(np.arange(W), np.arange(H))

    # 反投影到相机坐标系
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    X = (u - cx) * depth / fx
    Y = -(v - cy) * depth / fy  # Y 轴翻转（图像坐标系到相机坐标系）
    Z = -depth  # 深度值（相机前方）

    # 相机坐标系中的点 (H, W, 3)
    points_cam = np.stack([X, Y, Z], axis=-1)


    
    # 如果输入是 Blender 坐标系，需要转换到 PyTorch3D 坐标系
    
    R, t = c2w[:3, :3], c2w[:3, 3]

    # 展平、变换、重塑
    points_cam_flat = points_cam.reshape(-1, 3)
    points_world_flat = (R @ points_cam_flat.T).T + t
    if coordinate_system == 'blender':
        # Blender → PyTorch3D 转换矩阵
        # Blender: +X右, +Y前, +Z上
        # PyTorch3D: +X左, +Y上, +Z前
        # 转换: PyTorch3D_X = -Blender_X, PyTorch3D_Y = Blender_Z, PyTorch3D_Z = Blender_Y
        blender_to_pytorch3d = np.array([
            [-1,  0,  0],
            [ 0,  1,  0],
            [ 0,  0,  -1]
        ], dtype=np.float32)
        
        points_world_flat = (blender_to_pytorch3d @ points_world_flat.T).T

    point_map = points_world_flat.reshape(H, W, 3)


    

    return point_map


def load_image(image_path: str) -> Optional[np.ndarray]:
    """加载 RGB 图像"""
    if not os.path.exists(image_path):
        print(f"[WARN] Image not found: {image_path}")
        return None
    
    img = Image.open(image_path).convert('RGB')
    return np.array(img)


def extract_pointcloud_from_frames(
    frames: List[Dict],
    camera_params: Dict,
    base_dir: str,
    depth_dir: str = "depth",
    rgb_dir: str = "rgb",
    max_frames: int = -1,
    min_depth: float = 0.01,
    max_depth: float = 100.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    从多视角图像和深度图提取点云
    
    Args:
        frames: 帧列表
        camera_params: 相机参数
        base_dir: 数据集根目录
        depth_dir: 深度图子目录
        rgb_dir: RGB图像子目录
        max_frames: 最大处理帧数（-1表示全部）
        min_depth: 最小有效深度
        max_depth: 最大有效深度
        
    Returns:
        points: (N, 3) 点云坐标
        colors: (N, 3) 点云颜色 [0, 1]
    """
    all_points = []
    all_colors = []
    
    frame_limit = len(frames) if max_frames < 0 else min(max_frames, len(frames))
    
    for i, frame_data in enumerate(frames[:frame_limit]):
        print(f"[INFO] Processing frame {i+1}/{frame_limit}...")
        
        # 解析帧名称
        frame_name = frame_data.get('frame', frame_data.get('file_path', ''))
        if frame_name.startswith('./'):
            frame_name = frame_name[2:]
        frame_name = Path(frame_name).stem
        
        # 提取帧号
        frame_number_match = re.search(r'(\d+)$', frame_name)
        if frame_number_match:
            frame_number = frame_number_match.group(1)
            depth_filename = f"depth_{frame_number}.exr"
        else:
            depth_filename = f"{frame_name}.exr"
        
        # 构建路径
        depth_path = os.path.join(base_dir, depth_dir, depth_filename)
        
        # 查找 RGB 图像
        rgb_path = None
        search_dir = os.path.join(base_dir, rgb_dir)
        for ext in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG']:
            potential_path = os.path.join(search_dir, f"{frame_name}{ext}")
            if os.path.exists(potential_path):
                rgb_path = potential_path
                break
        
        if rgb_path is None:
            print(f"[WARN] RGB image not found for frame: {frame_name}")
            continue
        
        # 读取深度图
        try:
            depth = load_depth_exr(depth_path)
        except Exception as e:
            print(f"[ERROR] Failed to load depth for {frame_name}: {e}")
            continue
        
        # 读取 RGB 图像
        rgb_img = load_image(rgb_path)
        if rgb_img is None:
            continue
        
        color = rgb_img.astype(np.float32) / 255.0
        
        # 获取 w2c 矩阵
        w2c = None
        if 'c2w' in frame_data:
            c2w = np.array(frame_data['c2w'], dtype=np.float32)
            w2c = np.linalg.inv(c2w)
        elif 'w2c' in frame_data:
            w2c = np.array(frame_data['w2c'], dtype=np.float32)
        elif 'transform_matrix' in frame_data:
            transform_matrix = np.array(frame_data['transform_matrix'], dtype=np.float32)
            if 'camera_position' in frame_data:
                w2c = np.linalg.inv(transform_matrix)
            else:
                w2c = transform_matrix
        elif 'extrinsic' in frame_data:
            w2c = np.array(frame_data['extrinsic'], dtype=np.float32)
        else:
            print(f"[ERROR] No w2c or transform_matrix found for frame: {frame_name}")
            continue
        
        # 计算内参
        K, width, height = compute_intrinsics_from_params(camera_params, frame_data, rgb_img.shape[1], rgb_img.shape[0])
        
        # 确保尺寸一致
        if depth.shape[0] != rgb_img.shape[0] or depth.shape[1] != rgb_img.shape[1]:
            from scipy.ndimage import zoom
            scale_y = rgb_img.shape[0] / depth.shape[0]
            scale_x = rgb_img.shape[1] / depth.shape[1]
            depth = zoom(depth, (scale_y, scale_x), order=1)
        
        # 过滤有效深度
        valid = (depth > min_depth) & (depth < max_depth) & np.isfinite(depth)
        
        # 生成 3D point map (使用 Blender 坐标系转换)
        point_map = unproject_depth_to_points(depth, K, c2w, coordinate_system='blender')
        
        # 提取有效点
        valid_idx = np.where(valid)
        points = point_map[valid_idx]
        colors_frame = color[valid_idx]
        
        all_points.append(points)
        all_colors.append(colors_frame)
        
        print(f"[INFO] Frame {frame_name}: {len(points)} points")
    
    if not all_points:
        raise ValueError("No valid points extracted from any frame!")
    
    # 合并所有点
    points_combined = np.vstack(all_points)
    colors_combined = np.vstack(all_colors)
    
    print(f"\n[INFO] Total points: {len(points_combined)}")
    print(f"[INFO] Point bounds:")
    print(f"  X: [{points_combined[:, 0].min():.4f}, {points_combined[:, 0].max():.4f}]")
    print(f"  Y: [{points_combined[:, 1].min():.4f}, {points_combined[:, 1].max():.4f}]")
    print(f"  Z: [{points_combined[:, 2].min():.4f}, {points_combined[:, 2].max():.4f}]")
    
    # 调试：检查坐标系
    print(f"[DEBUG] Coordinate system check:")
    print(f"  Points with Z > 0: {np.sum(points_combined[:, 2] > 0)}")
    print(f"  Points with Z < 0: {np.sum(points_combined[:, 2] < 0)}")
    print(f"  Mean point: [{points_combined[:, 0].mean():.4f}, {points_combined[:, 1].mean():.4f}, {points_combined[:, 2].mean():.4f}]")
    
    return points_combined, colors_combined


def convert_blender_to_pytorch3d(
    position: Tuple[float, float, float],
    rotation_degrees: Optional[Tuple[float, float, float]] = None
) -> Tuple[Tuple[float, float, float], Optional[np.ndarray]]:
    """
    将 Blender 坐标系的位置和旋转转换为 PyTorch3D 坐标系
    
    坐标系定义：
    - Blender (右手系): +X右, +Y前, +Z上; 相机坐标系 +X右, +Y上, +Z朝内
    - PyTorch3D (左手系): +X左, +Y上, +Z前; 相机坐标系 +X左, +Y上, +Z朝外
    
    Args:
        position: (x, y, z) Blender 世界坐标系中的位置
        rotation_degrees: (rx, ry, rz) Blender 世界坐标系中的欧拉角旋转（度）
        
    Returns:
        position_pytorch3d: (x, y, z) PyTorch3D 世界坐标系中的位置
        rotation_matrix_pytorch3d: (3, 3) PyTorch3D 的 world-to-camera 旋转矩阵（如果提供了 rotation）
    """
    x_b, y_b, z_b = position
    
    # 1. 转换位置：Blender世界坐标 -> PyTorch3D世界坐标
    # PyTorch3D: +X左, +Y上, +Z前
    # Blender: +X右, +Y前, +Z上
    # 映射: PyTorch3D_X = -Blender_X, PyTorch3D_Y = Blender_Z, PyTorch3D_Z = Blender_Y
    x_p = -x_b  # 左 = -右
    y_p = z_b   # PyTorch3D的Y(上) = Blender的Z(上)
    z_p = y_b   # PyTorch3D的Z(前) = Blender的Y(前)
    
    position_pytorch3d = (x_p, y_p, z_p)
    
    if rotation_degrees is None:
        return position_pytorch3d, None
    
    # 2. 转换旋转
    rx_b, ry_b, rz_b = rotation_degrees
    
    # 在 Blender 坐标系中构建旋转矩阵
    rx_rad = np.radians(ry_b)
    ry_rad = np.radians(-rx_b)
    rz_rad = np.radians(rz_b+90)

    

    # Blender 坐标系中的旋转矩阵（世界坐标系下）
    Rx_b = np.array([
        [1, 0, 0],
        [0, np.cos(rx_rad), -np.sin(rx_rad)],
        [0, np.sin(rx_rad), np.cos(rx_rad)]
    ], dtype=np.float32)
    
    Ry_b = np.array([
        [np.cos(ry_rad), 0, np.sin(ry_rad)],
        [0, 1, 0],
        [-np.sin(ry_rad), 0, np.cos(ry_rad)]
    ], dtype=np.float32)
    
    Rz_b = np.array([
        [np.cos(rz_rad), -np.sin(rz_rad), 0],
        [np.sin(rz_rad), np.cos(rz_rad), 0],
        [0, 0, 1]
    ], dtype=np.float32)

    # R_default = default_Rz @ default_Ry @ default_Rx
    
    # 组合旋转（XYZ 顺序）
    R_blender = Rz_b @ Ry_b @ Rx_b

    # R_blender = R_blender @ R_default
    
    # 将 Blender 世界坐标系中的旋转矩阵转换到 PyTorch3D 世界坐标系
    # 坐标系转换矩阵
    blender_to_pytorch3d = np.array([
        [-1,  0,  0],  # PyTorch3D_X = -Blender_X
        [ 0,  0,  1],  # PyTorch3D_Y = Blender_Z
        [ 0,  1,  0]   # PyTorch3D_Z = Blender_Y
    ], dtype=np.float32)
    
    # 转换旋转矩阵: R_p = T @ R_b @ T^(-1)
    # 其中 T 是坐标系转换矩阵，T^(-1) 是其逆
    # 对于正交矩阵，T^(-1) = T^T
    R_pytorch3d = blender_to_pytorch3d @ R_blender @ blender_to_pytorch3d.T
    
    return position_pytorch3d, R_pytorch3d


def save_pointcloud_ply(points: np.ndarray, colors: np.ndarray, filepath: str):
    """
    保存点云为PLY格式
    
    Args:
        points: (N, 3) 点云坐标
        colors: (N, 3) 点云颜色 [0, 1]
        filepath: 输出PLY文件路径
    """
    import struct
    
    # 确保颜色在 [0, 255] 范围
    colors_uint8 = (colors * 255).astype(np.uint8)
    
    num_points = len(points)
    
    # 写入PLY文件
    with open(filepath, 'wb') as f:
        # 写入头部
        header = f"""ply
format binary_little_endian 1.0
element vertex {num_points}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
"""
        f.write(header.encode('ascii'))
        
        # 写入点云数据
        for i in range(num_points):
            # 位置 (float32 x 3)
            f.write(struct.pack('fff', points[i, 0], points[i, 1], points[i, 2]))
            # 颜色 (uint8 x 3)
            f.write(struct.pack('BBB', colors_uint8[i, 0], colors_uint8[i, 1], colors_uint8[i, 2]))
    
    print(f"[INFO] Saved point cloud to: {filepath} ({num_points} points)")


# ============================================================================
# 方法 1: 纯 PyTorch GPU 实现 (最快，适合大规模点云)
# ============================================================================

class TorchPanoramaRenderer:
    """
    高性能 GPU 全景渲染器
    
    特点：
    - 纯 PyTorch 实现，完全利用 GPU 并行
    - 支持 z-buffer 和 splatting 两种模式
    - 支持百万级点云实时渲染
    - 可微分（可用于优化）
    """
    
    def __init__(self, width: int = 2048, height: int = 1024, device: str = 'cuda'):
        self.width = width
        self.height = height
        self.device = torch.device(device)
        
    def render_equirectangular(
        self,
        points: torch.Tensor,  # (N, 3) 世界坐标
        colors: torch.Tensor,  # (N, 3) RGB [0, 1]
        cam_pos: Optional[torch.Tensor] = None,  # (3,)
        cam_rot: Optional[torch.Tensor] = None,  # (3, 3) rotation matrix
        splat_radius: int = 0,  # 0=point, >0=splat with radius
        background: Optional[torch.Tensor] = None,  # (H, W, 3) 背景图
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        渲染等距柱状投影全景图
        
        Args:
            points: (N, 3) 点云坐标
            colors: (N, 3) 点云颜色
            cam_pos: (3,) 相机位置，默认原点
            cam_rot: (3, 3) 相机旋转矩阵，默认单位矩阵
            splat_radius: 点扩散半径（像素），0表示点渲染
            background: 背景图像
            
        Returns:
            rgb: (H, W, 3) 渲染的全景图
            alpha: (H, W) 覆盖率/权重图
        """
        N = points.shape[0]
        if N == 0:
            empty = torch.zeros((self.height, self.width, 3), device=self.device)
            alpha = torch.zeros((self.height, self.width), device=self.device)
            return empty, alpha
            
        # 转换到相机坐标系
        if cam_pos is None:
            cam_pos = torch.zeros(3, device=self.device, dtype=points.dtype)
        if cam_rot is None:
            cam_rot = torch.eye(3, device=self.device, dtype=points.dtype)
            
        pts = points - cam_pos.unsqueeze(0)  # (N, 3)
        pts = (cam_rot @ pts.T).T  # (N, 3)
        
        # 过滤掉相机后面的点（假设相机朝向 +Z）
        valid_mask = pts[:, 2] > 1e-6
        pts = pts[valid_mask]
        cols = colors[valid_mask]
        
        if pts.shape[0] == 0:
            empty = torch.zeros((self.height, self.width, 3), device=self.device)
            alpha = torch.zeros((self.height, self.width), device=self.device)
            return empty, alpha
        
        # 球面坐标投影
        x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
        r = torch.norm(pts, dim=1)  # 距离
        theta = torch.asin(torch.clamp(y / (r + 1e-12), -1.0, 1.0))  # 纬度 [-π/2, π/2]
        phi = torch.atan2(x, z)  # 经度 [-π, π]
        
        # 映射到像素坐标
        u = (phi + pi) / (2 * pi) * self.width
        v = (pi / 2 - theta) / pi * self.height
        
        # 整数像素索引
        u_int = torch.round(u).long() % self.width
        v_int = torch.clamp(torch.round(v).long(), 0, self.height - 1)
        
        if splat_radius == 0:
            # 模式 1: 简单 z-buffer（最快）
            rgb, alpha = self._render_zbuffer(u_int, v_int, r, cols)
        else:
            # 模式 2: Splatting（更平滑，适合稀疏点云）
            rgb, alpha = self._render_splat(u, v, r, cols, splat_radius)
        
        # 应用背景
        if background is not None:
            bg = background.to(self.device)
            rgb = alpha.unsqueeze(-1) * rgb + (1 - alpha.unsqueeze(-1)) * bg
            
        return rgb, alpha
    
    def _render_zbuffer(
        self,
        u: torch.Tensor,  # (N,) 像素 x
        v: torch.Tensor,  # (N,) 像素 y
        depth: torch.Tensor,  # (N,) 深度
        colors: torch.Tensor,  # (N, 3)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Z-buffer 渲染（快速，适合密集点云）"""
        N = u.shape[0]
        
        # 展平索引
        pixel_idx = v * self.width + u  # (N,)
        
        # 初始化缓冲区
        num_pixels = self.height * self.width
        zbuffer = torch.full((num_pixels,), float('inf'), device=self.device, dtype=depth.dtype)
        rgb_flat = torch.zeros((num_pixels, 3), device=self.device, dtype=colors.dtype)
        alpha_flat = torch.zeros((num_pixels,), device=self.device, dtype=colors.dtype)
        
        # 使用 scatter_reduce 找到每个像素的最小深度
        # PyTorch 2.0+ 支持 scatter_reduce
        try:
            zbuffer = zbuffer.scatter_reduce(0, pixel_idx, depth, reduce='amin', include_self=True)
        except:
            # 降级到循环（较慢但兼容性好）
            for i in range(N):
                idx = pixel_idx[i]
                if depth[i] < zbuffer[idx]:
                    zbuffer[idx] = depth[i]
                    rgb_flat[idx] = colors[i]
                    alpha_flat[idx] = 1.0
            return rgb_flat.view(self.height, self.width, 3), alpha_flat.view(self.height, self.width)
        
        # 找到每个像素对应的最近点
        # 创建掩码：depth == zbuffer[pixel_idx]
        is_closest = (depth == zbuffer[pixel_idx])
        
        # 使用掩码选择颜色
        valid_indices = pixel_idx[is_closest]
        valid_colors = colors[is_closest]
        
        # Scatter 颜色
        rgb_flat.index_put_((valid_indices,), valid_colors, accumulate=False)
        alpha_flat.index_put_((valid_indices,), torch.ones_like(valid_indices, dtype=alpha_flat.dtype), accumulate=False)
        
        return rgb_flat.view(self.height, self.width, 3), alpha_flat.view(self.height, self.width)
    
    def _render_splat(
        self,
        u: torch.Tensor,  # (N,) 浮点像素 x
        v: torch.Tensor,  # (N,) 浮点像素 y
        depth: torch.Tensor,  # (N,) 深度
        colors: torch.Tensor,  # (N, 3)
        radius: int = 2,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Splat 渲染（平滑，适合稀疏点云）
        每个点扩散到周围半径像素内
        """
        N = u.shape[0]
        
        # 初始化累加缓冲区
        color_acc = torch.zeros((self.height, self.width, 3), device=self.device, dtype=colors.dtype)
        weight_acc = torch.zeros((self.height, self.width), device=self.device, dtype=colors.dtype)
        zbuffer = torch.full((self.height, self.width), float('inf'), device=self.device, dtype=depth.dtype)
        
        # 高斯核（可选：预计算）
        sigma = max(0.5, radius / 2.0)
        x = torch.arange(-radius, radius + 1, device=self.device, dtype=torch.float32)
        y = x.clone()
        xx, yy = torch.meshgrid(x, y, indexing='xy')
        kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma * sigma))
        kernel = kernel / (kernel.sum() + 1e-12)
        
        # 对每个点进行 splat（可优化为批处理）
        u_int = u.round().long()
        v_int = v.round().long()
        
        for i in range(N):
            cx, cy = u_int[i].item(), v_int[i].item()
            d = depth[i]
            
            # 计算 splat 区域
            x0, x1 = cx - radius, cx + radius + 1
            y0, y1 = cy - radius, cy + radius + 1
            
            # 裁剪到图像范围
            kx0 = max(0, -x0)
            ky0 = max(0, -y0)
            kx1 = kernel.shape[1] - max(0, x1 - self.width)
            ky1 = kernel.shape[0] - max(0, y1 - self.height)
            
            x0_clip = max(0, x0)
            y0_clip = max(0, y0)
            x1_clip = min(self.width, x1)
            y1_clip = min(self.height, y1)
            
            if x0_clip >= x1_clip or y0_clip >= y1_clip:
                continue
            
            # 获取核和区域
            k = kernel[ky0:ky1, kx0:kx1]
            region_z = zbuffer[y0_clip:y1_clip, x0_clip:x1_clip]
            
            # 深度测试（只接受更近的点）
            accept = d < (region_z + 1e-4)
            w = k * accept.float()
            
            # 累加
            weight_acc[y0_clip:y1_clip, x0_clip:x1_clip] += w
            color_acc[y0_clip:y1_clip, x0_clip:x1_clip] += w.unsqueeze(-1) * colors[i]
            zbuffer[y0_clip:y1_clip, x0_clip:x1_clip] = torch.where(accept, d, region_z)
        
        # 归一化
        mask = weight_acc > 0
        rgb = torch.zeros_like(color_acc)
        rgb[mask] = color_acc[mask] / weight_acc[mask].unsqueeze(-1)
        alpha = torch.clamp(weight_acc, 0.0, 1.0)
        
        return rgb, alpha


# ============================================================================
# 方法 2: PyTorch3D 实现（高质量，可微分）
# ============================================================================

def render_panorama_pytorch3d(
    points: torch.Tensor,
    colors: torch.Tensor,
    cam_pos: torch.Tensor,
    cam_rot: Optional[torch.Tensor] = None,
    width: int = 2048,
    height: int = 1024,
    method: str = 'cubemap',  # 'cubemap' 或 'equirect'
    point_radius: float = 0.01,
    points_per_pixel: int = 10,
) -> torch.Tensor:
    """
    使用 PyTorch3D 渲染全景图
    
    方法：
    1. cubemap: 渲染 6 个面然后转换为 equirectangular
    2. equirect: 直接渲染 equirectangular（需要自定义相机）
    
    Args:
        points: (N, 3) 点云
        colors: (N, 3) 颜色
        cam_pos: (3,) 相机位置
        cam_rot: (3, 3) 可选的相机旋转矩阵（world-to-camera），用于改变全景图朝向
                 如果为 None，使用默认朝向（看向 +Z）
        width, height: 输出尺寸
        method: 'cubemap' 或 'equirect'
        point_radius: 点渲染半径
        points_per_pixel: 每像素点数
    
    Returns:
        rgb: (H, W, 3) 全景图
    """
    try:
        from pytorch3d.structures import Pointclouds
        from pytorch3d.renderer import (
            PointsRasterizationSettings,
            PointsRenderer,
            PointsRasterizer,
            AlphaCompositor,
            look_at_view_transform,
            FoVPerspectiveCameras,
        )
    except ImportError:
        raise ImportError(
            "PyTorch3D 未安装。请运行: pip install pytorch3d\n"
            "或访问 https://github.com/facebookresearch/pytorch3d"
        )
    
    if method == 'cubemap':
        # 渲染 6 个立方体面（+X, -X, +Y, -Y, +Z, -Z）
        face_size = height  # 每个面的尺寸
        faces = []
        
        # 直接定义6个面的相机旋转矩阵
        # PyTorch3D坐标系：+X左，+Y上，+Z前（相机看向+Z）
        # PyTorch3D 的 R 是 world-to-camera 变换矩阵
        # 
        # 标准 cubemap 定义（OpenGL/WebGL 约定）：
        # - 所有面的 up 方向一致（除了 top/bottom）
        # - 使用右手坐标系
        # 
        # 对于旋转矩阵 R，满足 p_camera = R @ p_world
        # PyTorch3D 中，R.T 的列是相机轴在世界坐标系中的方向
        # 即：R.T = [cam_x_world, cam_y_world, cam_z_world]
        # 所以：R = [cam_x_world, cam_y_world, cam_z_world].T
        
        # 定义每个面的相机朝向（世界坐标系中的方向）
        # 格式：(cam_x_left, cam_y_up, cam_z_forward, name)
        # PyTorch3D 坐标系: +X左, +Y上, +Z前
        # 相机坐标系: cam_X 是左方向, cam_Y 是上方向, cam_Z 是看向方向(前方)
        # 使用右手定则验证: up × forward = left
        camera_directions = [
            # Face 0: 看向 +X (世界坐标系的左方)
            # up × forward = (+Y) × (+X) = -Z
            # left=-Z, up=+Y, forward=+X
            ([0, 0, -1], [0, 1, 0], [1, 0, 0], "face0(+X)"),
            
            # Face 1: 看向 -X (世界坐标系的右方)
            # up × forward = (+Y) × (-X) = +Z
            # left=+Z, up=+Y, forward=-X
            ([0, 0, 1], [0, 1, 0], [-1, 0, 0], "face1(-X)"),
            
            # Face 2: 看向 +Y (世界坐标系的上方)
            # up × forward = (-Z) × (+Y) = +X
            # left=+X, up=-Z, forward=+Y
            ([1, 0, 0], [0, 0, -1], [0, 1, 0], "face2(+Y)"),
            
            # Face 3: 看向 -Y (世界坐标系的下方)
            # up × forward = (+Z) × (-Y) = +X
            # left=+X, up=+Z, forward=-Y
            ([1, 0, 0], [0, 0, 1], [0, -1, 0], "face3(-Y)"),
            
            # Face 4: 看向 +Z (世界坐标系的前方)
            # up × forward = (+Y) × (+Z) = +X
            # left=+X, up=+Y, forward=+Z
            ([1, 0, 0], [0, 1, 0], [0, 0, 1], "face4(+Z)"),
            
            # Face 5: 看向 -Z (世界坐标系的后方)
            # up × forward = (+Y) × (-Z) = -X
            # left=-X, up=+Y, forward=-Z
            ([-1, 0, 0], [0, 1, 0], [0, 0, -1], "face5(-Z)"),
        ]
        
        device = points.device
        point_cloud = Pointclouds(points=[points], features=[colors])
        
        # 将相机位置转换为列表形式（用于平移点云）
        if isinstance(cam_pos, torch.Tensor):
            cam_pos_np = cam_pos.cpu().numpy()
        else:
            cam_pos_np = np.array(cam_pos)
        
        # 将点云平移，使相机位置成为原点
        # 注意：点云不做旋转，旋转应该施加在相机上
        points_centered = points - torch.from_numpy(cam_pos_np).to(device).float()
        point_cloud_centered = Pointclouds(points=[points_centered], features=[colors])
        
        # 检查点云是否为空，避免对空张量调用 max/min
        if points_centered.shape[0] == 0:
            # 返回空白全景图
            faces = [torch.zeros((face_size, face_size, 3), device=device, dtype=colors.dtype) for _ in range(6)]
            panorama = _cubemap_to_equirect(faces, width, height, device)
            return panorama
        
        # 如果提供了 cam_rot，将其转换到 device
        cam_rot_device = None
        if cam_rot is not None:
            cam_rot_device = cam_rot.to(device)
        
        # 计算点云的尺度
        max_extent = max(
            points_centered[:, 0].max() - points_centered[:, 0].min(),
            points_centered[:, 1].max() - points_centered[:, 1].min(),
            points_centered[:, 2].max() - points_centered[:, 2].min()
        )
        
        # 相机在原点，通过旋转矩阵设置不同朝向
        for i, (cam_x, cam_y, cam_z, face_name) in enumerate(camera_directions):
            
            # 构建基础相机旋转矩阵 R_base (world-to-camera)
            # R.T 的列向量是相机坐标系的轴在世界坐标系中的方向
            # 所以 R = [cam_x, cam_y, cam_z].T
            R_base = torch.tensor([cam_x, cam_y, cam_z], dtype=torch.float32, device=device).T  # (3, 3)
            
            # 如果提供了 cam_rot，将其应用到相机上
            # cam_rot 是用户指定的额外旋转（world-to-camera）
            # 最终的相机旋转 R = cam_rot @ R_base
            # 这样相当于先应用基础朝向，再应用用户旋转
            if cam_rot_device is not None:
                R = (cam_rot_device @ R_base).unsqueeze(0)  # (1, 3, 3)
            else:
                R = R_base.unsqueeze(0)  # (1, 3, 3)
            
            # 验证 R 是正交矩阵（旋转矩阵必须满足）
            R_check = R[0]  # (3, 3)
            identity_check = torch.matmul(R_check, R_check.T)
            det = torch.det(R_check)
            
            
            # 相机位置在原点
            T = torch.zeros((1, 3), dtype=torch.float32, device=device)
            
            cameras = FoVPerspectiveCameras(device=device, R=R, T=T, fov=90)
            
            raster_settings = PointsRasterizationSettings(
                image_size=face_size,
                radius=point_radius,
                points_per_pixel=points_per_pixel,
                bin_size=0,  # 使用 naive rasterization 避免 bin size 溢出
            )
            
            rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
            renderer = PointsRenderer(rasterizer=rasterizer, compositor=AlphaCompositor())
            
            face_image = renderer(point_cloud_centered)  # 使用居中的点云！
            faces.append(face_image[0, ..., :3])  # (H, W, 3)
            
            # 调试：保存每个面的图像和alpha掩码
            # face_rgb = face_image[0, ..., :3]
            # if face_image.shape[-1] > 3:
            #     face_alpha = face_image[0, ..., 3]

            
            # # 保存调试图像
            # from PIL import Image
            # import numpy as np
            
            # # 保存RGB图像
            # face_rgb_np = (face_rgb.cpu().numpy() * 255).astype(np.uint8)
            # face_img = Image.fromarray(face_rgb_np)
            # face_img.save(f"debug_face_{i+1}_{face_name.replace('(', '').replace(')', '').replace('+', 'p').replace('-', 'n')}.png")
            
            # # 保存alpha掩码（如果有的话）
            # if face_image.shape[-1] > 3:
            #     face_alpha_np = (face_alpha.cpu().numpy() * 255).astype(np.uint8)
            #     alpha_img = Image.fromarray(face_alpha_np, mode='L')
            #     alpha_img.save(f"debug_face_{i+1}_{face_name.replace('(', '').replace(')', '').replace('+', 'p').replace('-', 'n')}_alpha.png")
        
        # 创建cubemap可视化（十字形布局）
        # _save_cubemap_cross(faces, device)
        
        # 将 cubemap 转换为 equirectangular
        panorama = _cubemap_to_equirect(faces, width, height, device)
        return panorama
    
    else:
        raise NotImplementedError("Direct equirect rendering requires custom camera. Use cubemap method.")


def _save_cubemap_cross(faces: list, device: torch.device, filename: str = "debug_cubemap_cross.png"):
    """
    将6个cubemap面保存为十字形布局，便于可视化
    
    布局：
        [ ]  [+Y]  [ ]  [ ]
        [-X] [-Z]  [+X] [+Z]
        [ ]  [-Y]  [ ]  [ ]
    
    Args:
        faces: [+X, -X, +Y, -Y, +Z, -Z]
        device: torch device
        filename: 输出文件名
    """
    from PIL import Image
    import numpy as np
    
    face_size = faces[0].shape[0]
    
    # 创建 3x4 的布局
    cross = torch.zeros((face_size * 3, face_size * 4, 3), device=device)
    
    # 放置各个面：按照新的布局
    # faces 数组顺序: [+X, -X, +Y, -Y, +Z, -Z]
    #
    # 布局：
    #     [ ]  [+Y]  [ ]  [ ]
    #     [-X] [-Z]  [+X] [+Z]
    #     [ ]  [-Y]  [ ]  [ ]
    #
    cross[0:face_size, face_size:face_size*2] = faces[2]              # +Y (top)
    cross[face_size:face_size*2, 0:face_size] = faces[1]              # -X (left)
    cross[face_size:face_size*2, face_size:face_size*2] = faces[5]    # -Z (center)
    cross[face_size:face_size*2, face_size*2:face_size*3] = faces[0]  # +X (right)
    cross[face_size:face_size*2, face_size*3:face_size*4] = faces[4]  # +Z (far right)
    cross[face_size*2:face_size*3, face_size:face_size*2] = faces[3]  # -Y (bottom)
    
    # 转换为numpy并保存
    cross_np = (cross.cpu().numpy() * 255).astype(np.uint8)
    img = Image.fromarray(cross_np)
    img.save(filename)
    print(f"[DEBUG] Saved cubemap cross layout to: {filename}")


def _cubemap_to_equirect(
    faces: list,  # 6 个 (H, W, 3) 的面
    out_width: int,
    out_height: int,
    device: torch.device,
) -> torch.Tensor:
    """
    将 cubemap 转换为 equirectangular 投影
    
    布局顺序（水平环绕）: [-X] [-Z] [+X] [+Z]
    
    Args:
        faces: 6 个面的列表 [+X, -X, +Y, -Y, +Z, -Z]
        out_width, out_height: 输出全景图尺寸
    
    Returns:
        panorama: (out_height, out_width, 3)
    """

    
    # 生成全景图的每个像素对应的 3D 方向
    u = torch.linspace(0, 1, out_width, device=device)
    v = torch.linspace(0, 1, out_height, device=device)
    u, v = torch.meshgrid(u, v, indexing='xy')  # (H, W)
    
    # Equirectangular 投影
    # phi: 经度，从右到左 [0, 2π]
    # theta: 纬度 [π/2, -π/2]（从上到下）
    phi = (1 - u) * 2 * pi  # 反转方向: u=0 → phi=2π, u=1 → phi=0
    theta = (0.5 - v) * pi  # [π/2, -π/2]
    
    # 转换为 3D 方向向量（球坐标到笛卡尔坐标）
    # PyTorch3D 坐标系: +X左, +Y上, +Z前
    # phi=0 对应 +Z (前), phi=π/2 对应 +X (左), phi=π 对应 -Z (后), phi=3π/2 对应 -X (右)
    x = torch.cos(theta) * torch.sin(phi)
    y = torch.sin(theta)
    z = torch.cos(theta) * torch.cos(phi)
    
    # 确定每个方向属于哪个面，并采样
    abs_x = torch.abs(x)
    abs_y = torch.abs(y)
    abs_z = torch.abs(z)
    
    panorama = torch.zeros((out_height, out_width, 3), device=device)
    
    # 统计每个面被采样的像素数
    face_pixel_counts = []
    
    # Face 0 (+X): 看向 +X (世界左方)
    # camera_directions: left=-Z, up=+Y, forward=+X
    mask = (abs_x >= abs_y) & (abs_x >= abs_z) & (x > 0)
    if mask.any():
        # 在相机图像中: U从left到right, V从up到down
        # left=-Z: 当Z=-1时U=0, 当Z=+1时U=1 → u = (z + 1) / 2
        # up=+Y:   当Y=+1时V=0, 当Y=-1时V=1 → v = (-y + 1) / 2
        # 但是需要除以x来归一化到面上
        u_face = (z[mask] / x[mask] + 1) / 2
        v_face = (-y[mask] / x[mask] + 1) / 2
        panorama[mask] = _sample_face(faces[0], u_face, v_face)
        face_pixel_counts.append(("Face0(+X)", mask.sum().item()))
    else:
        face_pixel_counts.append(("Face0(+X)", 0))
    
    # Face 1 (-X): 看向 -X (世界右方)
    # camera_directions: left=+Z, up=+Y, forward=-X
    mask = (abs_x >= abs_y) & (abs_x >= abs_z) & (x < 0)
    if mask.any():
        # left=+Z: 当Z=+1时U=0, 当Z=-1时U=1 → u = (-z + 1) / 2
        # up=+Y:   当Y=+1时V=0, 当Y=-1时V=1 → v = (-y + 1) / 2
        # 除以 -x (因为 x < 0)
        u_face = (-z[mask] / -x[mask] + 1) / 2
        v_face = (-y[mask] / -x[mask] + 1) / 2
        panorama[mask] = _sample_face(faces[1], u_face, v_face)
        face_pixel_counts.append(("Face1(-X)", mask.sum().item()))
    else:
        face_pixel_counts.append(("Face1(-X)", 0))
    
    # Face 2 (+Y): 看向 +Y (世界上方)
    # camera_directions: left=+X, up=-Z, forward=+Y
    mask = (abs_y >= abs_x) & (abs_y >= abs_z) & (y > 0)
    if mask.any():
        # left=+X: 当X=+1时U=0, 当X=-1时U=1 → u = (-x + 1) / 2
        # up=-Z:   当Z=-1时V=0, 当Z=+1时V=1 → v = (z + 1) / 2
        # 除以 y
        u_face = (-x[mask] / y[mask] + 1) / 2
        v_face = (z[mask] / y[mask] + 1) / 2
        panorama[mask] = _sample_face(faces[2], u_face, v_face)
        face_pixel_counts.append(("Face2(+Y)", mask.sum().item()))
    else:
        face_pixel_counts.append(("Face2(+Y)", 0))
    
    # Face 3 (-Y): 看向 -Y (世界下方)
    # camera_directions: left=+X, up=+Z, forward=-Y
    mask = (abs_y >= abs_x) & (abs_y >= abs_z) & (y < 0)
    if mask.any():
        # left=+X: 当X=+1时U=0, 当X=-1时U=1 → u = (-x + 1) / 2
        # up=+Z:   当Z=+1时V=0, 当Z=-1时V=1 → v = (-z + 1) / 2
        # 除以 -y (因为 y < 0)
        u_face = (-x[mask] / -y[mask] + 1) / 2
        v_face = (-z[mask] / -y[mask] + 1) / 2
        panorama[mask] = _sample_face(faces[3], u_face, v_face)
        face_pixel_counts.append(("Face3(-Y)", mask.sum().item()))
    else:
        face_pixel_counts.append(("Face3(-Y)", 0))
    
    # Face 4 (+Z): 看向 +Z (世界前方)
    # camera_directions: left=+X, up=+Y, forward=+Z
    mask = (abs_z >= abs_x) & (abs_z >= abs_y) & (z > 0)
    if mask.any():
        # left=+X: 当X=+1时U=0, 当X=-1时U=1 → u = (-x + 1) / 2
        # up=+Y:   当Y=+1时V=0, 当Y=-1时V=1 → v = (-y + 1) / 2
        # 除以 z
        u_face = (-x[mask] / z[mask] + 1) / 2
        v_face = (-y[mask] / z[mask] + 1) / 2
        panorama[mask] = _sample_face(faces[4], u_face, v_face)
        face_pixel_counts.append(("Face4(+Z)", mask.sum().item()))
    else:
        face_pixel_counts.append(("Face4(+Z)", 0))
    
    # Face 5 (-Z): 看向 -Z (世界后方)
    # camera_directions: left=-X, up=+Y, forward=-Z
    mask = (abs_z >= abs_x) & (abs_z >= abs_y) & (z < 0)
    if mask.any():
        # left=-X: 当X=-1时U=0, 当X=+1时U=1 → u = (x + 1) / 2
        # up=+Y:   当Y=+1时V=0, 当Y=-1时V=1 → v = (-y + 1) / 2
        # 除以 -z (因为 z < 0)
        u_face = (x[mask] / -z[mask] + 1) / 2
        v_face = (-y[mask] / -z[mask] + 1) / 2
        panorama[mask] = _sample_face(faces[5], u_face, v_face)
        face_pixel_counts.append(("Face5(-Z)", mask.sum().item()))
    else:
        face_pixel_counts.append(("Face5(-Z)", 0))
        face_pixel_counts.append(("Face5(-Z)", 0))
    
    # 打印统计信息
    # print(f"[DEBUG] Cubemap face pixel distribution:")
    for face_name, count in face_pixel_counts:
        percentage = count / (out_width * out_height) * 100
        # print(f"  {face_name}: {count} pixels ({percentage:.1f}%)")
    
    return panorama


def _sample_face(face: torch.Tensor, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    从一个面采样颜色（双线性插值）
    
    Args:
        face: (H, W, 3)
        u, v: (N,) 归一化坐标 [0, 1]
    
    Returns:
        colors: (N, 3)
    """
    H, W = face.shape[:2]
    x = u * (W - 1)
    y = v * (H - 1)
    
    x0 = torch.floor(x).long().clamp(0, W - 2)
    x1 = x0 + 1
    y0 = torch.floor(y).long().clamp(0, H - 2)
    y1 = y0 + 1
    
    wx = x - x0.float()
    wy = y - y0.float()
    
    c00 = face[y0, x0]
    c01 = face[y0, x1]
    c10 = face[y1, x0]
    c11 = face[y1, x1]
    
    color = (
        c00 * (1 - wx).unsqueeze(-1) * (1 - wy).unsqueeze(-1) +
        c01 * wx.unsqueeze(-1) * (1 - wy).unsqueeze(-1) +
        c10 * (1 - wx).unsqueeze(-1) * wy.unsqueeze(-1) +
        c11 * wx.unsqueeze(-1) * wy.unsqueeze(-1)
    )
    
    return color


# ============================================================================
# 示例和测试代码
# ============================================================================

def create_test_pointcloud(N: int = 100000) -> Tuple[torch.Tensor, torch.Tensor]:
    """创建一个测试用的球形点云"""
    phi = torch.rand(N) * 2 * pi - pi
    theta = torch.rand(N) * pi - pi / 2
    r = 1.0 + 0.05 * torch.randn(N)
    
    x = r * torch.cos(theta) * torch.sin(phi)
    y = r * torch.sin(theta)
    z = r * torch.cos(theta) * torch.cos(phi)
    
    points = torch.stack([x, y, z], dim=1)
    colors = (points - points.min()) / (points.max() - points.min())
    
    return points, colors


def render_panorama_from_mvimages(
    data_dir: str,
    transform_json: str = "transform.json",
    depth_dir: str = "depth",
    rgb_dir: str = "rgb",
    output_path: str = "panorama.png",
    max_frames: int = -1,
    min_depth: float = 0.01,
    max_depth: float = 100.0,
    render_width: int = 2048,
    render_height: int = 1024,
    point_radius: float = 0.01,
    points_per_pixel: int = 10,
    cam_pos: Optional[Tuple[float, float, float]] = None,
    cam_rot: Optional[np.ndarray] = None,
    device: str = 'cuda',
):
    """
    完整流程：从多视角图像生成点云并渲染全景图（使用 PyTorch3D）
    
    Args:
        data_dir: 数据集根目录
        transform_json: transform.json 文件名或路径
        depth_dir: 深度图子目录
        rgb_dir: RGB图像子目录
        output_path: 输出全景图路径
        max_frames: 最大处理帧数（-1表示全部）
        min_depth: 最小有效深度
        max_depth: 最大有效深度
        render_width: 渲染图像宽度
        render_height: 渲染图像高度
        point_radius: 点渲染半径（PyTorch3D 参数）
        points_per_pixel: 每像素点数（PyTorch3D 参数）
        cam_pos: 全景相机位置 (x, y, z)，None表示原点
        cam_rot: 全景相机旋转矩阵 (3, 3) world-to-camera，None表示默认朝向
        device: 渲染设备
    """
    print("=" * 80)
    print("从多视角图像渲染全景图 (PyTorch3D)")
    print("=" * 80)
    
    # 1. 构建 transform.json 路径
    if os.path.isabs(transform_json):
        transform_path = transform_json
    else:
        potential_paths = [
            os.path.join(data_dir, rgb_dir, transform_json),
            os.path.join(data_dir, transform_json),
        ]
        
        transform_path = None
        for path in potential_paths:
            if os.path.exists(path):
                transform_path = path
                break
        
        if transform_path is None:
            raise FileNotFoundError(f"Cannot find {transform_json} in {data_dir}")
    
    print(f"[INFO] Using transform.json: {transform_path}")
    
    # 2. 加载 transform.json
    frames, camera_params = load_transform_json(transform_path)
    
    # 3. 从多视角图像提取点云
    print(f"\n[STEP 1] Extracting point cloud from multi-view images...")
    points_np, colors_np = extract_pointcloud_from_frames(
        frames=frames,
        camera_params=camera_params,
        base_dir=data_dir,
        depth_dir=depth_dir,
        rgb_dir=rgb_dir,
        max_frames=max_frames,
        min_depth=min_depth,
        max_depth=max_depth,
    )
    
    # 保存点云
    pointcloud_path = output_path.replace('.png', '_pointcloud.ply')
    save_pointcloud_ply(points_np, colors_np, pointcloud_path)
    
    # 4. 转换为 torch tensors
    print(f"\n[STEP 2] Preparing for rendering on {device}...")
    points = torch.from_numpy(points_np).float().to(device)
    colors = torch.from_numpy(colors_np).float().to(device)
    
    # 5. 设置相机位置和旋转
    if cam_pos is None:
        cam_pos = (0.0, 0.0, 0.0)
    cam_pos_tensor = torch.tensor(cam_pos, dtype=torch.float32, device=device)
    
    cam_rot_tensor = None
    if cam_rot is not None:
        cam_rot_tensor = torch.from_numpy(cam_rot).float()
        print(f"[INFO] Using camera rotation matrix:")
        print(cam_rot)
    
    # 6. 使用 PyTorch3D 渲染全景图
    print(f"\n[STEP 3] Rendering panorama with PyTorch3D ({render_height}x{render_width})...")
    print(f"[INFO] Point radius: {point_radius}, Points per pixel: {points_per_pixel}")
    
    import time
    start = time.time()
    
    rgb = render_panorama_pytorch3d(
        points=points,
        colors=colors,
        cam_pos=cam_pos_tensor,
        cam_rot=cam_rot_tensor,
        width=render_width,
        height=render_height,
        method='cubemap',
        point_radius=point_radius,
        points_per_pixel=points_per_pixel,
    )
    
    if device == 'cuda':
        torch.cuda.synchronize()
    elapsed = time.time() - start
    
    print(f"[INFO] Rendering completed in {elapsed:.3f}s")
    
    # 7. 保存结果
    rgb_np = (rgb.cpu().numpy() * 255).astype(np.uint8)
    
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    Image.fromarray(rgb_np).save(output_path)
    
    print(f"\n[SUCCESS] Saved panorama to: {output_path}")
    
    return rgb_np


def main():
    parser = argparse.ArgumentParser(
        description="从多视角图像(MV-images)反投影生成点云并渲染全景图"
    )
    
    # 输入数据
    parser.add_argument("--data_dir", type=str, required=True, 
                       help="数据集根目录（包含 rgb 和 depth 子目录）")
    parser.add_argument("--transform_json", type=str, default="transform.json", 
                       help="transform.json 文件名或路径")
    parser.add_argument("--depth_dir", type=str, default="depth", 
                       help="深度图子目录")
    parser.add_argument("--rgb_dir", type=str, default="rgb", 
                       help="RGB图像子目录")
    
    # 点云提取参数
    parser.add_argument("--max_frames", type=int, default=2, 
                       help="最大处理帧数（-1表示全部）")
    parser.add_argument("--min_depth", type=float, default=0.01, 
                       help="最小有效深度")
    parser.add_argument("--max_depth", type=float, default=100.0, 
                       help="最大有效深度")
    
    # 渲染参数
    parser.add_argument("--output", type=str, default="panorama.png", 
                       help="输出全景图路径")
    parser.add_argument("--render_width", type=int, default=1024, 
                       help="渲染图像宽度")
    parser.add_argument("--render_height", type=int, default=512, 
                       help="渲染图像高度")
    parser.add_argument("--point_radius", type=float, default=0.01, 
                       help="点渲染半径（PyTorch3D 参数）")
    parser.add_argument("--points_per_pixel", type=int, default=10, 
                       help="每像素点数（PyTorch3D 参数）")
    parser.add_argument("--cam_pos", type=str, default="-0.416674, -2.210811, -0.202782", 
                       help="全景相机位置 (x,y,z)，留空则使用原点")
    parser.add_argument("--cam_rot", type=str, default="0, 0, 71.62762000628658",
                       help="全景相机旋转角度 (rx,ry,rz) 单位为度，分别表示绕X、Y、Z轴的旋转。例如: '0,90,0' 表示绕Y轴旋转90度")
    parser.add_argument("--input_coordinate_system", type=str, default="blender", choices=["blender", "pytorch3d"],
                       help="输入的 cam_pos 和 cam_rot 所在的坐标系：'blender' (右手系: +X右 +Y前 +Z上) 或 'pytorch3d' (左手系: +X左 +Y上 +Z前)")
    parser.add_argument("--device", type=str, default="cuda", 
                       help="渲染设备（cuda 或 cpu）")
    parser.add_argument("--save_pointcloud", action="store_true",
                       help="保存生成的点云为PLY格式（默认自动保存）")
    
    # 测试模式
    parser.add_argument("--test", action="store_true", 
                       help="运行测试模式（生成测试点云）")
    
    args = parser.parse_args()
    

    cam_pos = None
    cam_pos_raw = None
    if args.cam_pos:
        try:
            coords = [float(x.strip()) for x in args.cam_pos.split(',')]
            if len(coords) != 3:
                raise ValueError("cam_pos must have 3 values")
            cam_pos_raw = tuple(coords)
            print(f"[INFO] Input camera position ({args.input_coordinate_system}): {cam_pos_raw}")
        except ValueError as e:
            print(f"[ERROR] Invalid cam_pos format: {e}")
            return
    
    cam_rot = None
    cam_rot_angles = None
    if args.cam_rot:
        try:
            angles = [float(x.strip()) for x in args.cam_rot.split(',')]
            if len(angles) != 3:
                raise ValueError("cam_rot must have 3 values (rx, ry, rz in degrees)")
            cam_rot_angles = tuple(angles)
            print(f"[INFO] Input camera rotation ({args.input_coordinate_system}): rx={angles[0]:.1f}°, ry={angles[1]:.1f}°, rz={angles[2]:.1f}°")
        except ValueError as e:
            print(f"[ERROR] Invalid cam_rot format: {e}")
            return
    
    # 坐标系转换
    if args.input_coordinate_system == 'blender':
        print(f"\n[INFO] Converting from Blender coordinate system to PyTorch3D...")
        if cam_pos_raw is not None:
            cam_pos, cam_rot = convert_blender_to_pytorch3d(cam_pos_raw, cam_rot_angles)
        else:
            cam_pos = None
            if cam_rot_angles is not None:
                _, cam_rot = convert_blender_to_pytorch3d((0, 0, 0), cam_rot_angles)
    else:
        # 已经是 PyTorch3D 坐标系
        cam_pos = cam_pos_raw
        if cam_rot_angles is not None:
            # 直接构建旋转矩阵
            rx, ry, rz = [np.radians(a) for a in cam_rot_angles]
            
            Rx = np.array([
                [1, 0, 0],
                [0, np.cos(rx), -np.sin(rx)],
                [0, np.sin(rx), np.cos(rx)]
            ], dtype=np.float32)
            
            Ry = np.array([
                [np.cos(ry), 0, np.sin(ry)],
                [0, 1, 0],
                [-np.sin(ry), 0, np.cos(ry)]
            ], dtype=np.float32)
            
            Rz = np.array([
                [np.cos(rz), -np.sin(rz), 0],
                [np.sin(rz), np.cos(rz), 0],
                [0, 0, 1]
            ], dtype=np.float32)
            
            cam_rot = Rz @ Ry @ Rx
            
            print(f"[INFO] Using camera rotation angles (XYZ order): rx={cam_rot_angles[0]:.1f}°, ry={cam_rot_angles[1]:.1f}°, rz={cam_rot_angles[2]:.1f}°")
            
            # 验证是否为旋转矩阵
            det = np.linalg.det(cam_rot)
            if abs(det - 1.0) > 0.01:
                print(f"[WARN] Rotation matrix determinant is {det:.6f}, should be close to 1.0")
    
    if cam_rot is not None:
        print(f"[INFO] Final rotation matrix (PyTorch3D coordinate system):")
        print(cam_rot)
    
    render_panorama_from_mvimages(
        data_dir=args.data_dir,
        transform_json=args.transform_json,
        depth_dir=args.depth_dir,
        rgb_dir=args.rgb_dir,
        output_path=args.output,
        max_frames=args.max_frames,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        render_width=args.render_width,
        render_height=args.render_height,
        point_radius=args.point_radius,
        points_per_pixel=args.points_per_pixel,
        cam_pos=cam_pos,
        cam_rot=cam_rot,
        device=args.device,
    )

    print("\n所有任务完成！")


def render_panorama_from_batch(
    batch_data,
    batch_idx=0,
    render_all=False,
    width=2048,
    height=1024,
    point_radius=0.01,
    points_per_pixel=10,
    device='cuda',
    save_ply=False,
    ply_output_dir=None,
    coordinate_system='blender',
):
    """
    Fuse the depth maps of all views into one point cloud and render an
    equirectangular panorama at the envmap position.

    Args:
        batch_data: dict containing
            - rgb_images:      List[List[Tensor(3, H, W)]] in [0, 1]
            - depth_maps:      List[List[Tensor(H, W)]]
            - intrinsics:      List[List[dict(fx, fy, cx, cy)]]
            - extrinsics:      List[List[np.ndarray(4, 4)]] camera-to-world
            - envmap_position: List[np.ndarray(3,)]
            - envmap_rotation: List[tuple(rx, ry, rz)] in degrees
        batch_idx: sample index used when render_all=False
        render_all: render every sample in the batch
        width, height: panorama resolution
        point_radius, points_per_pixel: PyTorch3D point rasterization settings
        save_ply: save the fused point cloud to `ply_output_dir/point_cloud.ply`

    Returns:
        render_all=False: Tensor (H, W, 3) in [0, 1]
        render_all=True:  Tensor (B, 1, 3, H, W) in [0, 1]
    """
    batch_indices = range(len(batch_data['rgb_images'])) if render_all else [batch_idx]
    panoramas = []

    for idx in batch_indices:
        rgb_images = batch_data['rgb_images'][idx]
        depth_maps = batch_data['depth_maps'][idx]
        intrinsics = batch_data['intrinsics'][idx]
        extrinsics = batch_data['extrinsics'][idx]
        envmap_position = batch_data['envmap_position'][idx]
        envmap_rotation = batch_data['envmap_rotation'][idx]

        # 1. Unproject every view to world space
        all_points, all_colors = [], []
        for rgb, depth, intrinsic, c2w in zip(rgb_images, depth_maps, intrinsics, extrinsics):
            rgb_np = rgb.permute(1, 2, 0).cpu().numpy() if isinstance(rgb, torch.Tensor) else rgb
            depth_np = depth.cpu().numpy() if isinstance(depth, torch.Tensor) else depth
            K = np.array([
                [intrinsic['fx'], 0, intrinsic['cx']],
                [0, intrinsic['fy'], intrinsic['cy']],
                [0, 0, 1]
            ], dtype=np.float32)
            point_map = unproject_depth_to_points(depth_np, K, c2w, coordinate_system='blender')
            all_points.append(point_map.reshape(-1, 3))
            all_colors.append(rgb_np.reshape(-1, 3))

        # 2. Fuse
        points_combined = np.vstack(all_points)
        colors_combined = np.vstack(all_colors)

        if save_ply and ply_output_dir is not None:
            os.makedirs(ply_output_dir, exist_ok=True)
            save_pointcloud_ply(points_combined, colors_combined, os.path.join(ply_output_dir, "point_cloud.ply"))

        points_torch = torch.from_numpy(points_combined).float().to(device)
        colors_torch = torch.from_numpy(colors_combined).float().to(device)

        # 3. Panorama camera
        if coordinate_system == 'blender':
            cam_pos, cam_rot = convert_blender_to_pytorch3d(
                position=tuple(envmap_position),
                rotation_degrees=envmap_rotation
            )
        else:
            cam_pos = envmap_position
            rx, ry, rz = [np.radians(a) for a in envmap_rotation]
            Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]], dtype=np.float32)
            Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]], dtype=np.float32)
            Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]], dtype=np.float32)
            cam_rot = Rz @ Ry @ Rx

        cam_pos_torch = torch.tensor(cam_pos, dtype=torch.float32, device=device)
        cam_rot_torch = torch.from_numpy(cam_rot).float().to(device) if cam_rot is not None else None

        # 4. Render
        panorama = render_panorama_pytorch3d(
            points=points_torch,
            colors=colors_torch,
            cam_pos=cam_pos_torch,
            cam_rot=cam_rot_torch,
            width=width,
            height=height,
            method='cubemap',
            point_radius=point_radius,
            points_per_pixel=points_per_pixel,
        )
        panoramas.append(panorama)

    if render_all:
        return torch.stack([pano.permute(2, 0, 1) for pano in panoramas], dim=0).unsqueeze(1)
    return panoramas[0]


if __name__ == "__main__":
    main()
