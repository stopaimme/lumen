import argparse
import os
import random
import sys

import numpy as np
import torch
from PIL import Image
from diffusers import DiffusionPipeline, FluxPriorReduxPipeline
from diffusers.models import FluxTransformer2DModel
from peft import LoraConfig, set_peft_model_state_dict

LUMEN_ROOT = os.path.dirname(os.path.abspath(__file__))

# Use a local VGGT checkout (LUMEN_ROOT/vggt) if present, otherwise the pip-installed `vggt` package
VGGT_BASE_PATH = os.path.join(LUMEN_ROOT, 'vggt')
if os.path.isdir(os.path.join(VGGT_BASE_PATH, 'vggt')):
    sys.path.insert(0, VGGT_BASE_PATH)

from fluxpaint_single.modules import CustomFluxTransformer2DModel
from pointcloud_panorama_renderer import render_panorama_from_batch

IMAGE_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')

# OpenCV 相机系 (x右, y下, z前) <-> OpenGL/Blender 相机系 (x右, y上, z后)
# render_panorama_from_batch 中的反投影使用的是后者: X=(u-cx)d/fx, Y=-(v-cy)d/fy, Z=-d
OPENCV_TO_OPENGL = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)


def collect_image_paths(image_paths=None, image_dir=None):
    """
    收集输入图片路径。第一张图片作为参考视角（世界坐标原点）。
    """
    paths = []
    if image_paths:
        paths.extend(image_paths)
    if image_dir:
        paths.extend(
            os.path.join(image_dir, f) for f in sorted(os.listdir(image_dir))
            if f.lower().endswith(IMAGE_EXTS)
        )
    if len(paths) == 0:
        raise ValueError("No input images. Use --image_path and/or --image_dir.")
    for p in paths:
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Image not found: {p}")
    return paths


def estimate_multiview_geometry(image_paths, device='cuda'):
    """
    使用 VGGT 对 1~N 张图片联合估计深度、内参和相机位姿。

    VGGT 输出的位姿以第一张图片为参考（第一张相机外参为单位阵），
    这里将其转换为 OpenGL/Blender 相机约定的 c2w，因此第一张相机的 c2w 仍为单位阵，
    与原单图流程（camera_position=0, camera_rotation=0）保持一致。

    Args:
        image_paths: 图片路径列表
        device: 计算设备

    Returns:
        rgb_images: [(H, W, 3) uint8]，与深度图逐像素对齐（取自 VGGT 预处理后的图像）
        depth_maps: [(H, W) float32]
        conf_maps:  [(H, W) float32] 深度置信度
        intrinsics: [{'fx', 'fy', 'cx', 'cy', 'width', 'height'}]
        extrinsics: [(4, 4) float32] c2w，OpenGL/Blender 相机约定
    """
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    num_views = len(image_paths)
    print(f"[INFO] Estimating geometry for {num_views} view(s) using VGGT...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+)
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    print(f"[INFO] Loading VGGT-1B model...")
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
    model.eval()

    # (N, 3, H, W), [0, 1]
    images = load_and_preprocess_images(image_paths).to(device)
    height, width = images.shape[-2:]
    print(f"[INFO] VGGT input size: {width}x{height}")

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            images_batch = images[None]  # (1, N, 3, H, W)
            aggregated_tokens_list, ps_idx = model.aggregator(images_batch)

        pose_enc = model.camera_head(aggregated_tokens_list)[-1]
        # extrinsic: (1, N, 3, 4) OpenCV w2c; intrinsic: (1, N, 3, 3)
        extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images_batch.shape[-2:])
        # depth: (1, N, H, W, 1); depth_conf: (1, N, H, W)
        depth, depth_conf = model.depth_head(aggregated_tokens_list, images_batch, ps_idx)

    extrinsic = extrinsic[0].float().cpu().numpy()
    intrinsic = intrinsic[0].float().cpu().numpy()
    depth = depth[0, ..., 0].float().cpu().numpy()
    depth_conf = depth_conf[0].float().cpu().numpy()
    rgb_np = (images.permute(0, 2, 3, 1).float().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)

    del model, aggregated_tokens_list, images, images_batch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    rgb_images, depth_maps, conf_maps, intrinsics, extrinsics = [], [], [], [], []
    for i in range(num_views):
        K = intrinsic[i]
        intrinsics.append({
            'fx': float(K[0, 0]),
            'fy': float(K[1, 1]),
            'cx': float(K[0, 2]),
            'cy': float(K[1, 2]),
            'width': int(width),
            'height': int(height),
        })

        w2c_cv = np.eye(4, dtype=np.float32)
        w2c_cv[:3, :] = extrinsic[i]
        c2w_cv = np.linalg.inv(w2c_cv)
        c2w_gl = OPENCV_TO_OPENGL @ c2w_cv @ OPENCV_TO_OPENGL
        extrinsics.append(c2w_gl.astype(np.float32))

        rgb_images.append(rgb_np[i])
        depth_maps.append(depth[i].astype(np.float32))
        conf_maps.append(depth_conf[i].astype(np.float32))

        print(f"[INFO] View {i}: {os.path.basename(image_paths[i])} | "
              f"fx={K[0, 0]:.1f}, fy={K[1, 1]:.1f}, cx={K[0, 2]:.1f}, cy={K[1, 2]:.1f} | "
              f"depth=[{depth[i].min():.3f}, {depth[i].max():.3f}] | "
              f"cam_pos={np.round(c2w_gl[:3, 3], 3).tolist()}")

    return rgb_images, depth_maps, conf_maps, intrinsics, extrinsics


def create_batch_data_from_views(
    rgb_images,
    depth_maps,
    intrinsics,
    extrinsics,
    envmap_position,
    envmap_rotation=(0, 0, 0),
):
    """
    从多视图（或单视图）重建结果创建 render_panorama_from_batch 所需的 batch_data

    Args:
        rgb_images: [(H, W, 3) uint8]
        depth_maps: [(H, W)]
        intrinsics: [dict]
        extrinsics: [(4, 4) c2w]
        envmap_position: (3,)
        envmap_rotation: (rx, ry, rz) 度

    Returns:
        batch_data: batch size = 1，num_views = N
    """
    envmap_position = np.asarray(envmap_position, dtype=np.float32)

    batch_data = {
        'rgb_images': [[torch.from_numpy(rgb.copy()).float().permute(2, 0, 1) / 255.0 for rgb in rgb_images]],
        'rgb_imgs': [[Image.fromarray(rgb.astype(np.uint8)) for rgb in rgb_images]],
        'depth_maps': [[torch.from_numpy(d.copy()).float() for d in depth_maps]],
        'num_views': [len(rgb_images)],
        'intrinsics': [intrinsics],
        'extrinsics': [extrinsics],
        'envmap_position': [envmap_position],
        'envmap_rotation': [envmap_rotation],
    }

    return batch_data


def get_redux_embeddings(pipe_prior_redux, images, mode='first', ref_index=0):
    """
    用 FLUX-Redux 编码输入图片。fluxpaint_single 的去噪 batch size 由 prompt_embeds.shape[0] 决定，
    因此多视图时需要把 N 份 embedding 合并为 1 份。

    mode:
        'first': 只用 ref_index 指定的视图
        'mean' : 对所有视图的 embedding 取平均
    """
    if mode == 'first' or len(images) == 1:
        out = pipe_prior_redux([images[ref_index]])
        return out["prompt_embeds"], out["pooled_prompt_embeds"]
    if mode == 'mean':
        out = pipe_prior_redux(images)
        prompt_embeds = out["prompt_embeds"].mean(dim=0, keepdim=True)
        pooled_prompt_embeds = out["pooled_prompt_embeds"].mean(dim=0, keepdim=True)
        return prompt_embeds, pooled_prompt_embeds
    raise ValueError(f"Unknown redux mode: {mode}")


def main(args):
    """
    主推理函数 - 从单张或多张图片生成 panorama
    """
    # 设置设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[INFO] Using device: {device}")

    image_paths = collect_image_paths(args.image_path, args.image_dir)
    print(f"[INFO] {len(image_paths)} input image(s), reference view: {image_paths[0]}")
    
    # 1. 估计深度、内参和位姿（多视图联合重建）
    print(f"\n{'='*60}")
    print("Step 1: Multi-view Depth / Intrinsics / Pose Estimation")
    print(f"{'='*60}")
    
    rgb_images, depth_maps, conf_maps, intrinsics, extrinsics = estimate_multiview_geometry(
        image_paths,
        device=device
    )

    
    # 5. 加载模型
    print(f"\n{'='*60}")
    print("Step 4: Loading Diffusion Models")
    print(f"{'='*60}")

    target_modules = [
        "attn.to_k",
        "attn.to_q",
        "attn.to_v",
        "attn.to_out.0",
        "attn.add_k_proj",
        "attn.add_q_proj",
        "attn.add_v_proj",
        "attn.to_add_out",
        "ff.net.0.proj",
        "ff.net.2",
        "ff_context.net.0.proj",
        "ff_context.net.2",
    ]

    flux_path = args.flux_path
    pretrained_model_name_or_path = os.path.join(flux_path, 'transformer')
    transformer = FluxTransformer2DModel.from_pretrained(pretrained_model_name_or_path)
    transformer = transformer.to(dtype=torch.bfloat16)
    custom_transformer = CustomFluxTransformer2DModel(transformer)
    pipe = DiffusionPipeline.from_pretrained(
        flux_path,
        transformer=custom_transformer,
        text_encoder=None,
        text_encoder_2=None,
        custom_pipeline=os.path.join(LUMEN_ROOT, 'fluxpaint_single'),
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )

    transformer_lora_config = LoraConfig(
                r=128,
                lora_alpha=128,
                init_lora_weights=True,
                target_modules=target_modules,
                lora_bias=False,
    )
    pipe.transformer.add_adapter(transformer_lora_config, adapter_name="lumen")
    lora_ckpt_path = args.lora_path
    if not os.path.exists(lora_ckpt_path):
        raise FileNotFoundError(f"LoRA checkpoint not found: {lora_ckpt_path}")
    lora_state_dict = CustomFluxTransformer2DModel.lora_state_dict(lora_ckpt_path)
    transformer_lora_state_dict = {
        f"{k.replace('transformer.', '', 1)}": v
        for k, v in lora_state_dict.items()
        if k.startswith("transformer.") and "lora" in k
    }
    if len(transformer_lora_state_dict) == 0:
        raise RuntimeError(f"No LoRA weights found in {lora_ckpt_path}")
    incompatible_keys = set_peft_model_state_dict(
        pipe.transformer, transformer_lora_state_dict, adapter_name="lumen"
    )
    print(f"[INFO] Loaded {len(transformer_lora_state_dict)} LoRA tensors from {lora_ckpt_path}")

    repo_redux = "black-forest-labs/FLUX.1-Redux-dev"
    pipe_prior_redux = FluxPriorReduxPipeline.from_pretrained(repo_redux, torch_dtype=torch.bfloat16)

    pipe = pipe.to(device)
    pipe_prior_redux.to(device)

    # Redux 条件只与输入图片有关，与 envmap 位置无关，循环外计算一次
    redux_images = [Image.open(p).convert("RGB") for p in image_paths]
    prompt_embeds, pooled_prompt_embeds = get_redux_embeddings(
        pipe_prior_redux, redux_images, mode=args.redux_mode, ref_index=0
    )

    # 6. 保存中间结果
    print(f"\n{'='*60}")
    print("Step 5: Saving Intermediate Results")
    print(f"{'='*60}")

    os.makedirs(args.output_path, exist_ok=True)
    envmap_position1 = np.array(args.envmap_position1, dtype=np.float32)
    envmap_position2 = np.array(args.envmap_position2, dtype=np.float32)
    envmap_rotation = tuple(args.envmap_rotation)
    num_positions = args.num_positions

    for i in range(num_positions):
        alpha = i / (num_positions - 1) if num_positions > 1 else 0.0
        envmap_position = envmap_position1 * (1 - alpha) + envmap_position2 * alpha
    
        batch_data = create_batch_data_from_views(
            rgb_images=rgb_images,
            depth_maps=depth_maps,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            envmap_position=envmap_position,
            envmap_rotation=envmap_rotation,
        )
        
        # 4. 渲染 panorama（多视图点云融合后渲染）
        print(f"\n{'='*60}")
        print(f"Step 3: Rendering Panorama from Point Cloud ({len(rgb_images)} views), position {i}: {np.round(envmap_position, 3).tolist()}")
        print(f"{'='*60}")
        
        panorama = render_panorama_from_batch(
            batch_data=batch_data,
            render_all=True,
            width=args.width,
            height=args.height,
            point_radius=args.point_radius,
            points_per_pixel=args.points_per_pixel,
            device=device,
            save_ply=(i == 0),
            ply_output_dir=args.output_path,
        )
        
        panorama_np = panorama[0, 0].permute(1, 2, 0).cpu().numpy()
        panorama_np = (panorama_np * 255).clip(0, 255).astype(np.uint8)
        Image.fromarray(panorama_np).save(f"{args.output_path}/pointrender{i}.png")
        print(f"[INFO] Panorama saved to: {args.output_path}/pointrender{i}.png")

        # 7. 运行 diffusion 生成
        print(f"\n{'='*60}")
        print("Step 6: Generating HDR Environment Map")
        print(f"{'='*60}")

        cached_condition = {
            "num_in_batch": 1,
            "panorama": [[Image.fromarray(panorama_np)]],
            "rgb_imgs": batch_data['rgb_imgs'],
        }
        
        if args.seed is None:
            generator = None
        else:
            generator = torch.Generator(device=device).manual_seed(args.seed)
        
        output = pipe(
            prompt=None,
            height=args.height,
            width=args.width,
            num_inference_steps=30,
            guidance_scale=3.0,
            generator=generator,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            output_type='np',
            **cached_condition,
        ).images
        
        # 8. 保存最终结果
        print(f"\n{'='*60}")
        print("Step 7: Saving Final Results")
        print(f"{'='*60}")
        
        LDR_np = output[0]
        
        
        # 转换 LDR 到 HDR
        LDR_np = np.clip(LDR_np, 0.0, 0.99999)
        HDR_np = LDR_np / (1.0 - LDR_np)
        
        # 保存 HDR 为 EXR 格式
        try:
            import OpenEXR
            import Imath
            
            height, width = HDR_np.shape[:2]
            header = OpenEXR.Header(width, height)
            header['channels'] = {
                'R': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)),
                'G': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)),
                'B': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))
            }
            
            exr_file = OpenEXR.OutputFile(f"{args.output_path}/predict_hdr{i}.exr", header)
            exr_file.writePixels({
                'R': HDR_np[:, :, 0].astype(np.float32).tobytes(),
                'G': HDR_np[:, :, 1].astype(np.float32).tobytes(),
                'B': HDR_np[:, :, 2].astype(np.float32).tobytes()
            })
            exr_file.close()
            print(f"[INFO] HDR image saved to: {args.output_path}/predict_hdr{i}.exr")
        except ImportError:
            print("[WARNING] OpenEXR not available, saving HDR as .npy")
            np.save(f"{args.output_path}/predict_hdr{i}.npy", HDR_np)
            print(f"[INFO] HDR data saved to: {args.output_path}/predict_hdr{i}.npy")
        
        print(f"\n{'='*60}")
        print("Processing Complete!")
        print(f"{'='*60}\n")

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate HDR panorama from one or more RGB images using multi-view reconstruction and point cloud rendering"
    )
    
    # Input/Output
    parser.add_argument("--image_path", type=str, nargs='+', default=None,
                        help="One or more input RGB images. The first one is the reference view (world origin).")
    parser.add_argument("--image_dir", type=str, default=None,
                        help="Directory of input RGB images (sorted by filename, appended after --image_path).")
    parser.add_argument("--output_path", type=str, default="outputs_vggt",
                        help="Output directory for results")

    # Model paths
    parser.add_argument("--flux_path", type=str, default=os.path.join(LUMEN_ROOT, 'ckpts/flux'),
                        help="Path to FLUX-dev diffusers directory")
    parser.add_argument("--lora_path", type=str,
                        default=os.path.join(LUMEN_ROOT, 'ckpts/lumen'),
                        help="Path to trained LoRA directory (containing pytorch_lora_weights.safetensors)")
    parser.add_argument("--redux_mode", type=str, default='first', choices=['first', 'mean'],
                        help="How to build Redux condition from multiple views: 'first' (reference view only) or 'mean' (average over views)")
    
    # Envmap position
    parser.add_argument("--envmap_position1", type=float, nargs=3, default=[-0.3, 0.0, -0.5],
                        help="Start envmap position [x, y, z] in the reference camera frame")
    parser.add_argument("--envmap_position2", type=float, nargs=3, default=[0.3, 0.0, -0.5],
                        help="End envmap position [x, y, z] in the reference camera frame")
    parser.add_argument("--num_positions", type=int, default=12,
                        help="Number of envmap positions interpolated between position1 and position2")
    parser.add_argument("--envmap_rotation", type=float, nargs=3, default=[-90, 0.0, 0.0],
                        help="Envmap rotation [rx, ry, rz] in degrees")
    
    # Panorama settings
    parser.add_argument("--width", type=int, default=1024,
                        help="Panorama width")
    parser.add_argument("--height", type=int, default=512,
                        help="Panorama height")
    
    # Rendering settings
    parser.add_argument("--point_radius", type=float, default=0.01,
                        help="Point rendering radius")
    parser.add_argument("--points_per_pixel", type=int, default=10,
                        help="Points per pixel for rendering")
    
    # Random seed
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")

    args = parser.parse_args()
    
    # 设置随机种子
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
    
    main(args)
