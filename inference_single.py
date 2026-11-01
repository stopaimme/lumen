import argparse
import torch
import os
import numpy as np
from PIL import Image
import sys

LUMEN_ROOT = os.path.dirname(os.path.abspath(__file__))

sys.path.append(os.path.join(LUMEN_ROOT, 'external'))
from external import equilib
from einops import rearrange
import math
from diffusers import DiffusionPipeline, FluxPriorReduxPipeline
from fluxpaint_single.modules import CustomFluxTransformer2DModel
from peft import LoraConfig, set_peft_model_state_dict
from diffusers.models import FluxTransformer2DModel

# Use a local VGGT checkout (LUMEN_ROOT/vggt) if present, otherwise the pip-installed `vggt` package
VGGT_BASE_PATH = os.path.join(LUMEN_ROOT, 'vggt')
if os.path.isdir(os.path.join(VGGT_BASE_PATH, 'vggt')):
    sys.path.insert(0, VGGT_BASE_PATH)


from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri


def get_masked_panorama(
    image: Image.Image,
    height: int,
    width: int,
    fov_x: float = 60.0,
    output_type: str = 'pt',
    device: str = 'cuda',
) -> torch.Tensor:
    pers = np.array(image.convert('RGB')).astype(np.float32)
    pers = rearrange(pers, 'h w c -> 1 c h w')
    rots = [{'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0}]
    equi = equilib.pers2equi(pers, rots=rots, height=height, width=width, fov_x=fov_x, clip_output=False)
    mask = equilib.pers2equi(np.ones_like(pers), rots=rots, height=height, width=width, fov_x=fov_x, clip_output=False)
    equi[mask < 0.5] = 0  # set to gray
    if output_type == 'pil':
        equi = (np.clip(equi[0].transpose(1, 2, 0), 0, 255).astype(np.uint8))
        return Image.fromarray(equi)
    elif output_type == 'np':
        return (equi / 255.0)[0].transpose(1, 2, 0)
    elif output_type == 'pt':
        equi = torch.tensor((equi / 127.5) - 1.0, device=device, dtype=torch.float32)
    
    return equi


def estimate_depth_and_intrinsics(image_path, model, device='cuda'):
    """
    使用 VGGT 模型估计单张图片的深度和内参
    
    Args:
        image_path: 输入图片路径
        device: 计算设备
        
    Returns:
        rgb_image: RGB 图片 (H, W, 3) numpy array
        depth_map: 深度图 (H, W) numpy array
        intrinsic: 内参字典 {'fx', 'fy', 'cx', 'cy', 'width', 'height'}
    """
    print(f"[INFO] Loading image from: {image_path}")
    
    # 读取图片
    if isinstance(image_path, str):
        image = Image.open(image_path).convert('RGB').resize((518, 518))
    else:
        image = image_path.convert('RGB')
    
    rgb_image = np.array(image)
    height, width = rgb_image.shape[:2]
    
    print(f"[INFO] Image size: {width}x{height}")
    print(f"[INFO] Estimating depth using VGGT...")

    

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+) 
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    # Load and preprocess example images
    image_names = [image_path]  
    images = load_and_preprocess_images(image_names).to(device)

    print(f"[INFO] Running VGGT inference...")
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            # Predict attributes including cameras, depth maps, and point maps.
            images_batch = images[None]  # add batch dimension
            aggregated_tokens_list, ps_idx = model.aggregator(images_batch)
                
        # Predict Cameras
        pose_enc = model.camera_head(aggregated_tokens_list)[-1]
        # Extrinsic and intrinsic matrices, following OpenCV convention (camera from world)
        extrinsic, intrinsic_matrix = pose_encoding_to_extri_intri(pose_enc, images_batch.shape[-2:])

        # Predict Depth Maps
        depth_tensor, depth_conf = model.depth_head(aggregated_tokens_list, images_batch, ps_idx)
    
    # 转换 intrinsic 矩阵为字典格式
    # intrinsic_matrix shape: [batch, 3, 3]

    intrinsic_np = intrinsic_matrix.squeeze(0).squeeze(0).cpu().numpy()  # [3, 3]
    cx = float(intrinsic_np[0, 2])
    fx = float(intrinsic_np[0, 0])
    intrinsic = {
        'fx': float(intrinsic_np[0, 0]),
        'fy': float(intrinsic_np[1, 1]),
        'cx': float(intrinsic_np[0, 2]),
        'cy': float(intrinsic_np[1, 2]),
        'width': width,
        'height': height
    }
    fov = 2 * math.atan(cx / fx) * 180 / math.pi

    return fov


def main(args):
    """
    主推理函数
    """
    # 设置设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[INFO] Using device: {device}")
    

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

    device = "cuda"
    custom_pipeline = "fluxpaint_single"

    flux_path = args.flux_path
    pretrained_model_name_or_path = os.path.join(flux_path, 'transformer')
    transformer = FluxTransformer2DModel.from_pretrained(pretrained_model_name_or_path)
    transformer = transformer.to(dtype=torch.bfloat16)
    custom_transformer = CustomFluxTransformer2DModel(transformer)
    pipe = DiffusionPipeline.from_pretrained(
        flux_path, # trained on dev2pro, infer on dev
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
                # modules_to_save=modules_to_save,
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

    # Initialize the model and load the pretrained weights.
    print(f"[INFO] Loading VGGT-1B model...")
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
    imgs = [
        f for f in os.listdir(args.scene_path)
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))
    ]
    for img_name in imgs:
        image_path = f"{args.scene_path}/{img_name}"
        fov = estimate_depth_and_intrinsics(image_path, model, device=device)
        image = Image.open(image_path).convert("RGB")
        # black_image = Image.new("RGB", image.size, (0, 0, 0))
        panorama = get_masked_panorama(image=image, height=args.height, width=args.width, fov_x=60, output_type='pil')
        panorama.save(f"{args.output_path}/{img_name.split('.')[0]}_panorama.png")
        

        cached_condition = {
            "num_in_batch": 1,
            "panorama": [[panorama]],
            "rgb_imgs": [[image]],
        }

        # for img in cached_condition["rgb_imgs"][0]:
        #     img = np.zeros_like(img)

        rgb_imgs_raw = cached_condition["rgb_imgs"][0]
        # Convert numpy arrays to PIL Images (imageio.imread returns numpy arrays)
        rgb_imgs = [Image.fromarray(img) if isinstance(img, np.ndarray) else img for img in rgb_imgs_raw]
        
        
        # Prepare embeddings
        pipe_prior_output = pipe_prior_redux(rgb_imgs)
        prompt_embeds = pipe_prior_output["prompt_embeds"]
        pooled_prompt_embeds = pipe_prior_output["pooled_prompt_embeds"]
        text_ids = torch.zeros(prompt_embeds.shape[1], 3).to(device=device, dtype=torch.bfloat16)
        

        if args.seed is None:
            generator = None
        else:
            generator = torch.Generator(device='cuda').manual_seed(args.seed)

        
        output = pipe(
            prompt=None,
            height=args.height,
            width=args.width,
            num_inference_steps=30,
            guidance_scale=3.0,
            generator=generator,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            output_type='np',  # 返回 numpy array
            **cached_condition,
        ).images
        
        # output[0] 现在是 numpy array，shape (H, W, 3)，值范围 [0, 1]
        LDR_np = output[0]  # numpy array, shape (H, W, 3), dtype float32, range [0, 1]
        
        # 保存 LDR 图像
        LDR_img = (LDR_np * 255).clip(0, 255).astype(np.uint8)
        # Image.fromarray(LDR_img).save(f"{args.output_path}/predict_ldr.png")
        
        # 转换 LDR 到 HDR
        # LDR_np 已经是归一化到 [0, 1] 的 numpy array
        
        # 使用反 Reinhard tone mapping: HDR = LDR / (1 - LDR)
        # 为避免除零，将 LDR 限制在 [0, 0.99999]
        # 设置 numpy 打印精度为 8 位小数（可选）
        np.set_printoptions(precision=8, suppress=True)
        min_v, max_v = float(LDR_np.min()), float(LDR_np.max())
        print(f"{min_v:.8f}, {max_v:.8f}")
        LDR_np = np.clip(LDR_np, 0.0000, 0.999)
        HDR_np = LDR_np / (1.0 - LDR_np)
        
        # 3. 保存 HDR 为 EXR 格式（需要 OpenEXR）
        # 如果要保存为 PNG，需要 tone mapping 回 LDR
        # 这里简单保存为 EXR
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
            exr_file_name = f"{args.output_path}/{img_name.split('.')[0]}.exr"
            exr_file = OpenEXR.OutputFile(exr_file_name, header)
            exr_file.writePixels({
                'R': HDR_np[:, :, 0].astype(np.float32).tobytes(),
                'G': HDR_np[:, :, 1].astype(np.float32).tobytes(),
                'B': HDR_np[:, :, 2].astype(np.float32).tobytes()
            })
            exr_file.close()
        except ImportError:
            print("[WARNING] OpenEXR not available, skipping HDR save")
            # 或者保存为 numpy array
            npy_file_name = f"{args.output_path}/{img_name.split('.')[0]}.npy"
            np.save(npy_file_name, HDR_np)
            print(f"[INFO] HDR data saved to: {npy_file_name}")

    


    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render panorama from scene data")
    parser.add_argument("--scene_path", type=str, required=True, 
                        help="Path to image directory")
    parser.add_argument("--width", type=int, default=1024,
                        help="Panorama width")
    parser.add_argument("--height", type=int, default=512,
                        help="Panorama height")
    parser.add_argument("--output_path", type=str, default="outputs",
                        help="Output path for rendered panorama")
    parser.add_argument("--flux_path", type=str, default=os.path.join(LUMEN_ROOT, 'ckpts/flux'),
                        help="Path to FLUX-dev diffusers directory")
    parser.add_argument("--lora_path", type=str, default=os.path.join(LUMEN_ROOT, 'ckpts/lumen'),
                        help="Path to trained LoRA directory (containing pytorch_lora_weights.safetensors)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    
    args = parser.parse_args()

    # args.output_path = args.scene_path
    os.makedirs(args.output_path, exist_ok=True)
    
    # 设置随机种子

    main(args)



