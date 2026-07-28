import os
import cv2
import torch
import lpips
import numpy as np
from datetime import datetime
from skimage.metrics import structural_similarity as calculate_ssim
from skimage.metrics import peak_signal_noise_ratio as calculate_psnr
from tqdm import tqdm

def image_to_tensor(img_np):
    img_float = img_np.astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_float).permute(2, 0, 1).unsqueeze(0)
    img_tensor = img_tensor * 2.0 - 1.0
    return img_tensor

def evaluate_folders(gt_dir, gen_dir, output_file="metrics_results.txt"):
    # 1. 初始化 LPIPS 模型
    print("正在加载 LPIPS 模型...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)
    
    # 2. 获取文件列表
    valid_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')
    gt_files = [f for f in os.listdir(gt_dir) if f.lower().endswith(valid_exts)]
    gen_files = os.listdir(gen_dir)
    
    psnr_list, ssim_list, lpips_list = [], [], []
    
    # 准备写入文件的数据头
    log_lines = []
    log_lines.append(f"评估时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_lines.append(f"GT 文件夹: {gt_dir}")
    log_lines.append(f"生成文件夹: {gen_dir}")
    log_lines.append("-" * 60)
    log_lines.append(f"{'Image Name':<30} | {'PSNR':>8} | {'SSIM':>8} | {'LPIPS':>8}")
    log_lines.append("-" * 60)

    print(f"开始处理 {len(gt_files)} 张图片...")
    
    # 3. 遍历计算
    for gt_file in tqdm(gt_files, desc="Processing"):
        gt_name_no_ext = os.path.splitext(gt_file)[0]
        matched_gen_file = next((f for f in gen_files if gt_name_no_ext in f and f.lower().endswith(valid_exts)), None)
                
        if not matched_gen_file:
            continue
            
        img_gt = cv2.cvtColor(cv2.imread(os.path.join(gt_dir, gt_file)), cv2.COLOR_BGR2RGB)
        img_gen = cv2.cvtColor(cv2.imread(os.path.join(gen_dir, matched_gen_file)), cv2.COLOR_BGR2RGB)
        
        if img_gt.shape != img_gen.shape:
            img_gen = cv2.resize(img_gen, (img_gt.shape[1], img_gt.shape[0]), interpolation=cv2.INTER_AREA)
            
        # 计算指标
        psnr_val = calculate_psnr(img_gt, img_gen, data_range=255)
        ssim_val = calculate_ssim(img_gt, img_gen, data_range=255, channel_axis=-1)
        
        tensor_gt = image_to_tensor(img_gt).to(device)
        tensor_gen = image_to_tensor(img_gen).to(device)
        with torch.no_grad():
            lpips_val = loss_fn_vgg(tensor_gt, tensor_gen).item()
            
        psnr_list.append(psnr_val)
        ssim_list.append(ssim_val)
        lpips_list.append(lpips_val)
        
        # 记录单张图片结果到列表
        log_lines.append(f"{gt_file:<30} | {psnr_val:8.4f} | {ssim_val:8.4f} | {lpips_val:8.4f}")
        
    # 4. 计算平均值并汇总
    if not psnr_list:
        print("未发现匹配图片。")
        return

    avg_psnr, avg_ssim, avg_lpips = np.mean(psnr_list), np.mean(ssim_list), np.mean(lpips_list)
    
    summary = [
        "-" * 60,
        f"最终平均结果 ({len(psnr_list)} images):",
        f"Average PSNR  : {avg_psnr:.4f}",
        f"Average SSIM  : {avg_ssim:.4f}",
        f"Average LPIPS : {avg_lpips:.4f}",
        "-" * 60
    ]
    
    # 打印到控制台
    for line in summary:
        print(line)
        
    # 写入到 TXT 文件
    with open(output_file, "w", encoding="utf-8") as f:
        for line in log_lines:
            f.write(line + "\n")
        for line in summary:
            f.write(line + "\n")
            
    print(f"\n结果已成功保存至: {os.path.abspath(output_file)}")

if __name__ == "__main__":
    # 配置路径
    GT_PATH = "/home/yanghaotian/server_data/yanghaotian/data/ubc_frames/frame300"
    GEN_PATH = "/home/yanghaotian/server_data/yanghaotian/test/MusePose/output/image-20260511/0706-/res"
    # GEN_PATH = "/home/yanghaotian/server_data/yanghaotian/test/Musepose_copy1/output/image-20260508/0708-pose_guider-1199450/res"
    # GEN_PATH = "/home/yanghaotian/server_data/yanghaotian/test/MusePose/output/image-20260509/1220-pose_guider-70800/res"
    SAVE_PATH = "./metric_output/evaluation_results.txt" # 你可以修改这个文件名
    
    evaluate_folders(GT_PATH, GEN_PATH, SAVE_PATH)