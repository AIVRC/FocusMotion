# FocusMotion

**FocusMotion** — 基于扩散模型的姿态驱动虚拟人视频生成框架，支持训练、推理、姿态对齐、复杂区域识别与多维度评估。

## 概述

FocusMotion 面向参考图像驱动的虚拟人动作视频生成任务：给定一张参考人物图像与一段目标姿态序列，生成该人物遵循姿态序列运动的视频。项目在原 MusePose 基础上进行了重构与扩展，新增了复杂区域识别、批量推理、多指标评估（FID / FVD / SSIM / PSNR / LPIPS）等模块，并整理了 TikTok、FashionDrive、UBC、FreeMan 等数据集的元信息。

## 核心功能

- **两阶段训练**：stage 1 训练 reference_unet + denoising_unet（不含 motion module），stage 2 加入 motion module 联合训练，支持多 GPU（accelerate + deepspeed zero2）。
- **姿态对齐**：`pose_align.py` 将任意舞蹈视频的 DWPose 对齐到任意参考图像，显著提升推理效果。
- **复杂区域识别**：`complex_region/` 模块用于识别与提取衣物、复杂区域，辅助精细化生成与评估。
- **批量推理**：`batch_gen*.py` / `batch_infer2.py` 支持大规模批量生成。
- **多维评估**：`video_metrics_pyiqa*` / `image_metrics.py` / `video_fvd.py` / `get_fvd.py` 提供帧级与视频级指标计算。
- **基准测试**：`benchmark/` 提供推理性能 profiling 工具。

## 目录结构

```
.
├── benchmark/              # 推理性能 profiling 工具
├── complex_region/         # 复杂区域识别与衣物提取
├── configs/                # 训练 / 推理配置 (yaml)
├── meta*/                  # 数据集元信息 (TikTok / FashionDrive / UBC / FreeMan)
├── metric_output/          # 评估结果输出
├── pose/                   # DWPose 配置与脚本
├── pretrained_weights/     # 预训练权重 (gitignored)
├── *.py                    # 训练 / 推理 / 评估 / 工具脚本
└── requirements.txt
```

## 安装

### 1. 构建环境

推荐 Python >= 3.10，CUDA 11.7。

```shell
pip install -r requirements.txt
```

### 2. 安装 mmlab 系列

```bash
pip install --no-cache-dir -U openmim
mim install mmengine
mim install "mmcv>=2.0.1"
mim install "mmdet>=3.1.0"
mim install "mmpose>=1.1.0"
```

### 3. 下载权重

所需权重组织在 `pretrained_weights/` 下：

```
./pretrained_weights/
├── MusePose/
│   ├── denoising_unet.pth
│   ├── motion_module.pth
│   ├── pose_guider.pth
│   └── reference_unet.pth
├── dwpose/
│   ├── dw-ll_ucoco_384.pth
│   └── yolox_l_8x8_300e_coco.pth
├── sd-image-variations-diffusers/
│   └── unet/
├── image_encoder/
├── sd-vae-ft-mse/
└── animatediff/
    └── mm_sd_v15_v2.ckpt
```

## 快速开始

### 推理

1. 准备参考图像与舞蹈视频：

```
./assets/
├── images/
│   └── ref.png
└── videos/
    └── dance.mp4
```

2. 姿态对齐：

```shell
python pose_align.py --imgfn_refer ./assets/images/ref.png --vidfn ./assets/videos/dance.mp4
```

3. 配置 `./configs/test_stage_2.yaml`：

```yaml
test_cases:
  "./assets/images/ref.png":
    - "./assets/poses/align/img_ref_video_dance.mp4"
```

4. 运行推理：

```shell
python test_stage_2.py --config ./configs/test_stage_2.yaml
```

结果输出至 `./output/`。可通过 `-W` `-H` 降低分辨率以减少显存占用（如 512x512 约 16GB VRAM）。

### 训练

1. 数据准备：

```shell
python extract_dwpose_keypoints.py --video_dir ./xxx
python draw_dwpose.py --video_dir ./xxx
python extract_meta_info_multiple_dataset.py --video_dirs ./xxx --dataset_name xxx
```

2. 配置 accelerate：

```shell
pip install accelerate
accelerate config
```

3. 修改 `./configs/train_stage_1.yaml` 与 `./configs/train_stage_2.yaml`。

4. 启动训练：

```shell
# stage 1
accelerate launch train_stage_1_multiGPU.py --config configs/train_stage_1.yaml
# stage 2
accelerate launch train_stage_2_multiGPU.py --config configs/train_stage_2.yaml
```

## 评估

```shell
# 帧级指标
python image_metrics.py
# 视频级指标 (pyiqa)
python video_metrics_pyiqa.py
# FVD
python video_fvd.py
```

## 致谢

本项目参考了 [AnimateAnyone](https://github.com/HumanAIGC/AnimateAnyone)、[Moore-AnimateAnyone](https://github.com/MooreThreads/Moore-AnimateAnyone)、[diffusers](https://github.com/huggingface/diffusers)、[AnimateDiff](https://animatediff.github.io/)、[DWPose](https://github.com/IDEA-Research/DWPose)、[Stable Diffusion](https://github.com/CompVis/stable-diffusion) 等开源工作。

## License

代码遵循 MIT License；模型权重仅供非商业研究用途。
