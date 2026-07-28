"""
复杂区域识别与提取模块

功能：
1. 包含完整的复杂区域识别算法（原 Complex_area_identification.py）。
2. 执行批量提取逻辑：读取 metadata，遍历视频，提取指定帧的复杂区域（原 extract_regions.py）。
"""

import os
import json
import math
import random
from dataclasses import dataclass, asdict, field
from datetime import datetime
from functools import lru_cache
from typing import List, Optional, Tuple, Dict, Any, NamedTuple
from contextlib import contextmanager

import cv2
import numpy as np

# matplotlib 后端设置
import matplotlib
# 尝试使用非交互式后端，避免在无显示服务的服务器上报错
# 如果需要显示窗口，可以改回 "TkAgg"
try:
    matplotlib.use("Agg")
except:
    pass
import matplotlib.pyplot as plt

import torch
from PIL import Image
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
from ultralytics import YOLO

# ============================================================
# PART 1: 核心算法类 (原 Complex_area_identification.py)
# ============================================================

# ============================================================
# 1) 配置管理类
# ============================================================
@dataclass(frozen=True)
class Config:
    """全局配置，使用 frozen 确保不被意外修改"""
    IMG_PATH: str = r"/home/yanghaotian/server_data/yanghaotian/data/applied_dataset/ref/00009_0001.png"
    IMG_DIR: str = r"../../data/TikTok_complex"
    USE_DIR_MODE: bool = True

    CALIB_DIR: str = r"F:\Data\clothes_calib"
    YOLO_SEG_WEIGHTS: str = "yolov8n-seg.pt"
    # 请确保此路径指向正确的模型目录
    PARSE_MODEL_DIR: str = r"/home/yanghaotian/server_data/yanghaotian/test/Musepose_copy1/complex_region/models/segformer-b2-human-parse-24"

    SAVE_DPI: int = 220
    BEST_JSON_NAME: str = "best_params.json"

    # 启发式搜索配置
    ENABLE_STRONG_HEURISTIC_SEARCH: bool = True
    CALIB_FALLBACK_FIRST_N: int = 24
    PSEUDO_POS_FRAC: float = 0.40
    PSEUDO_NEG_FRAC: float = 0.40

    # SA 搜索参数
    RANDOM_STAGE_TRIALS: int = 220
    RANDOM_STAGE_TOPK: int = 8
    SA_MULTI_START_ITERS: int = 220
    SA_T0: float = 0.18
    SA_TMIN: float = 0.01
    SA_ALPHA: float = 0.985
    SA_EARLY_PATIENCE: int = 60
    SA_SEED: int = 0

    # 目标函数权重
    W_POS_FOUND: float = 2.5
    W_POS_QUALITY: float = 1.2
    W_POS_STABLE: float = 0.20
    W_NEG_FP: float = 0.8
    W_NEG_RATIO: float = 0.5

    # ratio 约束
    RATIO_MIN: float = 0.02
    RATIO_MAX: float = 0.45

    # 优化模式: "default" / "auto_iterative" / "load_json"
    OPTIMIZATION_MODE: str = "load_json"

    # 迭代优化配置
    ITER_MAX_IMAGES: int = 30
    ITER_CONVERGENCE_N: int = 8
    ITER_NEIGHBOR_TRIALS: int = 20

    # 增强 Canny 开关
    USE_ENHANCED_CANNY: bool = True

    # 多数据集 JSON 路径列表
    META_JSON_PATHS: Tuple[str, ...] = (
        # "../meta2/Tik_meta2.json",
        # "../meta2/ubc_train_meta.json",
    #    "../meta3/fd_1000_meta.json",
        # "../meta3/fd_2000_meta.json",
        # "../meta3/fd_3000_meta.json",
        "../meta3/fd_4000_meta.json",
        "../meta3/fd_5000_meta.json",
        # "../meta3/fd_6000_meta.json",
        # "../meta3/fd_7000_meta.json",
        # "../meta3/fd_8000_meta.json",
        # "../meta3/fd_8512_meta.json",
        # "../meta3/Tik_meta.json",
        # "../meta3/ubc_meta.json",
        # "../meta3/single.json",

    )


# 衣物标签集合
CLOTH_LABELS = frozenset({
    "upper_only_torso_region",
    "dresses_only_torso_region",
    "coat_only_torso_region",
    "left_pants",
    "right_pants",
    "right_patns",  # 兼容原拼写
    "skirts",
    "left_sleeve_for_upper",
    "right_sleeve_for_upper",
})
# CLOTH_LABELS = frozenset({
#     # --- 1. 上身躯干 (Torso) ---
#     "upper_only_torso_region",    # 基础内搭/T恤/衬衫的躯干部分
#     "coat_only_torso_region",     # 外套/大衣的躯干部分
#     "dresses_only_torso_region",  # 连衣裙的躯干部分

#     # --- 2. 袖子 (Sleeves) - 必须包含，否则手臂部分会被切掉 ---
#     "left_sleeve_for_upper",      # 内搭左袖
#     "right_sleeve_for_upper",     # 内搭右袖
#     "left_sleeve_for_coat",       # 外套左袖
#     "right_sleeve_for_coat",      # 外套右袖
#     "left_sleeve_for_dress",      # 连衣裙左袖
#     "right_sleeve_for_dress",     # 连衣裙右袖

#     # --- 3. 下装 (Bottoms) ---
#     "left_pants",                 # 左裤腿
#     "right_pants",                # 右裤腿
#     "right_patns",                # (常见模型拼写错误兼容)
#     "skirts",                     # 裙子
#     # 注意：Legs 通常指露出的皮肤腿部，如果你想要连裤袜效果，可以把 legs 也加进去，
#     # 但通常 cloth cut 不包含皮肤。如果需要请取消下面两行的注释：
#     "left_leg",
#     "right_leg",

#     # --- 4. 鞋袜 (Feet) ---
#     "shoe",
#     "left_shoe",
#     "right_shoe",
#     "socks",

#     # --- 5. 头部与颈部配饰 (Head & Neck) ---
#     "hat",                        # 帽子
#     "scarf",                      # 围巾
#     # "hair",                     # 头发 (通常不作为衣物提取，如果需要请取消注释)

#     # --- 6. 其他配饰 (Accessories) ---
#     "belt",                       # 腰带
#     "bag",                        # 包 (手提包/背包)
#     "gloves",                     # 手套
#     "sunglasses",                 # 墨镜
#     "glasses",                    # 眼镜
# })
# 图像扩展名集合
IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})

# 全局配置实例
CONFIG = Config()


# ============================================================
# 2) 参数管理
# ============================================================
@dataclass
class Params:
    """算法参数类，支持序列化和归一化"""
    YOLO_MASK_THRESH: float = 0.7
    KEEP_COMPONENTS: int = 2
    MORPH_KERNEL: int = 9

    CANNY_T1: int = 20
    CANNY_T2: int = 80

    DENSITY_BLUR: int = 61
    TOP_PERCENT: float = 0.70

    BORDER_ERODE: int = 15
    BORDER_ERODE_ITER: int = 2
    PATTERN_MORPH: int = 7
    MIN_PATTERN_PIXELS: int = 300
    MIN_EDGE_PIXELS: int = 80
    ZSCORE_K: float = 0.45

    CLOTH_DENSITY_THRESH: float = 2.0
    EDGE_COMPLEXITY_THRESH: float = 0.03

    def normalize(self) -> "Params":
        """归一化参数到有效范围"""
        self.DENSITY_BLUR = int(np.clip(self.DENSITY_BLUR, 21, 91))
        self.DENSITY_BLUR = _ensure_odd(self.DENSITY_BLUR, mn=21)
        self.BORDER_ERODE = _ensure_odd(self.BORDER_ERODE, mn=3)
        self.BORDER_ERODE_ITER = int(np.clip(self.BORDER_ERODE_ITER, 1, 4))
        self.PATTERN_MORPH = _ensure_odd(self.PATTERN_MORPH, mn=1)
        self.MORPH_KERNEL = _ensure_odd(self.MORPH_KERNEL, mn=3)

        self.TOP_PERCENT = float(np.clip(self.TOP_PERCENT, 0.05, 0.50))
        self.CANNY_T1 = int(np.clip(self.CANNY_T1, 5, 60))
        self.CANNY_T2 = int(np.clip(self.CANNY_T2, 30, 180))
        if self.CANNY_T2 <= self.CANNY_T1 + 10:
            self.CANNY_T2 = self.CANNY_T1 + 40

        self.MIN_PATTERN_PIXELS = int(np.clip(self.MIN_PATTERN_PIXELS, 100, 800))
        self.MIN_EDGE_PIXELS = int(np.clip(self.MIN_EDGE_PIXELS, 40, 300))
        self.ZSCORE_K = float(np.clip(self.ZSCORE_K, 0.10, 1.20))

        self.YOLO_MASK_THRESH = float(np.clip(self.YOLO_MASK_THRESH, 0.35, 0.95))
        self.KEEP_COMPONENTS = int(np.clip(self.KEEP_COMPONENTS, 1, 3))
        
        self.CLOTH_DENSITY_THRESH = float(np.clip(self.CLOTH_DENSITY_THRESH, 0.0, 30.0))
        self.EDGE_COMPLEXITY_THRESH = float(np.clip(self.EDGE_COMPLEXITY_THRESH, 0.005, 0.15))
        return self

    def to_json(self, path: str) -> None:
        """保存参数到 JSON 文件"""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, ensure_ascii=False, indent=2)
        print(f"[SAVE] Params -> {path}")

    @classmethod
    def from_json(cls, path: str, fallback: "Params" = None) -> "Params":
        """从 JSON 文件加载参数"""
        if fallback is None:
            fallback = cls()

        if not os.path.exists(path):
            return fallback

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            params = cls(**{**asdict(fallback), **data})
            return params.normalize()
        except Exception as e:
            print(f"[WARN] load params failed: {path} -> {e}")
            return fallback


# ============================================================
# 3) 评估结果数据类
# ============================================================
class PatternEvalResult(NamedTuple):
    """图案评估结果"""
    ok: bool
    ratio: float
    contrast: float
    comp_count: int
    score_quality: float


class ObjectiveResult(NamedTuple):
    """目标函数结果"""
    score: float
    info: Dict[str, float]


# ============================================================
# 4) 工具函数
# ============================================================
def _ensure_odd(x: int, mn: int = 1) -> int:
    """确保数值为奇数且不小于最小值"""
    x = max(mn, int(x))
    return x if x % 2 == 1 else x + 1


def check_parse_model_dir(model_dir: str) -> None:
    """检查模型目录是否包含必需文件"""
    required_files = ["model.safetensors", "config.json", "preprocessor_config.json"]
    # 简单检查，避免路径不存在报错，只在真正加载时处理
    if not os.path.exists(model_dir):
         print(f"[WARN] Model directory not found: {model_dir}")
         return

    missing = [f for f in required_files if not os.path.exists(os.path.join(model_dir, f))]
    if missing:
        print(f"[WARN] Missing files in {model_dir}: {missing}")


def ensure_output_dir(folder_name: str = "outputs") -> str:
    """创建带时间戳的输出目录"""
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        base_dir = os.getcwd()

    root = os.path.join(base_dir, folder_name)
    os.makedirs(root, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(root, f"run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def list_images_in_dir(img_dir: str) -> List[str]:
    """列出目录中的所有图像文件"""
    if not os.path.isdir(img_dir):
        return []

    paths = [
        os.path.join(img_dir, name)
        for name in os.listdir(img_dir)
        if os.path.isfile(os.path.join(img_dir, name))
        and os.path.splitext(name.lower())[1] in IMAGE_EXTENSIONS
    ]
    paths.sort()
    return paths


# ============================================================
# 5) 图像处理核心类
# ============================================================
class ImageProcessor:
    """图像处理核心类，封装所有图像操作"""

    def __init__(self, params: Params, use_enhanced_canny: bool = False):
        self.params = params
        self.use_enhanced_canny = use_enhanced_canny
        self._kernel_cache: Dict[Tuple[int, int], np.ndarray] = {}

    def _get_kernel(self, ksize: int) -> np.ndarray:
        key = (ksize, ksize)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, key
            )
        return self._kernel_cache[key]

    @staticmethod
    def bbox_from_mask(mask_u8: np.ndarray, pad: int = 20) -> Optional[Tuple[int, int, int, int]]:
        ys, xs = np.nonzero(mask_u8)
        if xs.size == 0:
            return None

        H, W = mask_u8.shape
        x1 = max(0, xs.min() - pad)
        x2 = min(W - 1, xs.max() + pad)
        y1 = max(0, ys.min() - pad)
        y2 = min(H - 1, ys.max() + pad)
        return x1, y1, x2, y2

    def keep_largest_components(self, mask_u8: np.ndarray, keep_n: int = 2) -> np.ndarray:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask_u8, connectivity=8
        )
        if num_labels <= 1:
            return mask_u8

        areas = stats[1:, cv2.CC_STAT_AREA]
        keep_indices = np.argsort(areas)[::-1][:keep_n] + 1

        result = np.zeros_like(mask_u8)
        for idx in keep_indices:
            result[labels == idx] = 255
        return result

    def postprocess_mask(self, mask_u8: np.ndarray) -> np.ndarray:
        ksize = _ensure_odd(self.params.MORPH_KERNEL, mn=3)
        kernel = self._get_kernel(ksize)

        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel, iterations=1)
        return self.keep_largest_components(mask_u8, keep_n=self.params.KEEP_COMPONENTS)

    def compute_edge_density(
        self, gray: np.ndarray, mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self.use_enhanced_canny:
            return self._compute_edge_density_enhanced(gray, mask)

        edges = cv2.Canny(gray, self.params.CANNY_T1, self.params.CANNY_T2)
        edges = cv2.bitwise_and(edges, edges, mask=mask)

        k = _ensure_odd(self.params.DENSITY_BLUR, mn=21)
        blurred = cv2.GaussianBlur(edges, (k, k), 0)
        density = cv2.normalize(blurred, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        density = cv2.bitwise_and(density, density, mask=mask)

        return edges, density

    def _compute_edge_density_enhanced(
        self, gray: np.ndarray, mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """增强版：轻度双边滤波 + 收紧自适应阈值 + L2梯度，提高线稿精准度"""
        # 减小滤波强度，保留更多细节
        denoised = cv2.bilateralFilter(gray, d=5, sigmaColor=50, sigmaSpace=50)

        valid_pixels = denoised[mask > 0]
        if valid_pixels.size > 0:
            median_val = float(np.median(valid_pixels))
        else:
            median_val = 128.0

        # 收紧阈值比例，提取更精确的边缘
        auto_t1 = int(max(5, 0.50 * median_val))
        auto_t2 = int(min(255, 1.50 * median_val))

        edges = cv2.Canny(denoised, auto_t1, auto_t2, apertureSize=3, L2gradient=True)
        edges = cv2.bitwise_and(edges, edges, mask=mask)

        k = _ensure_odd(self.params.DENSITY_BLUR, mn=21)
        blurred = cv2.GaussianBlur(edges, (k, k), 0)
        density = cv2.normalize(blurred, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        density = cv2.bitwise_and(density, density, mask=mask)

        return edges, density

    def compute_threshold(self, vals: np.ndarray) -> float:
        top = max(0.05, min(0.60, float(self.params.TOP_PERCENT)))
        quantile_thr = float(np.quantile(vals, 1.0 - top))

        mean_val = float(vals.mean())
        std_val = float(vals.std() + 1e-6)
        zscore_thr = mean_val + float(self.params.ZSCORE_K) * std_val

        return max(quantile_thr, zscore_thr)

    def masked_canny_heatmap(
        self, img_bgr: np.ndarray, mask_u8: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        work = img_bgr.copy()
        work[mask_u8 == 0] = (255, 255, 255)

        gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)

        if self.use_enhanced_canny:
            denoised = cv2.bilateralFilter(gray, d=5, sigmaColor=50, sigmaSpace=50)
            valid_pixels = denoised[mask_u8 > 0]
            median_val = float(np.median(valid_pixels)) if valid_pixels.size > 0 else 128.0
            auto_t1 = int(max(5, 0.50 * median_val))
            auto_t2 = int(min(255, 1.50 * median_val))
            edges = cv2.Canny(denoised, auto_t1, auto_t2, apertureSize=3, L2gradient=True)
        else:
            gray = cv2.GaussianBlur(gray, (3, 3), 0)
            edges = cv2.Canny(gray, self.params.CANNY_T1, self.params.CANNY_T2)

        edges = cv2.bitwise_and(edges, edges, mask=mask_u8)

        k = _ensure_odd(self.params.DENSITY_BLUR, mn=21)
        blurred = cv2.GaussianBlur(edges, (k, k), 0)
        density = cv2.normalize(blurred, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        density = cv2.bitwise_and(density, density, mask=mask_u8)
        heatmap = cv2.applyColorMap(density, cv2.COLORMAP_JET)

        return edges, heatmap


# ============================================================
# 6) 模型管理类
# ============================================================
class ModelManager:
    """模型加载和管理类"""

    def __init__(self, config: Config = CONFIG):
        self.config = config
        self.yolo: Optional[YOLO] = None
        self.parse_model = None
        self.processor = None
        self.device: str = "cpu"
        self.cloth_ids: List[int] = []

    def load(self) -> "ModelManager":
        check_parse_model_dir(self.config.PARSE_MODEL_DIR)

        print("Loading YOLOv8 model...")
        try:
            self.yolo = YOLO(self.config.YOLO_SEG_WEIGHTS)
            print("YOLO loaded.")
        except Exception as e:
            print(f"Error loading Yolo: {e}")

        print("Loading Human Parsing model (local)...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        try:
            self.processor = AutoImageProcessor.from_pretrained(
                self.config.PARSE_MODEL_DIR, local_files_only=True, use_fast=True
            )
            self.parse_model = SegformerForSemanticSegmentation.from_pretrained(
                self.config.PARSE_MODEL_DIR, local_files_only=True
            ).to(self.device).eval()

            id2label = {int(k): v for k, v in self.parse_model.config.id2label.items()}
            self.cloth_ids = [i for i, name in id2label.items() if name in CLOTH_LABELS]
        except Exception as e:
            print(f"Error loading Segformer: {e}")

        print(f"Device: {self.device}")
        print(f"Cloth IDs: {self.cloth_ids}")
        return self


# ============================================================
# 7) 掩码构建器
# ============================================================
class MaskBuilder:
    """掩码构建类"""

    def __init__(self, models: ModelManager, image_processor: ImageProcessor):
        self.models = models
        self.img_processor = image_processor

    def build_person_mask(
        self, img_input: Any, img_shape_hw: Tuple[int, int]
    ) -> Optional[np.ndarray]:
        """使用 YOLO 构建人物掩码"""
        if self.models.yolo is None:
            return None
            
        H, W = img_shape_hw
        # Try YOLO first
        results = self.models.yolo(img_input, verbose=False)
        result = results[0]

        has_yolo_person = False
        if result.masks is not None:
            class_ids = result.boxes.cls.cpu().numpy().astype(int)
            person_indices = np.nonzero(class_ids == 0)[0]
            if person_indices.size > 0:
                has_yolo_person = True
                params = self.img_processor.params
                final_mask = np.zeros((H, W), dtype=np.uint8)
                for idx in person_indices:
                    mask_data = result.masks.data[idx].detach().cpu().numpy()
                    resized = cv2.resize(mask_data, (W, H), interpolation=cv2.INTER_NEAREST)
                    binary = (resized > params.YOLO_MASK_THRESH).astype(np.uint8) * 255
                    final_mask = cv2.bitwise_or(final_mask, binary)
                return final_mask

        # Fallback to Segformer if YOLO fails
        if not has_yolo_person:
            print("[INFO] YOLO failed to detect person. Falling back to Segformer.")
            if self.models.parse_model is None:
                return None
            
            # Read image for Segformer
            if isinstance(img_input, str):
                img_bgr = cv2.imread(img_input)
            elif isinstance(img_input, np.ndarray):
                img_bgr = img_input
            else:
                return None

            if img_bgr is None:
                return None
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            
            pil_img = Image.fromarray(img_rgb)
            inputs = self.models.processor(images=pil_img, return_tensors="pt")
            inputs = {k: v.to(self.models.device) for k, v in inputs.items()}

            with torch.no_grad():
                output = self.models.parse_model(**inputs)

            h0, w0 = img_rgb.shape[:2]
            logits = torch.nn.functional.interpolate(
                output.logits, size=(h0, w0), mode="bilinear", align_corners=False
            )
            seg = logits.argmax(dim=1)[0].detach().cpu().numpy().astype(np.uint8)
            
            # Define person labels (0 is background)
            # Generally: 1=hat, 2=hair, 3=glove, 4=sunglasses, 5=upperclothes, 6=dress, 7=coat, 8=socks, 9=pants, 10=jumpsuits, 11=scarf, 12=skirt, 13=face, 14=leftArm, 15=rightArm, 16=leftLeg, 17=rightLeg, 18=leftShoe, 19=rightShoe
            # We exclude background (0)
            person_mask = (seg > 0).astype(np.uint8) * 255
            
            # Post-process to keep largest component
            return self.img_processor.keep_largest_components(person_mask, keep_n=1)

        return None

    def build_cloth_mask(
        self, img_bgr: np.ndarray, person_mask: np.ndarray
    ) -> Optional[np.ndarray]:
        """使用 Human Parsing 模型构建衣物掩码"""
        if self.models.parse_model is None:
            return None

        H, W = img_bgr.shape[:2]
        roi = ImageProcessor.bbox_from_mask(person_mask, pad=20)
        if roi is None:
            return None

        x1, y1, x2, y2 = roi
        person_crop_rgb = cv2.cvtColor(img_bgr[y1:y2+1, x1:x2+1], cv2.COLOR_BGR2RGB)
        person_mask_crop = person_mask[y1:y2+1, x1:x2+1]

        pil_img = Image.fromarray(person_crop_rgb)
        inputs = self.models.processor(images=pil_img, return_tensors="pt")
        inputs = {k: v.to(self.models.device) for k, v in inputs.items()}

        with torch.no_grad():
            output = self.models.parse_model(**inputs)

        h0, w0 = person_crop_rgb.shape[:2]
        logits = torch.nn.functional.interpolate(
            output.logits, size=(h0, w0), mode="bilinear", align_corners=False
        )
        seg = logits.argmax(dim=1)[0].detach().cpu().numpy().astype(np.uint8)

        if not self.models.cloth_ids:
            return None

        cloth_mask_crop = (np.isin(seg, self.models.cloth_ids).astype(np.uint8) * 255)
        cloth_mask_crop = cv2.bitwise_and(
            cloth_mask_crop, cloth_mask_crop, mask=person_mask_crop
        )
        cloth_mask_crop = self.img_processor.postprocess_mask(cloth_mask_crop)

        cloth_mask = np.zeros((H, W), dtype=np.uint8)
        cloth_mask[y1:y2+1, x1:x2+1] = cloth_mask_crop
        return cloth_mask


# ============================================================
# 8) 复杂区域检测器
# ============================================================
class PatternDetector:
    """复杂纹理区域检测器"""

    def __init__(self, image_processor: ImageProcessor):
        self.img_processor = image_processor

    def detect_pattern_region(
        self, img_bgr: np.ndarray, cloth_mask: np.ndarray
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """检测衣物上的复杂纹理区域（强化边缘过滤 + 小区域限制）"""
        params = self.img_processor.params
        cloth_cut = cv2.bitwise_and(img_bgr, img_bgr, mask=cloth_mask)

        # 创建内部掩码（多尺度侵蚀 + 梯度感知边缘排除）
        inner_mask = self._create_inner_mask(cloth_mask, img_bgr)

        # 准备灰度图
        work = cloth_cut.copy()
        work[cloth_mask == 0] = (255, 255, 255)
        gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
        if not self.img_processor.use_enhanced_canny:
            gray = cv2.GaussianBlur(gray, (3, 3), 0)

        # 计算边缘密度
        edges, density = self.img_processor.compute_edge_density(gray, inner_mask)

        edge_count = np.count_nonzero(edges)
        if edge_count < params.MIN_EDGE_PIXELS:
            return None

        # 简单线稿检测：边缘像素占内部区域比例太低 → 线条简单，无复杂区域
        inner_area = float(np.count_nonzero(inner_mask))
        if inner_area > 0:
            edge_ratio = edge_count / inner_area
            if edge_ratio < params.EDGE_COMPLEXITY_THRESH:
                return None

        # 计算阈值并生成纹理掩码
        vals = density[inner_mask > 0]
        if vals.size == 0 or np.max(vals) == 0:
            return None

        threshold = self.img_processor.compute_threshold(vals)
        cloth_area = float(np.count_nonzero(cloth_mask))
        pattern_mask = self._create_pattern_mask(density, inner_mask, threshold, cloth_area)

        # 小区域限制：自适应阈值 = max(MIN_PATTERN_PIXELS, cloth_area * 3%)
        # 小点散布的情况，整体面积不够大就不算复杂区域
        pattern_area = np.count_nonzero(pattern_mask)
        adaptive_min = max(params.MIN_PATTERN_PIXELS, int(cloth_area * 0.03))
        if pattern_area < adaptive_min:
            return None

        pattern_cut = cv2.bitwise_and(img_bgr, img_bgr, mask=pattern_mask)
        return pattern_mask, pattern_cut

    def _create_inner_mask(
        self, cloth_mask: np.ndarray, img_bgr: np.ndarray = None
    ) -> np.ndarray:
        """创建去除边界的内部掩码（多尺度侵蚀 + 梯度感知边缘排除）

        增强策略：
        1. 多尺度侵蚀：先用大核粗略去边界轮廓，再用小核精细化
        2. 梯度感知：在边界附近检测强梯度（衣物与皮肤/背景交界处），
           排除这些高对比度边缘以避免误检
        3. 距离变换加权：离边界越远权重越高，边界附近逐渐衰减
        """
        params = self.img_processor.params
        border_erode = _ensure_odd(params.BORDER_ERODE, mn=3)
        erode_iter = max(1, int(params.BORDER_ERODE_ITER))

        # --- 第一阶段：基础多尺度侵蚀 ---
        # 大核粗略侵蚀，去掉外层轮廓
        large_k = _ensure_odd(border_erode + 4, mn=5)
        kernel_large = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (large_k, large_k)
        )
        coarse_mask = cv2.erode(cloth_mask, kernel_large, iterations=1)

        # 小核精细侵蚀，进一步收紧
        small_k = _ensure_odd(max(3, border_erode - 2), mn=3)
        kernel_small = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (small_k, small_k)
        )
        inner_mask = cv2.erode(coarse_mask, kernel_small, iterations=max(1, erode_iter - 1))

        # --- 第二阶段：梯度感知边缘排除 ---
        if img_bgr is not None:
            inner_mask = self._exclude_gradient_edges(inner_mask, cloth_mask, img_bgr)

        return inner_mask

    def _exclude_gradient_edges(
        self, inner_mask: np.ndarray, cloth_mask: np.ndarray, img_bgr: np.ndarray
    ) -> np.ndarray:
        """利用梯度信息排除衣物边界附近的高对比度伪边缘

        原理：衣物与皮肤/背景的交界处会产生强梯度，这些梯度容易被误判为
        衣物内部的复杂纹理。通过检测距离边界一定范围内的强梯度区域并排除，
        可以有效避免边界伪纹理污染最终结果。
        """
        params = self.img_processor.params

        # 1. 提取衣物边界附近带（cloth_mask 与 inner_mask 的差集区域）
        border_band = cv2.subtract(cloth_mask, inner_mask)

        # 扩大边界带范围（向内多覆盖一些）
        expand_k = _ensure_odd(max(5, params.BORDER_ERODE // 2), mn=5)
        expand_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (expand_k, expand_k)
        )
        border_band_expanded = cv2.dilate(border_band, expand_kernel, iterations=1)

        # 2. 计算灰度图梯度幅值
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = cv2.magnitude(grad_x, grad_y)

        # 3. 在扩展边界带内找高梯度区域
        grad_in_border = grad_mag.copy()
        grad_in_border[border_band_expanded == 0] = 0

        if np.max(grad_in_border) > 0:
            # 自适应阈值：边界带内梯度的 75th 百分位
            border_vals = grad_in_border[border_band_expanded > 0]
            if border_vals.size > 0:
                grad_thresh = float(np.percentile(border_vals, 75))
                high_grad_mask = (grad_in_border > grad_thresh).astype(np.uint8) * 255

                # 稍微膨胀高梯度区域以确保完全覆盖
                dilate_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                high_grad_mask = cv2.dilate(high_grad_mask, dilate_k, iterations=1)

                # 从 inner_mask 中排除这些区域
                inner_mask = cv2.subtract(inner_mask, high_grad_mask)

        return inner_mask

    def _create_pattern_mask(
        self, density: np.ndarray, inner_mask: np.ndarray,
        threshold: float, cloth_area: float = 0.0
    ) -> np.ndarray:
        """创建纹理掩码（含自适应小区域噪点过滤和边缘邻近排除）"""
        params = self.img_processor.params

        pattern_mask = (density >= threshold).astype(np.uint8) * 255
        pattern_mask = cv2.bitwise_and(pattern_mask, pattern_mask, mask=inner_mask)

        pm = _ensure_odd(params.PATTERN_MORPH, mn=1)
        if pm > 1:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (pm, pm))
            # 先闭运算填充纹理内小空洞
            pattern_mask = cv2.morphologyEx(pattern_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            # 再开运算去除小突起和噪点（增加迭代次数，更激进地去除小点）
            pattern_mask = cv2.morphologyEx(pattern_mask, cv2.MORPH_OPEN, kernel, iterations=2)
            pattern_mask = cv2.bitwise_and(pattern_mask, pattern_mask, mask=inner_mask)

        # 自适应噪点过滤：min_area = max(1500, cloth_area * 2%)
        # 小于此面积的散点全部移除
        adaptive_min_area = max(1500, int(cloth_area * 0.02)) if cloth_area > 0 else 1500
        pattern_mask = self._remove_small_components(pattern_mask, min_area=adaptive_min_area)

        return pattern_mask

    @staticmethod
    def _remove_small_components(mask: np.ndarray, min_area: int = 500) -> np.ndarray:
        """移除小型孤立连通区域（消除边缘小点点噪声）

        使用双重过滤策略：
        1. 绝对面积阈值：小于 min_area 像素的区域直接移除
        2. 相对面积阈值：小于最大连通区域 2% 的区域移除（防止碎片残留）
        """
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )
        if num_labels <= 1:
            return mask

        areas = stats[1:, cv2.CC_STAT_AREA]
        max_comp_area = float(np.max(areas)) if areas.size > 0 else 0.0
        # 相对阈值：最大连通区域面积的10%（小碎片相比主区域太小则移除）
        relative_min = int(max_comp_area * 0.10)
        effective_min = max(min_area, relative_min)

        result = np.zeros_like(mask)
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= effective_min:
                result[labels == i] = 255
        return result


# ============================================================
# 9) 纹理复杂度评估器
# ============================================================
class TextureEvaluator:
    """纹理复杂度评估器"""

    @staticmethod
    def compute_complexity_metric(img_bgr: np.ndarray, cloth_mask: np.ndarray) -> float:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        inner = cv2.erode(cloth_mask, kernel, iterations=1)
        area = float(np.count_nonzero(inner))

        if area < 300:
            return 0.0

        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        # 边缘密度
        edges = cv2.Canny(gray, 30, 100)
        edges = cv2.bitwise_and(edges, edges, mask=inner)
        edge_density = float(np.count_nonzero(edges)) / area

        # Laplacian 方差
        lap = cv2.Laplacian(gray, cv2.CV_32F)
        lap_vals = lap[inner > 0]
        lap_var = float(lap_vals.var()) if lap_vals.size else 0.0

        return float(edge_density * 3.0 + math.log1p(lap_var) * 0.15)

    def evaluate_pattern_quality(
        self, img_bgr: np.ndarray, cloth_mask: np.ndarray, params: Params
    ) -> PatternEvalResult:
        # 创建内部掩码
        border_erode = _ensure_odd(params.BORDER_ERODE, mn=1)
        if border_erode > 1:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (border_erode, border_erode)
            )
            inner = cv2.erode(cloth_mask, kernel, iterations=1)
        else:
            inner = cloth_mask.copy()

        cloth_area = int(np.count_nonzero(inner))
        if cloth_area < 300:
            return PatternEvalResult(False, 0.0, 0.0, 0, 0.0)

        # 复制密度计算逻辑
        cloth_cut = cv2.bitwise_and(img_bgr, img_bgr, mask=cloth_mask)
        work = cloth_cut.copy()
        work[cloth_mask == 0] = (255, 255, 255)

        gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        edges = cv2.Canny(gray, params.CANNY_T1, params.CANNY_T2)
        edges = cv2.bitwise_and(edges, edges, mask=inner)

        if int(np.count_nonzero(edges)) < int(params.MIN_EDGE_PIXELS):
            return PatternEvalResult(False, 0.0, 0.0, 0, 0.0)

        k = _ensure_odd(params.DENSITY_BLUR, mn=21)
        blurred = cv2.GaussianBlur(edges, (k, k), 0)
        density = cv2.normalize(blurred, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        density = cv2.bitwise_and(density, density, mask=inner)

        vals = density[inner > 0]
        if vals.size == 0 or np.max(vals) == 0:
            return PatternEvalResult(False, 0.0, 0.0, 0, 0.0)

        # 计算阈值
        top = float(params.TOP_PERCENT)
        qthr = float(np.quantile(vals, 1.0 - top))
        mu = float(vals.mean())
        sigma = float(vals.std() + 1e-6)
        thr = max(qthr, mu + float(params.ZSCORE_K) * sigma)

        # 创建纹理掩码
        pmask = (density >= thr).astype(np.uint8) * 255
        pmask = cv2.bitwise_and(pmask, pmask, mask=inner)

        pm = _ensure_odd(params.PATTERN_MORPH, mn=1)
        if pm > 1:
            mk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (pm, pm))
            pmask = cv2.morphologyEx(pmask, cv2.MORPH_CLOSE, mk, iterations=2)
            pmask = cv2.morphologyEx(pmask, cv2.MORPH_OPEN, mk, iterations=1)
            pmask = cv2.bitwise_and(pmask, pmask, mask=inner)

        pat_area = int(np.count_nonzero(pmask))
        # 自适应小区域限制
        adaptive_min = max(int(params.MIN_PATTERN_PIXELS), int(cloth_area * 0.01))
        if pat_area < adaptive_min:
            return PatternEvalResult(False, 0.0, 0.0, 0, 0.0)

        ratio = pat_area / float(cloth_area)

        # 计算对比度
        cloth = inner > 0
        pat = (pmask > 0) & cloth
        bg = cloth & (~pat)

        if pat.sum() == 0 or bg.sum() == 0:
            return PatternEvalResult(False, ratio, 0.0, 0, 0.0)

        contrast = float(density[pat].mean() - density[bg].mean()) / 255.0

        # 连通组件数量
        num_labels, _, _, _ = cv2.connectedComponentsWithStats(
            (pmask > 0).astype(np.uint8), connectivity=8
        )
        comp = int(num_labels - 1)

        # 质量分数
        target, sigma_r = 0.12, 0.10
        ratio_score = math.exp(-((ratio - target) ** 2) / (2 * sigma_r ** 2))
        contrast_score = max(0.0, min(1.0, contrast * 2.2))
        comp_penalty = max(0.0, comp / 25.0)

        score_quality = 0.60 * ratio_score + 0.65 * contrast_score - 0.35 * comp_penalty

        return PatternEvalResult(True, float(ratio), float(contrast), int(comp), float(score_quality))


# ============================================================
# 10) 启发式搜索器
# ============================================================
class HeuristicSearcher:
    """启发式参数搜索器"""

    def __init__(self, config: Config = CONFIG):
        self.config = config
        self.evaluator = TextureEvaluator()

    def pseudo_split_pos_neg(
        self, items: List[Tuple[str, np.ndarray, np.ndarray]]
    ) -> Tuple[List, List]:
        scores = np.array([
            TextureEvaluator.compute_complexity_metric(img, cm)
            for _, img, cm in items
        ], dtype=np.float32)

        if len(scores) < 2:
            # Calib set too small, treat all as positive
            return items, []

        idx = np.argsort(scores)
        n = len(items)
        n_pos = max(1, int(round(n * self.config.PSEUDO_POS_FRAC)))
        n_neg = max(1, int(round(n * self.config.PSEUDO_NEG_FRAC)))
        
        # Ensure at least one pos if n >= 1
        if n_pos < 1 and n >= 1:
             n_pos = 1

        neg_idx = idx[:n_neg].tolist()
        pos_idx = idx[-n_pos:].tolist()

        return [items[i] for i in pos_idx], [items[i] for i in neg_idx]

    def objective_strong(
        self, pos_items: List, neg_items: List, params: Params
    ) -> ObjectiveResult:
        pos_scores = []
        pos_quality = []
        pos_found = 0
        pos_ratios = []

        for _, img, cm in pos_items:
            result = self.evaluator.evaluate_pattern_quality(img, cm, params)
            if result.ok:
                pos_found += 1
                pos_quality.append(result.score_quality)
                pos_ratios.append(result.ratio)
            pos_scores.append(result.score_quality if result.ok else 0.0)

        neg_fp = 0
        neg_ratios = []
        for _, img, cm in neg_items:
            result = self.evaluator.evaluate_pattern_quality(img, cm, params)
            if result.ok:
                neg_fp += 1
                neg_ratios.append(result.ratio)

        pos_n = max(1, len(pos_items))
        neg_n = max(1, len(neg_items))

        pos_found_rate = pos_found / float(pos_n)
        pos_mean = float(np.mean(pos_scores)) if pos_scores else 0.0
        pos_std = float(np.std(pos_scores)) if pos_scores else 0.0
        pos_ratio_mean = float(np.mean(pos_ratios)) if pos_ratios else 0.0

        neg_fp_rate = neg_fp / float(neg_n) if neg_items else 0.0
        neg_ratio_mean = float(np.mean(neg_ratios)) if neg_ratios else 0.0

        # 轻硬约束
        ratio_penalty = 0.0
        if pos_ratios:
            if pos_ratio_mean < self.config.RATIO_MIN:
                ratio_penalty += (self.config.RATIO_MIN - pos_ratio_mean) * 2.0
            if pos_ratio_mean > self.config.RATIO_MAX:
                ratio_penalty += (pos_ratio_mean - self.config.RATIO_MAX) * 2.0

        # 目标函数
        obj = (
            self.config.W_POS_FOUND * pos_found_rate
            + self.config.W_POS_QUALITY * pos_mean
            - self.config.W_POS_STABLE * pos_std
            - self.config.W_NEG_FP * neg_fp_rate
            - self.config.W_NEG_RATIO * neg_ratio_mean
            - ratio_penalty
        )

        info = {
            "pos_found_rate": pos_found_rate,
            "pos_mean": pos_mean,
            "pos_std": pos_std,
            "pos_ratio_mean": pos_ratio_mean,
            "neg_fp_rate": neg_fp_rate,
            "neg_ratio_mean": neg_ratio_mean,
            "ratio_penalty": ratio_penalty,
        }
        return ObjectiveResult(float(obj), info)

    def _propose_neighbor(self, p: Params) -> Params:
        q = Params(**asdict(p))
        keys = random.sample(
            ["CANNY_T1", "CANNY_T2", "DENSITY_BLUR", "TOP_PERCENT",
             "BORDER_ERODE", "PATTERN_MORPH", "MIN_PATTERN_PIXELS",
             "MIN_EDGE_PIXELS", "ZSCORE_K"],
            k=random.randint(1, 3)
        )

        perturbations = {
            "CANNY_T1": lambda v: v + random.randint(-5, 5),
            "CANNY_T2": lambda v: v + random.randint(-10, 10),
            "DENSITY_BLUR": lambda v: v + random.choice([-6, -4, 4, 6]),
            "TOP_PERCENT": lambda v: v + random.choice([-0.04, -0.02, 0.02, 0.04]),
            "BORDER_ERODE": lambda v: v + random.choice([-2, 2]),
            "PATTERN_MORPH": lambda v: v + random.choice([-2, 2]),
            "MIN_PATTERN_PIXELS": lambda v: v + random.choice([-100, -50, 50, 100]),
            "MIN_EDGE_PIXELS": lambda v: v + random.choice([-40, -20, 20, 40]),
            "ZSCORE_K": lambda v: v + random.choice([-0.10, -0.05, 0.05, 0.10]),
        }

        for key in keys:
            if key in perturbations:
                setattr(q, key, perturbations[key](getattr(q, key)))

        return q.normalize()

    def _random_sample_params(self, base: Params) -> Params:
        p = Params(**asdict(base))
        p.CANNY_T1 = random.randint(15, 50)
        p.CANNY_T2 = random.randint(80, 150)
        if p.CANNY_T2 <= p.CANNY_T1 + 30:
            p.CANNY_T2 = p.CANNY_T1 + 50

        p.DENSITY_BLUR = random.choice([41, 51, 61, 71, 81])
        p.TOP_PERCENT = random.choice([0.12, 0.15, 0.18, 0.20, 0.25, 0.30])
        p.BORDER_ERODE = random.choice([5, 7, 9, 11])
        p.PATTERN_MORPH = random.choice([5, 7, 9, 11])
        p.MIN_PATTERN_PIXELS = random.choice([150, 200, 250, 300, 400, 500])
        p.MIN_EDGE_PIXELS = random.choice([40, 60, 80, 100, 120])
        p.ZSCORE_K = random.choice([0.25, 0.35, 0.50, 0.65, 0.80, 1.00, 1.20])

        return p.normalize()

    def _simulated_annealing(
        self, pos_items: List, neg_items: List, start: Params
    ) -> Tuple[Params, float, Dict]:
        cfg = self.config
        cur = start.normalize()
        result = self.objective_strong(pos_items, neg_items, cur)
        cur_obj, cur_info = result.score, result.info

        best = Params(**asdict(cur))
        best_obj = cur_obj
        best_info = dict(cur_info)

        T = cfg.SA_T0
        no_imp = 0

        for _ in range(cfg.SA_MULTI_START_ITERS):
            cand = self._propose_neighbor(cur)
            result = self.objective_strong(pos_items, neg_items, cand)
            cand_obj, cand_info = result.score, result.info

            accept = cand_obj >= cur_obj or random.random() < math.exp(
                (cand_obj - cur_obj) / max(1e-9, T)
            )

            if accept:
                cur, cur_obj, cur_info = cand, cand_obj, cand_info

            if cur_obj > best_obj:
                best = Params(**asdict(cur))
                best_obj = cur_obj
                best_info = dict(cur_info)
                no_imp = 0
            else:
                no_imp += 1

            if no_imp >= cfg.SA_EARLY_PATIENCE:
                break

            T = max(cfg.SA_TMIN, T * cfg.SA_ALPHA)

        return best, best_obj, best_info

    def search(
        self, pos_items: List, neg_items: List, base: Params
    ) -> Tuple[Params, float, Dict]:
        cfg = self.config

        # Stage 1: Random sampling
        scored = []
        for _ in range(cfg.RANDOM_STAGE_TRIALS):
            p = self._random_sample_params(base)
            result = self.objective_strong(pos_items, neg_items, p)
            scored.append((result.score, p, result.info))
        scored.sort(key=lambda x: x[0], reverse=True)

        topk = scored[:max(1, cfg.RANDOM_STAGE_TOPK)]
        print("\n[SEARCH] Random stage TOP candidates:")
        for i, (obj, p, info) in enumerate(topk, 1):
            print(f"  #{i} obj={obj:.4f}  pos_found={info['pos_found_rate']:.2f}")

        # Stage 2: SA
        best_global = None
        best_obj = -1e9
        best_info = None

        for si, (obj0, p0, _) in enumerate(topk, 1):
            best, obj, info = self._simulated_annealing(pos_items, neg_items, p0)
            if obj > best_obj:
                best_global, best_obj, best_info = best, obj, info

        return best_global, best_obj, best_info


# ============================================================
# 10.5) 自迭代参数优化器
# ============================================================
class IterativeOptimizer:
    """自迭代参数优化器：逐图学习，逐步收敛到最优参数"""

    def __init__(self, config: Config = CONFIG):
        self.config = config
        self.evaluator = TextureEvaluator()
        self.best_params: Params = Params().normalize()
        self.best_score: float = -1e9
        self.no_improve_count: int = 0
        self.history: List[Dict[str, Any]] = []

    def evaluate_single(
        self, img_bgr: np.ndarray, cloth_mask: np.ndarray, params: Params
    ) -> float:
        """评估单张图片在给定参数下的综合质量分数"""
        result = self.evaluator.evaluate_pattern_quality(img_bgr, cloth_mask, params)
        if not result.ok:
            return 0.0

        ratio_ok = 1.0 if self.config.RATIO_MIN <= result.ratio <= self.config.RATIO_MAX else 0.0
        score = (
            0.30 * 1.0
            + 0.30 * result.score_quality
            + 0.20 * min(1.0, result.contrast * 2.0)
            + 0.20 * ratio_ok
        )
        return float(score)

    def _propose_neighbor(self, p: Params) -> Params:
        """在当前参数附近生成邻居参数（覆盖所有影响复杂区域判断的参数）"""
        q = Params(**asdict(p))
        all_keys = [
            "CANNY_T1", "CANNY_T2", "DENSITY_BLUR", "TOP_PERCENT",
            "BORDER_ERODE", "BORDER_ERODE_ITER", "PATTERN_MORPH",
            "MIN_PATTERN_PIXELS", "MIN_EDGE_PIXELS", "ZSCORE_K",
            "CLOTH_DENSITY_THRESH", "EDGE_COMPLEXITY_THRESH",
            "MORPH_KERNEL", "YOLO_MASK_THRESH",
        ]
        keys = random.sample(all_keys, k=random.randint(1, 4))
        perturbations = {
            "CANNY_T1": lambda v: v + random.randint(-5, 5),
            "CANNY_T2": lambda v: v + random.randint(-10, 10),
            "DENSITY_BLUR": lambda v: v + random.choice([-6, -4, 4, 6]),
            "TOP_PERCENT": lambda v: v + random.choice([-0.04, -0.02, 0.02, 0.04]),
            "BORDER_ERODE": lambda v: v + random.choice([-4, -2, 2, 4]),
            "BORDER_ERODE_ITER": lambda v: v + random.choice([-1, 1]),
            "PATTERN_MORPH": lambda v: v + random.choice([-2, 2]),
            "MIN_PATTERN_PIXELS": lambda v: v + random.choice([-100, -50, 50, 100]),
            "MIN_EDGE_PIXELS": lambda v: v + random.choice([-40, -20, 20, 40]),
            "ZSCORE_K": lambda v: v + random.choice([-0.10, -0.05, 0.05, 0.10]),
            "CLOTH_DENSITY_THRESH": lambda v: v + random.choice([-2.0, -1.0, 1.0, 2.0]),
            "EDGE_COMPLEXITY_THRESH": lambda v: v + random.choice([-0.005, -0.01, 0.005, 0.01]),
            "MORPH_KERNEL": lambda v: v + random.choice([-2, 2]),
            "YOLO_MASK_THRESH": lambda v: v + random.choice([-0.05, 0.05]),
        }
        for key in keys:
            if key in perturbations:
                setattr(q, key, perturbations[key](getattr(q, key)))
        return q.normalize()

    def optimize_step(
        self, img_bgr: np.ndarray, cloth_mask: np.ndarray, step_idx: int
    ) -> bool:
        """对一张图执行优化步骤，返回是否已收敛"""
        current_score = self.evaluate_single(img_bgr, cloth_mask, self.best_params)

        improved = False
        best_cand_score = current_score
        best_cand_params = Params(**asdict(self.best_params))

        for _ in range(self.config.ITER_NEIGHBOR_TRIALS):
            candidate = self._propose_neighbor(self.best_params)
            score = self.evaluate_single(img_bgr, cloth_mask, candidate)
            if score > best_cand_score:
                best_cand_score = score
                best_cand_params = candidate
                improved = True

        if improved and best_cand_score > self.best_score:
            self.best_params = best_cand_params
            self.best_score = best_cand_score
            self.no_improve_count = 0
        else:
            self.no_improve_count += 1

        self.history.append({
            "step": step_idx,
            "score": round(best_cand_score, 6),
            "best_score": round(self.best_score, 6),
            "improved": improved and best_cand_score > self.best_score - 1e-9,
            "no_improve_count": self.no_improve_count,
        })

        tag = "\u2713 improved" if improved and best_cand_score > self.best_score - 1e-9 else "\u2014 no improve"
        print(f"  [ITER {step_idx:>3d}] score={best_cand_score:.4f}  "
              f"best={self.best_score:.4f}  {tag}  "
              f"({self.no_improve_count}/{self.config.ITER_CONVERGENCE_N})")

        return self.no_improve_count >= self.config.ITER_CONVERGENCE_N

    def run(
        self, items: List[Tuple[np.ndarray, np.ndarray]]
    ) -> Tuple[Params, List[Dict]]:
        """
        逐图迭代优化参数。
        Args:
            items: List of (img_bgr, cloth_mask) tuples
        Returns:
            (best_params, history)
        """
        print(f"\n{'='*60}")
        print(f"[ITERATIVE OPTIMIZER] Starting with {len(items)} images")
        print(f"  max_images      = {self.config.ITER_MAX_IMAGES}")
        print(f"  convergence_n   = {self.config.ITER_CONVERGENCE_N}")
        print(f"  neighbor_trials = {self.config.ITER_NEIGHBOR_TRIALS}")
        print(f"{'='*60}")

        if not items:
            print("[WARN] No items for optimization. Using default params.")
            return self.best_params, self.history

        # Step 0: 用默认参数处理第1张图作为基准
        img0, cm0 = items[0]
        self.best_score = self.evaluate_single(img0, cm0, self.best_params)
        self.history.append({
            "step": 0,
            "score": round(self.best_score, 6),
            "best_score": round(self.best_score, 6),
            "improved": True,
            "no_improve_count": 0,
        })
        print(f"  [ITER   0] Baseline score={self.best_score:.4f}")

        # Steps 1..N: 每张图在 best_params 基础上搜索更优参数
        max_n = min(len(items), self.config.ITER_MAX_IMAGES)
        for i in range(1, max_n):
            img, cm = items[i]
            converged = self.optimize_step(img, cm, i)
            if converged:
                print(f"\n[ITERATIVE OPTIMIZER] Converged at step {i}")
                break

        print(f"\n[ITERATIVE OPTIMIZER] Final best_score = {self.best_score:.4f}")
        print(f"  best_params = {self.best_params}")
        return self.best_params, self.history


# ============================================================
# 11) 可视化输出
# ============================================================
class Visualizer:
    @staticmethod
    def save_summary_2x3(
        img_bgr: np.ndarray,
        cloth_cut: np.ndarray,
        edges_raw: np.ndarray,
        heatmap_raw_bgr: np.ndarray,
        pattern_mask: np.ndarray,
        pattern_cut: np.ndarray,
        save_path: str,
        dpi: int = 220,
        has_complex_region: bool = True
    ) -> None:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

        # titles = ["Original", "Cloth (cut)", "Cloth Lineart (Canny)",
        #           "Cloth Heatmap (Density)", "Region Mask", "Region Cut"]
        titles = ["Original", "Cloth (cut)", "Cloth Lineart (Canny)",
                  "Cloth Heatmap (Density)", "Region Mask", "Region Cut"]
        
        images = [
            cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB),
            cv2.cvtColor(cloth_cut, cv2.COLOR_BGR2RGB),
            edges_raw,
            cv2.cvtColor(heatmap_raw_bgr, cv2.COLOR_BGR2RGB),
            pattern_mask,
            cv2.cvtColor(pattern_cut, cv2.COLOR_BGR2RGB),
        ]
        cmaps = [None, None, "gray", None, "gray", None]

        for ax, title, img, cmap in zip(axes.flat, titles, images, cmaps):
            ax.set_title(title)
            if cmap == "gray":
                ax.imshow(img, cmap=cmap, vmin=0, vmax=255)
            else:
                ax.imshow(img)
            ax.axis("off")

        # 添加状态标签
        if has_complex_region:
            label_text = "\u2713 Complex Region Detected"
            label_color = "#2ecc71"  # 绿色
        else:
            label_text = "\u2717 No Complex Region (Fallback to Cloth)"
            label_color = "#e74c3c"  # 红色

        fig.suptitle(label_text, fontsize=16, fontweight="bold", color=label_color, y=1.02)

        plt.tight_layout()
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)


# ============================================================
# 12) 主流程类
# ============================================================
class ComplexAreaIdentifier:
    """复杂区域识别主类"""

    def __init__(self, config: Config = CONFIG):
        self.config = config
        self.models: Optional[ModelManager] = None
        self.params: Optional[Params] = None

    def _initialize(self, params: Params) -> None:
        """初始化模型和处理器"""
        self.models = ModelManager(self.config).load()
        self.params = params
        self.img_processor = ImageProcessor(params, use_enhanced_canny=self.config.USE_ENHANCED_CANNY)
        self.mask_builder = MaskBuilder(self.models, self.img_processor)
        self.pattern_detector = PatternDetector(self.img_processor)

    def process_frame(
        self, frame_bgr: np.ndarray, save_path: str
    ) -> bool:
        """处理单帧图像并保存复杂区域（带密度阈值和回退机制）"""
        H, W = frame_bgr.shape[:2]

        # 1. 构建人物掩码
        person_mask = self.mask_builder.build_person_mask(frame_bgr, (H, W))
        if person_mask is None:
            print(f"[SKIP] No person mask detected by YOLO")
            return False

        # 2. 构建衣物掩码
        cloth_mask = self.mask_builder.build_cloth_mask(frame_bgr, person_mask)
        if cloth_mask is None:
            print(f"[SKIP] No cloth mask generated (ROI might be None)")
            return False
            
        if np.count_nonzero(cloth_mask) == 0:
            print(f"[SKIP] Cloth mask is empty (No matching CLOTH_LABELS found)")
            return False

        # 3. 计算热力图密度，判断是否为复杂区域
        #    注意：这里复用 masked_canny_heatmap 计算热力图，然后统计 cloth_mask 区域内的均值
        _, heatmap = self.img_processor.masked_canny_heatmap(frame_bgr, cloth_mask)
        
        #    转灰度图计算密度均值
        heatmap_gray = cv2.cvtColor(heatmap, cv2.COLOR_BGR2GRAY)
        
        mean_density = 0.0
        cloth_area = np.count_nonzero(cloth_mask)
        
        #    更为严谨的做法：重新计算 edge density (0-255)
        #    不过直接用 heatmap (也是基于 density) 均值也可以作为近似
        #    为了保持和 pattern_detector 内部逻辑一致，这里重新计算一下 raw density
        if cloth_area > 0:
             inner_mask = self.pattern_detector._create_inner_mask(cloth_mask)
             gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
             if not self.img_processor.use_enhanced_canny:
                 gray = cv2.GaussianBlur(gray, (3, 3), 0)
             _, density_raw = self.img_processor.compute_edge_density(gray, inner_mask)
             
             vals = density_raw[cloth_mask > 0]
             if vals.size > 0:
                 mean_density = float(vals.mean())

        # 4. 根据密度阈值判定
        use_fallback = False
        final_cut = None
        
        if mean_density < self.params.CLOTH_DENSITY_THRESH:
            print(f"[INFO] Low density ({mean_density:.2f} < {self.params.CLOTH_DENSITY_THRESH}). Fallback to cloth.")
            use_fallback = True
        else:
            # 尝试检测复杂区域
            region = self.pattern_detector.detect_pattern_region(frame_bgr, cloth_mask)
            if region is None:
                print(f"[INFO] No complex region detected. Fallback to cloth.")
                use_fallback = True
            else:
                _, final_cut = region

        # 5. 如果需要回退（密度低 或 没检测到），直接扣取衣物
        has_complex = not use_fallback
        if use_fallback:
            final_cut = cv2.bitwise_and(frame_bgr, frame_bgr, mask=cloth_mask)
            # 为了可视化，设置 pattern_mask = cloth_mask
            # 注意：这里的 pattern_mask 用于展示"识别出的区域"
            region_mask_vis = cloth_mask
        else:
            region_mask_vis = region[0] # pattern_mask
        
        # 6. 保存结果 (Combined Summary 2x3)
        if final_cut is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # 计算 display 用的 edge & heatmap
            edges_vis, heatmap_vis = self.img_processor.masked_canny_heatmap(frame_bgr, cloth_mask)
            
            # Create RGB cloth cut for visualization
            cloth_cut_vis = cv2.bitwise_and(frame_bgr, frame_bgr, mask=cloth_mask)
            
            # Visualizer.save_summary_2x3(
            #     frame_bgr, 
            #     cloth_cut_vis, 
            #     edges_vis, 
            #     heatmap_vis, 
            #     region_mask_vis, 
            #     final_cut, 
            #     save_path, 
            #     self.config.SAVE_DPI,
            #     has_complex_region=has_complex
            # )
            
            # Create RGBA cloth cut for saving
            b, g, r = cv2.split(frame_bgr)
            cloth_cut_rgba = cv2.merge([b, g, r, cloth_mask])

            # 单独保存 cloth_cut 图片
            cloth_cut_path = save_path.replace('.png', '_cloth_cut.png').replace('.jpg', '_cloth_cut.png') # Force png for transparency
            if cloth_cut_path == save_path:  # 如果没有匹配的扩展名
                cloth_cut_path = os.path.splitext(save_path)[0] + '_cloth_cut.png'
                
            cv2.imwrite(cloth_cut_path, cloth_cut_rgba)
            
            status = "COMPLEX" if has_complex else "FALLBACK"
            print(f"[OK] [{status}] Saved summary to {save_path}")
            return True
            
        return False

    def build_calib_items(
        self, calib_dir: str, fallback_dir: str, fallback_first_n: int
    ) -> List[Tuple[str, np.ndarray, np.ndarray]]:
        """构建校准集"""
        paths = list_images_in_dir(calib_dir)
        src = "CALIB_DIR" if paths else "FALLBACK"

        if not paths:
            all_paths = list_images_in_dir(fallback_dir)
            paths = all_paths[:max(6, fallback_first_n)]

        print(f"\n[CALIB] source={src} paths={len(paths)}")
        items = []

        for p in paths:
            img = cv2.imread(p)
            if img is None:
                continue

            H, W = img.shape[:2]
            pm = self.mask_builder.build_person_mask(p, (H, W))
            if pm is None:
                continue

            cm = self.mask_builder.build_cloth_mask(img, pm)
            if cm is None:
                continue

            items.append((p, img, cm))

        print(f"[CALIB] usable_items={len(items)}")
        return items

    def process_frame_debug(
        self, frame_bgr: np.ndarray, save_dir: str, file_prefix: str
    ) -> bool:
        """处理单帧图像并保存所有中间结果到指定目录"""
        H, W = frame_bgr.shape[:2]

        # 1. Save Original
        # cv2.imwrite(os.path.join(save_dir, f"{file_prefix}_original.png"), frame_bgr)

        # 构建人物掩码
        person_mask = self.mask_builder.build_person_mask(frame_bgr, (H, W))
        if person_mask is None:
            return False

        # 构建衣物掩码
        cloth_mask = self.mask_builder.build_cloth_mask(frame_bgr, person_mask)
        if cloth_mask is None:
            return False

        # 2. Save Lineart and Heatmap
        # Calculate edges and heatmap for saving
        # Note: masked_canny_heatmap returns edges and RGB heatmap
        edges, heatmap = self.img_processor.masked_canny_heatmap(frame_bgr, cloth_mask)
        # cv2.imwrite(os.path.join(save_dir, f"{file_prefix}_lineart.png"), edges)
        # cv2.imwrite(os.path.join(save_dir, f"{file_prefix}_heatmap.png"), heatmap)

        # 3. Density Threshold & Fallback Logic
        # Calculate mean density of the heatmap within the cloth mask
        density_gray = cv2.cvtColor(heatmap, cv2.COLOR_BGR2GRAY)
        
        # Calculate mean density strictly within the cloth area
        mean_density = 0.0
        cloth_area = np.count_nonzero(cloth_mask)
        if cloth_area > 0:
            # Re-compute exact density like in pattern detector
            _, density_raw = self.img_processor.compute_edge_density(
                cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY), 
                self.pattern_detector._create_inner_mask(cloth_mask)
            )
            mean_density = float(density_raw[cloth_mask > 0].mean())

        use_cloth_mask = False
        pattern_mask = None
        pattern_cut = None
        
        # Logic 1: Check Density Threshold
        if mean_density < self.params.CLOTH_DENSITY_THRESH:
            print(f"[{file_prefix}] Low density ({mean_density:.2f} < {self.params.CLOTH_DENSITY_THRESH}). Using cloth mask.")
            use_cloth_mask = True
            pattern_mask = cloth_mask
            pattern_cut = cv2.bitwise_and(frame_bgr, frame_bgr, mask=cloth_mask)
        
        else:
            # Logic 2: Attempt Complex Region Detection
            region = self.pattern_detector.detect_pattern_region(frame_bgr, cloth_mask)
            
            if region is None:
                # Fallback: No complex region found, use entire cloth
                print(f"[{file_prefix}] No complex region found. Fallback to cloth mask.")
                use_cloth_mask = True
                pattern_mask = cloth_mask
                pattern_cut = cv2.bitwise_and(frame_bgr, frame_bgr, mask=cloth_mask)
            else:
                # Success: Use detected region
                pattern_mask, pattern_cut = region
        
        # 3. Save Mask and Result (Suppressed)
        # cv2.imwrite(os.path.join(save_dir, f"{file_prefix}_mask.png"), pattern_mask)
        cv2.imwrite(os.path.join(save_dir, f"{file_prefix}_cloth_cut.png"), cloth_cut)
        
        # 4. Create Combined Image
        # Convert single channel images to BGR for concatenation
        # edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        # mask_bgr = cv2.cvtColor(pattern_mask, cv2.COLOR_GRAY2BGR)
        
        # Resize if necessary? Assuming all are same size (H, W)
        # Stack horizontally: Original | Lineart | Heatmap | Mask | Result
        # combined = np.hstack([frame_bgr, edges_bgr, heatmap, mask_bgr, pattern_cut])
        # cv2.imwrite(os.path.join(save_dir, f"{file_prefix}_combined.png"), combined)
        
        print(f"[OK] Saved combined view for {file_prefix}")
        return True


# ============================================================
# PART 2: 执行脚本 (原 extract_regions.py)
# ============================================================

def main():
    """主流程入口 — 支持 default / auto_iterative / load_json 三种模式"""
    config = CONFIG
    mode = config.OPTIMIZATION_MODE
    print(f"Initializing Complex Area Identifier... [MODE={mode}]")
    identifier = ComplexAreaIdentifier(config)

    # ========== 1. 参数初始化 ==========
    # 保存路径固定为当前目录
    best_params_path = "best_params.json"

    # load_json 模式下，额外搜索 IMG_DIR
    if mode == "load_json" and not os.path.exists(best_params_path):
        alt_path = os.path.join(config.IMG_DIR, config.BEST_JSON_NAME)
        if os.path.exists(alt_path):
            best_params_path = alt_path

    if mode == "load_json":
        params = Params.from_json(best_params_path)
        print(f"[load_json] Loaded params from {best_params_path}")
    else:
        params = Params().normalize()
        print(f"[{mode}] Starting with default params")

    identifier._initialize(params)

    # ========== 2. 加载所有 Metadata JSON ==========
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        script_dir = os.getcwd()

    all_datasets: List[Tuple[str, List[dict]]] = []  # (json_path, items)
    for rel_path in config.META_JSON_PATHS:
        # 尝试相对于当前目录
        mp = rel_path
        if not os.path.exists(mp):
            mp = os.path.join(script_dir, rel_path)
        if not os.path.exists(mp):
            print(f"[WARN] Metadata not found: {rel_path}, skipping.")
            continue
        try:
            with open(mp, 'r') as f:
                items = json.load(f)
            print(f"Loaded {len(items)} items from {os.path.basename(mp)}")
            all_datasets.append((os.path.abspath(mp), items))
        except Exception as e:
            print(f"[WARN] Failed to load {mp}: {e}")

    if not all_datasets:
        print("Error: No valid metadata JSON found.")
        return

    total_items = sum(len(items) for _, items in all_datasets)
    print(f"Total items across all datasets: {total_items}")

    # ========== 3. 帧路径解析辅助 ==========
    def resolve_video_path(video_rel_path: str, json_dir: str) -> Optional[str]:
        vp = os.path.normpath(os.path.join(json_dir, video_rel_path))
        if os.path.exists(vp):
            return vp
        # 备选路径
        for fallback_dir in [
            "/home/yanghaotian/server_data/yanghaotian/data/TikTok",
            "/home/yanghaotian/server_data/yanghaotian/digital_virtual/data/train",
        ]:
            alt = os.path.join(fallback_dir, os.path.basename(video_rel_path))
            if os.path.exists(alt):
                return alt
        return None

    def read_frame(video_path: str, idx: int) -> Optional[np.ndarray]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        cap.release()
        return frame if ret else None

    # ========== 4. 自迭代优化（如果开启） ==========
    if mode == "auto_iterative":
        print(f"\n[AUTO-ITERATIVE] Collecting calibration frames from all datasets...")
        calib_pairs: List[Tuple[np.ndarray, np.ndarray]] = []
        max_calib = config.ITER_MAX_IMAGES
        # 从所有数据集均匀采样
        per_ds = max(1, max_calib // len(all_datasets))

        for json_path, items in all_datasets:
            jdir = os.path.dirname(json_path)
            ds_name = os.path.basename(json_path)
            count = 0
            for item in items:
                if len(calib_pairs) >= max_calib:
                    break
                if count >= per_ds:
                    break
                video_rel = item.get('video_path')
                idx = item.get('idx')
                if not video_rel or idx is None:
                    continue
                vp = resolve_video_path(video_rel, jdir)
                if vp is None:
                    continue
                frame = read_frame(vp, idx)
                if frame is None:
                    continue

                H, W = frame.shape[:2]
                pm = identifier.mask_builder.build_person_mask(frame, (H, W))
                if pm is None:
                    continue
                cm = identifier.mask_builder.build_cloth_mask(frame, pm)
                if cm is None:
                    continue

                calib_pairs.append((frame, cm))
                count += 1
                print(f"  [{ds_name}] Collected {len(calib_pairs)}/{max_calib}: {os.path.basename(vp)}")

        if calib_pairs:
            random.seed(config.SA_SEED)
            optimizer = IterativeOptimizer(config)
            best_params, history = optimizer.run(calib_pairs)

            best_params.to_json(best_params_path)
            params = best_params

            identifier._initialize(params)
            print(f"[AUTO-ITERATIVE] Re-initialized with optimized params.")
        else:
            print("[WARN] No calibration pairs collected. Using default params.")

    # ========== 5. 创建输出目录 ==========
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_output_dir = os.path.join("out_put", f"run_{timestamp}")
    os.makedirs(run_output_dir, exist_ok=True)
    print(f"Created run output directory: {run_output_dir}")

    params.to_json(os.path.join(run_output_dir, "params.json"))
    if mode == "auto_iterative" and 'history' in dir():
        hist_path = os.path.join(run_output_dir, "optimization_history.json")
        with open(hist_path, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
        print(f"[SAVE] Optimization history -> {hist_path}")

    # ========== 6. 处理所有数据集的全部帧 ==========
    for json_path, items in all_datasets:
        jdir = os.path.dirname(json_path)
        ds_name = os.path.basename(json_path)
        print(f"\n{'='*60}")
        print(f"[DATASET] Processing {ds_name} ({len(items)} items)")
        print(f"{'='*60}")

        for item in items:
            video_rel_path = item.get('video_path')
            idx = item.get('idx')
            img_rel_path = item.get('img_path')

            if not video_rel_path or idx is None or not img_rel_path:
                continue

            video_path = resolve_video_path(video_rel_path, jdir)
            if video_path is None:
                print(f"Warning: Video not found: {video_rel_path}")
                continue

            print(f"Processing: {os.path.basename(video_path)} Frame: {idx}")
            frame = read_frame(video_path, idx)
            if frame is None:
                print(f"Error: Could not read frame {idx} from {video_path}")
                continue

            try:
                img_filename = os.path.basename(img_rel_path)
                save_path = os.path.join(run_output_dir, img_filename)
                success = identifier.process_frame(frame, save_path)
                if not success:
                    print(f"Failed to extract complex region for {os.path.basename(video_path)}")
            except Exception as e:
                print(f"Error processing frame: {e}")

    print(f"\n[DONE] All datasets processed. Output: {run_output_dir}")


if __name__ == "__main__":
    main()
