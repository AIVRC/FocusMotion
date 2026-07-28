"""
复杂区域识别模块

功能：识别衣物图像中的复杂花纹区域
主要步骤：
1. 使用 YOLOv8 检测人物区域
2. 使用 Human Parsing 模型分割衣物区域
3. 使用边缘检测和密度分析识别复杂纹理区域
4. 支持启发式参数搜索优化

版本：2.0 (优化版)
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

# matplotlib 后端必须在 pyplot 之前设置
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

import torch
from PIL import Image
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
from ultralytics import YOLO


# ============================================================
# 1) 配置管理类
# ============================================================
@dataclass(frozen=True)
class Config:
    """全局配置，使用 frozen 确保不被意外修改"""
    IMG_PATH: str = r"/home/yanghaotian/server_data/yanghaotian/data/applied_dataset/ref/00009_0001.png"
    IMG_DIR: str = r"../../data/TikTok_complex"
    USE_DIR_MODE: bool = False

    CALIB_DIR: str = r"F:\Data\clothes_calib"
    YOLO_SEG_WEIGHTS: str = "yolov8n-seg.pt"
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

    # 目标函数权重（调整：更重视正样本检出率，降低负样本惩罚避免过于保守）
    W_POS_FOUND: float = 2.5      # 正样本找到率权重（提高）
    W_POS_QUALITY: float = 1.2    # 正样本质量权重
    W_POS_STABLE: float = 0.20    # 稳定性惩罚（降低）
    W_NEG_FP: float = 0.8         # 负样本误检惩罚（大幅降低，避免过于保守）
    W_NEG_RATIO: float = 0.5      # 负样本 ratio 惩罚（降低）

    # ratio 约束
    RATIO_MIN: float = 0.02
    RATIO_MAX: float = 0.45


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

    CANNY_T1: int = 30
    CANNY_T2: int = 100

    DENSITY_BLUR: int = 61
    TOP_PERCENT: float = 0.20

    BORDER_ERODE: int = 7
    PATTERN_MORPH: int = 7
    MIN_PATTERN_PIXELS: int = 300
    MIN_EDGE_PIXELS: int = 80
    ZSCORE_K: float = 0.65

    def normalize(self) -> "Params":
        """归一化参数到有效范围"""
        self.DENSITY_BLUR = int(np.clip(self.DENSITY_BLUR, 21, 91))
        self.DENSITY_BLUR = _ensure_odd(self.DENSITY_BLUR, mn=21)
        self.BORDER_ERODE = _ensure_odd(self.BORDER_ERODE, mn=1)
        self.PATTERN_MORPH = _ensure_odd(self.PATTERN_MORPH, mn=1)
        self.MORPH_KERNEL = _ensure_odd(self.MORPH_KERNEL, mn=3)

        self.TOP_PERCENT = float(np.clip(self.TOP_PERCENT, 0.05, 0.50))
        self.CANNY_T1 = int(np.clip(self.CANNY_T1, 10, 60))
        self.CANNY_T2 = int(np.clip(self.CANNY_T2, 50, 180))
        if self.CANNY_T2 <= self.CANNY_T1 + 10:
            self.CANNY_T2 = self.CANNY_T1 + 40

        self.MIN_PATTERN_PIXELS = int(np.clip(self.MIN_PATTERN_PIXELS, 100, 800))
        self.MIN_EDGE_PIXELS = int(np.clip(self.MIN_EDGE_PIXELS, 40, 300))
        self.ZSCORE_K = float(np.clip(self.ZSCORE_K, 0.20, 1.20))

        self.YOLO_MASK_THRESH = float(np.clip(self.YOLO_MASK_THRESH, 0.35, 0.95))
        self.KEEP_COMPONENTS = int(np.clip(self.KEEP_COMPONENTS, 1, 3))
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
    missing = [f for f in required_files if not os.path.exists(os.path.join(model_dir, f))]

    if missing:
        raise FileNotFoundError(
            f"本地 Human Parsing 模型目录缺文件：\n"
            f"  目录: {model_dir}\n"
            f"  缺少: {missing}\n\n"
            f"请下载以下文件到同一目录：\n"
            + "\n".join(f"  - {f}" for f in required_files)
        )


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

    def __init__(self, params: Params):
        self.params = params
        # 缓存形态学核
        self._kernel_cache: Dict[Tuple[int, int], np.ndarray] = {}

    def _get_kernel(self, ksize: int) -> np.ndarray:
        """获取缓存的形态学核"""
        key = (ksize, ksize)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, key
            )
        return self._kernel_cache[key]

    @staticmethod
    def bbox_from_mask(mask_u8: np.ndarray, pad: int = 20) -> Optional[Tuple[int, int, int, int]]:
        """从掩码获取边界框"""
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
        """保留最大的 N 个连通组件"""
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask_u8, connectivity=8
        )
        if num_labels <= 1:
            return mask_u8

        # 获取组件面积（排除背景）
        areas = stats[1:, cv2.CC_STAT_AREA]
        keep_indices = np.argsort(areas)[::-1][:keep_n] + 1

        result = np.zeros_like(mask_u8)
        for idx in keep_indices:
            result[labels == idx] = 255
        return result

    def postprocess_mask(self, mask_u8: np.ndarray) -> np.ndarray:
        """后处理掩码：形态学操作 + 保留最大组件"""
        ksize = _ensure_odd(self.params.MORPH_KERNEL, mn=3)
        kernel = self._get_kernel(ksize)

        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel, iterations=1)
        return self.keep_largest_components(mask_u8, keep_n=self.params.KEEP_COMPONENTS)

    def compute_edge_density(
        self, gray: np.ndarray, mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """计算边缘密度图"""
        edges = cv2.Canny(gray, self.params.CANNY_T1, self.params.CANNY_T2)
        edges = cv2.bitwise_and(edges, edges, mask=mask)

        k = _ensure_odd(self.params.DENSITY_BLUR, mn=21)
        blurred = cv2.GaussianBlur(edges, (k, k), 0)
        density = cv2.normalize(blurred, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        density = cv2.bitwise_and(density, density, mask=mask)

        return edges, density

    def compute_threshold(self, vals: np.ndarray) -> float:
        """计算自适应阈值"""
        top = max(0.05, min(0.60, float(self.params.TOP_PERCENT)))
        quantile_thr = float(np.quantile(vals, 1.0 - top))

        mean_val = float(vals.mean())
        std_val = float(vals.std() + 1e-6)
        zscore_thr = mean_val + float(self.params.ZSCORE_K) * std_val

        return max(quantile_thr, zscore_thr)

    def masked_canny_heatmap(
        self, img_bgr: np.ndarray, mask_u8: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """计算带掩码的 Canny 边缘和热力图（仅在掩码区域内）"""
        # 1. 准备工作图：背景置白
        work = img_bgr.copy()
        work[mask_u8 == 0] = (255, 255, 255)

        # 2. 转灰度并根据 mask 提取边缘
        gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        edges = cv2.Canny(gray, self.params.CANNY_T1, self.params.CANNY_T2)
        edges = cv2.bitwise_and(edges, edges, mask=mask_u8)

        # 3. 计算热力图
        k = _ensure_odd(self.params.DENSITY_BLUR, mn=21)
        blurred = cv2.GaussianBlur(edges, (k, k), 0)
        density = cv2.normalize(blurred, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        density = cv2.bitwise_and(density, density, mask=mask_u8)
        heatmap = cv2.applyColorMap(density, cv2.COLORMAP_JET)

        return edges, heatmap

    def raw_canny_heatmap(self, img_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """(Deprecated) 计算原始 Canny 边缘和热力图"""
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        edges = cv2.Canny(gray, self.params.CANNY_T1, self.params.CANNY_T2)

        k = _ensure_odd(self.params.DENSITY_BLUR, mn=21)
        blurred = cv2.GaussianBlur(edges, (k, k), 0)
        density = cv2.normalize(blurred, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
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
        """加载所有模型"""
        check_parse_model_dir(self.config.PARSE_MODEL_DIR)

        print("Loading YOLOv8 model...")
        self.yolo = YOLO(self.config.YOLO_SEG_WEIGHTS)
        print("YOLO loaded.")

        print("Loading Human Parsing model (local)...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.processor = AutoImageProcessor.from_pretrained(
            self.config.PARSE_MODEL_DIR, local_files_only=True, use_fast=True
        )
        self.parse_model = SegformerForSemanticSegmentation.from_pretrained(
            self.config.PARSE_MODEL_DIR, local_files_only=True
        ).to(self.device).eval()

        id2label = {int(k): v for k, v in self.parse_model.config.id2label.items()}
        self.cloth_ids = [i for i, name in id2label.items() if name in CLOTH_LABELS]

        print(f"Device: {self.device}")
        print(f"Cloth IDs: {self.cloth_ids}")
        if not self.cloth_ids:
            print("[WARN] cloth_ids is empty. Check CLOTH_LABELS vs model id2label.")

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
        self, img_path: str, img_shape_hw: Tuple[int, int]
    ) -> Optional[np.ndarray]:
        """使用 YOLO 构建人物掩码"""
        H, W = img_shape_hw
        results = self.models.yolo(img_path)
        result = results[0]

        if result.masks is None:
            return None

        class_ids = result.boxes.cls.cpu().numpy().astype(int)
        person_indices = np.nonzero(class_ids == 0)[0]

        if person_indices.size == 0:
            return None

        params = self.img_processor.params
        final_mask = np.zeros((H, W), dtype=np.uint8)

        for idx in person_indices:
            mask_data = result.masks.data[idx].detach().cpu().numpy()
            resized = cv2.resize(mask_data, (W, H), interpolation=cv2.INTER_NEAREST)
            binary = (resized > params.YOLO_MASK_THRESH).astype(np.uint8) * 255
            final_mask = cv2.bitwise_or(final_mask, binary)

        return final_mask

    def build_cloth_mask(
        self, img_bgr: np.ndarray, person_mask: np.ndarray
    ) -> Optional[np.ndarray]:
        """使用 Human Parsing 模型构建衣物掩码"""
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
        """检测衣物上的复杂纹理区域"""
        params = self.img_processor.params
        cloth_cut = cv2.bitwise_and(img_bgr, img_bgr, mask=cloth_mask)

        # 创建内部掩码（去除边界）
        inner_mask = self._create_inner_mask(cloth_mask)

        # 准备灰度图
        work = cloth_cut.copy()
        work[cloth_mask == 0] = (255, 255, 255)
        gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        # 计算边缘密度
        edges, density = self.img_processor.compute_edge_density(gray, inner_mask)

        if np.count_nonzero(edges) < params.MIN_EDGE_PIXELS:
            return None

        # 计算阈值并生成纹理掩码
        vals = density[inner_mask > 0]
        if vals.size == 0 or np.max(vals) == 0:
            return None

        threshold = self.img_processor.compute_threshold(vals)
        pattern_mask = self._create_pattern_mask(density, inner_mask, threshold)

        if np.count_nonzero(pattern_mask) < params.MIN_PATTERN_PIXELS:
            return None

        pattern_cut = cv2.bitwise_and(img_bgr, img_bgr, mask=pattern_mask)
        return pattern_mask, pattern_cut

    def _create_inner_mask(self, cloth_mask: np.ndarray) -> np.ndarray:
        """创建去除边界的内部掩码"""
        params = self.img_processor.params
        border_erode = _ensure_odd(params.BORDER_ERODE, mn=1)

        if border_erode > 1:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (border_erode, border_erode)
            )
            return cv2.erode(cloth_mask, kernel, iterations=1)
        return cloth_mask.copy()

    def _create_pattern_mask(
        self, density: np.ndarray, inner_mask: np.ndarray, threshold: float
    ) -> np.ndarray:
        """创建纹理掩码"""
        params = self.img_processor.params

        pattern_mask = (density >= threshold).astype(np.uint8) * 255
        pattern_mask = cv2.bitwise_and(pattern_mask, pattern_mask, mask=inner_mask)

        pm = _ensure_odd(params.PATTERN_MORPH, mn=1)
        if pm > 1:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (pm, pm))
            pattern_mask = cv2.morphologyEx(pattern_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            pattern_mask = cv2.morphologyEx(pattern_mask, cv2.MORPH_OPEN, kernel, iterations=1)
            pattern_mask = cv2.bitwise_and(pattern_mask, pattern_mask, mask=inner_mask)

        return pattern_mask


# ============================================================
# 9) 纹理复杂度评估器
# ============================================================
class TextureEvaluator:
    """纹理复杂度评估器"""

    @staticmethod
    def compute_complexity_metric(img_bgr: np.ndarray, cloth_mask: np.ndarray) -> float:
        """计算纹理复杂度指标"""
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
        """评估纹理图案质量"""
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
        if pat_area < int(params.MIN_PATTERN_PIXELS):
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
        """伪标签划分正负样本"""
        scores = np.array([
            TextureEvaluator.compute_complexity_metric(img, cm)
            for _, img, cm in items
        ], dtype=np.float32)

        if len(scores) < 6:
            return items, []

        idx = np.argsort(scores)
        n = len(items)
        n_pos = max(2, int(round(n * self.config.PSEUDO_POS_FRAC)))
        n_neg = max(2, int(round(n * self.config.PSEUDO_NEG_FRAC)))

        neg_idx = idx[:n_neg].tolist()
        pos_idx = idx[-n_pos:].tolist()

        return [items[i] for i in pos_idx], [items[i] for i in neg_idx]

    def objective_strong(
        self, pos_items: List, neg_items: List, params: Params
    ) -> ObjectiveResult:
        """强目标函数"""
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
        """生成邻近参数"""
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
        """随机采样参数"""
        p = Params(**asdict(base))

        # 限制搜索范围，避免找到过于保守的参数
        p.CANNY_T1 = random.randint(15, 50)  # 收窄范围
        p.CANNY_T2 = random.randint(80, 150)  # 收窄范围
        if p.CANNY_T2 <= p.CANNY_T1 + 30:
            p.CANNY_T2 = p.CANNY_T1 + 50

        p.DENSITY_BLUR = random.choice([41, 51, 61, 71, 81])  # 限制最大模糊度
        p.TOP_PERCENT = random.choice([0.12, 0.15, 0.18, 0.20, 0.25, 0.30])  # 收窄范围
        p.BORDER_ERODE = random.choice([5, 7, 9, 11])
        p.PATTERN_MORPH = random.choice([5, 7, 9, 11])
        p.MIN_PATTERN_PIXELS = random.choice([150, 200, 250, 300, 400, 500])  # 限制上限，避免过大
        p.MIN_EDGE_PIXELS = random.choice([40, 60, 80, 100, 120])  # 限制上限
        p.ZSCORE_K = random.choice([0.25, 0.35, 0.50, 0.65, 0.80, 1.00, 1.20])

        return p.normalize()

    def _simulated_annealing(
        self, pos_items: List, neg_items: List, start: Params
    ) -> Tuple[Params, float, Dict]:
        """模拟退火算法"""
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
        """两阶段搜索：随机采样 + 多起点模拟退火"""
        cfg = self.config

        # Stage 1: 随机采样
        scored = []
        for _ in range(cfg.RANDOM_STAGE_TRIALS):
            p = self._random_sample_params(base)
            result = self.objective_strong(pos_items, neg_items, p)
            scored.append((result.score, p, result.info))
        scored.sort(key=lambda x: x[0], reverse=True)

        topk = scored[:max(1, cfg.RANDOM_STAGE_TOPK)]
        print("\n[SEARCH] Random stage TOP candidates:")
        for i, (obj, p, info) in enumerate(topk, 1):
            print(f"  #{i} obj={obj:.4f}  pos_found={info['pos_found_rate']:.2f} "
                  f"pos_mean={info['pos_mean']:.3f} neg_fp={info['neg_fp_rate']:.2f} "
                  f"neg_ratio={info['neg_ratio_mean']:.3f}")
            print(f"     {p}")

        # Stage 2: 多起点模拟退火
        best_global = None
        best_obj = -1e9
        best_info = None

        for si, (obj0, p0, _) in enumerate(topk, 1):
            print(f"\n[SEARCH] SA start {si}/{len(topk)} (obj0={obj0:.4f})")
            best, obj, info = self._simulated_annealing(pos_items, neg_items, p0)
            print(f"  [SA DONE] obj={obj:.4f}  pos_found={info['pos_found_rate']:.2f} "
                  f"pos_mean={info['pos_mean']:.3f} neg_fp={info['neg_fp_rate']:.2f} "
                  f"neg_ratio={info['neg_ratio_mean']:.3f}")
            print(f"           {best}")

            if obj > best_obj:
                best_global, best_obj, best_info = best, obj, info

        return best_global, best_obj, best_info


# ============================================================
# 11) 可视化输出
# ============================================================
class Visualizer:
    """结果可视化类"""

    @staticmethod
    def save_summary_2x3(
        img_bgr: np.ndarray,
        cloth_cut: np.ndarray,
        edges_raw: np.ndarray,
        heatmap_raw_bgr: np.ndarray,
        pattern_mask: np.ndarray,
        pattern_cut: np.ndarray,
        save_path: str,
        dpi: int = 220
    ) -> None:
        """保存 2x3 布局的汇总图"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

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
        self.img_processor = ImageProcessor(params)
        self.mask_builder = MaskBuilder(self.models, self.img_processor)
        self.pattern_detector = PatternDetector(self.img_processor)

    def process_frame(
        self, frame_bgr: np.ndarray, save_path: str
    ) -> bool:
        """处理单帧图像并保存复杂区域"""
        H, W = frame_bgr.shape[:2]

        # 构建人物掩码
        person_mask = self.mask_builder.build_person_mask(frame_bgr, (H, W)) # YOLO supports numpy array
        if person_mask is None:
            print(f"[SKIP] No person mask")
            return False

        # 构建衣物掩码
        cloth_mask = self.mask_builder.build_cloth_mask(frame_bgr, person_mask)
        if cloth_mask is None:
            print(f"[SKIP] No cloth mask")
            return False

        # 检测复杂区域
        region = self.pattern_detector.detect_pattern_region(frame_bgr, cloth_mask)
        if region is None:
            print(f"[SKIP] No complex region")
            return False

        _, pattern_cut = region
        
        # 确保保存目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, pattern_cut)
        print(f"[OK] Saved to {save_path}")
        return True

    def process_single_image(
        self, img_path: str, out_dir: str
    ) -> bool:
        """处理单张图像"""
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            print(f"[SKIP] cv2.imread failed: {img_path}")
            return False

        H, W = img_bgr.shape[:2]
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        save_path = os.path.join(out_dir, f"{base_name}_summary.png")

        # 构建人物掩码
        person_mask = self.mask_builder.build_person_mask(img_path, (H, W))
        if person_mask is None:
            print(f"[SKIP] No person mask: {img_path}")
            return False

        # 构建衣物掩码
        cloth_mask = self.mask_builder.build_cloth_mask(img_bgr, person_mask)
        if cloth_mask is None:
            print(f"[SKIP] No cloth mask: {img_path}")
            return False

        cloth_cut = cv2.bitwise_and(img_bgr, img_bgr, mask=cloth_mask)

        # 检测复杂区域
        region = self.pattern_detector.detect_pattern_region(img_bgr, cloth_mask)
        if region is None:
            print(f"[SKIP] No complex region: {os.path.basename(img_path)}")
            return False

        pattern_mask, pattern_cut = region
        
        # edges_raw, heatmap_raw = self.img_processor.raw_canny_heatmap(img_bgr)
        # 改为使用带 mask 的版本，以匹配 dd.py 的效果
        edges_raw, heatmap_raw = self.img_processor.masked_canny_heatmap(img_bgr, cloth_mask)

        # 保存结果
        Visualizer.save_summary_2x3(
            img_bgr, cloth_cut, edges_raw, heatmap_raw,
            pattern_mask, pattern_cut, save_path, self.config.SAVE_DPI
        )
        print(f"[OK] {os.path.basename(img_path)} -> {save_path}")
        return True

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

    def run(self) -> None:
        """执行主流程"""
        out_dir = ensure_output_dir()
        cfg = self.config

        # 构建图像列表
        if cfg.USE_DIR_MODE:
            img_list = list_images_in_dir(cfg.IMG_DIR)
            if not img_list:
                raise FileNotFoundError(f"No images found in: {cfg.IMG_DIR}")
        else:
            if not os.path.exists(cfg.IMG_PATH):
                raise FileNotFoundError(f"Image not found: {cfg.IMG_PATH}")
            img_list = [cfg.IMG_PATH]

        # 加载参数
        params_json_path = os.path.join(out_dir, cfg.BEST_JSON_NAME)
        default_params = Params().normalize()
        params = Params.from_json(params_json_path, default_params)

        # 初始化
        self._initialize(params)

        # 启发式搜索
        if cfg.ENABLE_STRONG_HEURISTIC_SEARCH:
            fallback_dir = cfg.IMG_DIR if cfg.USE_DIR_MODE else os.path.dirname(cfg.IMG_PATH)
            calib_items = self.build_calib_items(
                cfg.CALIB_DIR, fallback_dir, cfg.CALIB_FALLBACK_FIRST_N
            )

            if len(calib_items) >= 8:
                searcher = HeuristicSearcher(cfg)
                pos_items, neg_items = searcher.pseudo_split_pos_neg(calib_items)
                print(f"[PSEUDO] pos={len(pos_items)} neg={len(neg_items)} (mid dropped)")

                random.seed(cfg.SA_SEED)
                best_p, best_obj, best_info = searcher.search(pos_items, neg_items, params)
                params = best_p.normalize()
                params.to_json(params_json_path)

                print("\n[HEURISTIC BEST]")
                print(f"  obj={best_obj:.4f}")
                for k, v in best_info.items():
                    print(f"  {k}: {v}")
                print(f"  params={params}")

                # 更新处理器参数
                self.params = params
                self.img_processor = ImageProcessor(params)
                self.mask_builder = MaskBuilder(self.models, self.img_processor)
                self.pattern_detector = PatternDetector(self.img_processor)
            else:
                print("[WARN] calib usable items < 8, skip strong heuristic search.")

        # 处理所有图像
        print("\n=== RUN FULL DATA ===")
        print("[PARAMS]", self.params)

        success_count = 0
        for idx, p in enumerate(img_list, 1):
            print(f"\n[{idx}/{len(img_list)}] Processing: {p}")
            try:
                if self.process_single_image(p, out_dir):
                    success_count += 1
            except Exception as e:
                print(f"[ERROR] Failed on {p}\n  {type(e).__name__}: {e}")

        # 保存本次运行使用的参数
        self.params.to_json(params_json_path)

        print(f"\nDone. Processed {success_count}/{len(img_list)} images.")
        print(f"Outputs saved under:\n  {out_dir}")
        print(f"Params saved at:\n  {params_json_path}")


# ============================================================
# 主入口
# ============================================================
def main():
    """主函数入口"""
    identifier = ComplexAreaIdentifier()
    identifier.run()


if __name__ == "__main__":
    main()
