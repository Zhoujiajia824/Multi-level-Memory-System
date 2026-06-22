"""
DINOv2 图像特征提取器
=====================
使用真实 facebook/dinov2-base 权重提取图像 embedding。
不允许返回随机 embedding。
图像文件不存在时 hard fail。
权重加载失败时 hard fail，并提示运行 scripts/00_prepare_models.py。
特征向量用于中期记忆的 FAISS 向量检索。
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Union

import numpy as np

from src.vla_memory.perception.image_feature_extractor import ImageFeatureExtractor
from src.vla_memory.common.logging_utils import get_logger

logger = get_logger("dinov2_extractor")


class DINOv2Extractor(ImageFeatureExtractor):
    """DINOv2 图像特征提取器。

    使用 facebook/dinov2-base 模型提取图像 embedding。
    输出 L2 归一化的 768 维特征向量。

    Args:
        model_name: 模型名称，默认 facebook/dinov2-base。
        cache_dir: 模型缓存目录。
        device: 计算设备（cuda / cpu / auto）。
        normalize: 是否对特征向量进行 L2 归一化。
    """

    def __init__(
        self,
        model_name: str = "facebook/dinov2-base",
        cache_dir: str = ".cache/huggingface",
        device: str = "auto",
        normalize: bool = True,
    ):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.normalize = normalize
        self._model = None
        self._processor = None
        self._device = self._resolve_device(device)
        self._feature_dim = 768  # dinov2-base 默认

    @staticmethod
    def _resolve_device(device: str) -> str:
        """自动检测最佳计算设备。"""
        if device == "auto":
            try:
                import torch
                if torch.cuda.is_available():
                    return "cuda"
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    return "mps"
                else:
                    return "cpu"
            except ImportError:
                return "cpu"
        return device

    def load_model(self) -> None:
        """加载 DINOv2 模型权重（真实权重，不允许 mock）。

        Raises:
            RuntimeError: 模型加载失败。
        """
        try:
            import torch
            from transformers import AutoModel, AutoImageProcessor
        except ImportError as e:
            raise RuntimeError(
                f"缺少必要的依赖库: {e}\n"
                f"请安装: pip install torch transformers"
            )

        logger.info(f"正在加载 DINOv2 模型: {self.model_name}")
        logger.info(f"缓存目录: {self.cache_dir}")
        logger.info(f"设备: {self._device}")

        try:
            self._processor = AutoImageProcessor.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
            )
            self._model = AutoModel.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
            )
            self._model.to(self._device)
            self._model.eval()
            self._feature_dim = self._model.config.hidden_size
            logger.info(
                f"DINOv2 模型加载成功: dim={self._feature_dim}, device={self._device}"
            )
        except Exception as e:
            raise RuntimeError(
                f"DINOv2 模型加载失败: {e}\n"
                f"请先运行 python scripts/00_prepare_models.py 下载模型权重。"
            )

    def _ensure_loaded(self) -> None:
        """确保模型已加载，未加载则自动加载。"""
        if self._model is None:
            self.load_model()

    def extract(self, image_path: Union[str, Path]) -> np.ndarray:
        """从单张图像提取特征向量（真实 embedding，不允许随机）。

        Args:
            image_path: 图像文件路径。

        Returns:
            L2 归一化的特征向量 (feature_dim,)。

        Raises:
            FileNotFoundError: 图像文件不存在。
            RuntimeError: 特征提取失败。
        """
        self._ensure_loaded()

        import torch
        from PIL import Image

        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(
                f"图像文件不存在: {image_path}\n"
                f"请确认 nuScenes 数据集已正确放置。"
            )

        try:
            image = Image.open(str(image_path)).convert("RGB")
            inputs = self._processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self._device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self._model(**inputs)
                feature = outputs.last_hidden_state[:, 0, :].cpu().numpy().flatten()

            if self.normalize:
                norm = np.linalg.norm(feature)
                if norm > 0:
                    feature = feature / norm

            # 校验特征不是全零（可能表明模型加载异常）
            if np.all(feature == 0):
                raise RuntimeError(
                    f"提取的特征向量为全零: {image_path}\n"
                    f"模型可能加载异常，请重新运行 scripts/00_prepare_models.py"
                )

            return feature.astype(np.float32)

        except (FileNotFoundError, RuntimeError):
            raise
        except Exception as e:
            raise RuntimeError(f"图像特征提取失败 ({image_path}): {e}")

    def batch_extract(
        self,
        image_paths: List[Union[str, Path]],
        batch_size: int = 8,
    ) -> List[np.ndarray]:
        """批量提取图像特征。

        图像不存在时 hard fail，不返回随机 embedding。

        Args:
            image_paths: 图像文件路径列表。
            batch_size: 批量大小。

        Returns:
            特征向量列表。
        """
        self._ensure_loaded()

        import torch
        from PIL import Image

        features: list[np.ndarray] = []

        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i : i + batch_size]

            for path in batch_paths:
                path = Path(path)
                if not path.exists():
                    raise FileNotFoundError(
                        f"图像文件不存在: {path}\n"
                        f"批量提取中止。请确认数据集完整。"
                    )

                try:
                    img = Image.open(str(path)).convert("RGB")
                    inputs = self._processor(images=img, return_tensors="pt")
                    inputs = {k: v.to(self._device) for k, v in inputs.items()}

                    with torch.no_grad():
                        outputs = self._model(**inputs)
                        feature = outputs.last_hidden_state[:, 0, :].cpu().numpy().flatten()

                    if self.normalize:
                        norm = np.linalg.norm(feature)
                        if norm > 0:
                            feature = feature / norm

                    features.append(feature.astype(np.float32))

                except FileNotFoundError:
                    raise
                except Exception as e:
                    raise RuntimeError(f"特征提取失败 ({path}): {e}")

            logger.info(
                f"批量特征提取进度: "
                f"{min(i + batch_size, len(image_paths))}/{len(image_paths)}"
            )

        return features

    def get_feature_dim(self) -> int:
        """获取特征维度。"""
        return self._feature_dim
