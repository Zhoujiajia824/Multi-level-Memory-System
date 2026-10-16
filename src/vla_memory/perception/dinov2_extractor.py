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
from typing import List, Optional, Union

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
        local_files_only: 只从本地加载，不连接 Hugging Face。
        model_dir: 已下载的模型目录；提供时优先于 model_name，且始终禁止下载。
    """

    def __init__(
        self,
        model_name: str = "facebook/dinov2-base",
        cache_dir: str = ".cache/huggingface",
        device: str = "auto",
        normalize: bool = True,
        local_files_only: bool = False,
        model_dir: Optional[Union[str, Path]] = None,
    ):
        self.model_name = model_name
        self.cache_dir = str(Path(cache_dir).expanduser().resolve())
        self.normalize = normalize
        self.local_files_only = local_files_only or bool(model_dir)
        self.model_dir = Path(model_dir).expanduser().resolve() if model_dir else None
        self._model = None
        self._processor = None
        self._device = self._resolve_device(device)
        self._feature_dim = 768  # dinov2-base 默认

    def _pretrained_source(self) -> tuple[str, dict]:
        """Use an exported model or either supported Hugging Face cache layout."""
        if self.model_dir is not None:
            if not self.model_dir.is_dir() or not (self.model_dir / "config.json").is_file():
                raise FileNotFoundError(
                    f"DINOv2 本地模型目录无效: {self.model_dir}\n"
                    "model_dir 必须指向含 config.json、preprocessor_config.json 和权重的目录，"
                    "也可指向 Hugging Face 缓存的 snapshots/<版本> 目录。"
                )
            return str(self.model_dir), {"local_files_only": True}

        cache_dir = Path(self.cache_dir)
        repo_dir = "models--" + self.model_name.replace("/", "--")
        # HF_HOME contains hub/, while an explicit transformers cache_dir often
        # contains models--<owner>--<name> directly. Do not scan unrelated caches.
        if not (cache_dir / repo_dir).is_dir() and (cache_dir / "hub" / repo_dir).is_dir():
            cache_dir = cache_dir / "hub"
        return self.model_name, {
            "cache_dir": str(cache_dir),
            "local_files_only": self.local_files_only,
        }

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

        source, load_options = self._pretrained_source()
        logger.info(f"正在加载 DINOv2 模型: {source}")
        logger.info(f"缓存目录: {load_options.get('cache_dir', '本地模型目录')}")
        logger.info(f"仅使用本地文件: {self.local_files_only}")
        logger.info(f"设备: {self._device}")

        try:
            processor = AutoImageProcessor.from_pretrained(source, **load_options)
            model = AutoModel.from_pretrained(source, **load_options)
            model.to(self._device)
            model.eval()
            self._feature_dim = model.config.hidden_size
            self._processor = processor
            self._model = model
            logger.info(
                f"DINOv2 模型加载成功: dim={self._feature_dim}, device={self._device}"
            )
        except Exception as e:
            guidance = (
                "当前为离线模式，未尝试联网下载。请检查上传的缓存是否包含 refs、snapshots、blobs，"
                "以及符号链接目标是否完整；也可将 feature_extractor.model_dir 指向完整的模型目录。\n"
                "服务器验证命令: python scripts/00_prepare_models.py --offline"
                f' --cache-dir "{self.cache_dir}"'
                if self.local_files_only else
                "请先运行 python scripts/00_prepare_models.py 下载模型权重。"
            )
            raise RuntimeError(
                f"DINOv2 模型加载失败: {e}\n"
                f"模型来源: {source}\n{guidance}"
            ) from e

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
