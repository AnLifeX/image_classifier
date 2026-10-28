"""Incremental class-balanced KNN over fixed image embeddings."""

import logging
import math
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .contracts import AI_REMOVAL_LABEL, PredictionResult, PredictionSuggestion
from .feature_extractor import OnnxEmbeddingExtractor, read_rgb_image


TrainingSample = Tuple[str, str, str]
ProgressCallback = Callable[[int, int], None]


def inspect_feature_store(cache_path: Path) -> Dict[str, object]:
    """Validate a reusable feature store without loading its source images."""
    cache_path = Path(cache_path)
    try:
        with np.load(cache_path, allow_pickle=False) as stored:
            required = {"store_version", "model_id", "paths", "labels", "deep", "colors"}
            missing = required.difference(stored.files)
            if missing:
                raise ValueError(f"缺少字段：{', '.join(sorted(missing))}")
            store_version = int(stored["store_version"][0])
            model_id = str(stored["model_id"][0])
            paths = stored["paths"].astype(str)
            labels = stored["labels"].astype(str)
            deep = stored["deep"]
            colors = stored["colors"]
    except ValueError:
        raise
    except Exception as error:
        raise ValueError(f"无法读取 AI 特征库：{error}") from error

    if store_version != IncrementalEmbeddingClassifier.STORE_VERSION:
        raise ValueError(
            f"特征库版本不兼容（文件版本 {store_version}，"
            f"当前版本 {IncrementalEmbeddingClassifier.STORE_VERSION}）"
        )
    sample_count = len(paths)
    if len(labels) != sample_count:
        raise ValueError("图片标识与类别标签数量不一致")
    if deep.ndim != 2 or deep.shape[0] != sample_count:
        raise ValueError("视觉特征数量或维度不正确")
    if colors.ndim != 2 or colors.shape[0] != sample_count:
        raise ValueError("颜色特征数量或维度不正确")
    class_counts: Dict[str, int] = {}
    for label in labels.tolist():
        class_counts[label] = class_counts.get(label, 0) + 1
    return {
        "store_version": store_version,
        "model_id": model_id,
        "sample_count": sample_count,
        "class_counts": class_counts,
    }


class IncrementalEmbeddingClassifier:
    """A reusable engine whose public result is independent of model internals."""

    STORE_VERSION = 1
    RECENT_FEATURE_LIMIT = 64

    def __init__(
        self,
        model_dir: Path,
        cache_path: Path,
        preferred_provider: str = "auto",
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.logger = logger or logging.getLogger(__name__)
        self.extractor = OnnxEmbeddingExtractor(
            model_dir=model_dir,
            preferred_provider=preferred_provider,
            logger=self.logger,
        )
        classifier_config = self.extractor.manifest.get("classifier", {})
        self.neighbors_per_class = int(classifier_config.get("neighbors_per_class", 5))
        self.color_weight = float(classifier_config.get("spatial_color_weight", 0.6))
        self.uncertain_margin = float(classifier_config.get("uncertain_margin", 0.03))
        self.removal_uncertain_margin = float(
            classifier_config.get("removal_uncertain_margin", 0.05)
        )
        self.removal_min_similarity = float(
            classifier_config.get("removal_min_similarity", 0.65)
        )
        self.embedding_dimensions = int(self.extractor.manifest["output"]["dimensions"])
        self.color_dimensions = 512
        self.cache_path = Path(cache_path)
        self.paths: List[str] = []
        self.labels: List[str] = []
        self.deep_features = np.empty((0, self.embedding_dimensions), dtype=np.float32)
        self.color_features = np.empty((0, self.color_dimensions), dtype=np.float32)
        self.combined_features = np.empty(
            (0, self.embedding_dimensions + self.color_dimensions),
            dtype=np.float32,
        )
        self.color_mean = np.zeros((1, self.color_dimensions), dtype=np.float32)
        self.color_std = np.ones((1, self.color_dimensions), dtype=np.float32)
        self._recent_features: OrderedDict[str, Tuple[np.ndarray, np.ndarray]] = (
            OrderedDict()
        )
        self._lock = threading.RLock()
        self._extractor_lock = threading.Lock()
        self._save_lock = threading.Lock()
        self._load_store()

    @property
    def provider_label(self) -> str:
        return self.extractor.provider_label

    @property
    def uses_gpu(self) -> bool:
        return bool(
            getattr(
                self.extractor,
                "uses_gpu",
                self.provider_label.startswith("NVIDIA GPU"),
            )
        )

    @property
    def model_id(self) -> str:
        return self.extractor.model_id

    @property
    def sample_count(self) -> int:
        return len(self.paths)

    @property
    def class_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for label in self.labels:
            counts[label] = counts.get(label, 0) + 1
        return counts

    def rekey_samples(self, namespace: str) -> None:
        """Detach imported exemplars from source paths that may no longer exist."""
        with self._lock:
            self.paths = [
                f"__imported_ai_sample__/{namespace}/{index}"
                for index in range(len(self.paths))
            ]

    def _load_store(self) -> None:
        if not self.cache_path.exists():
            return
        try:
            with np.load(self.cache_path, allow_pickle=False) as stored:
                if int(stored["store_version"][0]) != self.STORE_VERSION:
                    return
                if str(stored["model_id"][0]) != self.extractor.model_id:
                    return
                self.paths = stored["paths"].astype(str).tolist()
                self.labels = stored["labels"].astype(str).tolist()
                self.deep_features = stored["deep"].astype(np.float32)
                self.color_features = stored["colors"].astype(np.float32)
            if self.deep_features.shape != (
                len(self.paths),
                self.embedding_dimensions,
            ):
                raise ValueError("深度特征缓存维度不匹配")
            if self.color_features.shape != (
                len(self.paths),
                self.color_dimensions,
            ):
                raise ValueError("颜色特征缓存维度不匹配")
            if len(self.labels) != len(self.paths):
                raise ValueError("特征缓存标签数量不匹配")
            self._rebuild_index()
        except Exception as error:
            self.logger.warning("AI 特征缓存读取失败，将重新建立: %s", error)
            self.paths = []
            self.labels = []
            self.deep_features = np.empty(
                (0, self.embedding_dimensions), dtype=np.float32
            )
            self.color_features = np.empty((0, self.color_dimensions), dtype=np.float32)

    def save(self) -> None:
        with self._lock:
            cache_path = self.cache_path
            model_id = self.extractor.model_id
            paths = np.asarray(self.paths)
            labels = np.asarray(self.labels)
            deep_features = self.deep_features.astype(np.float16)
            color_features = self.color_features.astype(np.float16)
        with self._save_lock:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            temporary_path = cache_path.with_suffix(".tmp")
            with open(temporary_path, "wb") as cache_file:
                np.savez_compressed(
                    cache_file,
                    store_version=np.asarray([self.STORE_VERSION]),
                    model_id=np.asarray([model_id]),
                    paths=paths,
                    labels=labels,
                    deep=deep_features,
                    colors=color_features,
                )
            temporary_path.replace(cache_path)

    def synchronize(
        self,
        samples: Sequence[TrainingSample],
        progress_callback: Optional[ProgressCallback] = None,
    ) -> Tuple[int, int]:
        """Align the local feature store with current human-confirmed labels."""
        sample_by_path = {
            logical_path: (actual_path, label)
            for logical_path, actual_path, label in samples
        }
        with self._lock:
            old_indices = {path: index for index, path in enumerate(self.paths)}
            retained_paths = []
            retained_labels = []
            retained_deep = []
            retained_colors = []
            missing = []
            for logical_path, (actual_path, label) in sample_by_path.items():
                old_index = old_indices.get(logical_path)
                if old_index is None:
                    missing.append((logical_path, actual_path, label))
                    continue
                retained_paths.append(logical_path)
                retained_labels.append(label)
                retained_deep.append(self.deep_features[old_index])
                retained_colors.append(self.color_features[old_index])

            self.paths = retained_paths
            self.labels = retained_labels
            self.deep_features = self._stack_or_empty(
                retained_deep, self.embedding_dimensions
            )
            self.color_features = self._stack_or_empty(
                retained_colors, self.color_dimensions
            )

        processed = 0
        extracted = 0
        total = len(missing)
        batch_size = 32 if self.uses_gpu else 16
        for start in range(0, total, batch_size):
            batch_samples = missing[start : start + batch_size]
            decoded_samples = []
            rgb_images = []
            for logical_path, actual_path, label in batch_samples:
                try:
                    rgb_images.append(read_rgb_image(actual_path))
                    decoded_samples.append((logical_path, label))
                except Exception as error:
                    self.logger.warning(
                        "跳过无法提取特征的图片 %s: %s", actual_path, error
                    )
            if rgb_images:
                with self._extractor_lock:
                    deep_batch, color_batch = self.extractor.extract_batch(rgb_images)
                with self._lock:
                    for item_index, (logical_path, label) in enumerate(decoded_samples):
                        self._append_feature(
                            logical_path,
                            label,
                            deep_batch[item_index],
                            color_batch[item_index],
                        )
                        extracted += 1
            processed = min(start + batch_size, total)
            if progress_callback is not None:
                progress_callback(processed, total)

        with self._lock:
            self._rebuild_index()
            self.save()
        return extracted, total - extracted

    def merge(
        self,
        samples: Sequence[TrainingSample],
        progress_callback: Optional[ProgressCallback] = None,
    ) -> Tuple[int, int]:
        """Add or relabel samples without discarding the existing feature store."""
        sample_by_path = {
            logical_path: (actual_path, label)
            for logical_path, actual_path, label in samples
        }
        missing = []
        with self._lock:
            old_indices = {path: index for index, path in enumerate(self.paths)}
            for logical_path, (actual_path, label) in sample_by_path.items():
                old_index = old_indices.get(logical_path)
                if old_index is None:
                    missing.append((logical_path, actual_path, label))
                else:
                    self.labels[old_index] = label

        processed = 0
        extracted = 0
        total = len(missing)
        batch_size = 32 if self.uses_gpu else 16
        for start in range(0, total, batch_size):
            batch_samples = missing[start : start + batch_size]
            decoded_samples = []
            rgb_images = []
            for logical_path, actual_path, label in batch_samples:
                try:
                    rgb_images.append(read_rgb_image(actual_path))
                    decoded_samples.append((logical_path, label))
                except Exception as error:
                    self.logger.warning(
                        "跳过无法提取特征的图片 %s: %s", actual_path, error
                    )
            if rgb_images:
                with self._extractor_lock:
                    deep_batch, color_batch = self.extractor.extract_batch(rgb_images)
                with self._lock:
                    for item_index, (logical_path, label) in enumerate(decoded_samples):
                        self._append_feature(
                            logical_path,
                            label,
                            deep_batch[item_index],
                            color_batch[item_index],
                        )
                        extracted += 1
            processed = min(start + batch_size, total)
            if progress_callback is not None:
                progress_callback(processed, total)

        with self._lock:
            self._rebuild_index()
            self.save()
        return extracted, total - extracted

    @staticmethod
    def _stack_or_empty(features: Iterable[np.ndarray], width: int) -> np.ndarray:
        feature_list = list(features)
        if not feature_list:
            return np.empty((0, width), dtype=np.float32)
        return np.stack(feature_list).astype(np.float32)

    def _append_feature(
        self,
        path: str,
        label: str,
        deep: np.ndarray,
        color: np.ndarray,
    ) -> None:
        self.paths.append(path)
        self.labels.append(label)
        self.deep_features = np.vstack(
            [self.deep_features, np.asarray(deep, dtype=np.float32).reshape(1, -1)]
        )
        self.color_features = np.vstack(
            [self.color_features, np.asarray(color, dtype=np.float32).reshape(1, -1)]
        )

    def _rebuild_index(self) -> None:
        if not self.paths:
            self.combined_features = np.empty(
                (0, self.embedding_dimensions + self.color_dimensions),
                dtype=np.float32,
            )
            return
        self.color_mean = self.color_features.mean(axis=0, keepdims=True)
        self.color_std = self.color_features.std(axis=0, keepdims=True)
        normalized_colors = (self.color_features - self.color_mean) / np.maximum(
            self.color_std, 1e-3
        )
        normalized_colors /= np.maximum(
            np.linalg.norm(normalized_colors, axis=1, keepdims=True), 1e-8
        )
        combined = np.concatenate(
            [self.deep_features, normalized_colors * self.color_weight], axis=1
        )
        combined /= np.maximum(np.linalg.norm(combined, axis=1, keepdims=True), 1e-8)
        self.combined_features = np.ascontiguousarray(combined, dtype=np.float32)

    def _remember_feature(self, path: str, deep: np.ndarray, color: np.ndarray) -> None:
        self._recent_features[path] = (deep.copy(), color.copy())
        self._recent_features.move_to_end(path)
        while len(self._recent_features) > self.RECENT_FEATURE_LIMIT:
            self._recent_features.popitem(last=False)

    def predict(
        self,
        request_id: int,
        image_path: str,
        rgb: np.ndarray,
        categories: Sequence[str],
    ) -> PredictionResult:
        started_at = time.perf_counter()
        with self._extractor_lock:
            deep, color = self.extractor.extract(rgb)
        with self._lock:
            self._remember_feature(image_path, deep, color)
            counts = self.class_counts
            allowed_categories = [
                category
                for category in categories
                if counts.get(category, 0) > 0
            ]
            if len(allowed_categories) < 2:
                elapsed = (time.perf_counter() - started_at) * 1000
                return PredictionResult(
                    request_id=request_id,
                    image_path=image_path,
                    suggestions=(),
                    uncertain=True,
                    provider=self.provider_label,
                    latency_ms=elapsed,
                    reason="至少需要两个各有 1 张有效人工样本的类别；零样本类别不参与预测",
                )

            normalized_color = (color.reshape(1, -1) - self.color_mean) / np.maximum(
                self.color_std, 1e-3
            )
            normalized_color /= max(float(np.linalg.norm(normalized_color)), 1e-8)
            combined = np.concatenate(
                [deep.reshape(1, -1), normalized_color * self.color_weight], axis=1
            )
            combined /= max(float(np.linalg.norm(combined)), 1e-8)
            similarities = (combined @ self.combined_features.T)[0]
            labels_array = np.asarray(self.labels)

            class_scores = []
            for category in allowed_categories:
                category_similarities = similarities[labels_array == category]
                take = min(self.neighbors_per_class, len(category_similarities))
                best = np.partition(category_similarities, -take)[-take:]
                class_scores.append((category, float(best.mean())))
            class_scores.sort(key=lambda item: item[1], reverse=True)

            suggestions = tuple(
                PredictionSuggestion(
                    category=category,
                    similarity=similarity,
                )
                for category, similarity in class_scores[:3]
            )
            margin = (
                class_scores[0][1] - class_scores[1][1]
                if len(class_scores) > 1
                else math.inf
            )
            minimum_margin = self.uncertain_margin
            minimum_similarity = 0.60
            if class_scores[0][0] == AI_REMOVAL_LABEL:
                minimum_margin = max(minimum_margin, self.removal_uncertain_margin)
                minimum_similarity = max(
                    minimum_similarity, self.removal_min_similarity
                )
            uncertain = (
                margin < minimum_margin
                or class_scores[0][1] < minimum_similarity
            )

        elapsed = (time.perf_counter() - started_at) * 1000
        return PredictionResult(
            request_id=request_id,
            image_path=image_path,
            suggestions=suggestions,
            uncertain=uncertain,
            provider=self.provider_label,
            latency_ms=elapsed,
        )

    def learn(
        self, image_path: str, label: str, actual_path: Optional[str] = None
    ) -> bool:
        """Apply immediate feedback using the most recently extracted feature."""
        recent = None
        with self._lock:
            recent = self._recent_features.get(image_path)
            existing_index = (
                self.paths.index(image_path) if image_path in self.paths else None
            )
            if existing_index is not None:
                self.labels[existing_index] = label
                self._rebuild_index()
                return True
        if recent is None and actual_path:
            rgb = read_rgb_image(actual_path)
            with self._extractor_lock:
                recent = self.extractor.extract(rgb)
        if recent is None:
            return False
        with self._lock:
            self._append_feature(image_path, label, recent[0], recent[1])
            self._rebuild_index()
            return True

    def forget(self, image_path: str) -> bool:
        with self._lock:
            if image_path not in self.paths:
                return False
            index = self.paths.index(image_path)
            self.paths.pop(index)
            self.labels.pop(index)
            self.deep_features = np.delete(self.deep_features, index, axis=0)
            self.color_features = np.delete(self.color_features, index, axis=0)
            self._rebuild_index()
            return True

    def forget_label(self, label: str) -> int:
        """Remove every learned sample for one opt-in label."""
        with self._lock:
            keep_indices = [
                index
                for index, stored_label in enumerate(self.labels)
                if stored_label != label
            ]
            removed_count = len(self.labels) - len(keep_indices)
            if removed_count == 0:
                return 0
            self.paths = [self.paths[index] for index in keep_indices]
            self.labels = [self.labels[index] for index in keep_indices]
            self.deep_features = self.deep_features[keep_indices]
            self.color_features = self.color_features[keep_indices]
            self._rebuild_index()
            return removed_count
