"""Tests for the model-independent incremental classification layer."""

from pathlib import Path

import numpy as np

import core.ai.incremental_classifier as classifier_module
from core.ai import AI_REMOVAL_LABEL
from core.ai.incremental_classifier import (
    IncrementalEmbeddingClassifier,
    inspect_feature_store,
)


def _deep_feature(index: int) -> np.ndarray:
    feature = np.zeros(512, dtype=np.float32)
    feature[index] = 1.0
    return feature


class FakeExtractor:
    model_id = "fake-embedding-v1"
    provider_label = "CPU"
    manifest = {
        "output": {"dimensions": 512},
        "classifier": {
            "neighbors_per_class": 2,
            "spatial_color_weight": 0.0,
            "uncertain_margin": 0.03,
            "minimum_samples_per_class": 1,
        },
    }

    def __init__(self, model_dir, preferred_provider="auto", logger=None):
        self.model_dir = Path(model_dir)

    def extract(self, rgb):
        index = int(rgb[0, 0, 0])
        return _deep_feature(index), np.zeros(512, dtype=np.float32)

    def extract_batch(self, rgb_images):
        extracted = [self.extract(rgb) for rgb in rgb_images]
        return (
            np.stack([deep for deep, _ in extracted]),
            np.stack([color for _, color in extracted]),
        )


def _make_classifier(monkeypatch, tmp_path):
    monkeypatch.setattr(classifier_module, "OnnxEmbeddingExtractor", FakeExtractor)
    return IncrementalEmbeddingClassifier(
        model_dir=tmp_path / "model",
        cache_path=tmp_path / "features.npz",
    )


def _add_sample(classifier, path, label, feature_index):
    classifier._append_feature(
        path,
        label,
        _deep_feature(feature_index),
        np.zeros(512, dtype=np.float32),
    )


def test_feature_store_can_be_reused_without_source_images(tmp_path):
    store_path = tmp_path / "old-project.npz"
    np.savez_compressed(
        store_path,
        store_version=np.asarray([1]),
        model_id=np.asarray(["fake-embedding-v1"]),
        paths=np.asarray(["missing/a.jpg", "missing/b.jpg"]),
        labels=np.asarray(["类别A", "类别B"]),
        deep=np.zeros((2, 512), dtype=np.float16),
        colors=np.zeros((2, 512), dtype=np.float16),
    )

    summary = inspect_feature_store(store_path)

    assert summary == {
        "store_version": 1,
        "model_id": "fake-embedding-v1",
        "sample_count": 2,
        "class_counts": {"类别A": 1, "类别B": 1},
    }


def test_predict_returns_top_suggestions_without_majority_bias(monkeypatch, tmp_path):
    classifier = _make_classifier(monkeypatch, tmp_path)
    for index in range(20):
        _add_sample(classifier, f"outside-{index}", "车位外", 1)
    _add_sample(classifier, "white-1", "白色车位内", 0)
    _add_sample(classifier, "white-2", "白色车位内", 0)
    _add_sample(classifier, "yellow-1", "黄色车位内", 2)
    _add_sample(classifier, "yellow-2", "黄色车位内", 2)
    classifier._rebuild_index()

    image = np.zeros((2, 2, 3), dtype=np.uint8)
    result = classifier.predict(
        request_id=7,
        image_path="query",
        rgb=image,
        categories=("车位外", "白色车位内", "黄色车位内"),
    )

    assert result.request_id == 7
    assert result.suggestions[0].category == "白色车位内"
    assert not result.uncertain
    assert len(result.suggestions) == 3


def test_feedback_is_immediate_and_persists(monkeypatch, tmp_path):
    classifier = _make_classifier(monkeypatch, tmp_path)
    _add_sample(classifier, "a", "类别A", 0)
    _add_sample(classifier, "b", "类别B", 1)
    classifier._rebuild_index()

    query = np.full((2, 2, 3), 2, dtype=np.uint8)
    classifier.predict(1, "new", query, ("类别A", "类别B"))
    assert classifier.learn("new", "类别C")
    assert classifier.class_counts["类别C"] == 1
    classifier.save()

    restored = _make_classifier(monkeypatch, tmp_path)
    assert restored.class_counts == {"类别A": 1, "类别B": 1, "类别C": 1}
    assert restored.sample_count == 3


def test_manual_feedback_can_extract_feature_without_prior_prediction(
    monkeypatch, tmp_path
):
    classifier = _make_classifier(monkeypatch, tmp_path)
    monkeypatch.setattr(
        classifier_module,
        "read_rgb_image",
        lambda path: np.full((2, 2, 3), 3, dtype=np.uint8),
    )

    assert classifier.learn("new", "类别A", actual_path="classified/new.jpg")

    assert classifier.class_counts == {"类别A": 1}


def test_insufficient_classes_returns_uncertain_reason(monkeypatch, tmp_path):
    classifier = _make_classifier(monkeypatch, tmp_path)
    _add_sample(classifier, "a", "唯一类别", 0)
    classifier._rebuild_index()

    image = np.zeros((2, 2, 3), dtype=np.uint8)
    result = classifier.predict(1, "query", image, ("唯一类别",))

    assert result.uncertain
    assert result.suggestions == ()
    assert "至少需要两个" in result.reason


def test_predict_includes_sparse_classes_despite_legacy_minimum(monkeypatch, tmp_path):
    monkeypatch.setitem(
        FakeExtractor.manifest["classifier"], "minimum_samples_per_class", 5
    )
    classifier = _make_classifier(monkeypatch, tmp_path)
    for index in range(5):
        _add_sample(classifier, f"a-{index}", "类别A", 0)
    for index in range(1):
        _add_sample(classifier, f"b-{index}", "类别B", 1)
    classifier._rebuild_index()

    result = classifier.predict(
        1,
        "query",
        np.ones((2, 2, 3), dtype=np.uint8),
        ("类别A", "类别B", "零样本类别"),
    )

    assert result.suggestions[0].category == "类别B"
    assert not result.uncertain
    assert {item.category for item in result.suggestions} == {"类别A", "类别B"}


def test_zero_sample_classes_do_not_count_towards_prediction_minimum(monkeypatch, tmp_path):
    classifier = _make_classifier(monkeypatch, tmp_path)
    _add_sample(classifier, "a", "类别A", 0)
    classifier._rebuild_index()

    result = classifier.predict(
        1, "query", np.zeros((2, 2, 3), dtype=np.uint8), ("类别A", "类别B")
    )

    assert result.suggestions == ()
    assert result.uncertain
    assert "至少需要两个" in result.reason


def test_removal_is_only_returned_as_a_prediction_label(monkeypatch, tmp_path):
    classifier = _make_classifier(monkeypatch, tmp_path)
    _add_sample(classifier, "outside", "车位外", 1)
    _add_sample(classifier, "removed", AI_REMOVAL_LABEL, 0)
    classifier._rebuild_index()

    image = np.zeros((2, 2, 3), dtype=np.uint8)
    result = classifier.predict(
        1,
        "query",
        image,
        ("车位外", AI_REMOVAL_LABEL),
    )

    assert result.suggestions[0].category == AI_REMOVAL_LABEL
    assert not result.uncertain
    assert classifier.class_counts[AI_REMOVAL_LABEL] == 1


def test_forget_label_purges_all_removed_samples(monkeypatch, tmp_path):
    classifier = _make_classifier(monkeypatch, tmp_path)
    _add_sample(classifier, "outside", "车位外", 1)
    _add_sample(classifier, "removed-1", AI_REMOVAL_LABEL, 0)
    _add_sample(classifier, "removed-2", AI_REMOVAL_LABEL, 2)
    classifier._rebuild_index()

    assert classifier.forget_label(AI_REMOVAL_LABEL) == 2
    assert classifier.paths == ["outside"]
    assert classifier.class_counts == {"车位外": 1}


def test_merge_adds_opt_in_samples_without_dropping_existing(
    monkeypatch, tmp_path
):
    classifier = _make_classifier(monkeypatch, tmp_path)
    _add_sample(classifier, "outside", "车位外", 1)
    classifier._rebuild_index()
    monkeypatch.setattr(
        classifier_module,
        "read_rgb_image",
        lambda path: np.full((2, 2, 3), 2, dtype=np.uint8),
    )

    classifier.merge(
        [("removed", "remove/removed.jpg", AI_REMOVAL_LABEL)]
    )

    assert classifier.class_counts == {
        "车位外": 1,
        AI_REMOVAL_LABEL: 1,
    }
