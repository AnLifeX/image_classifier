"""Interaction tests for automatic/manual AI prediction controls."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from PyQt6.QtCore import QEvent, Qt
from PyQt6.QtGui import QColor, QKeyEvent, QPixmap
from PyQt6.QtWidgets import QDialog, QLabel, QMainWindow, QPushButton

from core.ai import AI_REMOVAL_LABEL, default_ai_project_state, get_ai_model_profile
from ui.main_window import AITabEventFilter, ImageClassifier


class PredictionWindowHarness(ImageClassifier):
    """Small QMainWindow harness that reuses the production Tab interception."""

    def __init__(self):
        QMainWindow.__init__(self)
        self.ai_prediction_mode = "assist"
        self.tab_count = 0
        self.input_mode = False

    def _is_in_input_mode(self):
        return self.input_mode

    def _handle_ai_tab_once(self):
        self.tab_count += 1

    def closeEvent(self, event):
        event.accept()


class ImageLabelSpy:
    def __init__(self):
        self.image = None

    def set_image(self, image):
        self.image = image


class CachedImageDisplayHarness(ImageClassifier):
    """Capture the image forwarded by the cache-display path."""

    def __init__(self):
        QMainWindow.__init__(self)
        self.image_label = ImageLabelSpy()
        self.ai_ready_data = None

    def _on_ai_image_ready(self, image_data):
        self.ai_ready_data = image_data


def _tab_event(event_type, auto_repeat=False):
    return QKeyEvent(
        event_type,
        Qt.Key.Key_Tab,
        Qt.KeyboardModifier.NoModifier,
        "\t",
        auto_repeat,
        1,
    )


def test_tab_controls_ai_mode_once_per_physical_press(qtbot):
    window = PredictionWindowHarness()
    qtbot.addWidget(window)
    tab_filter = AITabEventFilter(window)

    with patch.object(window, "isActiveWindow", return_value=True), patch(
        "ui.main_window.QApplication.activeModalWidget", return_value=None
    ):
        assert tab_filter.eventFilter(
            window, _tab_event(QEvent.Type.KeyPress)
        )
        assert tab_filter.eventFilter(
            window, _tab_event(QEvent.Type.KeyPress, auto_repeat=True)
        )
        assert tab_filter.eventFilter(
            window, _tab_event(QEvent.Type.KeyRelease)
        )
    assert window.tab_count == 1

    window.ai_prediction_mode = "auto"
    with patch.object(window, "isActiveWindow", return_value=True), patch(
        "ui.main_window.QApplication.activeModalWidget", return_value=None
    ):
        tab_filter.eventFilter(window, _tab_event(QEvent.Type.KeyPress))
        tab_filter.eventFilter(window, _tab_event(QEvent.Type.KeyRelease))
    assert window.tab_count == 2

    window.ai_prediction_mode = "assist"
    window.input_mode = True
    with patch.object(window, "isActiveWindow", return_value=True), patch(
        "ui.main_window.QApplication.activeModalWidget", return_value=None
    ):
        assert not tab_filter.eventFilter(
            window, _tab_event(QEvent.Type.KeyPress)
        )
    assert window.tab_count == 2


def test_nested_delivery_of_same_tab_is_ignored(qtbot):
    window = PredictionWindowHarness()
    qtbot.addWidget(window)

    def nested_delivery():
        window.tab_count += 1
        window._handle_ai_tab()

    window._handle_ai_tab_once = nested_delivery
    window._handle_ai_tab()

    assert window.tab_count == 1


def test_tab_release_inside_confirmation_dialog_unlocks_next_press(qtbot):
    window = PredictionWindowHarness()
    qtbot.addWidget(window)
    tab_filter = AITabEventFilter(window)
    tab_filter._pressed = True

    with patch(
        "ui.main_window.QApplication.activeModalWidget", return_value=object()
    ):
        assert not tab_filter.eventFilter(
            window, _tab_event(QEvent.Type.KeyRelease)
        )

    assert not tab_filter._pressed

def test_cached_pixmap_display_is_forwarded_to_ai(qtbot):
    window = CachedImageDisplayHarness()
    qtbot.addWidget(window)
    pixmap = QPixmap(5, 4)
    pixmap.fill(QColor(12, 34, 56))

    window.display_image(pixmap, Path("cached.jpg"))

    assert window.image_label.image is pixmap
    assert window.ai_ready_data is pixmap


def test_qpixmap_is_converted_to_rgb_array_for_ai(qtbot):
    pixmap = QPixmap(5, 4)
    pixmap.fill(QColor(12, 34, 56))

    rgb = ImageClassifier._coerce_ai_rgb_image(None, pixmap)

    assert rgb.shape == (4, 5, 3)
    assert rgb.dtype == np.uint8
    assert np.all(rgb == np.array([12, 34, 56], dtype=np.uint8))


class PredictionLoadingHarness(ImageClassifier):
    def __init__(self):
        QMainWindow.__init__(self)
        self.statusBar = SimpleNamespace(showMessage=Mock(), clearMessage=Mock())
        self._ai_prediction_loading = False
        self._ai_auto_task = None
        self._ai_prediction_active_path = None
        self._ai_prediction_active_request_id = -1
        self.image_view_panel = SimpleNamespace(
            show_prediction_loading=Mock(),
            hide_prediction_loading=Mock(),
            delete_button=QPushButton(),
        )
        self.category_panel = SimpleNamespace(set_ai_status=Mock())

    def closeEvent(self, event):
        event.accept()

    def get_current_image_path(self):
        return Path("current-image.jpg")


def test_assist_prediction_shows_one_loading_overlay(qtbot):
    window = PredictionLoadingHarness()
    qtbot.addWidget(window)

    window._set_ai_prediction_loading(True, "assist")

    window.image_view_panel.show_prediction_loading.assert_called_once_with(
        "辅助预测中，请稍候…"
    )
    window.image_view_panel.hide_prediction_loading.assert_not_called()
    assert not window.image_view_panel.delete_button.isEnabled()


def test_full_auto_prediction_keeps_loading_overlay(qtbot):
    window = PredictionLoadingHarness()
    qtbot.addWidget(window)

    window._set_ai_prediction_loading(True, "full_auto")

    window.image_view_panel.show_prediction_loading.assert_called_once_with(
        "全自动中，请稍候…"
    )


def test_finished_prediction_restores_image_status_text(qtbot):
    window = PredictionLoadingHarness()
    qtbot.addWidget(window)
    window._ai_prediction_loading = True

    window._finish_ai_prediction_ui()

    window.statusBar.showMessage.assert_called_with("📷 current-image.jpg")
    assert not window._ai_prediction_loading


class DuplicateImageReadyHarness(ImageClassifier):
    def __init__(self):
        QMainWindow.__init__(self)
        self.ai_prediction_mode = "assist"
        self._ai_auto_task = None
        self._ai_assist_active = True
        self._ai_current_image_path = None
        self._ai_current_image_data = None
        self._ai_prediction_timer = SimpleNamespace(stop=Mock(), start=Mock())
        self._ai_manager = SimpleNamespace(is_ready=True)
        self.image_view_panel = SimpleNamespace(clear_prediction_overlay=Mock())
        self.category_panel = SimpleNamespace(set_ai_status=Mock())

    def get_current_image_path(self):
        return Path("same-image.jpg")

    def _is_active_ai_model_initialized(self):
        return True

    def _active_ai_resource_problem(self):
        return None

    def closeEvent(self, event):
        event.accept()


def test_duplicate_loader_callback_does_not_schedule_auto_prediction_twice(qtbot):
    window = DuplicateImageReadyHarness()
    qtbot.addWidget(window)
    image = np.zeros((4, 5, 3), dtype=np.uint8)

    window._on_ai_image_ready(image)
    window._on_ai_image_ready(image.copy())

    window._ai_prediction_timer.start.assert_called_once_with(
        window.AI_PREDICTION_DEBOUNCE_MS
    )
    window.image_view_panel.clear_prediction_overlay.assert_called_once_with()


def test_incremental_model_save_does_not_repredict_current_image():
    assert not ImageClassifier._ai_model_event_requires_prediction("updated")
    assert ImageClassifier._ai_model_event_requires_prediction("initialized")
    assert ImageClassifier._ai_model_event_requires_prediction("reinitialized")


class AutoTaskHarness(ImageClassifier):
    def __init__(self):
        QMainWindow.__init__(self)
        self.is_copy_mode = True
        self.is_multi_category = False
        self.image_files = [Path("a.jpg"), Path("b.jpg")]
        self.ordered_categories = ["站立", "躺倒"]
        self._ai_project_state = {
            "active_model": "balanced",
            "learn_removed_images": False,
            "models": {
                "balanced": {"class_counts": {"站立": 20, "躺倒": 20}}
            },
        }
        self._ai_auto_pause_requested = False
        self._ai_auto_task = {
            "active_path": "a.jpg",
            "processed": 0,
            "uncertain_paths": [],
            "auto_counts": {},
            "auto_removed": 0,
        }
        self._file_ops_manager = SimpleNamespace(
            move_to_category=Mock(), move_to_remove=Mock()
        )

    def closeEvent(self, event):
        event.accept()


def test_full_auto_requires_copy_mode_and_two_populated_classes(qtbot):
    window = AutoTaskHarness()
    qtbot.addWidget(window)
    assert window._ai_auto_preflight_problem() is None

    window.is_copy_mode = False
    assert "复制模式" in window._ai_auto_preflight_problem()
    window.is_copy_mode = True
    counts = window._ai_project_state["models"]["balanced"]["class_counts"]
    counts["站立"] = 1
    window.ordered_categories.append("零样本类别")
    assert window._ai_auto_preflight_problem() is None
    counts["站立"] = 0
    assert "至少需要两个" in window._ai_auto_preflight_problem()


def test_sample_warning_lists_only_short_categories_and_opt_in_removal(qtbot):
    window = AutoTaskHarness()
    qtbot.addWidget(window)
    counts = window._ai_project_state["models"]["balanced"]["class_counts"]
    counts["站立"] = 4
    assert "站立 4/5 张" in window._ai_sample_shortage_warning(5)
    assert "躺倒" not in window._ai_sample_shortage_warning(5)
    assert "移除" not in window._ai_sample_shortage_warning(5)
    window._ai_project_state["learn_removed_images"] = True
    assert "移除 0/5 张" in window._ai_sample_shortage_warning(5)
    counts["站立"] = 5
    counts[AI_REMOVAL_LABEL] = 5
    assert window._ai_sample_shortage_warning(5) == ""


def test_assist_shortage_confirmation_requires_red_force_button(qtbot):
    window = AutoTaskHarness()
    qtbot.addWidget(window)
    window._ai_project_state["models"]["balanced"]["class_counts"]["站立"] = 1

    def force_start(box):
        box.show()
        confirm = next(button for button in box.buttons() if button.text() == "强行开启")
        assert confirm.property("uiRole") == "danger"
        assert box.defaultButton().text() == "取消"
        confirm.click()

    with patch("ui.main_window.ThemedMessageBox.exec", force_start):
        assert window._confirm_ai_assist_start()
    with patch("ui.main_window.ThemedMessageBox.exec", return_value=0):
        assert not window._confirm_ai_assist_start()
    window._ai_project_state["models"]["balanced"]["class_counts"]["站立"] = 5
    with patch("ui.main_window.ThemedMessageBox.exec") as execute:
        assert window._confirm_ai_assist_start()
        execute.assert_not_called()


def test_full_auto_shortage_uses_existing_risk_dialog(qtbot):
    window = AutoTaskHarness()
    qtbot.addWidget(window)
    window._ai_project_state["models"]["balanced"]["class_counts"]["站立"] = 9

    def force_start(dialog):
        assert any("站立 9/20 张" in label.text() for label in dialog.findChildren(QLabel))
        confirm = next(button for button in dialog.findChildren(QPushButton) if "强行开启" in button.text())
        assert confirm.property("uiRole") == "danger"
        return QDialog.DialogCode.Accepted

    with patch("ui.main_window.QDialog.exec", force_start):
        assert window._confirm_ai_auto_start() is False
    with patch("ui.main_window.QDialog.exec", return_value=QDialog.DialogCode.Rejected):
        assert window._confirm_ai_auto_start() is None


def test_cancelled_sample_warning_leaves_assistance_disabled():
    window = SimpleNamespace(
        _is_active_ai_model_initialized=Mock(return_value=True),
        _active_ai_resource_problem=Mock(return_value=None),
        _ai_manager=SimpleNamespace(is_ready=True),
        ai_prediction_mode="assist",
        _ai_assist_active=False,
        _confirm_ai_assist_start=Mock(return_value=False),
        _ai_prediction_timer=Mock(),
    )

    ImageClassifier._handle_ai_tab_once(window)

    assert not window._ai_assist_active
    window._confirm_ai_assist_start.assert_called_once_with()
    window._ai_prediction_timer.stop.assert_not_called()


def test_full_auto_skips_uncertain_result_without_classifying(qtbot):
    window = AutoTaskHarness()
    qtbot.addWidget(window)
    result = SimpleNamespace(
        image_path="a.jpg", uncertain=True, suggestions=(SimpleNamespace(category="站立"),)
    )

    with patch("ui.main_window.QTimer.singleShot") as single_shot:
        window._handle_ai_auto_result(result)

    assert window._ai_auto_task["uncertain_paths"] == ["a.jpg"]
    window._file_ops_manager.move_to_category.assert_not_called()
    single_shot.assert_called_once()


def test_full_auto_commits_only_certain_prediction(qtbot):
    window = AutoTaskHarness()
    qtbot.addWidget(window)
    result = SimpleNamespace(
        image_path="a.jpg", uncertain=False, suggestions=(SimpleNamespace(category="站立"),)
    )

    with patch("ui.main_window.QTimer.singleShot"):
        window._handle_ai_auto_result(result)

    window._file_ops_manager.move_to_category.assert_called_once_with(
        "a.jpg", "站立"
    )
    assert window._ai_auto_task["auto_counts"] == {"站立": 1}


class AIReadinessHarness(ImageClassifier):
    def __init__(self, project_dir: Path):
        QMainWindow.__init__(self)
        source_dir = project_dir / "source"
        source_dir.mkdir()
        self.current_dir = source_dir
        self._ai_project_state = default_ai_project_state()
        profile = get_ai_model_profile("balanced")
        (project_dir / profile.project_model_file).touch()
        self._ai_project_state["models"]["balanced"] = {
            "initialized": True,
            "project_model_file": profile.project_model_file,
        }
        self._ai_manager = SimpleNamespace(
            preferred_provider="cpu",
            is_ready=False,
            clear_project=Mock(),
        )
        self.ai_prediction_mode = "auto"
        self._ai_auto_task = None
        self._ai_assist_active = False
        self.ai_prediction_mode_button = QPushButton()
        self.ai_model_button = QPushButton()
        self.setup_open_count = 0
        self.configure_count = 0
        self._ai_configured_project_dir = None
        self.category_panel = SimpleNamespace(set_ai_status=Mock())

    def closeEvent(self, event):
        event.accept()

    def show_ai_project_setup(self):
        self.setup_open_count += 1

    def _configure_ai_for_current_project(self, **_kwargs):
        self.configure_count += 1


def test_missing_base_resource_opens_setup_instead_of_entering_mode(
    qtbot, tmp_path
):
    window = AIReadinessHarness(tmp_path)
    qtbot.addWidget(window)

    with patch("ui.main_window.is_resource_installed", return_value=False):
        window._update_ai_prediction_mode_button()
        assert window.ai_prediction_mode_button.text() == "AI · 未就绪"

        window._on_ai_prediction_button_clicked()

    assert window.setup_open_count == 1
    assert window.ai_prediction_mode == "auto"


def test_project_restore_does_not_load_or_toast_when_resource_is_missing(
    qtbot, tmp_path
):
    window = AIReadinessHarness(tmp_path)
    qtbot.addWidget(window)

    with patch(
        "ui.main_window.is_resource_installed", return_value=False
    ), patch("ui.main_window.toast_warning") as warning_toast:
        window._restore_ai_for_current_project()

    assert window.configure_count == 0
    window._ai_manager.clear_project.assert_called_once_with()
    warning_toast.assert_not_called()
    status_message = window.category_panel.set_ai_status.call_args.args[0]
    assert "AI 未就绪" in status_message
