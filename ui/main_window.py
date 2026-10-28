"""
主窗口模块

包含应用程序的主窗口类ImageClassifier。
"""

import logging
import time
import psutil
import sys
import json
import threading
import shutil
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from PyQt6.QtWidgets import (QMainWindow, QVBoxLayout, QHBoxLayout, QWidget,
                            QSplitter, QLabel, QStatusBar, QToolBar
                            , QSizePolicy, QFileDialog,
                            QMessageBox, QApplication, QListView,
                            QPushButton, QAbstractItemView, QLineEdit, QTextEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox, QComboBox, QMenu, QDialog, QCheckBox)
# Phase 1.1: QListWidget已废弃，Model/View架构使用QListView
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QSize, QPoint, QObject, QEvent, QItemSelectionModel
from PyQt6.QtGui import QAction, QKeySequence, QPixmap, QColor, QIcon, QImage, QPainter, QPen, QBrush
from PyQt6.QtWidgets import QSystemTrayIcon
from .components.widgets import ExpandableSearch
# Phase 1.1: ImageListItem已废弃，Model/View架构不再需要
from .models.image_list_model import ImageListModel
from ._main_window.panels import CategoryPanel, ImageViewPanel, InfoPanel
from .delegates.image_list_delegate import ImageListDelegate
from .components.toast import toast_info, toast_success, toast_warning, toast_error
from .components.dialog_utils import (
    ThemedMessageBox,
    ThemedProgressDialog,
    configure_dialog,
    style_button,
    style_icon_button,
)
from .components.styles import ToolbarStyles, MainWindowStyles, WidgetStyles
from .components.styles.theme import default_theme
from .components.tutorial import TutorialManager
from .dialogs import (CategoryShortcutDialog, AddCategoriesDialog,
                     TabbedHelpDialog, ProgressDialog, SettingsDialog, ManageIgnoredCategoriesDialog,
                     UpdateCheckerThread, AIProjectSetupDialog)
from .managers import FileStateManager
from .managers.image_navigation_manager import ImageNavigationManager
from .managers.file_operation_manager import FileOperationManager as UIFileOperationManager
from .managers.category_manager import CategoryManager
from .managers.ai_classification_manager import AIClassificationManager
from .ai_resource_download import AIResourceDownloadManager
from .update_download import get_update_download_controller
from .update_popover import UpdateCenterPopover
from core.ai import (
    AI_PROJECT_STATE_KEY,
    AI_REMOVAL_LABEL,
    default_ai_project_state,
    get_ai_model_profile,
    get_model_resource,
    get_runtime_resource,
    is_resource_installed,
    is_project_model_initialized,
    normalize_ai_project_state,
    required_runtime_kind,
)
from core.config import Config
from utils.app_config import get_app_config
from core.scanner import FileScannerThread
from core.image_loader import HighPerformanceImageLoader
from utils.exceptions import ImageClassifierError, FileOperationError, ConfigError
from utils.file_operations import normalize_folder_name, retry_file_operation, is_network_path, remove_readonly
from core.file_manager import FileOperationManager
from core.update_utils import (
    load_ready_update,
)
from utils.performance import performance_monitor
from utils.paths import get_update_dir
from _version_ import RELEASE_NAME, __version__, compare_version, get_manifest_url
from product_channel import get_product_info


class DisabledButtonEventFilter(QObject):
    """禁用按钮的事件过滤器，用于捕获禁用按钮的点击事件"""

    def __init__(self, parent_window):
        super().__init__(parent_window)
        self.parent_window = parent_window

    def eventFilter(self, obj, event):
        """过滤事件，捕获禁用按钮的鼠标点击"""
        if event.type() == QEvent.Type.MouseButtonPress:
            # 检查按钮是否被禁用
            if not obj.isEnabled():
                # 获取当前主题模式 - 重新加载配置确保同步
                try:
                    app_config = get_app_config()
                    # 重新加载配置文件，确保获取最新值
                    app_config.reload_config()
                    theme_mode = app_config.theme_mode

                    if theme_mode == "auto":
                        toast_warning(self.parent_window, "当前为自动切换模式，请在设置中切换到手动模式")
                    elif theme_mode == "system":
                        toast_warning(self.parent_window, "当前为跟随系统模式，请在设置中切换到手动模式")
                except Exception as e:
                    self.parent_window.logger.error(f"获取主题模式失败: {e}")

                # 返回True表示事件已处理，不再传递
                return True

        # 对于其他事件，继续传递
        return super().eventFilter(obj, event)


class AITabEventFilter(QObject):
    """Consume one physical Tab press for AI without reacting to auto-repeat."""

    def __init__(self, parent_window):
        super().__init__(parent_window)
        self.parent_window = parent_window
        self._pressed = False

    def eventFilter(self, obj, event):
        if event.type() not in (QEvent.Type.KeyPress, QEvent.Type.KeyRelease):
            return False
        if event.key() != Qt.Key.Key_Tab:
            return False
        if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            return False
        # 启动确认框可能在 KeyPress 处理中同步打开，KeyRelease 随后会
        # 落到该模态框。释放必须优先解锁，不能被模态状态提前拦截。
        if event.type() == QEvent.Type.KeyRelease:
            self._pressed = False
            return QApplication.activeModalWidget() is None
        window = self.parent_window
        modal = QApplication.activeModalWidget()
        if modal is not None:
            return False
        if not window.isActiveWindow() or window._is_in_input_mode():
            return False
        if self._pressed or event.isAutoRepeat():
            return True
        self._pressed = True
        window._handle_ai_tab()
        return True


class NoSearchListView(QListView):
    """禁用键盘搜索的 QListView，避免与分类快捷键冲突"""
    def keyboardSearch(self, search: str):
        # 禁用默认的键盘搜索功能
        # 原因：数字/字母快捷键用于分类，不应触发列表跳转
        pass


class ImageClassifier(QMainWindow):
    """
    主图像分类器窗口

    通过实例变量和方法实现StateView/StateMutator/UIHooks/ImageLoader/ImageNavigator协议，
    使用鸭子类型（Protocol）而非显式继承。
    """
    
    # 信号定义
    file_moved = pyqtSignal(str, str)  # 文件移动信号(源路径, 目标路径)
    category_added = pyqtSignal(str)   # 类别添加信号
    AI_PREDICTION_DEBOUNCE_MS = 200
    
    def __init__(self):
        super().__init__()

        self.version = __version__
        self.logger = logging.getLogger(__name__)
        
        # 初始化焦点管理
        self._shortcuts_active = True
        self._last_focus_time = time.time()
        
        # 设置定期的快捷键状态检查定时器
        self._shortcut_monitor_timer = QTimer()
        self._shortcut_monitor_timer.timeout.connect(self._periodic_shortcut_check)
        self._shortcut_monitor_timer.start(30000)  # 每30秒检查一次
        
        # 初始化核心组件
        self.init_core_components()

        # 初始化状态变量
        self.init_state_variables()

        # 应用级更新下载控制器，允许进度窗口隐藏后继续在后台下载。
        self.update_download_controller = get_update_download_controller()

        # AI 资源下载同样由主窗口托管；模型配置弹窗关闭后任务继续。
        self._ai_resource_download_manager = AIResourceDownloadManager(self)
        self._ai_resource_download_manager.resource_installed.connect(
            self._on_ai_resource_installed
        )
        self._ai_resource_download_manager.resource_failed.connect(
            self._on_ai_resource_failed
        )

        # 初始化Manager（在状态变量之后）
        self._init_managers()

        # 创建用户界面
        self.init_ui()

        # UI初始化后，更新Manager的UI组件引用
        if self._nav_manager and hasattr(self, 'image_list') and hasattr(self, 'image_list_model'):
            self._nav_manager.set_ui_components(self.image_list, self.image_list_model)

        self._connect_update_download_controller()

        # 设置快捷键
        self.setup_shortcuts()

        # 初始化教程管理器
        try:
            self.tutorial_manager = TutorialManager(self)
        except Exception as e:
            self.logger.error(f"教程管理器初始化失败: {e}")
            self.tutorial_manager = None

        # 启动后固定启用静默自动检查更新。
        try:
            self._schedule_auto_update_check()
        except Exception as e:
            self.logger.debug(f"启动自动检查更新调度失败: {e}")

        # 初始化自动主题定时器
        self._auto_theme_timer = QTimer()
        self._auto_theme_timer.timeout.connect(self._check_and_apply_auto_theme)
        # 如果配置了自动模式，启动定时器（每1分钟检查一次）
        try:
            app_config = get_app_config()
            if app_config.theme_mode == "auto":
                self.start_auto_theme_timer()
                self.logger.info("自动主题模式已启用")
        except Exception as e:
            self.logger.debug(f"自动主题定时器初始化失败: {e}")

        # 延迟检查是否显示教程（确保窗口已完全初始化）
        QTimer.singleShot(1000, self._check_and_show_tutorial)

        # 注意：恢复目录的检查会在教程完成/跳过后自动触发，不在这里直接调用
    
    def _get_resource_path(self, relative_path):
        """获取资源文件路径，兼容开发环境和打包环境"""
        try:
            # PyInstaller 打包后的临时目录
            if hasattr(sys, '_MEIPASS'):
                base_path = Path(sys._MEIPASS)
                resource_path = base_path / relative_path
                if resource_path.exists():
                    self.logger.debug(f"使用打包环境资源路径: {resource_path}")
                    return resource_path
                
            # 开发环境 - 从当前文件位置查找
            base_path = Path(__file__).parent.parent
            resource_path = base_path / relative_path
            if resource_path.exists():
                self.logger.debug(f"使用开发环境资源路径: {resource_path}")
                return resource_path
                
            # 尝试从程序运行目录查找
            base_path = Path.cwd()
            resource_path = base_path / relative_path
            if resource_path.exists():
                self.logger.debug(f"使用运行目录资源路径: {resource_path}")
                return resource_path
                
            self.logger.warning(f"未找到资源文件: {relative_path}")
            return None
        except Exception as e:
            self.logger.error(f"获取资源路径失败: {e}")
            return None
        
        # 启动性能监控
        if self.enable_performance_logging:
            self.log_system_info()
    
    def init_core_components(self):
        """初始化核心组件"""
        try:
            # 配置管理器
            self.config = Config()

            # 应用配置管理器（优化8）
            from utils.app_config import get_app_config
            self.app_config = get_app_config()

            # 文件扫描器
            self.file_scanner = FileScannerThread()
            # 由 ImageNavigationManager 统一监听 scanner 信号，主窗口不再直连
            # self.file_scanner.files_found.connect(self.on_files_found)
            # self.file_scanner.initial_batch_ready.connect(self.on_initial_batch_ready)
            # self.file_scanner.scan_progress.connect(self.on_scan_progress)
            # self.file_scanner.scan_finished.connect(self.on_scan_completed)
            
            # 图像加载器
            self.image_loader = HighPerformanceImageLoader()
            # 连接图像加载器的信号
            self.image_loader.image_loaded.connect(self.on_image_loaded)
            self.image_loader.thumbnail_loaded.connect(self.on_thumbnail_loaded)
            self.image_loader.loading_progress.connect(self.on_loading_progress)

            # 文件操作管理器
            self.file_manager = FileOperationManager()


            # 线程池（用于文件操作）
            self.thread_pool = ThreadPoolExecutor(max_workers=4)

            self.logger.info("核心组件初始化完成")

        except Exception as e:
            self.logger.error(f"核心组件初始化失败: {e}")
            raise ImageClassifierError(f"核心组件初始化失败: {e}")

    def _init_managers(self):
        """
        初始化业务逻辑Manager

        Manager通过依赖注入接收接口（self实现StateView/StateMutator/UIHooks等），
        避免Manager直接访问主窗口（消除Parent Reaching反模式）。
        """
        try:
            # 图片导航管理器
            self._nav_manager = ImageNavigationManager(
                state=self,
                mutator=self,
                ui=self,
                loader=self.image_loader,  # 传入实际的image_loader而非self
                scanner=self.file_scanner,
                image_list=None,  # UI组件在init_ui后设置
                image_list_model=None,
                get_visible_indices=self._get_visible_indices,
                get_original_to_filtered_index=self._get_original_to_filtered_index_dict,
                update_category_selection_callback=self._update_category_selection_for_path,
                load_state_callback=self._load_state,
                log_image_info_callback=self._log_image_info,
                log_performance_info_callback=self._log_performance_info,
                is_network_path_callback=lambda p: is_network_path(str(p)) if p else False,
                show_loading_placeholder_callback=self._show_loading_placeholder_for_path,
                logger=self.logger
            )

            # 文件操作管理器
            self._file_ops_manager = UIFileOperationManager(
                state=self,
                mutator=self,
                ui=self,
                navigator=self,  # self实现ImageNavigator接口
                logger=self.logger
            )

            # 类别管理器
            self._category_manager = CategoryManager(
                state=self,
                mutator=self,
                ui=self,
                navigator=self._nav_manager,
                file_ops=self._file_ops_manager,
                logger=self.logger
            )

            # 模型文件保持在程序外部；缺失或不可用时只关闭 AI，不影响人工分类。
            self._ai_manager = AIClassificationManager(
                preferred_provider=get_app_config().ai_execution_provider,
                logger=self.logger,
                parent=self,
            )
            self._ai_manager.status_changed.connect(self._on_ai_status_changed)
            self._ai_manager.index_progress.connect(self._on_ai_index_progress)
            self._ai_manager.prediction_ready.connect(self._on_ai_prediction_ready)
            self._ai_manager.model_state_changed.connect(self._on_ai_model_state_changed)

            self.logger.info("Manager初始化完成")

            # 连接 Manager 信号（替代 scanner 直连）
            self._nav_manager.list_updated.connect(self.on_list_updated)
            self._nav_manager.scan_progress.connect(self.on_scan_progress)
            self._nav_manager.scan_completed.connect(self.on_scan_completed)

            # 连接 FileOperationManager 信号
            self._file_ops_manager.file_moved.connect(self.on_file_moved)
            self._file_ops_manager.file_removed.connect(self.on_file_removed)
            self._file_ops_manager.file_restored.connect(self.on_file_restored)
            self._file_ops_manager.mode_changed.connect(self.on_mode_changed)
            self._file_ops_manager.operation_failed.connect(self.on_operation_failed)

            # 连接 CategoryManager 信号
            self._category_manager.categories_changed.connect(self.on_categories_changed)
            self._category_manager.selection_changed.connect(self.on_category_selection_changed)
            self._category_manager.mode_changed.connect(self.on_category_mode_changed)
            self._category_manager.sort_mode_changed.connect(self.on_sort_mode_changed)

        except Exception as e:
            self.logger.error(f"Manager初始化失败: {e}")
            # Manager初始化失败不中断程序，降级到原有实现
            self._nav_manager = None
            self._file_ops_manager = None
            self._category_manager = None

    def _get_visible_indices(self) -> Optional[List[int]]:
        """获取可见图片的原始索引列表（供Manager回调）"""
        # 优先返回 apply_image_filter 维护的缓存
        if hasattr(self, '_visible_indices') and self._visible_indices:
            return self._visible_indices
        # 兜底：未过滤时返回 None
        return None

    def _get_original_to_filtered_index_dict(self) -> Optional[Dict[int, int]]:
        """获取原始索引到过滤后行号的映射（供Manager回调）"""
        # 优先返回 apply_image_filter 维护的缓存
        if hasattr(self, '_original_to_filtered_index') and self._original_to_filtered_index:
            return self._original_to_filtered_index
        # 兜底：未过滤时返回 None
        return None

    def _update_category_selection_for_path(self, img_path: str) -> None:
        """更新类别选中状态（带路径参数，供Manager回调）"""
        # 使用带路径参数的版本，正确更新分类状态
        self.update_category_selection_for_current_image(img_path)

    def _show_loading_placeholder_for_path(self, img_path: str) -> None:
        """显示加载占位符（带路径参数，供Manager回调）"""
        self.show_loading_placeholder()

    def _log_image_info(self, info) -> None:
        """记录图片信息（供Manager回调）"""
        self.logger.debug(f"图片信息: {info}")

    def _log_performance_info(self, *args) -> None:
        """记录性能信息（供Manager回调，支持多个参数）"""
        self.logger.debug(f"性能信息: {' '.join(str(arg) for arg in args)}")

    def _load_state(self) -> None:
        """加载状态（供Manager回调）"""
        self.load_state()

    def init_state_variables(self):
        """初始化状态变量"""
        # 文件和目录状态
        self.current_dir = None
        self.image_files = []
        self.all_image_files = []
        self.current_index = -1
        self.total_images = 0
        self.categories = set()
        self.ordered_categories = []
        self.category_buttons = []
        self.current_category_index = 0
        self.is_network_working_path = False  # 当前工作路径是否为网络路径（默认本地）

        # 操作模式
        self.is_copy_mode = True  # 复制/移动模式
        self.is_multi_category = False  # 多分类模式：False=单分类，True=多分类
        
        # 加载状态
        self.loading_in_progress = False
        self.initial_batch_loaded = False
        self.background_loading = False

        # AI 辅助/全自动任务状态。两种模式都必须由 Tab 显式启动。
        self.ai_prediction_mode = self.app_config.ai_prediction_mode
        self._ai_assist_active = False
        self._ai_auto_task = None
        self._ai_auto_pause_requested = False
        self._ai_auto_locked_widgets = []
        self._ai_tab_event_filter = AITabEventFilter(self)
        QApplication.instance().installEventFilter(self._ai_tab_event_filter)
        self._ai_current_image_data = None
        self._ai_current_image_path = None
        self._ai_prediction_loading = False
        self._ai_prediction_active_path = None
        self._ai_prediction_active_request_id = -1
        self._ai_navigation_notice_time = 0.0
        self._ai_prediction_timer = QTimer(self)
        self._ai_prediction_timer.setSingleShot(True)
        self._ai_prediction_timer.setInterval(self.AI_PREDICTION_DEBOUNCE_MS)
        self._ai_prediction_timer.timeout.connect(
            self._trigger_scheduled_ai_prediction
        )
        self._ai_project_state = default_ai_project_state()
        self._ai_pending_initialization = {}
        self._ai_configured_project_dir = None
        
        # 分类状态
        self.classified_images = {}  # 文件路径 -> 分类
        self.removed_images = set()  # 已移除的图片
        self.last_operation_category = None  # 上次操作的类别，用于保持选中状态
        self.selected_category = None
        self.last_move_time = 0
        self.saved_last_index = -1  # 从状态文件加载的上次图片索引
        
        # 性能监控
        self.enable_performance_logging = True
        self.performance_stats = {
            'switch_times': [],
            'total_switches': 0,
            'last_switch_start': 0,
            'current_image_info': {}
        }
        
        # 用户行为分析
        self.user_behavior = {
            'last_index': -1,
            'last_direction': None,
            'consecutive_forward': 0,
            'consecutive_backward': 0,
            'direction_history': []
        }
        
        # UI更新优化
        self.ui_update_lock = threading.Lock()
        self.pending_ui_updates = set()
        # Codex方案：F5刷新后等待批量更新完成再套用过滤（避免硬编码延迟）
        self._pending_reapply_filter = False

        # 文件状态管理器（延迟初始化）
        self.file_state_manager = None

        # 修复问题4：后台更新检查线程
        self.update_checker_thread = None
        self._main_update_state = "idle"
        self._main_update_message = ""
        self._pending_update_version = None
        self._pending_update_manifest = None
        self._latest_checked_version = ""
        self._update_check_manual = False
        self._update_check_animation_step = 0
        self._update_check_animation_timer = QTimer(self)
        self._update_check_animation_timer.setInterval(300)
        self._update_check_animation_timer.timeout.connect(
            self._advance_update_check_animation
        )

        # 快捷键管理器
        self.shortcut_manager = None
        self.ui_update_timer = QTimer()
        self.ui_update_timer.setSingleShot(True)
        self.ui_update_timer.timeout.connect(self.perform_batch_ui_update)

        # 高频分类只保留最后一次状态保存请求。旧实现用 singleShot(0)
        # 在 UI 线程堆积完整 JSON 写入，会阻塞万级列表的布局和滚动。
        self._pending_state_file = None
        self._state_save_timer = QTimer(self)
        self._state_save_timer.setSingleShot(True)
        self._state_save_timer.setInterval(600)
        self._state_save_timer.timeout.connect(self._flush_pending_state_save)
        self._state_save_max_timer = QTimer(self)
        self._state_save_max_timer.setSingleShot(True)
        self._state_save_max_timer.setInterval(3000)
        self._state_save_max_timer.timeout.connect(self._flush_pending_state_save)
        
        # 类别计数缓存
        self.category_counts = {}
        self.category_count_cache_time = 0
        self.category_count_cache_ttl = 30  # 30秒TTL

        # 过滤状态（ViewState）
        self.filter_unclassified_state = False
        self.filter_classified_state = False
        self.filter_removed_state = False
        self.focused_category = None
        self._category_focus_previous_classified_filter = None

        # 排序状态（ViewState）
        self.sort_mode_state = 'name'
        self.sort_ascending_state = True
        self.category_sort_mode_state = 'name'
        self.category_sort_ascending_state = True

        # 显示状态（ViewState）
        self.show_image_list_state = True
        self.show_category_panel_state = True
        self.show_status_bar_state = True

        # 预览状态（ViewState）
        self.preview_scale_state = 1.0
        self.fit_to_window_state = True

        # 搜索状态（ViewState）
        self.search_text_state = ''
        self.search_active_state = False

        # Manager实例（在init_core_components后初始化）
        self._nav_manager = None
        self._file_ops_manager = None
        self._category_manager = None

    # ==================== StateView 接口实现 ====================

    # Protocol使用鸭子类型，实例变量（如self.current_dir）自动满足接口要求
    # 以下只需实现接口中定义的非实例变量属性

    @property
    def base_dir(self) -> Optional[Path]:
        """基础目录（current_dir 的父目录）"""
        return self.current_dir.parent if self.current_dir else None

    @property
    def filter_unclassified(self) -> bool:
        """是否过滤未分类图片"""
        return self.filter_unclassified_state

    @filter_unclassified.setter
    def filter_unclassified(self, value: bool) -> None:
        """设置是否过滤未分类图片"""
        self.filter_unclassified_state = value

    @property
    def filter_classified(self) -> bool:
        """是否过滤已分类图片"""
        return self.filter_classified_state

    @filter_classified.setter
    def filter_classified(self, value: bool) -> None:
        """设置是否过滤已分类图片"""
        self.filter_classified_state = value

    @property
    def filter_removed(self) -> bool:
        """是否过滤已移除图片"""
        return self.filter_removed_state

    @filter_removed.setter
    def filter_removed(self, value: bool) -> None:
        """设置是否过滤已移除图片"""
        self.filter_removed_state = value

    @property
    def is_network_path(self) -> bool:
        """当前是否为网络路径"""
        return self.is_network_working_path

    def get_image_at_index(self, index: int) -> Optional[Path]:
        """获取指定索引的图片路径"""
        if 0 <= index < len(self.image_files):
            return Path(self.image_files[index])
        return None

    # ==================== StateMutator 接口实现 ====================

    def set_current_index(self, index: int) -> None:
        """设置当前图片索引"""
        self.current_index = index

    def set_classified_image(self, path: str, category) -> None:
        """设置图片的分类"""
        self.classified_images[path] = category

    def remove_classified_image(self, path: str) -> None:
        """移除图片的分类记录"""
        if path in self.classified_images:
            del self.classified_images[path]

    def add_removed_image(self, path: str) -> None:
        """添加到已移除列表"""
        self.removed_images.add(path)

    def remove_from_removed(self, path: str) -> None:
        """从已移除列表中移除"""
        self.removed_images.discard(path)

    def set_classified_images(self, images: dict) -> None:
        """设置已分类图片映射"""
        self.classified_images = images

    def set_removed_images(self, images: set) -> None:
        """设置已移除图片集合"""
        self.removed_images = images

    def set_copy_mode(self, is_copy: bool) -> None:
        """设置复制/移动模式"""
        self.is_copy_mode = is_copy

    def set_multi_category(self, is_multi: bool) -> None:
        """设置单/多分类模式"""
        self.is_multi_category = is_multi

    def set_current_dir(self, dir_path: Optional[Path]) -> None:
        """设置当前工作目录"""
        self.current_dir = dir_path

    def set_image_files(self, files: list) -> None:
        """设置当前图片文件列表"""
        self.image_files = files

    def set_all_image_files(self, files: list) -> None:
        """设置所有图片文件列表"""
        self.all_image_files = files

    def set_total_images(self, total: int) -> None:
        """设置图片总数"""
        self.total_images = total

    def set_categories(self, categories: set) -> None:
        """设置类别集合"""
        self.categories = categories

    def set_ordered_categories(self, categories: list) -> None:
        """设置排序后的类别列表"""
        self.ordered_categories = categories

    def add_category(self, category: str) -> None:
        """添加类别"""
        self.categories.add(category)

    def remove_category(self, category: str) -> None:
        """移除类别"""
        self.categories.discard(category)

    def set_last_operation_category(self, category: Optional[str]) -> None:
        """设置最后一次操作的类别"""
        self.last_operation_category = category

    def set_current_category_index(self, index: int) -> None:
        """设置当前选中的类别索引"""
        self.current_category_index = index

    def set_selected_category(self, category: Optional[str]) -> None:
        """设置当前选中的类别名"""
        self.selected_category = category

    def set_filter_unclassified(self, value: bool) -> None:
        """设置是否过滤未分类图片"""
        self.filter_unclassified_state = value

    def set_filter_classified(self, value: bool) -> None:
        """设置是否过滤已分类图片"""
        self.filter_classified_state = value

    def set_filter_removed(self, value: bool) -> None:
        """设置是否过滤已移除图片"""
        self.filter_removed_state = value

    def set_sort_mode(self, mode: str) -> None:
        """设置排序模式"""
        self.sort_mode_state = mode

    def set_sort_ascending(self, value: bool) -> None:
        """设置是否升序排序"""
        self.sort_ascending_state = value

    def set_category_sort_mode(self, mode: str) -> None:
        """设置类别排序模式"""
        self.category_sort_mode_state = mode

    def set_category_sort_ascending(self, value: bool) -> None:
        """设置类别是否升序排序"""
        self.category_sort_ascending_state = value

    def set_show_image_list(self, value: bool) -> None:
        """设置是否显示图片列表"""
        self.show_image_list_state = value

    def set_show_category_panel(self, value: bool) -> None:
        """设置是否显示类别面板"""
        self.show_category_panel_state = value

    def set_show_status_bar(self, value: bool) -> None:
        """设置是否显示状态栏"""
        self.show_status_bar_state = value

    def set_preview_scale(self, scale: float) -> None:
        """设置预览图缩放比例"""
        self.preview_scale_state = scale

    def set_fit_to_window(self, value: bool) -> None:
        """设置是否适应窗口大小"""
        self.fit_to_window_state = value

    def set_search_text(self, text: str) -> None:
        """设置搜索文本"""
        self.search_text_state = text

    def set_search_active(self, value: bool) -> None:
        """设置搜索是否激活"""
        self.search_active_state = value

    def set_current_requested_image(self, path: str) -> None:
        """设置当前请求加载的图片路径（用于异步回调验证）"""
        self._current_requested_image = path

    # ==================== UIHooks 接口实现 ====================

    def save_state_sync(self) -> None:
        """同步保存状态"""
        self._save_state_sync()

    def update_status_bar(self, message: str) -> None:
        """更新状态栏"""
        if hasattr(self, 'statusBar') and self.statusBar:
            self.statusBar.showMessage(message)

    def show_toast(self, toast_type: str, message: str) -> None:
        """显示Toast通知"""
        if toast_type == 'info':
            toast_info(self, message)
        elif toast_type == 'success':
            toast_success(self, message)
        elif toast_type == 'warning':
            toast_warning(self, message)
        elif toast_type == 'error':
            toast_error(self, message)

    def show_message_box(self, title: str, message: str, msg_type: str = 'info') -> None:
        """显示消息框"""
        icons = {
            'info': QMessageBox.Icon.Information,
            'warning': QMessageBox.Icon.Warning,
            'error': QMessageBox.Icon.Critical,
        }
        box = ThemedMessageBox(self)
        box.setIcon(icons.get(msg_type, QMessageBox.Icon.Information))
        box.setWindowTitle(title)
        box.setText(message)
        box.setStandardButtons(QMessageBox.StandardButton.Ok)
        box.exec()

    def show_question(
        self,
        title: str,
        message: str,
        destructive: bool = False,
        informative_text: str = "",
    ) -> bool:
        """显示确认对话框（中文按钮）"""
        msgBox = self._create_styled_message_box(
            QMessageBox.Icon.Question,
            title,
            message,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if informative_text:
            msgBox.setInformativeText(informative_text)
        msgBox.set_destructive(destructive)
        msgBox.setDefaultButton(QMessageBox.StandardButton.No)
        return msgBox.exec() == QMessageBox.StandardButton.Yes

    def show_progress_dialog(self, title: str, message: str, maximum: int = 100):
        """显示进度对话框"""
        progress = ThemedProgressDialog(message, "取消", 0, maximum, self)
        progress.setWindowTitle(title)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.show()
        return progress

    def refresh_image_list(self) -> None:
        """刷新图片列表"""
        if hasattr(self, 'image_list_model') and self.image_list_model:
            self.image_list_model.layoutChanged.emit()

    def display_image(self, image_data, path: Path) -> None:
        """显示图片"""
        if hasattr(self, 'image_label'):
            if isinstance(image_data, QPixmap):
                self.image_label.set_image(image_data)
            else:
                pixmap = self.convert_to_pixmap(image_data)
                self.image_label.set_image(pixmap)
            # 导航器命中内存缓存时加载器不会再次发 image_loaded 信号，
            # 因此已显示的缓存图片必须在这里直接交给 AI。
            self._on_ai_image_ready(image_data)

    # ==================== ImageLoader 接口实现 ====================

    def load_image(self, path: Path, priority: bool = False) -> None:
        """请求加载图片"""
        self.image_loader.load_image(str(path), priority=priority)

    def preload_images(self, paths: list) -> None:
        """预加载多张图片"""
        self.image_loader.preload_images([str(p) for p in paths])

    def is_cached(self, path: Path) -> bool:
        """检查图片是否已缓存"""
        return self.image_loader.is_cached(str(path))

    def get_from_cache(self, path: Path):
        """从缓存获取图片"""
        return self.image_loader._get_from_cache(str(path))

    def clear_cache(self) -> None:
        """清理缓存"""
        self.image_loader.clear_cache()

    # ==================== ImageNavigator 接口实现 ====================

    def select_after_removal(self, original_index: int) -> None:
        """删除图片后智能选择下一张可见图片"""
        if self._nav_manager:
            self._nav_manager.select_after_removal(original_index)
        else:
            # 降级处理：直接设置索引
            if self.image_files:
                new_index = min(original_index, len(self.image_files) - 1)
                self.current_index = max(0, new_index)
            else:
                self.current_index = -1
    
    def init_ui(self):
        """初始化美化的用户界面"""
        try:
            self.setWindowTitle(f"{RELEASE_NAME} v{self.version}")
            # 普通窗口状态下也要完整容纳核心操作区。
            self.setMinimumSize(1024, 600)
            self._start_maximized = False

            # 尝试恢复上次保存的窗口位置和大小
            if not self._restore_window_geometry():
                self._set_default_window_geometry()
            
            # 设置应用程序图标
            try:
                icon_name = get_product_info()["icon_basename"]
                icon_path = self._get_resource_path(f'assets/{icon_name}.ico')
                if icon_path and icon_path.exists():
                    app_icon = QIcon(str(icon_path))
                    self.setWindowIcon(app_icon)
                    # 同时设置应用程序图标
                    QApplication.instance().setWindowIcon(app_icon)
                    self.logger.debug(f"程序图标已加载: {icon_path}")
                else:
                    self.logger.warning(f"图标文件不存在: {icon_path}")
            except Exception as e:
                self.logger.warning(f"加载程序图标失败: {e}")
            
            # 应用主窗口样式
            self.setStyleSheet(MainWindowStyles.get_main_window_style())
            
            # 创建中央控件
            central_widget = QWidget()
            central_widget.setObjectName("central_widget")  # 教程系统需要
            self.setCentralWidget(central_widget)
            
            # 创建外层垂直布局（用于工具栏展开条 + 主内容区）
            outer_layout = QVBoxLayout(central_widget)
            outer_layout.setContentsMargins(5, 5, 5, 5)
            outer_layout.setSpacing(0)

            # 工具栏展开条（初始隐藏，工具栏折叠后显示）
            self._toolbar_expand_bar = QPushButton("⋯ 展开工具栏")
            self._toolbar_expand_bar.setObjectName("toolbar_expand_bar")
            self._toolbar_expand_bar.setFixedHeight(20)
            self._toolbar_expand_bar.setCursor(Qt.CursorShape.PointingHandCursor)
            self._toolbar_expand_bar.setToolTip("点击展开工具栏 (Ctrl+T)")
            self._toolbar_expand_bar.clicked.connect(self._toggle_toolbar)
            self._toolbar_expand_bar.hide()
            outer_layout.addWidget(self._toolbar_expand_bar)

            # 主内容区
            content_layout = QHBoxLayout()
            content_layout.setContentsMargins(0, 0, 0, 0)
            content_layout.setSpacing(5)
            outer_layout.addLayout(content_layout, 1)

            # 创建分割器
            self.splitter = QSplitter(Qt.Orientation.Horizontal)
            content_layout.addWidget(self.splitter)

            # 创建左侧面板（图片显示） - 使用ImageViewPanel
            self.image_view_panel = ImageViewPanel(self)
            self.splitter.addWidget(self.image_view_panel)

            # 连接ImageViewPanel信号
            self.image_view_panel.remove_requested.connect(self.move_to_remove)

            # 获取image_label引用（保持向后兼容）
            self.image_label = self.image_view_panel.get_image_label()
            self.image_scroll_area = self.image_view_panel.image_scroll_area

            # 创建右侧面板（控制区域）
            self.create_right_panel(self.splitter)

            # 设置分割器比例，允许右侧面板折叠
            self.splitter.setSizes([800, 400])
            self.splitter.setCollapsible(0, False)  # 左侧不可折叠
            self.splitter.setCollapsible(1, True)    # 右侧可拖拽折叠
            self._saved_splitter_sizes = None  # 记录折叠前的分割比例
            
            # 创建工具栏
            self.create_toolbar()
            
            # 创建状态栏
            self.create_status_bar()

            # 应用主题到所有UI组件（在所有组件创建完成后）
            try:
                self.apply_theme()
                self.logger.info(f"已应用主题: {default_theme.get_current_theme()}")
            except Exception as e:
                self.logger.error(f"应用主题失败: {e}")

            self.logger.info("用户界面初始化完成")

        except Exception as e:
            self.logger.error(f"用户界面初始化失败: {e}")
            raise ImageClassifierError(f"用户界面初始化失败: {e}")

    def create_right_panel(self, parent):
        """创建简洁的右侧控制面板"""
        self.right_widget = QWidget()
        self.right_widget.setObjectName("right_panel")  # 设置对象名用于精确样式选择
        self.right_widget.setMaximumWidth(380)
        self.right_widget.setMinimumWidth(220)
        right_layout = QVBoxLayout(self.right_widget)
        right_layout.setContentsMargins(6, 6, 6, 6)
        right_layout.setSpacing(6)
        
        # 图片列表区域 - 设置拉伸权重
        self.create_image_list_area(right_layout)

        # 类别面板 - 使用CategoryPanel组件
        self.category_panel = CategoryPanel(self.config, self)
        right_layout.addWidget(self.category_panel, 1)  # 设置拉伸权重1

        # 连接CategoryPanel信号
        self.category_panel.category_selected.connect(self._on_category_selected)
        self.category_panel.category_confirmed.connect(self.move_to_category)
        self.category_panel.operation_requested.connect(self._on_category_operation_requested)
        self.category_panel.sort_mode_changed.connect(self.change_category_sort_mode)
        self.category_panel.sort_direction_toggled.connect(self.toggle_sort_direction)

        # 信息面板 - 包含统计信息和提示文本
        self.info_panel = InfoPanel(self)
        right_layout.addWidget(self.info_panel, 0)  # 不拉伸

        # 保留statistics_panel引用（向后兼容）
        self.statistics_panel = self.info_panel.statistics_panel

        parent.addWidget(self.right_widget)
    
    def create_image_list_area(self, layout):
        """创建简洁的图片列表区域"""
        # 图片列表标题行 - 包含标题和文件夹图标按钮
        list_title_container = QWidget()
        list_title_container.setObjectName("list_title_container")
        list_title_container.setFixedHeight(40)
        list_title_layout = QHBoxLayout(list_title_container)
        list_title_layout.setContentsMargins(6, 4, 6, 4)
        list_title_layout.setSpacing(8)
        list_title_layout.setAlignment(Qt.AlignmentFlag.AlignVCenter)

        # 图片列表标题
        self.image_list_title_label = QLabel("📂 图片列表")
        list_title_layout.addWidget(self.image_list_title_label)

        # 添加弹性空间，推送右侧按钮到最右边
        list_title_layout.addStretch()

        # 类别聚焦属于图片列表筛选，因此状态展示在图片列表标题栏。
        # 具体类别从类别按钮右键菜单选择，避免和“当前分类目标”混淆。
        self.category_focus_button = QPushButton()
        self.category_focus_button.setObjectName("category_focus_button")
        self.category_focus_button.setFixedHeight(28)
        self.category_focus_button.setMaximumWidth(180)
        self.category_focus_button.setSizePolicy(
            QSizePolicy.Policy.Maximum,
            QSizePolicy.Policy.Fixed,
        )
        self.category_focus_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.category_focus_button.clicked.connect(self.clear_category_focus)
        self.category_focus_button.hide()
        self._apply_category_focus_button_style()
        list_title_layout.addWidget(self.category_focus_button)

        # 搜索组件 - 可展开的文件名搜索
        self.image_search_widget = ExpandableSearch()
        self.image_search_widget.search_confirmed.connect(self._on_image_search)
        self.image_search_widget.search_cleared.connect(self._on_image_search_cleared)
        list_title_layout.addWidget(self.image_search_widget)

        # 筛选按钮 - 筛选图片显示条件
        self.filter_button = self.create_toolbar_button('▼', 'filter_button',
                                                       '筛选图片显示条件',
                                                       self.show_filter_menu,
                                                       size=(32, 32))
        # 初始化过滤器状态（默认全部显示）
        self.filter_unclassified = True
        self.filter_classified = True
        self.filter_removed = True
        self._image_search_text = ""  # 图片列表搜索关键字

        style_icon_button(self.filter_button)
        list_title_layout.addWidget(self.filter_button)

        # 文件夹图标按钮 - 打开目录功能
        self.folder_button = self.create_toolbar_button(
            '📁',
            'folder_button',
            '选择包含图片的目录',
            self.open_directory,
            size=(32, 32),
        )
        style_icon_button(self.folder_button)
        list_title_layout.addWidget(self.folder_button)

        layout.addWidget(list_title_container, 0)  # 不拉伸

        # 图片列表容器 - 可随窗口拉伸
        # Phase 1.1 Migration: Replace QListWidget with QListView for performance
        # 使用 NoSearchListView 禁用键盘搜索，避免与分类快捷键冲突
        self.image_list = NoSearchListView()
        self.image_list.setObjectName("image_list")  # 设置对象名以便教程系统找到
        self.image_list.setMinimumHeight(60)  # 降低最小高度，兼容小窗口

        # Initialize empty Model and Delegate
        self.image_list_model = ImageListModel([], {}, set(), set(), self)
        self.image_list.setModel(self.image_list_model)
        self.image_list.setItemDelegate(ImageListDelegate(self))

        # 固定列表项尺寸后使用 SinglePass 一次计算稳定的滚动范围。
        # Batched 在中间行增删后会暂时让 visualRect 为空，快速分类期间
        # 50ms 校准又持续被重启，从而出现列表空白和滚动条来回跳动。
        # 实测 25k 行 SinglePass 初始化约 74ms，可接受且视窗稳定。
        self.image_list.setUniformItemSizes(True)
        self.image_list.setLayoutMode(QListView.LayoutMode.SinglePass)

        # Phase 1.1: 配置滚动条行为 - 支持长文件名横向滚动
        self.image_list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.image_list.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.image_list.setWordWrap(False)  # 禁用文字换行，允许横向滚动

        # 列表外观由 apply_theme 统一设置，避免初始化和切换主题各维护一套样式。
        # Signal changed from itemClicked(QListWidgetItem) to clicked(QModelIndex)
        self.image_list.clicked.connect(self.on_image_list_item_clicked)
        layout.addWidget(self.image_list, 1)  # 设置拉伸权重1

    def create_toolbar_button(self, text: str, object_name: str, tooltip: str,
                             click_handler=None, size=(32, 32)) -> QPushButton:
        """创建标准化的工具栏按钮

        Args:
            text: 按钮显示文本/图标
            object_name: 按钮对象名称（用于样式）
            tooltip: 提示文本
            click_handler: 点击事件处理函数（可选）
            size: 按钮尺寸，默认(32, 32)

        Returns:
            QPushButton: 配置好的按钮实例
        """
        button = QPushButton()
        button.setText(text)
        button.setObjectName(object_name)
        button.setToolTip(tooltip)
        button.setFixedSize(*size)

        # 应用统一样式 - 使用新的样式系统
        style_icon_button(button)

        # 绑定点击事件
        if click_handler:
            button.clicked.connect(click_handler)

        return button

    def create_toolbar(self):
        """创建工具栏"""
        self.toolbar = QToolBar()
        self.toolbar.setObjectName("toolbar")  # 设置对象名以便教程系统找到
        self.toolbar.setMovable(False)
        self.toolbar.setContextMenuPolicy(Qt.ContextMenuPolicy.PreventContextMenu)  # 禁用右键菜单
        # 应用工具栏样式
        self.toolbar.setStyleSheet(ToolbarStyles.get_main_toolbar_style())
        self.addToolBar(self.toolbar)
        
        # 打开目录
        open_action = QAction('📁 打开目录', self)
        open_action.triggered.connect(self.open_directory)
        open_action.setToolTip('选择包含图片的目录')
        self.toolbar.addAction(open_action)
        # 获取toolbar中对应的widget并设置objectName
        self.open_directory_toolbar_button = self.toolbar.widgetForAction(open_action)
        if self.open_directory_toolbar_button:
            self.open_directory_toolbar_button.setObjectName('open_directory_toolbar_button')

        # 添加类别
        add_category_action = QAction('➕ 添加类别', self)
        add_category_action.triggered.connect(self.show_add_category_dialog)
        add_category_action.setToolTip('批量添加分类类别')
        self.toolbar.addAction(add_category_action)
        # 获取toolbar中对应的widget并设置objectName
        self.add_category_toolbar_button = self.toolbar.widgetForAction(add_category_action)
        if self.add_category_toolbar_button:
            self.add_category_toolbar_button.setObjectName('add_category_toolbar_button')

        # 添加弹性空间 - 推送右侧按钮到最右边
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.toolbar.addWidget(spacer)

        # 模式选择
        self.create_mode_button(self.toolbar)

        # 分类模式按钮（单分类/多分类）
        self.create_category_mode_button(self.toolbar)

        # 模型预测模式（自动 / 手动Tab触发）
        self.create_ai_prediction_mode_button(self.toolbar)

        self.toolbar.addSeparator()

        # 刷新按钮 - 使用统一样式
        self.refresh_button = self.create_toolbar_button('↻', 'refresh_button',
                                                   '刷新类别目录，同步外部变化 (F5)',
                                                   self.refresh_categories)
        self.toolbar.addWidget(self.refresh_button)

        # 从应用配置中读取主题设置
        current_theme = "light"
        try:
            app_config = get_app_config()
            current_theme = app_config.theme
            self.logger.info(f"从应用配置加载主题: {current_theme}")
        except Exception as e:
            self.logger.warning(f"加载主题配置失败，使用默认: {e}")
        default_theme.set_theme(current_theme)

        # 主题切换按钮 - 使用简洁线条图标
        theme_icon = '☾' if current_theme == "light" else '☼'  # ☾ 月亮(暗色) ☼ 太阳(亮色)
        theme_tooltip = '切换到暗色主题' if current_theme == "light" else '切换到亮色主题'
        self.theme_button = self.create_toolbar_button(theme_icon, 'theme_button',
                                                      theme_tooltip,
                                                      self.toggle_theme)
        self.toolbar.addWidget(self.theme_button)

        # 安装事件过滤器，使禁用的按钮也能响应点击
        self.theme_button_filter = DisabledButtonEventFilter(self)
        self.theme_button.installEventFilter(self.theme_button_filter)

        # 根据theme_mode设置主题按钮的启用状态
        try:
            theme_mode = app_config.theme_mode
            if theme_mode in ("auto", "system"):
                self.theme_button.setEnabled(False)
                mode_name = "自动切换" if theme_mode == "auto" else "跟随系统"
                self.theme_button.setToolTip(f"已启用{mode_name}，点击查看提示")
        except Exception as e:
            self.logger.warning(f"设置主题按钮状态失败: {e}")

        # 设置按钮 - 使用齿轮图标
        self.settings_button = self.create_toolbar_button('⚙', 'settings_button',
                                                     '打开设置',
                                                     self.show_settings_dialog)
        self.toolbar.addWidget(self.settings_button)

        # 帮助按钮 - 使用统一样式
        self.help_button = self.create_toolbar_button('?', 'help_button',
                                                '查看使用指南和快捷键',
                                                self.show_help_dialog)
        self.toolbar.addWidget(self.help_button)

        self.toolbar.addSeparator()

        # 侧栏折叠按钮
        self._sidebar_toggle_btn = self.create_toolbar_button('»', 'sidebar_toggle_button',
                                                              '收起侧栏 (Ctrl+B)',
                                                              self._toggle_sidebar)
        self.toolbar.addWidget(self._sidebar_toggle_btn)

        # 工具栏折叠按钮
        self._toolbar_collapse_btn = self.create_toolbar_button('△', 'toolbar_collapse_button',
                                                                '收起工具栏 (Ctrl+T)',
                                                                self._toggle_toolbar)
        self.toolbar.addWidget(self._toolbar_collapse_btn)
    
    def create_mode_button(self, toolbar):
        """创建图标化的模式选择按钮 - 直接点击切换"""
        # 使用统一样式创建按钮
        self.mode_button = self.create_toolbar_button('⧉', 'mode_button',
                                                     '复制模式 - 点击切换到移动模式',
                                                     lambda: self.set_mode(not self.is_copy_mode))

        # 添加到工具栏
        toolbar.addWidget(self.mode_button)
        self.set_mode(self.is_copy_mode)

    def create_ai_prediction_mode_button(self, toolbar):
        """创建辅助/全自动模型预测模式切换按钮。"""
        self.ai_prediction_mode_button = QPushButton()
        self.ai_prediction_mode_button.setObjectName("ai_prediction_mode_button")
        self.ai_prediction_mode_button.clicked.connect(
            self._on_ai_prediction_button_clicked
        )
        toolbar.addWidget(self.ai_prediction_mode_button)

        self.ai_model_button = QPushButton()
        self.ai_model_button.setObjectName("ai_model_button")
        self.ai_model_button.clicked.connect(self.show_ai_project_setup)
        toolbar.addWidget(self.ai_model_button)
        self._update_ai_prediction_mode_button()

    def _update_ai_prediction_mode_button(self):
        button = getattr(self, "ai_prediction_mode_button", None)
        if button is None:
            return
        initialized = self._is_active_ai_model_initialized()
        resource_problem = (
            self._active_ai_resource_problem() if initialized else None
        )
        manager = getattr(self, "_ai_manager", None)
        ready = bool(
            initialized
            and resource_problem is None
            and manager is not None
            and manager.is_ready
        )
        is_auto = getattr(self, "ai_prediction_mode", "assist") == "auto"
        active = bool(
            getattr(self, "_ai_auto_task", None)
            if is_auto
            else getattr(self, "_ai_assist_active", False)
        )
        if ready:
            button.setText(
                ("AI · 自动" if is_auto else "AI · 辅助")
                + (" · 运行中" if active else "")
            )
            button.setToolTip(
                "全自动分类：切换模式后按 Tab 校验并启动"
                if is_auto
                else "辅助预测：按 Tab 开启或关闭，点击切换为全自动模式"
            )
        elif initialized and resource_problem:
            button.setText("AI · 未就绪")
            button.setToolTip(f"{resource_problem}；点击打开模型配置")
        elif initialized:
            button.setText("AI · 加载中")
            button.setToolTip("项目和资源已就绪，正在后台加载 AI 模型")
        else:
            button.setText("AI · 启用")
            button.setToolTip("首次启用：选择基础模型以及从零开始或导入旧标注")
        style_button(
            button,
            "primary" if ready and active else "secondary",
            "compact",
            min_width=92,
        )

        model_button = getattr(self, "ai_model_button", None)
        if model_button is not None:
            project_state = getattr(
                self, "_ai_project_state", default_ai_project_state()
            )
            model_key = project_state.get("active_model", "balanced")
            profile = get_ai_model_profile(model_key)
            model_button.setText(f"模型 · {profile.short_name}")
            model_button.setToolTip("项目 AI 设置：切换速度、均衡或精度模型")
            style_button(model_button, "secondary", "compact", min_width=92)

    def _on_ai_prediction_button_clicked(self):
        if not self._is_active_ai_model_initialized():
            self.show_ai_project_setup()
            return
        resource_problem = self._active_ai_resource_problem()
        if resource_problem:
            self.show_ai_project_setup()
            return
        manager = getattr(self, "_ai_manager", None)
        if manager is None or not manager.is_ready:
            toast_info(self, "AI 模型正在后台加载，请稍候")
            return
        self._toggle_ai_prediction_mode()

    def _toggle_ai_prediction_mode(self):
        mode = (
            "assist"
            if getattr(self, "ai_prediction_mode", "assist") == "auto"
            else "auto"
        )
        self.set_ai_prediction_mode(mode)

    def set_ai_prediction_mode(self, mode: str, notify: bool = True):
        """切换预测模式并刷新当前图片的预测调度。"""
        if mode not in ("assist", "auto"):
            return
        if not self._is_active_ai_prediction_ready():
            self._update_ai_prediction_mode_button()
            if notify:
                problem = self._active_ai_resource_problem()
                toast_warning(
                    self,
                    problem or "AI 模型正在后台加载，请稍候",
                )
            return
        if getattr(self, "_ai_prediction_loading", False):
            self._notify_ai_navigation_locked()
            return

        self.ai_prediction_mode = mode
        self.app_config.ai_prediction_mode = mode
        self._ai_assist_active = False
        self._ai_prediction_timer.stop()
        if hasattr(self, "image_view_panel"):
            self.image_view_panel.clear_prediction_overlay()
        self._update_ai_prediction_mode_button()

        if mode == "auto":
            if hasattr(self, "category_panel"):
                self.category_panel.set_ai_status(
                    "AI 全自动 · 按 Tab 校验并启动", "ready"
                )
            if notify:
                toast_info(self, "已切换到全自动模式，按 Tab 启动")
        else:
            if hasattr(self, "category_panel"):
                self.category_panel.set_ai_status(
                    "AI 辅助 · 按 Tab 开启", "ready"
                )
            if notify:
                toast_info(self, "已切换到辅助模式，按 Tab 开启")
    
    def create_status_bar(self):
        """创建状态栏"""
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("准备就绪")

        # 版本号是唯一更新入口，点击后在按钮上方向上展开更新中心。
        self.update_download_button = QPushButton(f"v{__version__}")
        self.update_download_button.setObjectName("update_download_button")
        self.update_download_button.setToolTip("打开更新中心")
        self.update_download_button.clicked.connect(
            self._handle_update_button_clicked
        )
        self._apply_update_button_style("idle")
        self.update_download_button.show()
        self.statusBar.addPermanentWidget(self.update_download_button)
        self.update_center_popover = UpdateCenterPopover(self)
        self.update_center_popover.check_requested.connect(
            lambda: self._start_main_update_check(manual=True)
        )
        self.update_center_popover.download_requested.connect(
            self._start_pending_update_download
        )
        self.update_center_popover.pause_requested.connect(
            self.update_download_controller.pause
        )
        self.update_center_popover.resume_requested.connect(
            self.update_download_controller.resume
        )
        self.update_center_popover.cancel_requested.connect(
            self.update_download_controller.cancel
        )
        self.update_center_popover.restart_requested.connect(
            self._restart_and_install_update
        )
        self._sync_update_popover()
    
    @performance_monitor
    def setup_shortcuts(self):
        """设置快捷键 - 改进版本，增加错误处理和状态检查"""
        try:
            self.logger.debug("开始设置快捷键...")

            # 清除现有的快捷键，但保留系统默认的
            existing_actions = self.actions()
            for action in existing_actions[:]:  # 使用副本避免迭代时修改列表
                try:
                    self.removeAction(action)
                    action.deleteLater()  # 确保清理资源
                except Exception as e:
                    self.logger.warning(f"清除快捷键失败: {e}")

            # 设置基本导航快捷键
            shortcuts = {
                Qt.Key.Key_Left: self.prev_image,
                Qt.Key.Key_Right: self.next_image,
                Qt.Key.Key_Up: self.prev_category,
                Qt.Key.Key_Down: self.next_category,
                Qt.Key.Key_Return: self.confirm_category,
                Qt.Key.Key_Delete: self.move_to_remove,
                Qt.Key.Key_F5: self.refresh_categories,
            }

            # 设置组合快捷键（用于图像控制）
            combo_shortcuts = {
                'Ctrl+F': lambda: self.image_label.fit_to_window() if hasattr(self, 'image_label') else None,
                'Ctrl+=': lambda: self.image_label.zoom_in() if hasattr(self, 'image_label') else None,
                'Ctrl+-': lambda: self.image_label.zoom_out() if hasattr(self, 'image_label') else None,
                'Ctrl+0': lambda: self.image_label.reset_zoom() if hasattr(self, 'image_label') else None,
                'Ctrl+B': self._toggle_sidebar,
                'Ctrl+T': self._toggle_toolbar,
            }

            # 创建基本快捷键
            success_count = 0
            for key, func in shortcuts.items():
                try:
                    action = QAction(self)
                    action.setShortcut(QKeySequence(key))
                    # 使用安全的连接方式
                    action.triggered.connect(lambda checked, f=func: self._safe_execute_shortcut(f))
                    self.addAction(action)
                    success_count += 1
                except Exception as e:
                    self.logger.error(f"创建基本快捷键失败 {key}: {e}")

            # 创建组合快捷键
            for key_combo, func in combo_shortcuts.items():
                try:
                    action = QAction(self)
                    action.setShortcut(QKeySequence(key_combo))
                    action.triggered.connect(lambda checked, f=func: self._safe_execute_shortcut(f))
                    self.addAction(action)
                    success_count += 1
                except Exception as e:
                    self.logger.error(f"创建组合快捷键失败 {key_combo}: {e}")

            # 设置类别快捷键
            if hasattr(self, 'config') and self.config and hasattr(self, 'categories'):
                for category_name, key in self.config.category_shortcuts.items():
                    try:
                        if key and category_name in self.categories:
                            action = QAction(self)
                            action.setShortcut(QKeySequence(key))
                            action.triggered.connect(lambda checked, name=category_name: self._safe_execute_shortcut(lambda: self.quick_classify_by_name(name)))
                            self.addAction(action)
                            success_count += 1
                    except Exception as e:
                        self.logger.error(f"创建类别快捷键失败 {category_name}->{key}: {e}")

            self.logger.debug(f"快捷键设置完成，成功创建 {success_count} 个快捷键")

        except Exception as e:
            self.logger.error(f"设置快捷键时发生严重错误: {e}")
            # 尝试恢复基本功能
            self._setup_minimal_shortcuts()

    def _safe_execute_shortcut(self, func):
        """安全执行快捷键函数"""
        try:
            if self._ai_auto_task is not None:
                self.logger.debug("全自动任务运行中，忽略非 Tab 快捷键")
                return
            if not self._shortcuts_active:
                self.logger.debug("快捷键被禁用，跳过执行")
                return

            # 检查是否在输入模式（输入框聚焦时不执行快捷键）
            if self._is_in_input_mode():
                self.logger.debug("输入模式，跳过快捷键执行")
                # 手动触发QLineEdit的returnPressed信号（QAction会拦截事件导致信号不触发）
                focused = self.focusWidget()
                if focused and isinstance(focused, QLineEdit):
                    focused.returnPressed.emit()
                return

            if func:
                # INFO级别记录QAction快捷键触发（简化记录）
                func_name = getattr(func, '__name__', 'lambda')
                if func_name == '<lambda>':
                    func_name = 'lambda'
                self.logger.info(f"快捷键执行: {func_name}")
                func()
        except Exception as e:
            self.logger.error(f"执行快捷键函数失败: {e}")

    def _setup_minimal_shortcuts(self):
        """设置最小化快捷键集合（紧急恢复用）"""
        try:
            self.logger.info("尝试设置最小化快捷键...")
            minimal_shortcuts = {
                Qt.Key.Key_Left: self.prev_image,
                Qt.Key.Key_Right: self.next_image,
                Qt.Key.Key_F5: self.refresh_categories,
            }

            for key, func in minimal_shortcuts.items():
                try:
                    action = QAction(self)
                    action.setShortcut(QKeySequence(key))
                    action.triggered.connect(func)
                    self.addAction(action)
                except Exception as e:
                    self.logger.error(f"最小化快捷键设置失败 {key}: {e}")

        except Exception as e:
            self.logger.error(f"最小化快捷键设置严重失败: {e}")

    def quick_classify_by_name(self, category):
        """通过类别名称快速分类"""
        if self.current_dir and category in self.categories:
            self.move_to_category(category)

    def _is_defined_shortcut(self, key: int, modifiers) -> bool:
        """检查是否为已定义的快捷键"""
        try:
            # 基本快捷键列表
            basic_shortcuts = {
                Qt.Key.Key_Left, Qt.Key.Key_Right, Qt.Key.Key_Up, Qt.Key.Key_Down,
                Qt.Key.Key_Return, Qt.Key.Key_Delete, Qt.Key.Key_Tab, Qt.Key.Key_F5
            }

            # 检查基本快捷键
            if key in basic_shortcuts:
                return True

            # 检查类别快捷键
            if hasattr(self, 'config') and self.config and hasattr(self, 'categories'):
                for category_name, shortcut_key in self.config.category_shortcuts.items():
                    if shortcut_key and category_name in self.categories:
                        try:
                            sequence = QKeySequence(shortcut_key)
                            if len(sequence) > 0:
                                key_combination = sequence[0]
                                if (key_combination.key() == key and
                                    key_combination.keyboardModifiers() == modifiers):
                                    return True
                        except Exception:
                            continue

            return False

        except Exception as e:
            self.logger.error(f"检查快捷键定义失败: {e}")
            return False

    def _is_network_path(self, path):
        """检查是否为网络路径"""
        path_str = str(path)
        return path_str.startswith('\\\\') or '://' in path_str
    
    def log_performance_info(self, operation, **kwargs):
        """记录详细的性能信息"""
        if not self.enable_performance_logging:
            return
            
        try:
            # 获取内存信息
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024
            
            # 获取系统内存信息
            system_memory = psutil.virtual_memory()
            system_available_mb = system_memory.available / 1024 / 1024
            
            # 获取缓存信息
            cache_info = {}
            if hasattr(self, 'image_loader'):
                cache_info = self.image_loader.get_cache_info()
            
            # 格式化基本信息
            base_info = f"[{operation}] "
            base_info += f"内存:{memory_mb:.1f}MB "
            base_info += f"系统可用:{system_available_mb:.1f}MB "
            
            if cache_info:
                base_info += f"缓存:{cache_info.get('cache_size', 0)}/{cache_info.get('max_size', 0)} "
                base_info += f"内存:{cache_info.get('memory_usage_mb', 0):.1f}/{cache_info.get('max_memory_mb', 0):.1f}MB "
                base_info += f"命中率:{cache_info.get('hit_rate', '0%')}"
            
            # 添加额外信息
            extra_info = ""
            for key, value in kwargs.items():
                extra_info += f" {key}:{value}"

            self.logger.debug(base_info + extra_info)
            
        except Exception as e:
            self.logger.debug(f"性能日志记录失败: {e}")
    
    def log_system_info(self):
        """输出系统和程序配置信息"""
        try:
            # 系统信息
            memory = psutil.virtual_memory()
            cpu_count = psutil.cpu_count()
            
            # 图片加载器配置
            cache_size_mb = self.image_loader.max_cache_memory / 1024 / 1024
            
            self.logger.info("=" * 80)
            self.logger.info(f"[系统配置] 图片分类工具 v{self.version} - 性能监控已启用")
            self.logger.info(f"[系统信息] CPU核心:{cpu_count} 系统内存:{memory.total / 1024 / 1024 / 1024:.1f}GB "
                        f"可用内存:{memory.available / 1024 / 1024 / 1024:.1f}GB")
            self.logger.info(f"[缓存配置] 图片缓存:{self.image_loader.cache_size}张 "
                        f"内存上限:{cache_size_mb:.0f}MB 缩略图缓存:{self.image_loader.thumbnail_cache_size}张")
            self.logger.info(f"[加载策略] OpenCV优先:{self.image_loader.use_opencv} "
                        f"网络优化:启用 滑动窗口:智能预加载")
            self.logger.info("[性能监控] 详细日志:启用 切换计时:启用 内存监控:启用 预加载追踪:启用")
            self.logger.info("=" * 80)
            
        except Exception as e:
            self.logger.error(f"系统信息输出失败: {e}")
    
    # ===== 窗口几何信息保存与恢复 =====

    def _set_default_window_geometry(self):
        """首次启动使用居中的普通恢复尺寸，并默认以最大化状态显示。"""
        self._start_maximized = True
        screen = QApplication.primaryScreen()
        if screen is None:
            return
        available = screen.availableGeometry()
        init_width = min(max(1024, int(available.width() * 0.88)), 1500)
        init_height = min(max(600, int(available.height() * 0.88)), 960)
        init_width = min(init_width, available.width())
        init_height = min(init_height, available.height())
        self.resize(init_width, init_height)
        x = available.x() + (available.width() - init_width) // 2
        y = available.y() + (available.height() - init_height) // 2
        self.move(x, y)

    def _save_window_geometry(self):
        """保存普通窗口几何信息、所在屏幕和最大化状态。"""
        try:
            app_config = get_app_config()
            if not app_config.remember_window_geometry:
                return

            maximized = self.isMaximized()
            normal_geo = self.normalGeometry()
            geo = (
                normal_geo
                if maximized and normal_geo.isValid()
                else self.geometry()
            )
            screen = self.screen()
            screen_name = screen.name() if screen else ""
            app_config.window_geometry = {
                "x": geo.x(),
                "y": geo.y(),
                "width": geo.width(),
                "height": geo.height(),
                "screen_name": screen_name,
                "maximized": maximized,
            }
            self.logger.debug(
                f"窗口几何信息已保存: {geo.width()}x{geo.height()} "
                f"at ({geo.x()},{geo.y()}) screen={screen_name} "
                f"maximized={maximized}"
            )
        except Exception as e:
            self.logger.warning(f"保存窗口几何信息失败: {e}")

    def _restore_window_geometry(self) -> bool:
        """从应用配置恢复窗口位置和大小

        Returns:
            True 表示恢复成功，False 表示恢复失败或未启用
        """
        try:
            app_config = get_app_config()
            if not app_config.remember_window_geometry:
                return False

            geo = app_config.window_geometry
            if not geo or not isinstance(geo, dict):
                return False

            saved_x = geo.get("x")
            saved_y = geo.get("y")
            saved_w = geo.get("width")
            saved_h = geo.get("height")
            if saved_w is None or saved_h is None:
                return False

            # 收集所有可用屏幕的联合区域
            screens = QApplication.screens()
            if not screens:
                return False

            # 优先按显示器名称恢复，显示器变化时再按坐标查找。
            target_screen = None
            saved_screen_name = str(geo.get("screen_name", "") or "")
            if saved_screen_name:
                target_screen = next(
                    (
                        screen
                        for screen in screens
                        if screen.name() == saved_screen_name
                    ),
                    None,
                )
            if target_screen is None and saved_x is not None and saved_y is not None:
                from PyQt6.QtCore import QPoint as QP
                target_point = QP(saved_x + saved_w // 2, saved_y + saved_h // 2)
                target_screen = QApplication.screenAt(target_point)

            # 如果目标位置不在任何屏幕上，回退到主屏幕
            if target_screen is None:
                target_screen = QApplication.primaryScreen()

            available = target_screen.availableGeometry()

            saved_maximized = geo.get("maximized")
            if saved_maximized is None:
                # 兼容旧配置：几何尺寸已接近铺满屏幕时视为最大化。
                saved_maximized = (
                    saved_w >= int(available.width() * 0.95)
                    and saved_h >= int(available.height() * 0.95)
                )
            self._start_maximized = bool(saved_maximized)

            # 如果保存的大小超过当前屏幕，使用默认大小
            if saved_w > available.width() or saved_h > available.height():
                self.logger.info(f"保存的窗口大小 {saved_w}x{saved_h} 超过当前屏幕 {available.width()}x{available.height()}，使用默认大小")
                return False

            self.resize(saved_w, saved_h)

            # 恢复位置：确保窗口至少有一部分在可见区域内
            if saved_x is not None and saved_y is not None:
                # 窗口至少 100px 在屏幕可见区域内
                clamped_x = max(available.x() - saved_w + 100, min(saved_x, available.right() - 100))
                clamped_y = max(available.y(), min(saved_y, available.bottom() - 100))
                self.move(clamped_x, clamped_y)
            else:
                # 没有保存位置信息，居中显示
                x = available.x() + (available.width() - saved_w) // 2
                y = available.y() + (available.height() - saved_h) // 2
                self.move(x, y)

            self.logger.info(
                f"窗口几何信息已恢复: {saved_w}x{saved_h} "
                f"at ({self.x()},{self.y()}) "
                f"maximized={self._start_maximized}"
            )
            return True

        except Exception as e:
            self.logger.warning(f"恢复窗口几何信息失败: {e}")
            return False

    # ===== 面板折叠控制 =====

    def _toggle_sidebar(self):
        """切换右侧面板的显示/隐藏"""
        if self.right_widget.isVisible():
            # 记录当前分割比例，然后隐藏
            self._saved_splitter_sizes = self.splitter.sizes()
            self.right_widget.hide()
            self._sidebar_toggle_btn.setText('«')
            self._sidebar_toggle_btn.setToolTip('展开侧栏 (Ctrl+B)')
        else:
            # 恢复显示和分割比例
            self.right_widget.show()
            if self._saved_splitter_sizes:
                self.splitter.setSizes(self._saved_splitter_sizes)
            self._sidebar_toggle_btn.setText('»')
            self._sidebar_toggle_btn.setToolTip('收起侧栏 (Ctrl+B)')

    def _toggle_toolbar(self):
        """切换工具栏的显示/隐藏"""
        if self.toolbar.isVisible():
            self.toolbar.hide()
            self._toolbar_expand_bar.show()
        else:
            self.toolbar.show()
            self._toolbar_expand_bar.hide()

    # ===== 响应式布局处理方法 =====
    
    def resizeEvent(self, event):
        """处理窗口大小改变事件"""
        try:
            super().resizeEvent(event)
            
            # 记录窗口大小变化
            if hasattr(self, 'logger'):
                self.logger.debug(f"窗口大小改变: {event.size().width()}x{event.size().height()}")

            tutorial_manager = getattr(self, 'tutorial_manager', None)
            if tutorial_manager is not None:
                tutorial_manager.handle_parent_resized()
               
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.error(f"窗口大小改变处理失败: {e}")
    
    # ===== 目录和文件处理方法 =====
    
    def open_directory(self):
        """打开图片目录"""

        if self._ai_prediction_loading:
            self._notify_ai_navigation_locked()
            return

        dir_path = QFileDialog.getExistingDirectory(self, '选择图片目录')
        if dir_path:
            if getattr(self, '_ai_manager', None):
                self._ai_manager.clear_project()
            self.current_dir = Path(dir_path)

            # 检测微信目录并提醒（只提醒一次）
            if not hasattr(self, '_wechat_warned'):
                self._wechat_warned = False
            if not self._wechat_warned and self._check_wechat_directory(self.current_dir):
                self._wechat_warned = True
                self._show_wechat_warning()

            current_drive = self.current_dir.drive  # 例如 "D:"

            # 尝试从全局配置中获取盘符缓存
            try:
                app_config = get_app_config()
                cached_drive = app_config.get_last_opened_drive()  # 从路径中实时提取盘符
                cached_is_network = app_config.last_opened_drive_is_network

                # 如果缓存命中（盘符相同），使用缓存值
                if cached_drive == current_drive:
                    is_network = cached_is_network
                    self.logger.debug(f"盘符缓存命中: {current_drive} ({'网络' if is_network else '本地'})")
                else:
                    # 缓存未命中，需要检测
                    is_network = is_network_path(self.current_dir)
                    self.logger.debug(f"盘符缓存未命中，重新检测: {current_drive} ({'网络' if is_network else '本地'})")
            except Exception as e:
                # 如果获取缓存失败，回退到直接检测
                self.logger.warning(f"获取盘符缓存失败: {e}，使用直接检测")
                is_network = is_network_path(self.current_dir)

            # Task 1.1：通知图像加载器当前工作路径类型
            # 本地路径将跳过磁盘缓存清理，节省100%清理开销
            try:
                self.image_loader.set_working_path(self.current_dir)
            except Exception as e:
                self.logger.warning(f"设置图像加载器工作路径失败: {e}")

            # 设置配置文件路径为图片目录的父目录
            # 这个操作可能会因为权限问题失败（尤其是网络路径）
            try:
                self.config.set_base_dir(str(self.current_dir.parent))
            except ConfigError as e:
                # 检查是否是权限错误
                if "Permission denied" in str(e) or "权限" in str(e):
                    self.logger.error(f"目录访问权限不足: {self.current_dir.parent}")
                    toast_error(self, f"⚠️ 无法打开目录：该路径没有写入权限\n路径：{self.current_dir.parent}")
                else:
                    self.logger.error(f"配置初始化失败: {e}")
                    toast_error(self, f"⚠️ 打开目录失败：{e}")

                # 重置当前目录，取消打开操作
                self.current_dir = None
                return  # 提前退出，不执行后续操作

            # 保存路径类型到实例变量，供循环翻页等功能使用
            self.is_network_working_path = is_network

            if is_network:
                # 网络路径提醒
                toast_info(self,'🚀 检测到网络路径，已启用专项优化')
                self.logger.info(f"[网络路径] 用户选择: {self.current_dir}")
            else:
                self.logger.info(f"[本地路径] 用户选择: {self.current_dir}")
                # 显示本地目录打开成功通知
                toast_success(self,f"已打开目录：{self.current_dir.name}")

            # 保存最后打开的目录和盘符信息到全局配置
            try:
                app_config = get_app_config()
                app_config.update_last_opened_drive_info(str(self.current_dir), is_network)
            except Exception as e:
                self.logger.error(f"保存最后打开的目录信息失败: {e}")

            # 清除搜索状态
            self._image_search_text = ""
            if hasattr(self, 'image_search_widget'):
                self.image_search_widget.clear_and_collapse()

            # 类别聚焦只在当前目录内有效，切换任务时恢复显示全部类别。
            self._deactivate_category_focus()

            # 先启动图片扫描，让UI立即响应
            self.load_images()

            # 延迟类别加载和同步操作
            QTimer.singleShot(100, self._delayed_load_categories)

    def _delayed_load_categories(self):
        """延迟加载类别，避免阻塞UI"""
        try:
            # 先快速加载类别（不启动同步操作）
            self._load_categories_only()
            
            # 显示当前图片
            self.show_current_image()
            
            # 进一步延迟同步操作，确保UI完全就绪后再进行
            QTimer.singleShot(1000, self._delayed_start_sync)
            
        except Exception as e:
            self.logger.error(f"延迟加载类别失败: {e}")
    
    def _load_categories_only(self):
        """加载类别（通过CategoryManager）"""
        if not self._category_manager:
            self.logger.error("CategoryManager未初始化")
            return
        if not self.current_dir:
            self.logger.warning("当前目录未设置，无法加载类别")
            return

        self._category_manager.load_categories()
        # Manager会触发categories_changed信号，UI刷新在on_categories_changed中处理
    
    def load_categories(self):
        """公共的加载类别方法，供对话框调用"""
        try:
            self._load_categories_only()
            self.logger.info("类别列表已刷新")
        except Exception as e:
            self.logger.error(f"刷新类别失败: {e}")
    
    def _delayed_start_sync(self):
        """延迟启动同步操作，确保不阻塞UI"""
        try:
            if self.categories and hasattr(self, 'file_manager'):
                self.logger.info("🔄 开始后台同步操作...")
                self.statusBar.showMessage("🔄 后台同步分类状态中...")
                
                # 连接同步完成信号（使用UniqueConnection防止重复连接）
                if hasattr(self.file_manager, 'file_sync'):
                    try:
                        self.file_manager.file_sync.sync_completed.connect(
                            self._on_sync_completed, Qt.ConnectionType.UniqueConnection)
                        self.file_manager.file_sync.sync_progress.connect(
                            self._on_sync_progress, Qt.ConnectionType.UniqueConnection)
                    except TypeError:
                        # 已连接，忽略
                        pass
                
                # 启动真正的同步操作
                try:
                    self.file_manager.start_background_sync(
                        self.current_dir, 
                        self.categories, 
                        self.classified_images, 
                        self.removed_images, 
                        quick_mode=True
                    )
                except Exception as sync_error:
                    self.logger.error(f"启动同步失败: {sync_error}")
                    self._update_status_with_current_image()
            else:
                self.logger.debug("无类别或文件管理器，跳过文件同步")
                self._update_status_with_current_image()
        except Exception as e:
            self.logger.error(f"延迟同步启动失败: {e}")
            self.statusBar.showMessage("❌ 同步启动失败，请查看日志")
    
    def _on_sync_completed(self, result):
        """同步完成回调"""
        try:
            self.logger.info("后台同步操作完成")
            self._update_status_with_current_image()
        except Exception as e:
            self.logger.error(f"处理同步完成回调失败: {e}")
    
    def _on_sync_progress(self, message):
        """同步进度回调"""
        try:
            # 更新状态栏显示同步进度
            self.statusBar.showMessage(f"🔄 {message}")
        except Exception as e:
            self.logger.error(f"处理同步进度回调失败: {e}")
    
    def _update_status_with_current_image(self):
        """使用当前图片信息更新状态栏"""
        try:
            if self.image_files and 0 <= self.current_index < len(self.image_files):
                current_image = self.image_files[self.current_index]
                image_name = Path(current_image).name
                self.statusBar.showMessage(f"📷 {image_name}")
            else:
                self.statusBar.showMessage("✅ 图片分类工具已就绪")
        except Exception as e:
            self.logger.error(f"更新状态栏失败: {e}")
            self.statusBar.showMessage("✅ 图片分类工具已就绪")
    
    def load_images(self):
        """开始智能加载目录下的图片（委托给ImageNavigationManager）"""
        # 设置加载状态，确保信号处理正常工作
        self.loading_in_progress = True
        self.background_loading = False  # 初始为False，on_initial_batch_ready中设为True

        if self._nav_manager:
            self._nav_manager.load_images()
        elif self.current_dir:
            # 降级：Manager未初始化
            self.logger.warning("[降级模式] ImageNavigationManager未初始化，使用直接扫描")
            self.image_loader.clear_cache()
            self.file_scanner.scan_directory(self.current_dir)
    
    # ===== 文件扫描事件处理 =====

    def on_list_updated(self, file_list):
        """接收 Manager 的列表更新信号，执行 UI 层刷新"""
        # Manager已经更新了状态（image_files, all_image_files, total_images）
        # 这里只需要刷新UI
        self.schedule_ui_update('image_list', 'statistics')

    def on_initial_batch_ready(self, initial_files):
        """处理初始批次文件"""
        if not self.loading_in_progress:
            return

        # 立即设置后台加载标记，确保files_found信号不会被丢弃
        self.background_loading = True

        self.logger.info(f"接收到初始批次: {len(initial_files)} 个文件")
        
        # 设置初始显示文件
        self.image_files = initial_files.copy()
        self.all_image_files = initial_files.copy()
        self.current_index = 0 if self.image_files else -1
        self.initial_batch_loaded = True
        
        # 立即设置总数，让UI完全可用
        self.total_images = len(initial_files)
        
        # 设置图片加载器的文件列表引用
        self.image_loader.set_image_files_reference(self.image_files)
        
        # 立即完全启用UI
        self.loading_in_progress = False
        
        # 异步加载状态文件，避免阻塞UI
        QTimer.singleShot(50, self._delayed_load_state)
        
        # 立即更新UI组件（不包括current_image，避免重复刷新）
        self.schedule_ui_update('image_list', 'statistics', 'ui_state')
        
        # 立即显示成功信息
        path_str = str(self.current_dir) if self.current_dir else ""
        is_network_path = path_str.startswith('\\\\')
        location_type = "网络路径" if is_network_path else "本地路径"
        
        self.statusBar.showMessage(f"✅ {location_type}已就绪 {len(initial_files)} 张图片，后台继续扫描...")

        self.logger.info("🚀 程序UI已完全启用，用户可立即使用")

    def _delayed_load_state(self):
        """延迟加载状态文件，避免阻塞UI"""
        try:
            self.load_state()

            # 检查并修复当前列表中的重复文件
            self._remove_duplicates_from_current_list()

            # 移动模式下，补充已分类和已移除的图片到列表
            self._supplement_moved_files_to_list()

            # Bug修复：状态加载后重新计算类别计数
            # 确保 categories 和 classified_images 都已加载后再计算
            if self.categories:
                self.init_category_counts()

            # 应用过滤器（如果按钮已创建）
            if hasattr(self, 'filter_button'):
                self.apply_image_filter()
            else:
                # 状态加载完成后，更新UI显示（不包括current_image，避免重复刷新）
                self.schedule_ui_update('image_list', 'statistics')

            self.logger.debug("状态文件异步加载完成")
        except Exception as e:
            self.logger.error(f"延迟加载状态文件失败: {e}")
    
    def _remove_duplicates_from_current_list(self):
        """移除当前文件列表中的重复项"""
        if not self.image_files:
            return
            
        original_count = len(self.image_files)
        
        # 去重处理
        unique_files = []
        seen_paths = set()
        for file_path in self.image_files:
            path_str = str(file_path)
            if path_str not in seen_paths:
                unique_files.append(file_path)
                seen_paths.add(path_str)
        
        # 更新文件列表
        self.image_files = unique_files
        self.all_image_files = unique_files.copy()
        self.total_images = len(unique_files)
        
        # 更新图片加载器的文件列表引用
        self.image_loader.set_image_files_reference(self.image_files)
        
        if original_count != len(unique_files):
            removed_count = original_count - len(unique_files)
            self.logger.info(f"[去重修复] 移除了 {removed_count} 个重复文件，剩余 {len(unique_files)} 个文件")
            self.statusBar.showMessage(f"✅ 已修复重复文件问题，移除 {removed_count} 个重复项")

    def _supplement_moved_files_to_list(self):
        """移动模式下，补充已分类和已移除的图片到列表"""
        if not self.image_files:
            return

        # 复制模式下不需要补充，因为文件本来就在原目录
        if self.is_copy_mode:
            return

        # 收集当前列表中已有的路径
        existing_paths = {str(f) for f in self.image_files}
        added_files = []

        # 补充已分类的图片
        for img_path in self.classified_images.keys():
            if img_path not in existing_paths:
                # 验证文件实际存在
                real_path = self.get_real_file_path(img_path)
                if real_path.exists():
                    added_files.append(Path(img_path))
                    self.logger.debug(f"补充已分类图片到列表: {Path(img_path).name}")

        # 补充已移除的图片
        for img_path in self.removed_images:
            if img_path not in existing_paths:
                # 验证文件实际存在
                real_path = self.get_real_file_path(img_path)
                if real_path.exists():
                    added_files.append(Path(img_path))
                    self.logger.debug(f"补充已移除图片到列表: {Path(img_path).name}")

        # 将补充的文件添加到列表
        if added_files:
            # 保存当前查看的图片路径（如果有）
            current_image_path = None
            if self.image_files and 0 <= self.current_index < len(self.image_files):
                current_image_path = str(self.image_files[self.current_index])

            self.image_files.extend(added_files)
            self.all_image_files.extend(added_files)

            # 补充后重新排序，保持文件的自然顺序（按文件名）
            self.image_files.sort(key=lambda p: p.name.lower())
            self.all_image_files.sort(key=lambda p: p.name.lower())

            self.total_images = len(self.image_files)

            # 如果之前在查看某张图片，重新定位到该图片
            if current_image_path:
                try:
                    self.current_index = next(i for i, p in enumerate(self.image_files) if str(p) == current_image_path)
                    self.logger.debug(f"排序后重新定位到图片: {Path(current_image_path).name}，新索引: {self.current_index}")
                except StopIteration:
                    # 如果找不到原图片，保持索引不变或重置为0
                    if self.current_index >= len(self.image_files):
                        self.current_index = 0

            # 更新图片加载器的文件列表引用
            self.image_loader.set_image_files_reference(self.image_files)

            self.logger.info(f"[移动模式补充] 添加了 {len(added_files)} 个已移动的图片到列表，总计 {self.total_images} 个文件（已重新排序）")

    def on_files_found(self, file_batch):
        """处理后续发现的文件批次"""
        if not self.background_loading:
            return
            
        # 静默扩展文件列表 - 去重处理
        existing_paths = {str(f) for f in self.image_files}
        new_files = [f for f in file_batch if str(f) not in existing_paths]
        
        if new_files:
            self.image_files.extend(new_files)
            self.all_image_files.extend(new_files)
            
            # 更新图片加载器的文件列表引用
            self.image_loader.set_image_files_reference(self.image_files)
            
            # 减少UI更新频率
            current_total = len(self.all_image_files)
            
            # 只在每发现1000个文件时才更新UI
            if current_total % 1000 == 0:
                self.schedule_ui_update('image_list')
                self.statusBar.showMessage(f"📈 后台发现 {current_total} 张图片...")
                self.logger.info(f"[后台扫描] 里程碑更新: 总计 {current_total} 个文件")
    
    def on_scan_completed(self, total_count):
        """全量扫描完成处理"""
        self.background_loading = False
        self.initial_batch_loaded = True  # 标记扫描完成
        
        # 最终数据同步和去重处理
        # 对文件列表进行最终去重
        unique_files = []
        seen_paths = set()
        for file_path in self.image_files:
            path_str = str(file_path)
            if path_str not in seen_paths:
                unique_files.append(file_path)
                seen_paths.add(path_str)
        
        # 更新文件列表
        self.image_files = unique_files
        self.all_image_files = unique_files.copy()
        
        # 更新总数为实际去重后的数量
        actual_count = len(self.image_files)
        self.total_images = actual_count
        
        # 最终更新图片加载器的文件列表引用
        self.image_loader.set_image_files_reference(self.image_files)

        # 移动模式下，补充已分类和已移除的图片到列表
        self._supplement_moved_files_to_list()

        # 更新实际总数（可能在补充后增加）
        actual_count = len(self.image_files)
        self.total_images = actual_count

        # 补充后再次更新图片加载器的文件列表引用
        self.image_loader.set_image_files_reference(self.image_files)

        # 延迟启动同步操作
        if hasattr(self, 'categories') and self.categories:
            QTimer.singleShot(2000, lambda: self._delayed_start_sync())

        self.update_category_counts()

        # 重新更新UI以显示完整列表
        self.schedule_ui_update('image_list', 'statistics')

        # 静默完成通知
        path_str = str(self.current_dir) if self.current_dir else ""
        is_network_path = path_str.startswith('\\\\')
        location_type = "网络路径" if is_network_path else "本地路径"

        self.statusBar.showMessage(f"🎯 {location_type}扫描完成，总计 {actual_count} 张图片")

        # 强制更新UI列表
        self.schedule_ui_update('image_list', 'statistics')

        self.logger.info(f"🎯 后台扫描完成: 实际文件 {actual_count} 个")

        # 更新主题按钮状态（根据图片数量）
        self.update_theme_button_state()

        # 检查是否需要提示用户跳转到上次位置
        QTimer.singleShot(500, self._check_and_prompt_last_position)

        # 优化8：启动缓存预热（仅网络路径且已启用）
        if is_network_path and self.app_config.cache_warmup_enabled:
            warmup_count = self.app_config.cache_warmup_count
            if actual_count > 0:
                QTimer.singleShot(1000, lambda: self._start_cache_warmup(warmup_count))

    def _check_and_prompt_last_position(self):
        """检查并提示用户是否跳转到上次处理的位置"""
        try:
            # 检查是否有保存的索引
            if not hasattr(self, 'saved_last_index'):
                return

            last_index = self.saved_last_index

            # 检查索引是否有效（大于0且小于图片总数）
            if last_index <= 0 or not self.image_files or last_index >= len(self.image_files):
                self.logger.debug(f"last_index 无效或为初始值: {last_index}, 不提示跳转")
                # 修复：索引无效时显示第一张图片（如果有图片）
                if self.image_files and self.current_index >= 0:
                    self.logger.debug("显示第一张图片")
                    self.show_current_image()
                return

            # 弹出确认对话框

            msg_box = ThemedMessageBox(self)
            msg_box.setWindowTitle("恢复上次位置")
            msg_box.setText(f"检测到上次处理到第 {last_index + 1} 张图片")
            msg_box.setInformativeText("是否跳转到上次处理的位置继续工作？")
            msg_box.setIcon(QMessageBox.Icon.Question)

            # 设置程序图标
            try:
                icon_name = get_product_info()["icon_basename"]
                icon_path = self._get_resource_path(f'assets/{icon_name}.ico')
                if icon_path and icon_path.exists():
                    msg_box.setWindowIcon(QIcon(str(icon_path)))
            except Exception:
                pass

            # 添加按钮
            yes_btn = msg_box.addButton("跳转", QMessageBox.ButtonRole.YesRole)
            msg_box.addButton("从头开始", QMessageBox.ButtonRole.NoRole)
            msg_box.setDefaultButton(yes_btn)

            # 显示对话框并获取用户选择
            msg_box.exec()
            clicked_button = msg_box.clickedButton()

            if clicked_button == yes_btn:
                # 用户选择跳转
                self.logger.info(f"用户选择跳转到上次位置: 索引 {last_index}")
                self.jump_to_image(last_index)
                toast_info(self, f"已跳转到第 {last_index + 1} 张图片")
            else:
                # 用户选择从头开始（修复：显示第一张图片）
                self.logger.info("用户选择从头开始")
                if self.image_files and self.current_index >= 0:
                    self.show_current_image()
                    toast_info(self, "从第一张图片开始")

        except Exception as e:
            self.logger.error(f"检查上次位置失败: {e}")

    def _start_cache_warmup(self, count: int):
        """启动缓存预热（优化8 + 循环翻页末尾预热）

        Args:
            count: 预热图片数量
        """
        try:
            if not self.image_files:
                return

            # 检查是否开启网络路径循环翻页
            enable_tail_warmup = (
                self.app_config.network_loop_enabled and
                hasattr(self, 'is_network_working_path') and
                self.is_network_working_path
            )

            # 调用图片加载器的预热方法
            success = self.image_loader.warmup_cache(
                self.image_files,
                count=count,
                enable_tail_warmup=enable_tail_warmup,
                callback=self._on_warmup_progress
            )

            if success:
                if enable_tail_warmup:
                    tail_count = count // 2
                    total_warmup = count + tail_count
                    self.logger.info(f"[循环翻页] 网络循环已开启，预热前{count}张+末尾{tail_count}张，共{total_warmup}张")
                    toast_info(self, f"开始预热缓存（前{count}张+末尾{tail_count}张）...")
                else:
                    self.logger.info(f"[优化8-预热] 开始预热前 {count} 张图片...")
                    toast_info(self, f"开始预热缓存（{count}张图片）...")
            else:
                self.logger.info("[优化8-预热] 本地路径无需预热，已跳过")

        except Exception as e:
            self.logger.error(f"[优化8-预热] 启动预热失败: {e}")

    def _on_warmup_progress(self, current: int, total: int, filename: str):
        """预热进度回调（优化8）

        Args:
            current: 当前进度
            total: 总数
            filename: 当前文件名

        注意：此方法在后台线程中调用，Toast等UI操作需要通过QTimer发送到主线程
        """
        try:
            if current == total:
                # 预热完成
                self.logger.info(f"[优化8-预热] 预热完成: {total}/{total}张图片")
                # 修复BUG：使用QTimer将Toast发送到主线程
                QTimer.singleShot(0, lambda: toast_success(self, f"缓存预热完成（{total}张图片）"))
            elif current % 10 == 0:
                # 每10张输出一次进度日志
                self.logger.debug(f"[优化8-预热] 预热进度: {current}/{total} | {filename}")
        except Exception as e:
            self.logger.error(f"[优化8-预热] 进度回调错误: {e}")

    def jump_to_image(self, index):
        """跳转到指定索引的图片（委托给ImageNavigationManager）"""
        if not self._prepare_ai_navigation():
            return
        self.logger.info(f"[跳转] 请求跳转到索引 {index}, 当前文件数 {len(self.image_files)}")

        # 先设置当前请求的图片路径，用于回调时判断
        if 0 <= index < len(self.image_files):
            img_path = str(self.image_files[index])
            real_path = str(self.get_real_file_path(img_path))
            self._current_requested_image = str(Path(real_path).resolve())

        if self._nav_manager:
            self._nav_manager.jump_to_image(index)
        else:
            # 降级：直接设置索引并显示
            self.logger.warning("[降级模式] ImageNavigationManager未初始化")
            if 0 <= index < len(self.image_files):
                self.current_index = index
                self.show_current_image()

    def on_scan_progress(self, message):
        """处理扫描进度"""
        self.statusBar.showMessage(message)

    # ===== AI 辅助分类事件处理 =====

    def _current_ai_project_dir(self) -> Optional[Path]:
        current_dir = getattr(self, "current_dir", None)
        return current_dir.parent if current_dir else None

    def _is_active_ai_model_initialized(self) -> bool:
        project_dir = self._current_ai_project_dir()
        if project_dir is None:
            return False
        project_state = getattr(
            self, "_ai_project_state", default_ai_project_state()
        )
        model_key = project_state.get("active_model", "balanced")
        return is_project_model_initialized(
            project_state, project_dir, model_key
        )

    def _active_ai_resource_problem(self) -> Optional[str]:
        """Explain the first missing prerequisite for the active AI model."""
        project_dir = self._current_ai_project_dir()
        if project_dir is None:
            return "请先打开图片目录"
        project_state = getattr(
            self, "_ai_project_state", default_ai_project_state()
        )
        model_key = project_state.get("active_model", "balanced")
        if not is_project_model_initialized(
            project_state, project_dir, model_key
        ):
            return "当前项目尚未初始化 AI"

        model_resource = get_model_resource(model_key)
        if not is_resource_installed(model_resource):
            return f"{model_resource.display_name}尚未下载"

        profile = get_ai_model_profile(model_key)
        manager = getattr(self, "_ai_manager", None)
        provider = (
            manager.preferred_provider
            if manager is not None
            else get_app_config().ai_execution_provider
        )
        runtime_kind = required_runtime_kind(profile, provider)
        if runtime_kind:
            runtime_resource = get_runtime_resource(runtime_kind)
            if not is_resource_installed(runtime_resource):
                return f"{runtime_resource.display_name}尚未下载"
        return None

    def _is_active_ai_prediction_ready(self) -> bool:
        manager = getattr(self, "_ai_manager", None)
        return bool(
            self._active_ai_resource_problem() is None
            and manager is not None
            and manager.is_ready
        )

    def _collect_current_ai_samples(self, include_removed: Optional[bool] = None):
        """Collect usable, human-confirmed samples from the active project."""
        if include_removed is None:
            include_removed = bool(
                self._ai_project_state.get("learn_removed_images", False)
            )
        samples = []
        for logical_path, category in getattr(
            self, "classified_images", {}
        ).items():
            if not isinstance(category, str):
                continue
            actual_path = self.get_real_file_path(logical_path)
            if Path(actual_path).is_file():
                samples.append((str(logical_path), str(actual_path), category))
        if include_removed:
            samples.extend(self._collect_removed_ai_samples())
        return samples

    def _collect_removed_ai_samples(self):
        """Collect readable removed images only when the user opts into learning them."""
        samples = []
        for logical_path in getattr(self, "removed_images", ()):
            actual_path = self.get_real_file_path(logical_path)
            if Path(actual_path).is_file():
                samples.append(
                    (str(logical_path), str(actual_path), AI_REMOVAL_LABEL)
                )
        return samples

    @staticmethod
    def _resolve_imported_ai_image(
        state_dir: Path, logical_path: str, category: str
    ) -> Optional[Path]:
        logical = Path(logical_path)
        candidates = [logical]
        if category != AI_REMOVAL_LABEL:
            candidates.extend(
                [
                    state_dir / category / logical.name,
                    state_dir / "分类结果" / category / logical.name,
                ]
            )
        else:
            candidates.extend(
                state_dir / folder / logical.name
                for folder in ("已移除", "removed", "待删除")
            )
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        return None

    def _load_imported_ai_samples(
        self, state_file: Path, include_removed: bool = False
    ):
        """Read old human classifications without changing the current UI state."""
        with open(state_file, "r", encoding="utf-8") as file:
            imported = json.load(file)
        samples = []
        state_dir = state_file.parent
        for logical_path, category in imported.get("classified_images", {}).items():
            if not isinstance(category, str):
                continue
            actual_path = self._resolve_imported_ai_image(
                state_dir, logical_path, category
            )
            if actual_path is not None:
                samples.append((str(actual_path), str(actual_path), category))
        if include_removed:
            for logical_path in imported.get("removed_images", []):
                actual_path = self._resolve_imported_ai_image(
                    state_dir, logical_path, AI_REMOVAL_LABEL
                )
                if actual_path is not None:
                    samples.append(
                        (str(actual_path), str(actual_path), AI_REMOVAL_LABEL)
                    )
        return samples

    def show_ai_project_setup(self):
        """Show first-use setup or switch to another project model profile."""
        project_dir = self._current_ai_project_dir()
        if project_dir is None:
            toast_warning(self, "请先打开图片目录")
            return
        classified_samples = self._collect_current_ai_samples(
            include_removed=False
        )
        removed_samples = self._collect_removed_ai_samples()
        dialog = AIProjectSetupDialog(
            project_dir=project_dir,
            ai_state=self._ai_project_state,
            existing_sample_count=len(classified_samples),
            removed_sample_count=len(removed_samples),
            resource_download_manager=getattr(
                self, "_ai_resource_download_manager", None
            ),
            parent=self,
        )
        self._ai_setup_dialog_active = True
        try:
            dialog_result = dialog.exec()
        finally:
            self._ai_setup_dialog_active = False
        if dialog_result != QDialog.DialogCode.Accepted:
            return
        model_key = dialog.selected_model_key
        if model_key is None:
            return
        execution_provider = dialog.selected_execution_provider
        learn_removed_images = dialog.selected_learn_removed_images
        current_samples = list(classified_samples)
        if learn_removed_images:
            current_samples.extend(removed_samples)

        profile = get_ai_model_profile(model_key)
        already_initialized = is_project_model_initialized(
            self._ai_project_state, project_dir, model_key
        )
        force_reinitialize = bool(
            already_initialized and dialog.selected_reinitialize
        )
        needs_initialization = not already_initialized or force_reinitialize
        if (
            profile.recommended_gpu
            and needs_initialization
            and execution_provider == "cpu"
        ):
            if not self.show_question(
                "CPU 性能提示",
                "当前选择仅使用 CPU。\n"
                "精度优先模型首次处理大量旧标注会比较慢。\n"
                "完成后可直接复用，不需要重新建库。\n\n"
                "是否继续初始化？",
            ):
                return

        # 只有用户通过所有确认后才保存设备偏好，避免取消初始化时意外改动配置。
        get_app_config().ai_execution_provider = execution_provider
        self._ai_manager.set_preferred_provider(execution_provider)

        samples = []
        source_kind = "reuse"
        source_path = None
        model_store_path = None
        if needs_initialization:
            source_kind = dialog.selected_source_kind
            if source_kind == "existing":
                samples = current_samples
            elif source_kind == "import":
                source_path = dialog.selected_import_path
                try:
                    samples = self._load_imported_ai_samples(
                        source_path,
                        include_removed=learn_removed_images,
                    )
                except Exception as error:
                    self.logger.exception("导入旧 AI 样本失败")
                    toast_error(self, f"无法导入旧标注：{error}")
                    return
                if not samples:
                    toast_warning(
                        self,
                        "JSON 只保存分类记录，无法在原图片缺失时重建特征；"
                        "请改选旧项目中的 AI .npz 特征库",
                    )
                    return
            elif source_kind == "model_store":
                model_store_path = dialog.selected_model_store_path

        models = self._ai_project_state.setdefault("models", {})
        record = models.setdefault(model_key, {})
        if needs_initialization:
            record.update(
                {
                    "initialized": already_initialized,
                    "rebuilding": force_reinitialize,
                    "project_model_file": profile.project_model_file,
                    "model_id": profile.expected_model_id,
                    "source": source_kind,
                    "source_state_file": str(source_path) if source_path else None,
                    "source_model_file": (
                        str(model_store_path) if model_store_path else None
                    ),
                    "requested_sample_count": len(samples),
                }
            )
        self._ai_project_state["enabled"] = True
        self._ai_project_state["active_model"] = model_key
        self._ai_project_state["learn_removed_images"] = learn_removed_images
        self._ai_pending_initialization = {
            "project_dir": str(project_dir),
            "model_key": model_key,
        }
        self._save_state_sync()
        self._update_ai_prediction_mode_button()
        self._configure_ai_for_current_project(
            samples=samples,
            model_key=model_key,
            migrate_legacy=source_kind == "existing",
            force_reinitialize=force_reinitialize,
            merge_samples=(
                removed_samples
                if (
                    already_initialized
                    and not force_reinitialize
                    and learn_removed_images
                )
                else ()
            ),
            excluded_labels=(
                () if learn_removed_images else (AI_REMOVAL_LABEL,)
            ),
            import_store_path=model_store_path,
        )

    def _configure_ai_for_current_project(
        self,
        samples=None,
        model_key: Optional[str] = None,
        migrate_legacy: bool = False,
        force_reinitialize: bool = False,
        merge_samples=None,
        excluded_labels=None,
        import_store_path: Optional[Path] = None,
    ):
        """Load or initialize one project-local model store."""
        manager = getattr(self, '_ai_manager', None)
        if manager is None or not self.current_dir:
            return
        project_dir = self.current_dir.parent
        model_key = model_key or self._ai_project_state.get(
            "active_model", "balanced"
        )
        self._ai_configured_project_dir = (str(project_dir), model_key)
        manager.configure_project(
            project_dir,
            samples or (),
            model_key=model_key,
            migrate_legacy=migrate_legacy,
            force_reinitialize=force_reinitialize,
            merge_samples=merge_samples or (),
            excluded_labels=excluded_labels or (),
            import_store_path=import_store_path,
        )

    def _restore_ai_for_current_project(self):
        """Reuse an initialized project model without rebuilding its sample bank."""
        manager = getattr(self, '_ai_manager', None)
        project_dir = self._current_ai_project_dir()
        if manager is None or project_dir is None:
            return
        model_key = self._ai_project_state.get("active_model", "balanced")
        initialized = is_project_model_initialized(
            self._ai_project_state, project_dir, model_key
        )
        resource_problem = (
            self._active_ai_resource_problem() if initialized else None
        )
        if initialized and resource_problem is None:
            token = (str(project_dir), model_key)
            if self._ai_configured_project_dir != token or not manager.is_ready:
                self._configure_ai_for_current_project(
                    model_key=model_key,
                    excluded_labels=(
                        ()
                        if self._ai_project_state.get(
                            "learn_removed_images", False
                        )
                        else (AI_REMOVAL_LABEL,)
                    ),
                )
        else:
            manager.clear_project()
            self._ai_configured_project_dir = None
            if hasattr(self, "category_panel"):
                if initialized and resource_problem:
                    message = f"AI 未就绪 · {resource_problem} · 点击模型按钮继续"
                else:
                    models = self._ai_project_state.get("models", {})
                    record = models.get(model_key, {})
                    if record.get("initialized"):
                        message = "AI 项目模型文件缺失 · 点击模型按钮重新初始化"
                    else:
                        message = "AI 未初始化 · 点击“AI · 启用”开始"
                self.category_panel.set_ai_status(message, "warning")
        self._update_ai_prediction_mode_button()

    def _on_ai_resource_installed(self, resource) -> None:
        """Report a background install and resume an already initialized project."""
        toast_success(self, f"{resource.display_name}已在后台安装完成")
        self._update_ai_prediction_mode_button()
        if (
            not getattr(self, "_ai_setup_dialog_active", False)
            and self._is_active_ai_model_initialized()
            and self._active_ai_resource_problem() is None
        ):
            self._restore_ai_for_current_project()

    def _on_ai_resource_failed(self, resource, error: str) -> None:
        self._update_ai_prediction_mode_button()
        toast_warning(self, f"{resource.display_name}下载失败：{error}")

    def _on_ai_model_state_changed(self, snapshot: dict):
        """Persist initialization and incremental-learning metadata in project JSON."""
        project_dir = self._current_ai_project_dir()
        if project_dir is None or snapshot.get("project_dir") != str(project_dir):
            return
        model_key = snapshot.get("model_key")
        if not model_key:
            return
        models = self._ai_project_state.setdefault("models", {})
        record = models.setdefault(model_key, {})
        first_initialization = not record.get("initialized")
        model_event = snapshot.get("event", "updated")
        record.update(
            {
                "initialized": True,
                "model_id": snapshot.get("model_id"),
                "project_model_file": snapshot.get("project_model_file"),
                "sample_count": snapshot.get("sample_count", 0),
                "class_counts": snapshot.get("class_counts", {}),
                "provider": snapshot.get("provider", "CPU"),
                "requested_provider": snapshot.get(
                    "requested_provider", "auto"
                ),
                "provider_fallback_reason": snapshot.get(
                    "provider_fallback_reason"
                ),
                "rebuilding": False,
                "updated_at": datetime.now().isoformat(timespec="seconds"),
            }
        )
        record.setdefault("initialized_at", record["updated_at"])
        self._ai_project_state["enabled"] = True
        is_active_model = (
            self._ai_project_state.get("active_model") == model_key
        )
        pending = getattr(self, "_ai_pending_initialization", {})
        if pending.get("model_key") == model_key:
            self._ai_pending_initialization = {}
        self._update_ai_prediction_mode_button()
        if first_initialization:
            self._save_state_sync()
            toast_success(
                self,
                f"AI {get_ai_model_profile(model_key).display_name}初始化完成 · "
                f"{record['sample_count']} 张样本",
            )
        elif model_event == "reinitialized":
            self.save_state()
            toast_success(
                self,
                f"AI {get_ai_model_profile(model_key).display_name}重新初始化完成 · "
                f"{record['sample_count']} 张样本",
            )
        elif model_event == "reinitialize_failed":
            self.save_state()
            toast_warning(self, "AI 重新初始化失败，已继续使用旧模型")
        else:
            self.save_state()
        if (
            is_active_model
            # 普通 updated 事件来自上一张分类后的增量学习保存。此时当前
            # 图片已经由翻页流程完成预测，不能再次启动自动预测。
            and self._ai_model_event_requires_prediction(model_event)
            and
            self.ai_prediction_mode == "assist"
            and self._ai_assist_active
            and self._ai_current_image_data is not None
            and not self._ai_prediction_loading
        ):
            self._ai_prediction_timer.start(self.AI_PREDICTION_DEBOUNCE_MS)

    @staticmethod
    def _ai_model_event_requires_prediction(model_event: str) -> bool:
        """Only model availability changes warrant predicting the current image."""
        return model_event in ("initialized", "reinitialized")

    def _on_ai_status_changed(self, message: str, tone: str):
        if hasattr(self, 'category_panel'):
            self.category_panel.set_ai_status(message, tone)
        self._update_ai_prediction_mode_button()
        if tone in ("disabled", "error"):
            self._finish_ai_prediction_ui()
            if tone == "error" and getattr(self, "_ai_auto_task", None) is not None:
                self._finish_ai_auto_task(cancelled=True)
            if tone == "error":
                toast_error(self, message)
            else:
                toast_warning(self, message)
        elif (
            tone == "ready"
            and self.ai_prediction_mode == "assist"
            and self._ai_assist_active
            and self._ai_current_image_data is not None
            and not self._ai_prediction_loading
        ):
            self._ai_prediction_timer.start(self.AI_PREDICTION_DEBOUNCE_MS)

    def _on_ai_index_progress(self, current: int, total: int):
        if total <= 0 or not hasattr(self, 'category_panel'):
            return
        self.category_panel.set_ai_status(
            f"AI 正在学习现有标注 {current}/{total}…", "working"
        )

    def _on_ai_image_ready(self, image_data):
        """记录当前图片，并依据模式调度预测。"""
        current_path = self.get_current_image_path()
        current_path_text = str(current_path) if current_path else None
        auto_task = getattr(self, "_ai_auto_task", None)
        if (
            auto_task is not None
            and current_path_text != auto_task.get("active_path")
        ):
            # 文件操作管理器分类后会自行翻到下一张；全自动队列尚未正式
            # 接管该图片前忽略这个过渡回调，避免并发提交错误图片。
            return
        # 同一张图可能先由导航缓存直接显示，随后又收到仍在途的异步
        # 预加载完成信号。第二个回调不能重启自动预测，否则会连续出现
        # 两次相同的“自动预测中”遮罩。
        if (
            current_path_text is not None
            and self._ai_current_image_path == current_path_text
            and self._ai_current_image_data is not None
        ):
            return

        self._ai_prediction_timer.stop()
        self._ai_current_image_path = current_path_text
        self._ai_current_image_data = self._coerce_ai_rgb_image(image_data)
        if hasattr(self, "image_view_panel"):
            self.image_view_panel.clear_prediction_overlay()

        if self._ai_current_image_data is None:
            if hasattr(self, "category_panel"):
                self.category_panel.set_ai_status(
                    "AI 无法读取当前图片数据", "warning"
                )
            return

        if not self._is_active_ai_model_initialized():
            if hasattr(self, "category_panel"):
                self.category_panel.set_ai_status(
                    "AI 未初始化 · 点击“AI · 启用”选择模型和样本来源",
                    "warning",
                )
            return

        resource_problem = self._active_ai_resource_problem()
        if resource_problem:
            if hasattr(self, "category_panel"):
                self.category_panel.set_ai_status(
                    f"AI 未就绪 · {resource_problem} · 点击模型按钮继续",
                    "warning",
                )
            return

        manager = getattr(self, "_ai_manager", None)
        if manager is None or not manager.is_ready:
            if hasattr(self, "category_panel"):
                self.category_panel.set_ai_status(
                    "AI 模型正在后台加载…", "working"
                )
            return

        if getattr(self, "_ai_auto_task", None) is not None or (
            self.ai_prediction_mode == "assist"
            and getattr(self, "_ai_assist_active", False)
        ):
            if hasattr(self, "category_panel"):
                self.category_panel.set_ai_status(
                    (
                        "AI 全自动 · 正在处理…"
                        if getattr(self, "_ai_auto_task", None) is not None
                        else "AI 辅助 · 等待图片稳定 200ms…"
                    ),
                    "working",
                )
            self._ai_prediction_timer.start(self.AI_PREDICTION_DEBOUNCE_MS)
        elif hasattr(self, "category_panel"):
            self.category_panel.set_ai_status(
                (
                    "AI 全自动 · 按 Tab 校验并启动"
                    if self.ai_prediction_mode == "auto"
                    else "AI 辅助 · 按 Tab 开启"
                ),
                "ready",
            )

    def _coerce_ai_rgb_image(self, image_data):
        """把加载器可能返回的 NumPy/PIL/QPixmap 数据统一为 RGB uint8。"""
        if isinstance(image_data, np.ndarray):
            if image_data.ndim != 3 or image_data.shape[2] < 3:
                return None
            rgb = image_data[..., :3]
            if rgb.dtype != np.uint8:
                rgb = np.clip(rgb, 0, 255).astype(np.uint8)
            return np.ascontiguousarray(rgb)

        if isinstance(image_data, QPixmap):
            if image_data.isNull():
                return None
            image_data = image_data.toImage()

        if isinstance(image_data, QImage):
            if image_data.isNull():
                return None
            rgb_image = image_data.convertToFormat(QImage.Format.Format_RGB888)
            bits = rgb_image.bits()
            bits.setsize(rgb_image.sizeInBytes())
            rows = np.frombuffer(bits, dtype=np.uint8).reshape(
                rgb_image.height(), rgb_image.bytesPerLine()
            )
            rgb = rows[:, : rgb_image.width() * 3].reshape(
                rgb_image.height(), rgb_image.width(), 3
            )
            return np.ascontiguousarray(rgb.copy())

        if hasattr(image_data, "convert"):
            try:
                return np.ascontiguousarray(
                    np.asarray(image_data.convert("RGB"), dtype=np.uint8)
                )
            except Exception:
                self.logger.debug(
                    "AI 无法转换图片数据类型: %s", type(image_data).__name__
                )
        return None

    def _trigger_scheduled_ai_prediction(self):
        source = "full_auto" if getattr(self, "_ai_auto_task", None) is not None else "assist"
        self._trigger_ai_prediction(source)

    def _handle_ai_tab(self):
        """Handle one Tab action and reject nested delivery of the same key press."""
        if getattr(self, "_ai_tab_handling", False):
            return
        self._ai_tab_handling = True
        try:
            self._handle_ai_tab_once()
        finally:
            self._ai_tab_handling = False

    def _handle_ai_tab_once(self):
        """Tab toggles assistance or starts/pauses a full-auto task."""
        if (
            not self._is_active_ai_model_initialized()
            or self._active_ai_resource_problem()
        ):
            self.show_ai_project_setup()
            return
        manager = getattr(self, "_ai_manager", None)
        if manager is None or not manager.is_ready:
            toast_info(self, "AI 模型正在后台加载，请稍候")
            return

        if self.ai_prediction_mode == "auto":
            if self._ai_auto_task is not None:
                self._request_ai_auto_pause()
            else:
                self._start_ai_auto_task()
            return

        if not self._ai_assist_active and not self._confirm_ai_assist_start():
            return
        self._ai_assist_active = not self._ai_assist_active
        self._ai_prediction_timer.stop()
        self._update_ai_prediction_mode_button()
        if self._ai_assist_active:
            if hasattr(self, "category_panel"):
                self.category_panel.set_ai_status(
                    "AI 辅助已开启 · 停止翻页 200ms 后分析", "ready"
                )
            if self._ai_current_image_data is not None:
                self._ai_prediction_timer.start(self.AI_PREDICTION_DEBOUNCE_MS)
            toast_info(self, "AI 辅助已开启，再按 Tab 关闭")
        else:
            if hasattr(self, "image_view_panel"):
                self.image_view_panel.clear_prediction_overlay()
            if hasattr(self, "category_panel"):
                self.category_panel.set_ai_status("AI 辅助 · 按 Tab 开启", "ready")
            toast_info(self, "AI 辅助已关闭")

    def _ai_sample_shortage_warning(self, recommended: int) -> str:
        model_key = self._ai_project_state.get("active_model", "balanced")
        record = self._ai_project_state.get("models", {}).get(model_key, {})
        counts = record.get("class_counts", {}) if isinstance(record, dict) else {}
        labels = list(self.ordered_categories)
        if self._ai_project_state.get("learn_removed_images", False):
            labels.append(AI_REMOVAL_LABEL)
        missing = [
            f"{'移除' if label == AI_REMOVAL_LABEL else label} {int(counts.get(label, 0))}/{recommended} 张"
            for label in labels
            if int(counts.get(label, 0)) < recommended
        ]
        if not missing:
            return ""
        return (
            f"样本数量不足，建议每个类别至少 {recommended} 张有效人工样本：\n"
            + "\n".join(missing)
            + "\n\n强行开启可能导致误判；零样本类别无法识别，至少两个类别有样本才能预测。"
        )

    def _confirm_ai_assist_start(self) -> bool:
        warning = self._ai_sample_shortage_warning(5)
        if not warning:
            return True
        box = ThemedMessageBox(self)
        box.setWindowTitle("启动 AI 辅助标注")
        box.setIcon(QMessageBox.Icon.Warning)
        box.setText(warning)
        confirm = box.addButton("强行开启", QMessageBox.ButtonRole.AcceptRole)
        cancel = box.addButton("取消", QMessageBox.ButtonRole.RejectRole)
        box.setDefaultButton(cancel)
        box.setEscapeButton(cancel)
        box.set_destructive(True)
        box.exec()
        return box.clickedButton() is confirm

    def _ai_auto_preflight_problem(self) -> Optional[str]:
        if not self.is_copy_mode:
            return "全自动分类只能在复制模式下使用"
        if self.is_multi_category:
            return "全自动分类仅支持单分类模式"
        if not self.image_files:
            return "当前目录没有可处理的图片"
        model_key = self._ai_project_state.get("active_model", "balanced")
        record = self._ai_project_state.get("models", {}).get(model_key, {})
        counts = record.get("class_counts", {}) if isinstance(record, dict) else {}
        required_labels = list(self.ordered_categories)
        if self._ai_project_state.get("learn_removed_images", False):
            required_labels.append(AI_REMOVAL_LABEL)
        if sum(int(counts.get(label, 0)) > 0 for label in required_labels) < 2:
            return "至少需要两个各有 1 张有效人工样本的类别；零样本类别不参与预测"
        return None

    def _confirm_ai_auto_start(self) -> Optional[bool]:
        """Return whether existing labels should be reprocessed, or None on cancel."""
        dialog = QDialog(self)
        dialog.setWindowTitle("启动全自动分类")
        dialog.setModal(True)
        layout = QVBoxLayout(dialog)
        configure_dialog(dialog, layout)
        title = QLabel("⚠️ 全自动分类风险确认")
        title.setProperty("uiRole", "title")
        layout.addWidget(title)
        shortage = self._ai_sample_shortage_warning(20)
        warning = QLabel(
            (shortage + "\n\n" if shortage else "")
            + "全自动会依据当前样本库直接复制图片到预测类别。预测不确定的图片会跳过，"
            "任务期间本窗口的翻页、分类、模式切换和鼠标操作都会被锁定。\n\n"
            "有有效样本的类别均参与预测；零样本类别无法识别，样本少的类别更容易误判。\n\n"
            "请确认样本质量和类别定义可靠；AI 结果仍可能出错。窗口最小化不会暂停任务。"
        )
        warning.setWordWrap(True)
        layout.addWidget(warning)
        reprocess = QCheckBox("重新处理已经标注或移除的图片（默认跳过）")
        reprocess.setChecked(False)
        layout.addWidget(reprocess)
        buttons = QHBoxLayout()
        cancel = QPushButton("取消")
        confirm = QPushButton("我已了解风险，强行开启" if shortage else "我已了解风险，开始全自动")
        style_button(cancel, "secondary")
        style_button(confirm, "danger", min_width=200)
        cancel.clicked.connect(dialog.reject)
        cancel.setDefault(True)
        confirm.clicked.connect(dialog.accept)
        buttons.addStretch()
        buttons.addWidget(cancel)
        buttons.addWidget(confirm)
        layout.addLayout(buttons)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return None
        return reprocess.isChecked()

    def _start_ai_auto_task(self) -> None:
        problem = self._ai_auto_preflight_problem()
        if problem:
            toast_warning(self, problem)
            return
        reprocess_existing = self._confirm_ai_auto_start()
        if reprocess_existing is None:
            return
        self._ai_assist_active = False
        self._ai_auto_pause_requested = False
        self._ai_auto_task = {
            "paths": [str(path) for path in self.image_files],
            "position": 0,
            "total": len(self.image_files),
            "processed": 0,
            "skipped_existing": 0,
            "uncertain_paths": [],
            "auto_counts": {},
            "auto_removed": 0,
            "reprocess_existing": bool(reprocess_existing),
            "active_path": None,
        }
        self._set_ai_auto_ui_locked(True)
        self._update_ai_prediction_mode_button()
        self.statusBar.showMessage("🤖 全自动分类正在运行；按 Tab 可请求暂停")
        QTimer.singleShot(0, self._advance_ai_auto_task)

    def _set_ai_auto_ui_locked(self, locked: bool) -> None:
        names = (
            "image_list", "image_search_widget", "filter_button", "folder_button",
            "open_directory_toolbar_button", "refresh_button", "mode_button",
            "ai_prediction_mode_button", "ai_model_button", "category_panel",
        )
        if locked:
            self._ai_auto_locked_widgets = []
            for name in names:
                widget = getattr(self, name, None)
                if widget is not None:
                    self._ai_auto_locked_widgets.append((widget, widget.isEnabled()))
                    widget.setEnabled(False)
            if hasattr(self, "image_view_panel"):
                button = self.image_view_panel.delete_button
                self._ai_auto_locked_widgets.append((button, button.isEnabled()))
                button.setEnabled(False)
        else:
            for widget, was_enabled in self._ai_auto_locked_widgets:
                widget.setEnabled(was_enabled)
            self._ai_auto_locked_widgets = []

    def _advance_ai_auto_task(self) -> None:
        task = self._ai_auto_task
        if task is None or self._ai_auto_pause_requested:
            return
        paths = task["paths"]
        while task["position"] < len(paths):
            path = paths[task["position"]]
            task["position"] += 1
            if not task["reprocess_existing"] and (
                path in self.classified_images or path in self.removed_images
            ):
                task["skipped_existing"] += 1
                continue
            matching_index = next(
                (i for i, item in enumerate(self.image_files) if str(item) == path),
                None,
            )
            if matching_index is None:
                continue
            task["active_path"] = path
            self._ai_current_image_path = None
            self._ai_current_image_data = None
            self.current_index = matching_index
            self._nav_manager.show_current_image()
            return
        self._finish_ai_auto_task()

    def _handle_ai_auto_result(self, result) -> None:
        task = self._ai_auto_task
        if task is None or result.image_path != task.get("active_path"):
            return
        if self._ai_auto_pause_requested:
            self._show_ai_auto_pause_confirmation(result)
            return
        task["processed"] += 1
        if not result.suggestions or result.uncertain:
            task["uncertain_paths"].append(result.image_path)
            QTimer.singleShot(0, self._advance_ai_auto_task)
            return
        top = result.suggestions[0]
        if top.category == AI_REMOVAL_LABEL:
            task["auto_removed"] += 1
            self._file_ops_manager.move_to_remove(result.image_path)
        elif top.category in self.ordered_categories:
            counts = task["auto_counts"]
            counts[top.category] = counts.get(top.category, 0) + 1
            self._file_ops_manager.move_to_category(result.image_path, top.category)
        else:
            task["uncertain_paths"].append(result.image_path)
        QTimer.singleShot(0, self._advance_ai_auto_task)

    def _request_ai_auto_pause(self) -> None:
        if self._ai_auto_pause_requested:
            return
        self._ai_auto_pause_requested = True
        self._ai_prediction_timer.stop()
        self.statusBar.showMessage("⏸ 全自动任务已暂停，等待退出确认")
        if not self._ai_prediction_loading:
            self._show_ai_auto_pause_confirmation()

    def _show_ai_auto_pause_confirmation(self, pending_result=None) -> None:
        if self._ai_auto_task is None:
            return
        box = ThemedMessageBox(self)
        box.setWindowTitle("全自动任务已暂停")
        box.setIcon(QMessageBox.Icon.Warning)
        box.setText(
            "是否结束本次全自动任务？\n\n选择“否”将继续任务，当前进度不会丢失。"
        )
        box.setStandardButtons(
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        box.setDefaultButton(QMessageBox.StandardButton.No)
        box.set_destructive(True)
        answer = box.exec()
        if answer == QMessageBox.StandardButton.Yes:
            self._finish_ai_auto_task(cancelled=True)
            return
        self._ai_auto_pause_requested = False
        self.statusBar.showMessage("🤖 全自动分类正在运行；按 Tab 可请求暂停")
        if pending_result is not None:
            self._handle_ai_auto_result(pending_result)
        else:
            QTimer.singleShot(0, self._advance_ai_auto_task)

    @staticmethod
    def _count_classifications(classified_images) -> Dict[str, int]:
        counts = {}
        for value in classified_images.values():
            labels = value if isinstance(value, (list, tuple, set)) else (value,)
            for label in labels:
                counts[label] = counts.get(label, 0) + 1
        return counts

    def _finish_ai_auto_task(self, cancelled: bool = False) -> None:
        task = self._ai_auto_task
        if task is None:
            return
        self._ai_prediction_timer.stop()
        self._ai_auto_task = None
        self._ai_auto_pause_requested = False
        if self._ai_prediction_loading:
            self._finish_ai_prediction_ui()
        self._set_ai_auto_ui_locked(False)
        self._update_ai_prediction_mode_button()
        if hasattr(self, "category_panel"):
            self.category_panel.set_ai_status(
                "AI 全自动 · 按 Tab 校验并启动", "ready"
            )
        current_path = self.get_current_image_path()
        if current_path is not None:
            self.statusBar.showMessage(f"📷 {Path(current_path).name}")
        else:
            self.statusBar.clearMessage()
        final_counts = self._count_classifications(self.classified_images)
        category_text = "，".join(
            f"{category} {final_counts.get(category, 0)}"
            for category in self.ordered_categories
        ) or "无"
        auto_text = "，".join(
            f"{category} {task['auto_counts'].get(category, 0)}"
            for category in self.ordered_categories
        ) or "无"
        summary = (
            f"{'任务已提前结束' if cancelled else '任务已完成'}\n\n"
            f"总图片：{task['total']}\n"
            f"当前分类：{category_text}\n"
            f"当前移除：{len(self.removed_images)}\n\n"
            f"本次自动标注：{auto_text}\n"
            f"本次自动移除：{task['auto_removed']}\n"
            f"不确定：{len(task['uncertain_paths'])}\n"
            f"跳过已处理：{task['skipped_existing']}"
        )
        box = ThemedMessageBox(self)
        box.setWindowTitle("全自动分类结果")
        box.setIcon(QMessageBox.Icon.Information)
        box.setText(summary)
        box.setStandardButtons(QMessageBox.StandardButton.Ok)
        box.exec()
        if self.app_config.ai_auto_notify_on_complete:
            self._show_ai_auto_system_notification(summary.split("\n\n", 1)[0])
        if task["uncertain_paths"]:
            first_path = task["uncertain_paths"][0]
            index = next(
                (i for i, item in enumerate(self.image_files) if str(item) == first_path),
                None,
            )
            if index is not None:
                self.current_index = index
                self._nav_manager.show_current_image()

    def _show_ai_auto_system_notification(self, message: str) -> None:
        if not QSystemTrayIcon.isSystemTrayAvailable():
            return
        tray = QSystemTrayIcon(self.windowIcon(), self)
        tray.setToolTip(RELEASE_NAME)
        tray.show()
        tray.showMessage(
            "全自动分类任务完成",
            message,
            QSystemTrayIcon.MessageIcon.Information,
            8000,
        )
        self._ai_auto_notification_tray = tray
        QTimer.singleShot(10000, tray.hide)

    def _trigger_ai_prediction(self, source: str):
        """提交当前图片预测，并在真正推理期间锁定导航。"""
        if self._active_ai_resource_problem() is not None:
            return
        manager = getattr(self, '_ai_manager', None)
        current_path = self.get_current_image_path()
        if manager is None or current_path is None or self._ai_prediction_loading:
            if self._ai_prediction_loading and source == "manual":
                self._notify_ai_navigation_locked()
            return
        if (
            self._ai_current_image_path != str(current_path)
            or not isinstance(self._ai_current_image_data, np.ndarray)
        ):
            if source == "manual":
                toast_warning(self, "当前图片仍在加载，暂时无法预测")
            return
        if not manager.is_ready:
            message = (
                "AI 模型不可用"
                if manager.is_disabled
                else "AI 正在建立样本库，完成后即可预测"
            )
            if hasattr(self, "category_panel"):
                self.category_panel.set_ai_status(message, "warning")
            if source == "manual":
                toast_warning(self, message)
            return

        prediction_categories = tuple(self.ordered_categories)
        if self._ai_project_state.get("learn_removed_images", False):
            prediction_categories += (AI_REMOVAL_LABEL,)
        request_id = manager.submit_prediction(
            str(current_path),
            self._ai_current_image_data,
            prediction_categories,
        )
        if request_id < 0:
            toast_warning(self, "AI 模型当前不可用")
            return
        self._ai_prediction_active_path = str(current_path)
        self._ai_prediction_active_request_id = request_id
        self._set_ai_prediction_loading(True, source)

    def _set_ai_prediction_loading(self, loading: bool, source: str = "auto"):
        self._ai_prediction_loading = loading
        controls_enabled = not loading and getattr(self, "_ai_auto_task", None) is None
        if hasattr(self, "image_view_panel"):
            if loading:
                mode_text = "全自动" if source == "full_auto" else "辅助预测"
                self.image_view_panel.show_prediction_loading(
                    f"{mode_text}中，请稍候…"
                )
            else:
                self.image_view_panel.hide_prediction_loading()
            self.image_view_panel.delete_button.setEnabled(controls_enabled)
        locked_controls = (
            "image_list",
            "image_search_widget",
            "filter_button",
            "folder_button",
            "open_directory_toolbar_button",
            "refresh_button",
            "ai_prediction_mode_button",
        )
        for control_name in locked_controls:
            control = getattr(self, control_name, None)
            if control is not None:
                control.setEnabled(controls_enabled)
        if loading and hasattr(self, 'category_panel'):
            self.category_panel.set_ai_status("AI 分析中…", "working")
        if loading:
            self.statusBar.showMessage("⏳ 模型正在预测，完成前暂时锁定翻页")

    def _finish_ai_prediction_ui(self):
        if not self._ai_prediction_loading:
            return
        self._set_ai_prediction_loading(False)
        self._ai_prediction_active_path = None
        self._ai_prediction_active_request_id = -1
        if getattr(self, "_ai_auto_task", None) is not None:
            self.statusBar.showMessage("🤖 全自动分类正在运行；按 Tab 可请求暂停")
            return
        current_path = self.get_current_image_path()
        if current_path is not None:
            self.statusBar.showMessage(f"📷 {Path(current_path).name}")
        else:
            self.statusBar.clearMessage()

    def _notify_ai_navigation_locked(self):
        now = time.monotonic()
        self.statusBar.showMessage("⏳ 模型正在预测，暂时不能翻页")
        if now - self._ai_navigation_notice_time >= 1.0:
            self._ai_navigation_notice_time = now
            toast_info(self, "模型正在预测，请等待当前结果")

    def _prepare_ai_navigation(self) -> bool:
        """返回是否允许导航，并清理尚未触发的旧图片预测。"""
        if self._ai_prediction_loading:
            self._notify_ai_navigation_locked()
            return False
        self._ai_prediction_timer.stop()
        self._ai_current_image_data = None
        self._ai_current_image_path = None
        if hasattr(self, "image_view_panel"):
            self.image_view_panel.clear_prediction_overlay()
        if hasattr(self, "category_panel"):
            resource_problem = self._active_ai_resource_problem()
            manager = getattr(self, "_ai_manager", None)
            if resource_problem:
                self.category_panel.set_ai_status(
                    f"AI 未就绪 · {resource_problem} · 点击模型按钮继续",
                    "warning",
                )
            elif manager is None or not manager.is_ready:
                self.category_panel.set_ai_status(
                    "AI 模型正在后台加载…", "working"
                )
            elif self._ai_auto_task is not None:
                self.category_panel.set_ai_status(
                    "AI 全自动 · 等待当前图片显示…", "working"
                )
            elif self.ai_prediction_mode == "assist" and self._ai_assist_active:
                self.category_panel.set_ai_status(
                    "AI 辅助 · 等待当前图片显示…", "working"
                )
            else:
                self.category_panel.set_ai_status(
                    (
                        "AI 全自动 · 按 Tab 校验并启动"
                        if self.ai_prediction_mode == "auto"
                        else "AI 辅助 · 按 Tab 开启"
                    ),
                    "ready",
                )
        return True

    def _on_ai_prediction_ready(self, result):
        is_active_request = result.request_id == self._ai_prediction_active_request_id
        if is_active_request:
            self._finish_ai_prediction_ui()
        current_path = self.get_current_image_path()
        if current_path is None or str(current_path) != result.image_path:
            return
        if self._ai_auto_task is not None:
            self._handle_ai_auto_result(result)
            return
        if not hasattr(self, 'category_panel'):
            return
        if not result.suggestions:
            reason = result.reason or "AI 样本不足"
            self.category_panel.set_ai_status(reason, "warning")
            if hasattr(self, "image_view_panel"):
                self.image_view_panel.show_prediction_result(
                    "AI 暂无建议", reason, [], "warning"
                )
            if hasattr(self, "image_view_panel"):
                self.image_view_panel.show_prediction_toast(reason, "warning")
            return

        top = result.suggestions[0]

        def display_category(category: str) -> str:
            return "建议移除" if category == AI_REMOVAL_LABEL else category

        alternatives = [
            f"{index}. {display_category(suggestion.category)}"
            f"（相似度 {suggestion.similarity * 100:.0f}%）"
            for index, suggestion in enumerate(result.suggestions, start=1)
        ]
        self.category_panel.show_ai_prediction(
            category=display_category(top.category),
            similarity=top.similarity,
            uncertain=result.uncertain,
            latency_ms=result.latency_ms,
            provider=result.provider,
            alternatives=alternatives,
            is_removal=top.category == AI_REMOVAL_LABEL,
        )

        is_removal = top.category == AI_REMOVAL_LABEL
        display_top = display_category(top.category)
        if is_removal:
            title = "AI 建议移除" if not result.uncertain else "AI 可能建议移除"
            confirmation = "Delete 确认" if not result.uncertain else "请人工判断"
        else:
            title = (
                f"AI 建议 · {display_top}"
                if not result.uncertain
                else f"AI 不确定 · {display_top}"
            )
            confirmation = "Enter 确认" if not result.uncertain else "请人工判断"
        other_suggestions = " · ".join(
            f"{display_category(suggestion.category)} "
            f"相似度 {suggestion.similarity * 100:.0f}%"
            for suggestion in result.suggestions[1:]
        )
        detail = (
            f"相似度 {top.similarity * 100:.0f}% · "
            f"{result.provider} "
            f"{result.latency_ms:.0f}ms · {confirmation}"
        )
        if other_suggestions:
            detail += f"\n备选：{other_suggestions}"
        overlay_tone = "warning" if result.uncertain or is_removal else "ready"
        if hasattr(self, "image_view_panel"):
            self.image_view_panel.show_prediction_result(
                title, detail, alternatives, overlay_tone
            )

        if result.uncertain:
            if hasattr(self, "image_view_panel"):
                self.image_view_panel.show_prediction_toast(
                    f"AI 预测不确定：{display_top}，请人工判断", "warning"
                )
        elif is_removal:
            if hasattr(self, "image_view_panel"):
                self.image_view_panel.show_prediction_toast(
                    "AI 建议移除当前图片，请按 Delete 确认", "warning"
                )
        else:
            if hasattr(self, "image_view_panel"):
                self.image_view_panel.show_prediction_toast(
                    f"AI 建议：{display_top}（相似度 "
                    f"{top.similarity * 100:.0f}%）",
                    duration=1800,
                )

        # 高可信建议只预选，不执行文件操作；Enter/快捷键仍由用户确认。
        if (
            not result.uncertain
            and result.image_path not in self.classified_images
            and top.category in self.ordered_categories
        ):
            category_index = self.ordered_categories.index(top.category)
            self.category_panel.update_selection(category_index)
            self.current_category_index = category_index
            self.selected_category = top.category

    # ===== 文件操作事件处理（来自FileOperationManager）=====

    def on_file_moved(self, src: str, dst: str):
        """文件移动完成"""
        # 优化：只更新受影响图片的状态，不重建整个列表
        # 分类操作不改变列表内容，只改变状态显示
        if hasattr(self, 'image_list_model'):
            is_classified = src in self.classified_images
            is_multi = isinstance(self.classified_images.get(src), list) and len(self.classified_images.get(src, [])) > 1
            self.image_list_model.update_status_by_path(src, is_classified, is_removed=False, is_multi=is_multi)
        # 更新统计和计数（不包含 image_list，避免重建列表）
        self.schedule_ui_update('statistics', 'category_counts')
        self.statusBar.showMessage(f"✅ 已分类到 {Path(dst).parent.name}")
        category = self.classified_images.get(src)
        if isinstance(category, str) and getattr(self, '_ai_manager', None):
            self._ai_manager.learn(src, category, actual_path=dst)

    def on_file_removed(self, path: str):
        """文件移除完成"""
        # 优化：只更新受影响图片的状态，不重建整个列表
        if hasattr(self, 'image_list_model'):
            self.image_list_model.update_status_by_path(path, is_classified=False, is_removed=True, is_multi=False)
        # 更新统计（不包含 image_list，避免重建列表）
        self.schedule_ui_update('statistics')
        self.statusBar.showMessage("✅ 已移除")
        if getattr(self, '_ai_manager', None):
            if self._ai_project_state.get("learn_removed_images", False):
                actual_path = self.get_real_file_path(path)
                self._ai_manager.learn(
                    path, AI_REMOVAL_LABEL, actual_path=str(actual_path)
                )
            else:
                self._ai_manager.forget(path)

    def on_file_restored(self, path: str):
        """文件恢复完成"""
        # 优化：先更新单个项的状态
        if hasattr(self, 'image_list_model'):
            is_classified = path in self.classified_images
            is_removed = path in self.removed_images
            is_multi = isinstance(self.classified_images.get(path), list) and len(self.classified_images.get(path, [])) > 1
            self.image_list_model.update_status_by_path(path, is_classified, is_removed, is_multi)
        # 撤销操作可能影响过滤结果，直接应用过滤（内部会调用 show_current_image）
        self.apply_image_filter()
        # 更新统计和计数
        self.schedule_ui_update('statistics', 'category_counts', 'category_buttons')
        # 刷新按钮样式（撤销分类后需要更新已分类状态显示）
        self.refresh_category_buttons_style()
        self.statusBar.showMessage("✅ 已撤销")
        if path not in self.classified_images and getattr(self, '_ai_manager', None):
            self._ai_manager.forget(path)

    def on_mode_changed(self, is_copy_mode: bool):
        """操作模式变更"""
        mode_text = "复制模式" if is_copy_mode else "移动模式"
        self.statusBar.showMessage(f"✅ 已切换到{mode_text}")
        self.schedule_ui_update('ui_state')

    def on_operation_failed(self, path: str, reason: str):
        """文件操作失败"""
        from ui.components.toast import toast_error
        toast_error(self, f"操作失败: {reason}")

    # ===== 类别管理事件处理（来自CategoryManager）=====

    def on_categories_changed(self, categories: list):
        """类别列表变更"""
        if (
            getattr(self, 'focused_category', None)
            and self.focused_category not in categories
        ):
            self._deactivate_category_focus()
        # Manager已更新状态，这里只需UI刷新
        self.update_category_buttons()
        self.schedule_ui_update('statistics', 'category_counts')
        # 类别变化后重新绑定类别快捷键
        self.setup_shortcuts()

    def on_category_selection_changed(self, index: int, name: str):
        """类别选中状态变更"""
        self.current_category_index = index
        self.selected_category = name

    def on_category_mode_changed(self, is_multi: bool):
        """多分类模式变更"""
        self.is_multi_category = is_multi
        self.schedule_ui_update('ui_state', 'category_buttons')

    def on_sort_mode_changed(self, sort_mode: str, ascending: bool):
        """排序模式变更"""
        self.schedule_ui_update('category_buttons')
        self.statusBar.showMessage(f"✅ 排序已更新：{sort_mode}")

    # ===== UI更新和显示方法 =====
    
    def schedule_ui_update(self, *components):
        """调度UI更新，批量处理以提高性能"""
        if not components:
            return
            
        with self.ui_update_lock:
            for component in components:
                self.pending_ui_updates.add(component)
        
        # 延迟200ms后执行批量更新
        if not self.ui_update_timer.isActive():
            self.ui_update_timer.start(200)
        
    def perform_batch_ui_update(self):
        """执行批量UI更新 - 优化防止重复更新"""
        with self.ui_update_lock:
            if not self.pending_ui_updates:
                return
            components_to_update = self.pending_ui_updates.copy()
            self.pending_ui_updates.clear()

        # Codex方案：检查是否需要在image_list更新后重新应用过滤
        reapply_filter = False

        try:
            # 优化：如果包含current_image更新，只执行一次，并移除其他可能冲突的更新
            if 'current_image' in components_to_update:
                self._show_current_image_internal()
                # 移除其他可能导致重复刷新的组件
                components_to_update.discard('current_image')
                components_to_update.discard('ui_state')  # ui_state会在current_image中处理

            # 定义组件更新优先级，确保 category_counts 在 category_buttons 之前执行
            # 这样 category_panel.update_data() 拿到的是最新的计数值
            update_priority = {
                'category_counts': 0,    # 最先执行，更新计数字典
                'statistics': 1,         # 其次更新统计面板
                'category_buttons': 2,   # 再更新类别按钮（依赖最新计数）
                'image_list': 3,         # 图片列表
                'ui_state': 4,           # UI状态
            }
            # 按优先级排序，未定义的组件放最后
            ordered_updates = sorted(
                components_to_update,
                key=lambda item: update_priority.get(item, len(update_priority))
            )

            for component in ordered_updates:
                if component == 'category_buttons':
                    # 使用CategoryPanel更新按钮
                    if hasattr(self, 'category_panel'):
                        # 获取当前图片的分类状态
                        current_category = None
                        if self.image_files and 0 <= self.current_index < len(self.image_files):
                            current_path = str(self.image_files[self.current_index])
                            current_category = self.classified_images.get(current_path)

                        self.category_panel.update_data(
                            self.ordered_categories,
                            self.category_counts,
                            self.current_category_index,
                            self.categories,
                            current_category
                        )
                elif component == 'category_counts':
                    self._update_category_counts_internal()
                elif component == 'image_list':
                    self._update_image_list_internal()
                elif component == 'statistics':
                    self._update_statistics_internal()
                elif component == 'ui_state':
                    self._update_ui_state_internal()

            # Codex方案：本轮包含image_list更新时，刷新完成后再统一套用过滤
            if self._pending_reapply_filter and 'image_list' in components_to_update:
                reapply_filter = True

        except Exception as e:
            self.logger.error(f"批量更新UI失败: {e}")

        # Codex方案：在批量更新完成后再应用过滤（避免被_update_image_list_internal覆盖）
        if reapply_filter:
            self._pending_reapply_filter = False
            self.apply_image_filter()

    def _update_image_list_internal(self):
        """内部图片列表更新方法 - 完整显示所有图片"""
        try:
            # Phase 1.1 Migration: 使用 Model API 替代 QListWidget API
            # 不再需要 clear()，_init_data 会重置模型

            if not self.image_files:
                # 如果没有文件，初始化空模型以清空视图
                if hasattr(self, 'image_list_model'):
                    self.image_list_model._init_data([], {}, set(), set())
                return

            total_count = len(self.image_files)

            # 图片列表始终显示完整内容，不使用滑动窗口
            # 滑动窗口只用于内部预加载优化，不影响用户可见的列表
            if hasattr(self, 'initial_batch_loaded') and not self.initial_batch_loaded:
                # 初始加载阶段：只显示前200张，快速启动
                start_index = 0
                end_index = min(200, total_count)
                self.logger.info(f"初始加载阶段，显示前 {end_index} 张图片")
            else:
                # 扫描完成后：始终显示完整列表
                start_index = 0
                end_index = total_count
                self.logger.info(f"显示完整图片列表，共 {total_count} 张图片")

            # 1. 准备数据
            # 将 Path 对象转换为字符串，适配 Model 接口
            display_files = [str(f) for f in self.image_files[start_index:end_index]]

            # 2. 获取有效分类记录（只包含类别仍然存在的分类）
            valid_classified_images = self._get_valid_classified_images()

            # 3. 预计算多分类集合 (O(N) -> Set lookup)
            multi_classified = {
                p for p, c in valid_classified_images.items()
                if isinstance(c, list) and len(c) > 1
            }

            # 4. 批量初始化 Model (高性能，使用有效分类记录)
            self.image_list_model._init_data(
                display_files,
                valid_classified_images,
                self.removed_images,
                multi_classified
            )

            # 5. 恢复选中状态
            # 确保当前索引在显示范围内
            if 0 <= self.current_index < len(display_files):
                # 获取 ModelIndex (Row=current_index, Col=0)
                idx = self.image_list_model.index(self.current_index, 0)
                if idx.isValid():
                    # 使用 SelectionModel API 设置选中
                    self.image_list.selectionModel().setCurrentIndex(
                        idx,
                        QItemSelectionModel.SelectionFlag.ClearAndSelect
                    )
                    # 使用 View API 滚动
                    self.image_list.scrollTo(idx, QAbstractItemView.ScrollHint.EnsureVisible)

            # 如果列表是完整的，同步当前选中状态
            if start_index == 0 and end_index == total_count:
                QTimer.singleShot(100, self.sync_image_list_selection)
                
        except Exception as e:
            self.logger.error(f"更新图片列表失败: {e}")

    def _get_valid_classified_images(self) -> dict:
        """获取有效的分类记录（只包含类别仍然存在的分类）

        当类别被忽略后，对应的分类记录不应该显示为"已分类"状态。
        此方法过滤掉类别已不存在的分类记录。

        Returns:
            dict: 有效的分类记录 {path: category}
        """
        valid_classified = {}
        for path, cat in self.classified_images.items():
            if isinstance(cat, list):
                # 多分类：只保留仍然存在的类别
                valid_cats = [c for c in cat if c in self.categories]
                if valid_cats:
                    valid_classified[path] = valid_cats
            elif cat in self.categories:
                valid_classified[path] = cat
        return valid_classified

    def _update_statistics_internal(self):
        """内部统计更新方法"""
        try:
            total_images = len(self.image_files)
            # 使用有效分类记录（只计算类别仍然存在的分类）
            valid_classified = self._get_valid_classified_images()
            classified_count = len(valid_classified)
            removed_count = len(self.removed_images)
            
            # 统计面板显示真实的数据，不显示滑动窗口信息
            display_count = None  # 不显示单独的显示数量
            
            if hasattr(self, 'initial_batch_loaded') and not self.initial_batch_loaded:
                # 初始加载阶段：显示加载进度
                display_count = min(200, total_images)
                self.logger.debug(f"统计显示：初始加载阶段 {display_count}/{total_images}")
            else:
                # 扫描完成后：始终显示完整统计，不显示滑动窗口信息
                display_count = None
                self.logger.debug(f"统计显示：完整列表 {total_images}")
            
            self.statistics_panel.update_statistics(
                total=total_images,
                classified=classified_count, 
                removed=removed_count,
                display_count=display_count
            )
        except Exception as e:
            self.logger.error(f"更新统计信息失败: {e}")
    
    def _update_ui_state_internal(self):
        """内部UI状态更新方法"""
        # 更新窗口标题
        if self.current_dir and self.image_files:
            title = f"{RELEASE_NAME} v{self.version} - {self.current_dir.name} ({self.current_index + 1}/{len(self.image_files)})"
            self.setWindowTitle(title)
    
    def _update_category_counts_internal(self):
        """内部类别计数更新方法"""
        # 重新计算类别计数
        self.category_counts.clear()

        for img_path, category in self.classified_images.items():
            # 处理多分类情况
            if isinstance(category, list):
                for cat in category:
                    if cat in self.categories:
                        self.category_counts[cat] = self.category_counts.get(cat, 0) + 1
            elif category in self.categories:
                self.category_counts[category] = self.category_counts.get(category, 0) + 1

        # Bug修复：如果按数量排序，需要重新排序并重建按钮
        sort_mode = getattr(self.config, 'category_sort_mode', 'name')
        if sort_mode == 'count' and hasattr(self, 'category_panel'):
            # 按数量排序模式：重新排序类别列表
            self.ordered_categories = self.config.get_sorted_categories(
                self.categories, category_counts=self.category_counts
            )
            # 重建按钮（会自动使用新的排序顺序和计数）
            self.update_category_buttons()
        elif hasattr(self, 'category_panel'):
            # 其他排序模式：只更新计数标签
            self.category_panel.update_counts(self.category_counts)
    
    # ===== 状态管理方法 =====
    
    def load_state(self):
        """加载分类状态 - 从图片同级目录"""
        try:
            if not self.current_dir:
                return

            self._ai_project_state = default_ai_project_state()

            # 从图片目录的父目录加载状态文件
            parent_dir = self.current_dir.parent
            state_file = parent_dir / 'classification_state.json'

            if state_file.exists():
                with open(state_file, 'r', encoding='utf-8') as f:
                    state = json.load(f)

                self.classified_images = state.get('classified_images', {})
                self.removed_images = set(state.get('removed_images', []))
                self._ai_project_state = normalize_ai_project_state(
                    state.get(AI_PROJECT_STATE_KEY)
                )

                # 恢复上次的图片索引
                self.saved_last_index = state.get('last_index', -1)
                self.logger.debug(f"加载的 last_index: {self.saved_last_index}")

                # 恢复操作模式状态
                saved_copy_mode = state.get('is_copy_mode', True)  # 默认为复制模式
                self._set_mode_direct(saved_copy_mode)  # 直接设置模式，不触发迁移逻辑

                # 恢复分类模式状态
                saved_multi_category = state.get('is_multi_category', False)  # 默认为单分类模式
                self.is_multi_category = saved_multi_category
                # 延迟更新按钮状态，确保UI组件已完全初始化
                QTimer.singleShot(10, lambda: self._update_category_mode_button_state())

                modes = []
                if saved_copy_mode:
                    modes.append("复制")
                else:
                    modes.append("移动")

                if saved_multi_category:
                    modes.append("多分类")
                else:
                    modes.append("单分类")

                self.logger.info(f"状态加载完成: {len(self.classified_images)} 个分类记录，操作模式: {' + '.join(modes)}")
                self.logger.debug(f"状态文件路径: {state_file}")
            else:
                self.logger.info("状态文件不存在，使用空状态")
                self.saved_last_index = -1
                # 确保按钮状态正确（默认单分类模式）
                self.is_multi_category = False
                QTimer.singleShot(10, lambda: self._update_category_mode_button_state())

            self._restore_ai_for_current_project()

        except Exception as e:
            self.logger.error(f"加载状态失败: {e}")
            self.saved_last_index = -1
    
    def _update_category_mode_button_state(self):
        """更新分类模式按钮状态 - 统一方法"""
        self.update_category_mode_button()

    def update_category_mode_button(self):
        """统一的分类模式按钮状态更新方法"""
        try:
            if hasattr(self, 'category_mode_button') and self.category_mode_button:
                # 更新按钮图标和提示
                if self.is_multi_category:
                    self.category_mode_button.setText('⇶')  # 多箭头表示多分类
                    self.category_mode_button.setToolTip('多分类模式 - 点击切换到单分类模式')
                else:
                    self.category_mode_button.setText('→')  # 单箭头表示单分类
                    self.category_mode_button.setToolTip('单分类模式 - 点击切换到多分类模式')

                mode_desc = "多分类" if self.is_multi_category else "单分类"
                self.logger.debug(f"分类模式按钮状态已更新: {mode_desc}")
            else:
                self.logger.warning("分类模式按钮不存在，无法更新状态")
        except Exception as e:
            self.logger.error(f"更新分类模式按钮状态失败: {e}")
    
    def init_category_counts(self):
        """初始化类别计数"""
        self.category_counts.clear()

        # 从 classified_images 获取分类记录计数
        for img_path, category in self.classified_images.items():
            # 处理多分类情况
            if isinstance(category, list):
                for cat in category:
                    if cat in self.categories:
                        self.category_counts[cat] = self.category_counts.get(cat, 0) + 1
            elif category in self.categories:
                self.category_counts[category] = self.category_counts.get(category, 0) + 1

        # Bug修复：如果按数量排序，需要重新排序类别列表
        sort_mode = getattr(self.config, 'category_sort_mode', 'name')
        if sort_mode == 'count':
            self.ordered_categories = self.config.get_sorted_categories(
                self.categories, category_counts=self.category_counts
            )

        # 更新按钮显示
        self.update_category_buttons()
    
    def update_category_counts(self):
        """更新类别计数"""
        self.schedule_ui_update('category_counts')
    
    def update_category_buttons(self):
        """更新类别按钮"""
        self.schedule_ui_update('category_buttons')
    
    # ===== 图片显示和导航方法 =====

    def get_real_file_path(self, original_path: str) -> Path:
        """
        根据操作模式和分类状态计算文件的实际路径

        Args:
            original_path: JSON中存储的原始路径

        Returns:
            文件的实际路径
        """
        original_file = Path(original_path)

        # 复制模式：文件始终在原目录（包括多分类模式）
        if self.is_copy_mode:
            return original_file

        # 移动模式：检查分类状态
        # 已移除：文件在 remove 目录
        if original_path in self.removed_images:
            remove_dir = self.current_dir.parent / 'remove'
            return remove_dir / original_file.name

        # 已分类：文件在分类目录
        if original_path in self.classified_images:
            category = self.classified_images[original_path]
            category_dir = self.current_dir.parent / category
            return category_dir / original_file.name

        # 未分类：文件在原目录
        return original_file

    @performance_monitor
    def show_current_image(self):
        """显示当前图片（委托给ImageNavigationManager）"""
        # 先设置当前请求的图片路径，用于回调时判断
        if 0 <= self.current_index < len(self.image_files):
            img_path = str(self.image_files[self.current_index])
            real_path = str(self.get_real_file_path(img_path))
            self._current_requested_image = str(Path(real_path).resolve())

        if self._nav_manager:
            self._nav_manager.show_current_image()
        else:
            # 降级：直接调用内部方法
            self.logger.debug("[降级模式] ImageNavigationManager未初始化，使用内部方法显示图片")
            self._show_current_image_internal()
        
    def _show_current_image_internal(self):
        """内部显示当前图片方法 - 优化防止多图刷新"""
        if 0 <= self.current_index < len(self.image_files):
            img_path = str(self.image_files[self.current_index])

            # 计算文件的实际路径（根据操作模式和分类状态）
            real_path = str(self.get_real_file_path(img_path))

            # 记录图片文件信息（使用真实路径）
            self.log_image_info(real_path)

            # 立即更新窗口标题和状态信息
            self.update_window_title(img_path)

            # 设置当前图片索引用于智能缓存
            self.image_loader.set_current_image_index(self.current_index)

            # 检查缓存命中情况（使用真实路径）
            cache_key = self.image_loader._get_cache_key(real_path)
            is_cached = cache_key in self.image_loader.cache

            self.log_performance_info(
                "显示图片_开始",
                文件=Path(img_path).name,
                索引=f"{self.current_index + 1}/{len(self.image_files)}",
                缓存命中=is_cached
            )

            # 检查当前图片的分类状态并更新类别选择
            self.update_category_selection_for_current_image(img_path)

            # 如果缓存命中，直接显示；否则显示占位符
            if is_cached:
                # 直接从缓存加载，避免闪烁
                cached_data = self.image_loader._get_from_cache(cache_key)
                # 安全检查缓存数据类型
                if cached_data is not None:
                    # 检查是否为QPixmap类型
                    if isinstance(cached_data, QPixmap) and not cached_data.isNull():
                        self.image_label.set_image(cached_data)
                        self.statusBar.showMessage(f"📷 {Path(img_path).name}")
                    else:
                        # 缓存中可能是numpy数组或其他格式，转换为QPixmap后显示
                        cached_pixmap = self.convert_to_pixmap(cached_data)
                        if cached_pixmap is not None and not cached_pixmap.isNull():
                            self.image_label.set_image(cached_pixmap)
                            self.statusBar.showMessage(f"📷 {Path(img_path).name}")
                        else:
                            # 转换失败，显示占位符等待异步加载
                            self.show_loading_placeholder(img_path)
                else:
                    self.show_loading_placeholder(img_path)
            else:
                self.show_loading_placeholder(img_path)

            # 记录当前请求的图片路径（用于回调时判断）
            self._current_requested_image = str(Path(real_path).resolve())

            # 异步加载完整图片（使用真实路径）
            self.image_loader.load_image(real_path, priority=True)
            
            # 延迟预加载相邻图片，避免影响当前图片显示
            is_network_current = self._is_network_path(img_path)
            delay_time = 300 if is_network_current else 100
            
            QTimer.singleShot(delay_time, self.preload_adjacent_images)
            
            # 延迟更新图片列表高亮选中状态，避免阻塞当前图片显示
            QTimer.singleShot(50, self.sync_image_list_selection)
            
            # 调度UI状态更新
            self.schedule_ui_update('ui_state')
    
    def update_category_selection_for_current_image(self, image_path):
        """根据当前图片的分类状态更新类别选择"""
        try:
            # 获取当前图片的分类状态
            current_category = self.classified_images.get(image_path)

            # 使用 CategoryPanel 更新分类状态显示
            if hasattr(self, 'category_panel'):
                self.category_panel.update_classification_state(
                    current_category,
                    self.last_operation_category
                )
                # 同步 current_category_index
                self.current_category_index = self.category_panel._current_category_index

            # 记录日志
            if isinstance(current_category, list) and current_category:
                self.logger.debug(f"图片 {Path(image_path).name} 已分类到多个类别: {current_category}")
            elif current_category:
                self.logger.debug(f"图片 {Path(image_path).name} 已分类到: {current_category}")
            else:
                self.logger.debug(f"图片 {Path(image_path).name} 未分类")

        except Exception as e:
            self.logger.error(f"更新图片类别选择状态失败: {e}")
      
    def update_window_title(self, image_path):
        """更新窗口标题 - 顶部显示目录名+进度"""
        try:
            if self.current_dir and self.image_files:
                # 窗口标题：目录名 + 进度
                dir_name = self.current_dir.name
                progress = f"({self.current_index + 1}/{len(self.image_files)})"
                title = f"{RELEASE_NAME} v{self.version} - {dir_name} {progress}"
                self.setWindowTitle(title)
                
                # 状态栏：图片名称
                filename = Path(image_path).name
                self.statusBar.showMessage(f"📷 {filename}")
                
        except Exception as e:
            self.logger.debug(f"更新窗口标题失败: {e}")
    
    def show_loading_placeholder(self, image_path=None):
        """显示加载占位符"""
        try:
            # 创建简单的加载占位符
            placeholder_size = QSize(400, 300)
            placeholder = QPixmap(placeholder_size)
            placeholder.fill(QColor(240, 240, 240))

            # 立即显示占位符
            self.image_label.set_image(placeholder)

            # 更新状态栏
            if image_path:
                self.statusBar.showMessage(f"🔄 正在加载: {Path(image_path).name}")
            else:
                self.statusBar.showMessage("🔄 正在加载图片...")

        except Exception as e:
            self.logger.debug(f"显示加载占位符失败: {e}")

    def sync_image_list_selection(self):
        """同步图片列表的选中状态（委托给ImageNavigationManager）"""
        if self._nav_manager:
            self._nav_manager.sync_image_list_selection()

    def preload_adjacent_images(self):
        """预加载相邻图片（委托给ImageNavigationManager）"""
        if self._nav_manager:
            self._nav_manager.preload_adjacent_images()

    def log_image_info(self, image_path, pixmap=None):
        """记录图片详细信息"""
        if not self.enable_performance_logging:
            return
            
        try:
            file_path = Path(image_path)
            if file_path.exists():
                file_size = file_path.stat().st_size
                file_size_mb = file_size / 1024 / 1024
                
                # 获取图片尺寸
                width, height = 0, 0
                if pixmap and not pixmap.isNull():
                    width, height = pixmap.width(), pixmap.height()
                    resolution_mp = (width * height) / 1000000
                    
                    self.performance_stats['current_image_info'] = {
                        'path': str(file_path),
                        'size_mb': file_size_mb,
                        'width': width,
                        'height': height,
                        'resolution_mp': resolution_mp
                    }
                    
                    self.log_performance_info(
                        "图片信息",
                        文件=file_path.name,
                        大小=f"{file_size_mb:.2f}MB",
                        分辨率=f"{width}x{height}",
                        像素=f"{resolution_mp:.1f}MP"
                    )
                else:
                    self.log_performance_info(
                        "图片文件",
                        文件=file_path.name,
                        大小=f"{file_size_mb:.2f}MB"
                    )
        except Exception as e:
            self.logger.debug(f"图片信息记录失败: {e}")
    
    # ===== 导航方法 =====

    def _should_enable_loop(self) -> bool:
        """判断当前是否应该启用循环翻页

        Returns:
            bool: True表示启用循环，False表示不启用
        """
        try:
            # 动态刷新配置，确保开关即时生效（避免重启才能应用）
            if hasattr(self, 'app_config'):
                self.app_config.reload_config()
        except Exception as e:
            self.logger.debug(f"[循环翻页] 刷新配置失败: {e}")

        # 检查是否为网络路径
        is_network = hasattr(self, 'is_network_working_path') and self.is_network_working_path

        if is_network:
            # 网络路径：使用网络循环开关
            loop_enabled = self.app_config.network_loop_enabled
            self.logger.debug(f"[循环翻页] 网络路径，循环开关：{'开启' if loop_enabled else '关闭'}")
            return loop_enabled
        else:
            # 本地路径：使用本地循环开关
            loop_enabled = self.app_config.local_loop_enabled
            self.logger.debug(f"[循环翻页] 本地路径，循环开关：{'开启' if loop_enabled else '关闭'}")
            return loop_enabled

    def prev_image(self):
        """上一张图片（委托给ImageNavigationManager）"""
        if not self._prepare_ai_navigation():
            return
        if self._nav_manager:
            self._nav_manager.prev_image()

    def _record_direction(self, direction):
        """记录翻页方向（优化3：智能预加载）"""
        if 'direction_history' not in self.user_behavior:
            self.user_behavior['direction_history'] = []
        self.user_behavior['direction_history'].append(direction)
        if len(self.user_behavior['direction_history']) > 10:
            self.user_behavior['direction_history'] = self.user_behavior['direction_history'][-10:]

    def next_image(self):
        """下一张图片（委托给ImageNavigationManager）"""
        if not self._prepare_ai_navigation():
            return
        if self._nav_manager:
            self._nav_manager.next_image()

    def prev_category(self):
        """选择上一个类别 - 委托给CategoryPanel"""
        if hasattr(self, 'category_panel'):
            self.category_panel.prev_category()

    def next_category(self):
        """选择下一个类别 - 委托给CategoryPanel"""
        if hasattr(self, 'category_panel'):
            self.category_panel.next_category()

    def confirm_category(self):
        """确认当前选中的类别 - 委托给CategoryPanel"""
        if hasattr(self, 'category_panel'):
            self.category_panel.confirm_category()

    def refresh_categories(self):
        """刷新类别目录（F5快捷键 / 工具栏按钮回调）"""
        self.refresh_and_sync()

    # ===== 事件处理方法 =====
    
    def on_image_list_item_clicked(self, index):
        """处理图片列表项点击

        Args:
            index: QModelIndex, 点击的列表项索引
        """
        if not self._prepare_ai_navigation():
            return
        # Phase 1.1 Migration: 适配 QListView 的 clicked 信号 (传递 QModelIndex)
        if index.isValid():
            # 从自定义数据角色获取原始图片索引
            img_idx = index.data(ImageListModel.ROLE_IMAGE_INDEX)

            if img_idx is not None:
                self.current_index = img_idx
                self.show_current_image()

    def create_filter_status_icon(self, status_type, is_checked=False):
        """创建筛选菜单的状态图标（带checkbox）

        Args:
            status_type: 状态类型，'unclassified', 'classified', 'removed'
            is_checked: 是否选中

        Returns:
            QIcon: 状态图标
        """
        # 创建72x40的图标（左侧checkbox + 右侧状态图标）

        pixmap = QPixmap(72, 40)
        pixmap.fill(Qt.GlobalColor.transparent)

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # 绘制checkbox（左侧，增大尺寸）
        checkbox_x = 4
        checkbox_y = 6
        checkbox_size = 28

        # Checkbox背景
        if is_checked:
            painter.setPen(QPen(QColor("#0D6EFD"), 2))
            painter.setBrush(QBrush(QColor("#0D6EFD"), Qt.BrushStyle.SolidPattern))
        else:
            painter.setPen(QPen(QColor("#ADB5BD"), 2))
            painter.setBrush(QBrush(QColor("#FFFFFF"), Qt.BrushStyle.SolidPattern))

        painter.drawRoundedRect(checkbox_x, checkbox_y, checkbox_size, checkbox_size, 5, 5)

        # 绘制勾选标记（调整位置和大小）
        if is_checked:
            painter.setPen(QPen(Qt.GlobalColor.white, 3))
            painter.drawLine(checkbox_x + 7, checkbox_y + 14, checkbox_x + 11, checkbox_y + 19)
            painter.drawLine(checkbox_x + 11, checkbox_y + 19, checkbox_x + 21, checkbox_y + 8)

        # 绘制状态图标（右侧，偏移36像素）
        offset_x = 36

        # 根据状态设置颜色和图标
        if status_type == 'classified':
            # 已分类 - 绿色勾选图标
            color = QColor("#4CAF50")
            shadow_color = QColor("#2E7D32")
        elif status_type == 'removed':
            # 已移除 - 红色删除图标
            color = QColor("#F44336")
            shadow_color = QColor("#C62828")
        else:  # unclassified
            # 待处理 - 橙色警告图标
            color = QColor("#FF9800")
            shadow_color = QColor("#F57C00")

        # 绘制阴影效果（垂直居中）
        painter.setPen(QPen(shadow_color, 2))
        painter.setBrush(QBrush(shadow_color, Qt.BrushStyle.SolidPattern))
        painter.drawEllipse(offset_x + 3, 7, 28, 28)

        # 绘制主圆形（垂直居中）
        painter.setPen(QPen(color, 2))
        painter.setBrush(QBrush(color, Qt.BrushStyle.SolidPattern))
        painter.drawEllipse(offset_x + 2, 6, 28, 28)

        # 绘制状态符号（调整y坐标以居中）
        painter.setPen(QPen(Qt.GlobalColor.white, 3))
        if status_type == 'classified':
            # 绘制√ - 更优雅的勾选
            painter.drawLine(offset_x + 8, 20, offset_x + 14, 26)
            painter.drawLine(offset_x + 14, 26, offset_x + 26, 14)
        elif status_type == 'removed':
            # 绘制× - 删除符号
            painter.drawLine(offset_x + 10, 14, offset_x + 24, 28)
            painter.drawLine(offset_x + 24, 14, offset_x + 10, 28)
        else:  # unclassified
            # 绘制! - 待处理警告
            painter.setPen(QPen(Qt.GlobalColor.white, 2))
            painter.drawLine(offset_x + 16, 12, offset_x + 16, 22)  # 竖线
            painter.drawEllipse(offset_x + 14, 26, 4, 4)  # 点

        painter.end()
        return QIcon(pixmap)

    def create_checkbox_icon(self, checked=False):
        """创建复选框图标

        Args:
            checked: 是否选中状态
        """

        # 创建18x18的图标
        pixmap = QPixmap(18, 18)
        pixmap.fill(Qt.GlobalColor.transparent)

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        if checked:
            # 选中状态 - 蓝色背景 + 白色打勾
            painter.setPen(QPen(QColor(default_theme.colors.PRIMARY_DARK), 2))
            painter.setBrush(QBrush(QColor(default_theme.colors.PRIMARY)))
            painter.drawRoundedRect(1, 1, 16, 16, 3, 3)

            # 绘制白色打勾符号
            painter.setPen(QPen(Qt.GlobalColor.white, 2.5, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
            painter.drawLine(4, 9, 7, 13)
            painter.drawLine(7, 13, 14, 5)
        else:
            # 未选中状态 - 灰色边框 + 空白背景
            painter.setPen(QPen(QColor(default_theme.colors.BORDER_MEDIUM), 2))
            painter.setBrush(QBrush(QColor(default_theme.colors.BACKGROUND_CARD)))
            painter.drawRoundedRect(1, 1, 16, 16, 3, 3)

        painter.end()
        return QIcon(pixmap)

    def _on_image_search(self, text: str):
        """处理图片列表搜索

        Args:
            text: 搜索关键字
        """
        if not self._prepare_ai_navigation():
            return
        self._image_search_text = text.strip()
        self.apply_image_filter()

        # 搜索后自动选中第一个匹配项
        if self.image_list_model.rowCount() > 0:
            first_index = self.image_list_model.index(0, 0)
            self.image_list.setCurrentIndex(first_index)
            # 获取原始索引并跳转到该图片
            original_idx = first_index.data(ImageListModel.ROLE_IMAGE_INDEX)
            if original_idx is not None:
                self.current_index = original_idx
                self.show_current_image()

        # 显示搜索结果提示
        total_count = len(self.image_files) if hasattr(self, 'image_files') else 0
        match_count = self.image_list_model.rowCount()
        toast_info(self, f"搜索 \"{text}\"：找到 {match_count}/{total_count} 个匹配项")

    def _on_image_search_cleared(self):
        """处理搜索清除/关闭"""
        if not self._prepare_ai_navigation():
            return
        if self._image_search_text:  # 仅在有搜索词时才需要刷新
            self._image_search_text = ""
            self.apply_image_filter()

    def show_filter_menu(self):
        """显示筛选菜单"""

        menu = QMenu(self)
        menu.setStyleSheet(WidgetStyles.get_context_menu_style())

        # 三个可勾选的菜单项 - 使用自定义复选框图标
        action_unclassified = QAction(self.create_checkbox_icon(self.filter_unclassified), "⚠️ 显示未分类图片", self)
        action_unclassified.triggered.connect(lambda: self.toggle_filter('unclassified'))
        menu.addAction(action_unclassified)

        action_classified = QAction(self.create_checkbox_icon(self.filter_classified), "✅ 显示已分类图片", self)
        action_classified.triggered.connect(lambda: self.toggle_filter('classified'))
        menu.addAction(action_classified)

        action_removed = QAction(self.create_checkbox_icon(self.filter_removed), "❌ 显示已移除图片", self)
        action_removed.triggered.connect(lambda: self.toggle_filter('removed'))
        menu.addAction(action_removed)

        menu.addSeparator()

        # 统计信息（不可点击）
        if hasattr(self, 'image_files') and self.image_files:
            stats = self.get_filter_stats()
            stats_text = f"未分类: {stats['unclassified']} | 已分类: {stats['classified']} | 已移除: {stats['removed']}"
            action_stats = QAction(stats_text, self)
            action_stats.setEnabled(False)  # 不可点击
            menu.addAction(action_stats)

        # 在筛选按钮下方显示菜单 - 智能定位防止超出窗口
        # 先让菜单调整大小以获得准确的尺寸
        menu.adjustSize()

        # 获取按钮的右下角位置（改为右对齐）
        button_global_rect = self.filter_button.mapToGlobal(self.filter_button.rect().bottomRight())
        menu_size = menu.sizeHint()

        # 获取主窗口的实际可见区域
        window_rect = self.rect()
        window_global_pos = self.mapToGlobal(window_rect.topLeft())
        window_bottom = window_global_pos.y() + window_rect.height()

        # 初始位置：按钮右下角，菜单右对齐
        x = button_global_rect.x() - menu_size.width()
        y = button_global_rect.y()

        # 确保菜单不超出窗口左边界
        if x < window_global_pos.x():
            x = window_global_pos.x() + 5

        # 如果菜单超出窗口底部，显示在按钮上方
        if y + menu_size.height() > window_bottom - 10:  # 留10px底部边距
            y = self.filter_button.mapToGlobal(self.filter_button.rect().topRight()).y() - menu_size.height()

        menu.exec(QPoint(x, y))

    def toggle_filter(self, filter_type):
        """切换过滤器状态"""
        # 切换状态
        if filter_type == 'unclassified':
            self.filter_unclassified = not self.filter_unclassified
        elif filter_type == 'classified':
            self.filter_classified = not self.filter_classified
        elif filter_type == 'removed':
            self.filter_removed = not self.filter_removed

        # 至少保留一个选项
        if not (self.filter_unclassified or self.filter_classified or self.filter_removed):
            toast_warning(self, "至少需要选择一种图片类型")
            # 恢复刚才的选择
            if filter_type == 'unclassified':
                self.filter_unclassified = True
            elif filter_type == 'classified':
                self.filter_classified = True
            elif filter_type == 'removed':
                self.filter_removed = True
            return

        if filter_type == 'classified' and not self.filter_classified and self.focused_category:
            # “只看某类别”天然属于已分类图片。用户主动隐藏已分类时，
            # 同步退出类别聚焦，避免产生互相矛盾的筛选条件。
            self._deactivate_category_focus(restore_classified_filter=False)

        # 应用过滤
        self.apply_image_filter()

        # 更新按钮状态（有过滤时高亮）
        is_filtering = not (self.filter_unclassified and self.filter_classified and self.filter_removed)
        self.filter_button.setProperty("active", is_filtering)
        self.filter_button.style().unpolish(self.filter_button)
        self.filter_button.style().polish(self.filter_button)

    def _apply_category_focus_button_style(self):
        """应用图片列表标题栏中的类别聚焦标签样式。"""
        button = getattr(self, 'category_focus_button', None)
        if button is None:
            return
        c = default_theme.colors
        button.setStyleSheet(f"""
            QPushButton#category_focus_button {{
                color: {c.TEXT_PRIMARY};
                background-color: {c.PRIMARY_LIGHT};
                border: 1px solid {c.PRIMARY};
                border-radius: 5px;
                padding: 3px 8px;
                font-size: 12px;
                font-weight: 600;
            }}
            QPushButton#category_focus_button:hover {{
                background-color: {c.BACKGROUND_HOVER};
                border-color: {c.PRIMARY_DARK};
            }}
        """)

    def _update_category_focus_button(self):
        """同步类别聚焦标签的文本和可见性。"""
        button = getattr(self, 'category_focus_button', None)
        if button is None:
            return

        category_name = getattr(self, 'focused_category', None)
        if not category_name:
            button.hide()
            button.setText("")
            button.setToolTip("")
            return

        elided_name = button.fontMetrics().elidedText(
            category_name,
            Qt.TextElideMode.ElideRight,
            100,
        )
        button.setText(f"🎯 {elided_name}  ×")
        button.setToolTip(f"当前只显示类别“{category_name}”的图片，点击清除聚焦")
        button.show()

    def set_category_focus(self, category_name: str):
        """让图片列表只显示一个指定类别，新的选择会替换旧选择。"""
        category_name = str(category_name).strip()
        if not category_name or category_name not in self.categories:
            toast_warning(self, f"类别“{category_name}”不存在或已被忽略")
            return

        if not self.focused_category:
            self._category_focus_previous_classified_filter = self.filter_classified
        self.focused_category = category_name
        self.filter_classified = True
        self._update_category_focus_button()
        self.apply_image_filter()

        match_count = self.image_list_model.rowCount()
        if match_count:
            toast_info(self, f"已聚焦“{category_name}”：显示 {match_count} 张图片")

    def clear_category_focus(self):
        """清除类别聚焦，保留文件名搜索和图片状态筛选。"""
        category_name = getattr(self, 'focused_category', None)
        if not category_name:
            return
        self._deactivate_category_focus()
        self.apply_image_filter()
        toast_info(self, f"已取消类别“{category_name}”的聚焦")

    def _deactivate_category_focus(self, restore_classified_filter: bool = True):
        """退出类别聚焦，并按需恢复进入聚焦前的已分类筛选状态。"""
        previous_filter = getattr(
            self,
            '_category_focus_previous_classified_filter',
            None,
        )
        self.focused_category = None
        self._category_focus_previous_classified_filter = None
        if restore_classified_filter and previous_filter is not None:
            self.filter_classified = previous_filter
        self._update_category_focus_button()

    def _matches_category_focus(self, classification) -> bool:
        """判断一条分类记录是否符合当前的单类别聚焦条件。"""
        focused_category = getattr(self, 'focused_category', None)
        return ImageClassifier._classification_matches_focus(
            classification,
            focused_category,
        )

    @staticmethod
    def _classification_matches_focus(classification, focused_category) -> bool:
        """纯函数形式的类别匹配，供轻量 UIHooks 和测试替身复用。"""
        if not focused_category:
            return True
        if isinstance(classification, (list, tuple, set)):
            return focused_category in classification
        return classification == focused_category

    def apply_image_filter(self, suppress_show: bool = False):
        """
        应用过滤器到图片列表（支持抑制显示，避免重复刷新）

        Args:
            suppress_show: 是否抑制 show_current_image 调用（默认 False）
        """
        # Codex Review修复：空数据时也要更新Model，避免保留旧数据
        if not hasattr(self, 'image_files') or not self.image_files:
            # 清空Model
            if hasattr(self, 'image_list_model'):
                self.image_list_model.update_data([], {}, set(), set())
                self._visible_indices = []
                self._original_to_filtered_index = {}
            # 清空图像显示
            self.current_index = -1
            self.image_label.clear()
            return

        # 保存当前选中的图片路径（优先使用状态索引，确保导航后正确定位）
        current_path = None
        if self.image_files and 0 <= self.current_index < len(self.image_files):
            # 优先使用 current_index（导航后的新索引）
            current_path = str(self.image_files[self.current_index])
        else:
            # 回退到列表选中项
            current_index = self.image_list.currentIndex()
            if current_index.isValid():
                current_path = current_index.data(ImageListModel.ROLE_FULL_PATH)

        # Bug修复：只考虑类别仍然有效的分类记录
        # 当类别被忽略后，对应的分类记录不应该显示为"已分类"状态
        valid_classified_images = self._get_valid_classified_images()

        # Phase 1.1: 计算多分类集合（只考虑有效分类）
        multi_classified = {
            p for p, c in valid_classified_images.items()
            if isinstance(c, list) and len(c) > 1
        }

        # 过滤图片列表并测量最大文本宽度（Codex+Gemini方案：供横向滚动宽度计算）
        filtered_files = []
        original_indices = []  # 保存原始索引映射
        font_metrics = self.image_list.fontMetrics()
        max_text_width = 0

        # 搜索关键字预处理（小写，用于大小写不敏感匹配）
        search_keyword = self._image_search_text.lower() if self._image_search_text else ""

        for idx, img_path in enumerate(self.image_files):
            img_path_str = str(img_path)
            file_name = Path(img_path_str).name

            # 判断状态（使用有效分类记录）
            is_removed = img_path_str in self.removed_images
            is_classified = img_path_str in valid_classified_images
            is_unclassified = not is_removed and not is_classified

            # 检查状态过滤
            status_show = (
                (is_unclassified and self.filter_unclassified) or
                (is_classified and self.filter_classified) or
                (is_removed and self.filter_removed)
            )

            # 检查搜索关键字匹配（大小写不敏感）
            search_match = (not search_keyword) or (search_keyword in file_name.lower())

            # 类别聚焦与状态筛选、文件名搜索取交集；多分类图片只要
            # 包含目标类别就会显示。
            category_match = ImageClassifier._classification_matches_focus(
                valid_classified_images.get(img_path_str),
                getattr(self, 'focused_category', None),
            )

            # 同时满足状态过滤、搜索和类别聚焦
            should_show = status_show and search_match and category_match

            if should_show:
                filtered_files.append(img_path)
                original_indices.append(idx)
                # Codex方案：记录最长文件名宽度（单次遍历，24k规模可接受）
                file_name = Path(img_path_str).name
                max_text_width = max(max_text_width, font_metrics.horizontalAdvance(file_name))

        # 更新Model数据（使用有效分类记录，保留缩略图缓存，传递原始索引）
        self.image_list_model.update_data(
            filtered_files,
            valid_classified_images,  # 使用有效分类记录
            self.removed_images,
            multi_classified,  # 使用局部计算的多分类集合
            original_indices  # 传递原始索引，确保ROLE_IMAGE_INDEX返回正确值
        )

        # Codex+Gemini方案：根据最长文件名设置列表项宽度，保证横向滚动显示完整
        self._update_image_list_grid_width(max_text_width)

        if self.is_image_filter_active():
            # original_index -> filtered_row
            self._original_to_filtered_index = {
                original: row
                for row, original in enumerate(original_indices)
            }
            # 有筛选时保存可见索引，供 next/prev 跳过隐藏图片。
            self._visible_indices = original_indices
        else:
            # 全部显示并且未搜索时必须保持 None，导航直接使用原始索引，
            # 避免万级列表每次翻页都执行 list.index()。
            self._original_to_filtered_index = None
            self._visible_indices = None

        # 恢复选中
        # Codex优化：使用Model的_path_map进行O(1)查找而不是线性扫描
        # Bug修复-P0：处理当前图片被过滤掉的情况（row=None）
        if current_path:
            # O(1)查找路径对应的行
            row = self.image_list_model._path_map.get(str(current_path))
            if row is not None and 0 <= row < self.image_list_model.rowCount():
                # 当前图片仍在过滤列表中，恢复选中
                model_index = self.image_list_model.index(row, 0)
                self.image_list.setCurrentIndex(model_index)
                # 更新current_index为原始索引（从Model获取）
                self.current_index = model_index.data(ImageListModel.ROLE_IMAGE_INDEX)
            elif self.image_list_model.rowCount() > 0:
                # 当前图片被过滤掉（row=None），回退到第一张可见图片
                # Gemini建议：自动跳转比清空预览更符合用户工作流
                model_index = self.image_list_model.index(0, 0)
                self.image_list.setCurrentIndex(model_index)
                self.current_index = model_index.data(ImageListModel.ROLE_IMAGE_INDEX)
        elif self.image_list_model.rowCount() > 0:
            # 没有当前路径（首次加载），默认选中第一项
            model_index = self.image_list_model.index(0, 0)
            self.image_list.setCurrentIndex(model_index)
            # 更新current_index为原始索引（从Model获取）
            self.current_index = model_index.data(ImageListModel.ROLE_IMAGE_INDEX)

        # 更新图片显示
        if self.image_list_model.rowCount() > 0:
            if not suppress_show:  # 抑制时不显示
                self.show_current_image()
        else:
            self.current_index = -1
            self.image_label.clear()
            if not suppress_show:  # 抑制时不显示toast
                toast_info(self, "当前过滤条件下没有图片")

    def is_image_filter_active(self) -> bool:
        """返回当前筛选或搜索是否会改变图片列表内容。"""
        all_statuses_visible = (
            self.filter_unclassified
            and self.filter_classified
            and self.filter_removed
        )
        search_active = bool(getattr(self, '_image_search_text', '').strip())
        category_focus_active = bool(getattr(self, 'focused_category', None))
        return not all_statuses_visible or search_active or category_focus_active

    def refresh_image_filter_path(self, path: str) -> str:
        """分类后只增量更新一个列表条目，不重建万级图片模型。"""
        if not hasattr(self, 'image_list_model') or not self.image_list_model:
            return "unchanged"

        path_str = str(path)
        original_index = None
        if (
            0 <= self.current_index < len(self.image_files)
            and str(self.image_files[self.current_index]) == path_str
        ):
            original_index = self.current_index
        else:
            existing_row = self.image_list_model._path_map.get(path_str)
            if existing_row is not None:
                existing_index = self.image_list_model.index(existing_row, 0)
                original_index = existing_index.data(ImageListModel.ROLE_IMAGE_INDEX)
            else:
                original_index = next(
                    (
                        index
                        for index, image_path in enumerate(self.image_files)
                        if str(image_path) == path_str
                    ),
                    None,
                )

        if original_index is None:
            return "unchanged"

        category = self.classified_images.get(path_str)
        if isinstance(category, list):
            valid_categories = [item for item in category if item in self.categories]
            is_classified = bool(valid_categories)
            is_multi = len(valid_categories) > 1
        else:
            is_classified = category in self.categories if category else False
            is_multi = False

        is_removed = path_str in self.removed_images
        is_unclassified = not is_removed and not is_classified
        search_keyword = getattr(self, '_image_search_text', '').strip().lower()
        search_match = (
            not search_keyword
            or search_keyword in Path(path_str).name.lower()
        )
        category_match = ImageClassifier._classification_matches_focus(
            category,
            getattr(self, 'focused_category', None),
        )
        should_be_visible = search_match and category_match and (
            (is_unclassified and self.filter_unclassified)
            or (is_classified and self.filter_classified)
            or (is_removed and self.filter_removed)
        )

        action = self.image_list_model.update_filtered_item(
            path_str,
            int(original_index),
            should_be_visible,
            is_classified,
            is_removed,
            is_multi,
        )

        if self.is_image_filter_active():
            self._visible_indices = [
                item.index for item in self.image_list_model._data
            ]
            self._original_to_filtered_index = {
                original: row
                for row, original in enumerate(self._visible_indices)
            }
        else:
            # “全部显示”应走真正的未过滤导航，避免每次 next_image
            # 在线性列表中查找当前索引。
            self._visible_indices = None
            self._original_to_filtered_index = None

        return action

    def get_filter_stats(self):
        """获取过滤统计信息"""
        stats = {
            'unclassified': 0,
            'classified': 0,
            'removed': 0
        }

        if not hasattr(self, 'image_files') or not self.image_files:
            return stats

        for img_path in self.image_files:
            img_path_str = str(img_path)
            if img_path_str in self.removed_images:
                stats['removed'] += 1
            elif img_path_str in self.classified_images:
                stats['classified'] += 1
            else:
                stats['unclassified'] += 1

        return stats

    def _update_image_list_grid_width(self, max_text_width: int):
        """
        Codex+Gemini方案：根据最长文件名计算列表项宽度，保证横向滚动显示完整

        Args:
            max_text_width: 最长文件名的像素宽度（由QFontMetrics.horizontalAdvance计算）
        """
        try:
            # 获取Delegate的布局常量（紧凑模式：无缩略图）
            delegate = self.image_list.itemDelegate()
            padding = getattr(delegate, 'PADDING', 6)
            icon_size = getattr(delegate, 'ICON_SIZE', 20)

            # 计算基础宽度：padding + 图标 + padding + 文本 + padding（紧凑模式）
            base_width = padding + icon_size + padding + padding
            item_width = base_width + max_text_width

            # Gemini建议：设置最大宽度上限（400-500px），防止超长文件名撑爆布局
            MAX_WIDTH = 500
            item_width = min(item_width, MAX_WIDTH)
            # 确保最小宽度
            item_width = max(item_width, 200)

            # 获取行高（紧凑模式：44px）
            if self.image_list_model.rowCount() > 0:
                row_height = self.image_list.sizeHintForRow(0)
            else:
                row_height = max(44, icon_size + 2 * padding)

            # 设置统一的GridSize，配合setUniformItemSizes(True)保持性能
            from PyQt6.QtCore import QSize
            self.image_list.setGridSize(QSize(item_width, row_height))
            self.image_list.updateGeometry()

        except Exception as e:
            self.logger.debug(f"更新列表项宽度失败: {e}")

    def _on_category_selected(self, category_name: str):
        """处理CategoryPanel的category_selected信号"""
        self.selected_category = category_name
        if category_name in self.ordered_categories:
            self.current_category_index = self.ordered_categories.index(category_name)

    def _on_category_operation_requested(self, operation: str, data: dict):
        """处理CategoryPanel的operation_requested信号

        Phase 2.5 扩展：支持 CategoryButton 发出的所有业务操作
        """
        if operation == 'add_category':
            # 显示添加类别对话框
            self.show_add_category_dialog()
        elif operation == 'rename_category':
            # 重命名类别
            old_name = data.get('old_name')
            new_name = data.get('new_name')
            if old_name and new_name:
                self.rename_category(old_name, new_name)
        elif operation == 'focus_category':
            category_name = data.get('category_name')
            if category_name:
                self.set_category_focus(category_name)
        elif operation == 'change_shortcut':
            # 修改快捷键
            category_name = data.get('category_name')
            if category_name:
                self._handle_shortcut_change(category_name)
        elif operation == 'ignore_category':
            # 忽略类别
            category_name = data.get('category_name')
            if category_name:
                self.ignore_category(category_name)
        elif operation == 'delete_category':
            # 删除类别
            category_name = data.get('category_name')
            if category_name:
                self.delete_category(category_name)
        elif operation == 'manage_ignored':
            # 打开管理忽略类别对话框
            self.show_manage_ignored_dialog()
        else:
            self.logger.warning(f"未知的类别操作: {operation}")

    def _handle_shortcut_change(self, category_name: str):
        """处理快捷键修改请求

        Phase 2.5: 从 CategoryButton 移动到主窗口的快捷键修改逻辑
        """
        try:
            dialog = CategoryShortcutDialog(self.config, category_name, self)
            if dialog.exec():
                # 保存配置
                self.config.save_config()
                self.logger.info(f"快捷键已修改并保存: {category_name}")

                # 重新设置快捷键
                self.setup_shortcuts()

                # 重新计算排序列表（特别是"按快捷键排序"模式）
                if self.categories is not None:
                    category_counts = None
                    if self.config.category_sort_mode == "count":
                        category_counts = self._get_category_counts()
                    self.ordered_categories = self.config.get_sorted_categories(
                        self.categories, category_counts=category_counts
                    )

                # 更新类别按钮列表（重新排序）
                self.update_category_buttons()
                self.logger.info("类别按钮列表已更新")

        except Exception as e:
            self.logger.error(f"修改快捷键失败: {e}")
            toast_error(self, f"修改快捷键失败: {e}")

    def move_to_category(self, category_name):
        """分类当前图片到指定类别（通过FileOperationManager）"""
        if self._ai_prediction_loading:
            self._notify_ai_navigation_locked()
            return
        if not self._file_ops_manager:
            self.logger.error("FileOperationManager未初始化")
            return
        current_path = self.get_current_image_path()
        if current_path:
            self._file_ops_manager.move_to_category(str(current_path), category_name)
            # Manager会触发file_moved信号，UI刷新在on_file_moved中处理

    def get_current_image_path(self):
        """获取当前图片路径"""
        if 0 <= self.current_index < len(self.image_files):
            return self.image_files[self.current_index]
        return None

    def _update_single_image_status(self, image_index, image_path):
        """更新单个图片在列表中的状态图标"""
        try:
            # Phase 1.1 Migration: 使用 Model API O(1) 更新状态
            # 避免遍历 View，提高交互响应速度

            if not hasattr(self, 'image_list_model'):
                return

            # 1. 计算新状态
            is_classified = image_path in self.classified_images
            is_removed = image_path in self.removed_images

            is_multi = False
            if is_classified:
                current_category = self.classified_images.get(image_path)
                is_multi = isinstance(current_category, list) and len(current_category) > 1

            # 2. 调用 Model 接口更新 (O(1) 查找 + 自动刷新 Delegate)
            self.image_list_model.update_status(image_index, is_classified, is_removed, is_multi)

        except Exception as e:
            self.logger.debug(f"更新单个图片状态失败: {e}")
    
    def move_to_remove(self):
        """移除当前图片（通过FileOperationManager）"""
        if self._ai_prediction_loading:
            self._notify_ai_navigation_locked()
            return
        if not self._file_ops_manager:
            self.logger.error("FileOperationManager未初始化")
            return
        current_path = self.get_current_image_path()
        if current_path:
            self._file_ops_manager.move_to_remove(str(current_path))
            # Manager会触发file_removed信号，UI刷新在on_file_removed中处理

    def _undo_classification(self, image_path, category):
        """撤销分类（通过FileOperationManager）"""
        if not self._file_ops_manager:
            self.logger.error("FileOperationManager未初始化")
            return
        self._file_ops_manager._undo_classification(image_path, category)
        # Manager会触发file_restored信号，UI刷新在on_file_restored中处理

    def refresh_and_sync(self):
        """刷新类别并同步文件状态"""
        if not self._category_manager:
            self.logger.error("CategoryManager未初始化")
            return
        if not self.current_dir:
            self.logger.warning("当前目录未设置，跳过刷新")
            return
        try:
            self.logger.info("开始刷新并同步目录状态...")

            # 先重新加载类别
            self._load_categories_only()

            # 同步文件状态
            sync_results = self._sync_file_states()

            # 强制重新计算类别计数
            self.init_category_counts()

            # 健康检查并修复快捷键（这是关键修复）
            QTimer.singleShot(100, self._check_and_fix_shortcuts)

            # 更新UI组件（category_buttons和category_counts由categories_changed信号触发，避免重复）
            self.schedule_ui_update('image_list', 'statistics')

            # Bug修复-P1：F5刷新后重新应用过滤状态（Codex方案：用标志位替代硬编码延迟）
            # 在perform_batch_ui_update完成image_list更新后自动调用apply_image_filter
            self._pending_reapply_filter = True

            # 显示同步结果（只有在有变化时才显示）
            if sync_results.get('changes_detected'):
                # 有变化时才持久化，减少无效写盘
                self.save_state()
                self._show_sync_results(sync_results)
                toast_success(self, "目录状态已刷新并同步")
            else:
                # 即使没有检测到变化，也要显示刷新完成的信息
                self.statusBar.showMessage("🔄 目录状态已刷新")
                toast_success(self, "目录状态已刷新")

            self.logger.info("目录状态刷新完成")

        except Exception as e:
            self.logger.error(f"刷新目录状态失败: {e}")
            toast_error(self, f"刷新失败: {str(e)}")
    
    def _check_and_fix_shortcuts(self):
        """检查并修复快捷键"""
        if self.shortcut_manager:
            self.shortcut_manager.check_and_fix_shortcuts()
    
    def _periodic_shortcut_check(self):
        """定期快捷键检查"""
        if self.shortcut_manager:
            self.shortcut_manager.periodic_shortcut_check()
    
    def _ensure_file_state_manager(self):
        """确保FileStateManager实例是最新的"""
        if (self.current_dir is None or
            self.file_state_manager is None or
            self.file_state_manager.current_dir != self.current_dir):

            if self.current_dir is not None:
                self.file_state_manager = FileStateManager(
                    current_dir=self.current_dir,
                    categories=list(self.categories),
                    classified_images=self.classified_images,
                    removed_images=self.removed_images
                )

    def _sync_file_states(self):
        """同步文件状态与实际目录"""
        # 确保FileStateManager实例是最新的
        self._ensure_file_state_manager()

        if self.file_state_manager is None:
            self.logger.warning("FileStateManager未初始化，跳过文件状态同步")
            return {
                'changes_detected': False,
                'removed_files': [],
                'moved_files': [],
                'new_classifications': [],
                'invalid_classifications': []
            }

        try:
            # 使用FileStateManager进行状态同步
            sync_results = self.file_state_manager.sync_file_states()

            # 如果有变化，保存状态
            if sync_results['changes_detected']:
                self.save_state()

            return sync_results

        except Exception as e:
            self.logger.error(f"同步文件状态失败: {e}")
            return {
                'changes_detected': False,
                'removed_files': [],
                'moved_files': [],
                'new_classifications': [],
                'invalid_classifications': []
            }
    
    def _show_sync_results(self, sync_results):
        """显示同步结果"""
        try:
            messages = []
            
            if sync_results['removed_files']:
                count = len(sync_results['removed_files'])
                messages.append(f"🗑️ 发现 {count} 个文件被删除或移动")
            
            if sync_results['moved_files']:
                count = len(sync_results['moved_files'])
                messages.append(f"📦 发现 {count} 个文件位置变化")
            
            if sync_results['new_classifications']:
                count = len(sync_results['new_classifications'])
                messages.append(f"✅ 发现 {count} 个新的分类文件")
            
            if sync_results['invalid_classifications']:
                count = len(sync_results['invalid_classifications'])
                messages.append(f"❌ 清理了 {count} 个无效分类记录")
            
            if messages:
                toast_success(self,"目录同步完成，分类状态已更新")        
        except Exception as e:
            self.logger.error(f"显示同步结果失败: {e}")
    
    def _is_migration_needed(self):
        """检查是否需要迁移"""
        return len(self.classified_images) > 0 or len(self.removed_images) > 0

    def _show_migration_confirmation_dialog(self, target_mode):
        """显示迁移确认对话框"""

        dialog = QDialog(self)
        dialog.setWindowTitle("模式切换迁移确认")
        dialog.setModal(True)
        dialog.setMinimumSize(520, 420)

        layout = QVBoxLayout(dialog)
        configure_dialog(dialog, layout)

        # 标题
        title_label = QLabel(f"⚠️ 检测到分类记录，切换到{'复制' if target_mode else '移动'}模式需要数据迁移")
        title_label.setStyleSheet(
            f"font-size: 14px; font-weight: bold; color: {default_theme.colors.WARNING};"
        )
        layout.addWidget(title_label)

        # 说明文本
        explanation = QTextEdit()
        explanation.setReadOnly(True)
        explanation.setMaximumHeight(200)

        current_mode = "复制" if self.is_copy_mode else "移动"
        target_mode_name = "复制" if target_mode else "移动"

        explanation_text = f"""当前模式：{current_mode}模式
目标模式：{target_mode_name}模式

检测到的数据：
• 已分类图片：{len(self.classified_images)} 张
• 已移出图片：{len(self.removed_images)} 张

迁移说明：
"""

        if target_mode:  # 切换到复制模式
            explanation_text += """• 从移动模式切换到复制模式
• 会将分类目录和移除目录中的文件复制回原目录
• 原始目录将恢复所有图片（包括已分类和已移出的）
• 分类记录会被保留，但文件会存在于多个位置"""
        else:  # 切换到移动模式
            explanation_text += """• 从复制模式切换到移动模式
• 会删除原始目录中已分类和已移出的图片
• 图片只保留在对应的分类目录中
• 这是一个不可逆的操作，请谨慎选择"""

        explanation.setPlainText(explanation_text)
        layout.addWidget(explanation)

        # 警告
        warning_label = QLabel("⚠️ 此操作会修改文件系统，建议在操作前备份重要数据")
        warning_label.setStyleSheet(
            f"color: {default_theme.colors.ERROR}; font-weight: bold;"
        )
        layout.addWidget(warning_label)

        # 按钮布局（修复：危险操作按钮在左侧）
        button_layout = QHBoxLayout()

        # 危险操作始终使用统一的红色语义按钮。
        confirm_button = QPushButton(f"确认迁移到{target_mode_name}模式")
        confirm_button.clicked.connect(dialog.accept)
        style_button(confirm_button, "danger", min_width=160)

        cancel_button = QPushButton("取消")
        cancel_button.clicked.connect(dialog.reject)
        style_button(cancel_button, "secondary")

        button_layout.addStretch()
        button_layout.addWidget(cancel_button)
        button_layout.addWidget(confirm_button)

        layout.addLayout(button_layout)

        return dialog.exec() == QDialog.DialogCode.Accepted

    def set_mode(self, is_copy):
        """设置操作模式"""
        # 如果要切换到移动模式，但当前是多分类模式，拒绝切换（最高优先级）
        if not is_copy and self.is_multi_category:
            toast_warning(self,"移动模式不支持多分类功能，请先切换为单分类模式")
            # 拒绝切换，保持原来的复制模式
            return

        # 检查是否需要迁移
        if self.is_copy_mode != is_copy and self._is_migration_needed():
            # 显示迁移确认对话框
            if not self._show_migration_confirmation_dialog(is_copy):
                # 用户取消迁移，保持原模式
                return

            # 用户确认迁移，执行迁移逻辑
            try:
                if is_copy:
                    # 从移动模式迁移到复制模式
                    self._migrate_move_to_copy()
                else:
                    # 从复制模式迁移到移动模式
                    self._migrate_copy_to_move()
            except Exception as e:
                self.logger.error(f"模式迁移失败: {e}")
                toast_error(self, f"模式迁移失败: {e}")
                return

        self.is_copy_mode = is_copy

        # 更新按钮图标和提示
        if is_copy:
            self.mode_button.setText('⧉')  # 重叠方块表示复制
            self.mode_button.setToolTip('复制模式 - 点击切换到移动模式')
            # 显示模式切换Toast
            toast_info(self,"已切换到复制模式")
        else:
            self.mode_button.setText('✂')  # 剪刀表示移动
            self.mode_button.setToolTip('移动模式 - 点击切换到复制模式')
            # 显示模式切换Toast
            toast_info(self,"已切换到移动模式")

        self.logger.info(f"操作模式已切换为: {'复制' if is_copy else '移动'}")

        # 立即同步保存状态到文件，确保模式状态与文件系统状态保持一致
        self._save_state_sync()

    def _set_mode_direct(self, is_copy):
        """直接设置操作模式，不触发迁移逻辑（用于状态恢复）"""
        self.is_copy_mode = is_copy

        # 更新按钮图标和提示
        if is_copy:
            self.mode_button.setText('⧉')  # 重叠方块表示复制
            self.mode_button.setToolTip('复制模式 - 点击切换到移动模式')
        else:
            self.mode_button.setText('✂')  # 剪刀表示移动
            self.mode_button.setToolTip('移动模式 - 点击切换到复制模式')

        self.logger.info(f"操作模式已恢复为: {'复制' if is_copy else '移动'}")

    def _migrate_copy_to_move(self):
        """从复制模式迁移到移动模式"""
        self.logger.info("开始从复制模式迁移到移动模式")

        # 收集需要从原目录删除的文件
        files_to_delete = []
        for file_path in self.classified_images.keys():
            if Path(file_path).exists():
                files_to_delete.append(file_path)
        for file_path in self.removed_images:
            if Path(file_path).exists():
                files_to_delete.append(file_path)

        if not files_to_delete:
            self.logger.info("没有需要删除的文件")
            toast_info(self, "没有需要迁移的文件")
            return

        # 显示进度对话框
        progress_dialog = ProgressDialog("模式迁移 - 复制到移动", self)
        progress_dialog.update_main_text(f"正在迁移到移动模式，将删除原目录中的 {len(files_to_delete)} 个文件...")
        progress_dialog.show()
        QApplication.processEvents()

        # 删除原目录中的文件（带错误处理和回滚）
        deleted_files = []  # 记录已删除的文件，用于回滚
        deleted_count = 0
        total_files = len(files_to_delete)

        try:
            for i, file_path in enumerate(files_to_delete):
                # 检查是否被取消
                if progress_dialog.is_cancelled():
                    progress_dialog.force_close()
                    self.logger.info("用户取消了迁移操作")
                    toast_warning(self, "迁移操作已取消")
                    return

                try:
                    # 更新进度
                    progress_dialog.update_progress(i + 1, total_files)
                    progress_dialog.update_detail_text(f"正在删除: {Path(file_path).name}")
                    QApplication.processEvents()

                    # 在删除前验证对应分类文件存在
                    file_path_obj = Path(file_path)
                    if str(file_path) in self.classified_images:
                        category = self.classified_images[str(file_path)]
                        category_name = category[0] if isinstance(category, list) and category else category
                        if category_name:
                            parent_dir = self.current_dir.parent
                            category_dir = parent_dir / normalize_folder_name(category_name)
                            category_file = category_dir / file_path_obj.name
                            if not category_file.exists():
                                raise FileOperationError(f"分类目录中找不到对应文件: {category_file}")

                    # 备份文件信息用于可能的回滚
                    file_backup_info = {
                        'path': file_path,
                        'stat': file_path_obj.stat() if file_path_obj.exists() else None
                    }

                    # 删除原文件（修复：移除只读属性后再删除）
                    try:
                        file_path_obj.unlink()
                    except PermissionError:
                        # 只读文件，尝试移除只读属性后重试
                        self.logger.warning(f"文件只读，尝试移除只读属性: {file_path}")
                        remove_readonly(file_path_obj)
                        retry_file_operation(file_path_obj.unlink, max_retries=2)

                    deleted_files.append(file_backup_info)
                    deleted_count += 1
                    self.logger.debug(f"删除原文件: {file_path}")

                except Exception as e:
                    self.logger.error(f"删除文件失败 {file_path}: {e}")

                    # 尝试回滚已删除的文件
                    self._rollback_copy_to_move_migration(deleted_files)

                    progress_dialog.force_close()
                    raise FileOperationError(f"删除文件失败，已尝试回滚: {e}")

            # 完成
            progress_dialog.update_main_text("迁移完成")
            progress_dialog.update_detail_text(f"已删除 {deleted_count} 个文件")
            progress_dialog.update_progress(total_files, total_files)
            QApplication.processEvents()

            QTimer.singleShot(1000, progress_dialog.force_close)  # 1秒后关闭

            self.logger.info(f"复制到移动模式迁移完成，删除了 {deleted_count} 个文件")
            toast_success(self, f"迁移完成，已删除原目录中的 {deleted_count} 个文件")

        except Exception as e:
            self.logger.error(f"迁移过程中发生严重错误: {e}")
            progress_dialog.force_close()
            toast_error(self, f"迁移失败: {e}")
            raise

    def _rollback_copy_to_move_migration(self, deleted_files):
        """回滚复制到移动模式的迁移操作"""
        self.logger.warning("开始回滚复制到移动模式迁移操作")

        rollback_count = 0
        for file_info in deleted_files:
            try:
                file_path = file_info['path']

                # 从分类目录或移除目录找回文件
                source_file = None

                # 先尝试从分类目录找回
                if str(file_path) in self.classified_images:
                    category = self.classified_images[str(file_path)]
                    category_name = category[0] if isinstance(category, list) and category else category
                    if category_name:
                        parent_dir = self.current_dir.parent
                        category_dir = parent_dir / normalize_folder_name(category_name)
                        category_file = category_dir / Path(file_path).name
                        if category_file.exists():
                            source_file = category_file

                # 再尝试从移除目录找回
                if not source_file and str(file_path) in self.removed_images:
                    parent_dir = self.current_dir.parent
                    remove_dir = parent_dir / "remove"
                    remove_file = remove_dir / Path(file_path).name
                    if remove_file.exists():
                        source_file = remove_file

                # 恢复文件
                if source_file:
                    shutil.copy2(str(source_file), str(file_path))
                    rollback_count += 1
                    self.logger.debug(f"回滚恢复文件: {source_file} -> {file_path}")
                else:
                    self.logger.warning(f"无法找到回滚源文件: {file_path}")

            except Exception as e:
                self.logger.error(f"回滚文件失败 {file_info['path']}: {e}")

        self.logger.info(f"回滚完成，恢复了 {rollback_count} 个文件")
        if rollback_count > 0:
            toast_info(self, f"已回滚恢复 {rollback_count} 个文件")

    def _migrate_move_to_copy(self):
        """从移动模式迁移到复制模式"""
        self.logger.info("开始从移动模式迁移到复制模式")

        # 收集需要复制的文件
        files_to_copy = []

        # 收集分类文件
        parent_dir = self.current_dir.parent
        for file_path, category in self.classified_images.items():
            category_name = category[0] if isinstance(category, list) and category else category
            if category_name:
                category_dir = parent_dir / normalize_folder_name(category_name)
                category_file = category_dir / Path(file_path).name
                if category_file.exists() and not Path(file_path).exists():
                    files_to_copy.append((category_file, Path(file_path), 'classified'))

        # 收集移除文件
        remove_dir = parent_dir / "remove"
        if remove_dir.exists():
            for file_path in self.removed_images:
                remove_file = remove_dir / Path(file_path).name
                if remove_file.exists() and not Path(file_path).exists():
                    files_to_copy.append((remove_file, Path(file_path), 'removed'))

        if not files_to_copy:
            self.logger.info("没有需要复制的文件")
            toast_info(self, "没有需要迁移的文件")
            return

        # 显示进度对话框
        progress_dialog = ProgressDialog("模式迁移 - 移动到复制", self)
        progress_dialog.update_main_text(f"正在迁移到复制模式，将复制 {len(files_to_copy)} 个文件回原目录...")
        progress_dialog.show()
        QApplication.processEvents()

        # 复制文件（带错误处理和回滚）
        copied_files = []  # 记录已复制的文件，用于回滚
        copied_count = 0
        total_files = len(files_to_copy)

        try:
            for i, (source_file, target_file, file_type) in enumerate(files_to_copy):
                # 检查是否被取消
                if progress_dialog.is_cancelled():
                    progress_dialog.force_close()
                    self.logger.info("用户取消了迁移操作")
                    toast_warning(self, "迁移操作已取消")
                    return

                try:
                    # 更新进度
                    progress_dialog.update_progress(i + 1, total_files)
                    file_type_name = '分类' if file_type == 'classified' else '移除'
                    progress_dialog.update_detail_text(f"正在复制{file_type_name}文件: {target_file.name}")
                    QApplication.processEvents()

                    # 验证源文件存在
                    if not source_file.exists():
                        raise FileOperationError(f"源文件不存在: {source_file}")

                    # 检查目标文件是否已存在（避免覆盖）
                    if target_file.exists():
                        self.logger.warning(f"目标文件已存在，跳过: {target_file}")
                        continue

                    # 确保目标目录存在
                    target_file.parent.mkdir(parents=True, exist_ok=True)

                    # 验证磁盘空间
                    source_size = source_file.stat().st_size
                    free_space = shutil.disk_usage(target_file.parent).free
                    if source_size > free_space:
                        raise FileOperationError(f"磁盘空间不足，需要 {source_size} 字节，可用 {free_space} 字节")

                    # 复制文件
                    shutil.copy2(str(source_file), str(target_file))

                    # 验证复制是否成功
                    if not target_file.exists():
                        raise FileOperationError(f"文件复制后验证失败: {target_file}")

                    # 验证文件大小
                    if target_file.stat().st_size != source_file.stat().st_size:
                        raise FileOperationError(f"文件大小不匹配: {source_file} vs {target_file}")

                    copied_files.append(target_file)
                    copied_count += 1
                    self.logger.debug(f"复制文件: {source_file} -> {target_file}")

                except Exception as e:
                    self.logger.error(f"复制文件失败 {source_file} -> {target_file}: {e}")

                    # 尝试回滚已复制的文件
                    self._rollback_move_to_copy_migration(copied_files)

                    progress_dialog.force_close()
                    raise FileOperationError(f"复制文件失败，已尝试回滚: {e}")

            # 完成
            progress_dialog.update_main_text("迁移完成")
            progress_dialog.update_detail_text(f"已复制 {copied_count} 个文件回原目录")
            progress_dialog.update_progress(total_files, total_files)
            QApplication.processEvents()

            QTimer.singleShot(1000, progress_dialog.force_close)  # 1秒后关闭

            self.logger.info(f"移动到复制模式迁移完成，复制了 {copied_count} 个文件")
            toast_success(self, f"迁移完成，已复制 {copied_count} 个文件回原目录")

        except Exception as e:
            self.logger.error(f"迁移过程中发生严重错误: {e}")
            progress_dialog.force_close()
            toast_error(self, f"迁移失败: {e}")
            raise

    def _rollback_move_to_copy_migration(self, copied_files):
        """回滚移动到复制模式的迁移操作"""
        self.logger.warning("开始回滚移动到复制模式迁移操作")

        rollback_count = 0
        for target_file in copied_files:
            try:
                if target_file.exists():
                    target_file.unlink()
                    rollback_count += 1
                    self.logger.debug(f"回滚删除文件: {target_file}")
                else:
                    self.logger.warning(f"回滚时文件不存在: {target_file}")

            except Exception as e:
                self.logger.error(f"回滚删除文件失败 {target_file}: {e}")

        self.logger.info(f"回滚完成，删除了 {rollback_count} 个文件")
        if rollback_count > 0:
            toast_info(self, f"已回滚删除 {rollback_count} 个文件")

    def create_category_mode_button(self, toolbar):
        """创建图标化的分类模式切换按钮 - 单分类/多分类"""
        # 使用统一样式创建按钮
        self.category_mode_button = self.create_toolbar_button('→', 'category_mode_button',
                                                              '单分类模式 - 点击切换到多分类模式',
                                                              self.toggle_category_mode)

        # 添加到工具栏
        toolbar.addWidget(self.category_mode_button)
    
    def toggle_category_mode(self):
        """切换单分类/多分类模式"""
        # 移动模式不支持多分类
        if not self.is_copy_mode and not self.is_multi_category:
            toast_warning(self,"移动模式不支持多分类功能，请先切换到复制模式")
            return

        # 从多分类切到单分类前：检查是否存在多标签图片
        if self.is_multi_category:
            multi_tagged = [
                p for p, c in self.classified_images.items()
                if isinstance(c, list) and len(c) > 1
            ]
            if multi_tagged:
                toast_warning(
                    self,
                    f"检测到 {len(multi_tagged)} 张图片被分配了多个标签，无法切换到单分类模式。\n"
                    "请先移除多余标签后再切换。"
                )
                return

            # 归一化：List[str] (len==1) -> str，避免单分类逻辑拿到 list 造成异常
            for p, c in list(self.classified_images.items()):
                if isinstance(c, list):
                    if len(c) == 1:
                        self.classified_images[p] = c[0]
                    elif len(c) == 0:
                        del self.classified_images[p]

        self.is_multi_category = not self.is_multi_category

        # 使用统一的按钮状态更新方法
        self.update_category_mode_button()

        # 显示Toast通知
        if self.is_multi_category:
            toast_info(self, "已切换到多分类模式")
        else:
            toast_info(self, "已切换到单分类模式")

        # 保存分类模式状态
        self.save_state()

        self.logger.info(f"分类模式已切换为: {'多分类' if self.is_multi_category else '单分类'}")

    def change_category_sort_mode(self, new_mode):
        """切换类别排序模式

        Args:
            new_mode: "name", "shortcut" 或 "count"
        """
        try:
            if new_mode not in ["name", "shortcut", "count"]:
                self.logger.error(f"无效的排序模式: {new_mode}")
                return

            # 更新配置
            self.config.category_sort_mode = new_mode
            self.config.save_config()

            # 重新排序类别（count模式需要传入分类数量统计）
            category_counts = self._get_category_counts() if new_mode == "count" else None
            self.ordered_categories = self.config.get_sorted_categories(
                self.categories, category_counts=category_counts
            )

            # 更新UI
            self.update_category_buttons()

            # 更新方向按钮的tooltip（因为排序模式变了）
            self._update_direction_button_tooltip()

            # 显示提示（统一格式）
            current_text, _ = self._get_sort_tooltip_texts()
            toast_success(self, f"已切换：{current_text}")
            self.logger.info(f"类别排序模式已切换为: {new_mode}")

        except Exception as e:
            self.logger.error(f"切换排序模式失败: {e}")
            toast_error(self, f"切换排序模式失败: {str(e)}")

    def toggle_sort_direction(self):
        """切换排序方向（升序/降序）"""
        try:
            # 切换方向
            self.config.sort_ascending = not self.config.sort_ascending
            self.config.save_config()

            # 更新按钮图标和tooltip
            self.category_panel.sort_direction_button.setText('↑' if self.config.sort_ascending else '↓')
            self._update_direction_button_tooltip()

            # 重新排序类别
            category_counts = self._get_category_counts() if self.config.category_sort_mode == "count" else None
            self.ordered_categories = self.config.get_sorted_categories(
                self.categories, category_counts=category_counts
            )

            # 更新UI
            self.update_category_buttons()

            # 显示提示（统一格式，显示当前状态）
            current_text, _ = self._get_sort_tooltip_texts()
            toast_success(self, f"已切换：{current_text}")
            self.logger.info("排序方向已切换")

        except Exception as e:
            self.logger.error(f"切换排序方向失败: {e}")
            toast_error(self, f"切换排序方向失败: {str(e)}")

    def _get_sort_tooltip_texts(self):
        """获取当前排序状态的tooltip文案

        Returns:
            tuple: (当前状态简短描述, 切换后状态简短描述)
        """
        mode = self.config.category_sort_mode
        is_asc = self.config.sort_ascending

        # 状态描述映射表 (mode, is_ascending) -> (当前状态, 切换后状态)
        # 统一格式："{模式} {方向}"
        status_map = {
            ('name', True):     ("名称 A→Z", "名称 Z→A"),
            ('name', False):    ("名称 Z→A", "名称 A→Z"),
            ('shortcut', True): ("快捷键 1→Z", "快捷键 Z→1"),
            ('shortcut', False):("快捷键 Z→1", "快捷键 1→Z"),
            ('count', True):    ("数量 少→多", "数量 多→少"),
            ('count', False):   ("数量 多→少", "数量 少→多"),
        }

        return status_map.get((mode, is_asc), ("未知状态", "切换方向"))

    def _update_direction_button_tooltip(self):
        """根据当前排序模式和方向，动态更新方向按钮的tooltip"""
        current_text, action_text = self._get_sort_tooltip_texts()
        tooltip = f"当前：{current_text}\n点击切换为：{action_text}"
        self.category_panel.sort_direction_button.setToolTip(tooltip)

    def _sync_sort_button_state(self):
        """同步排序方向按钮的UI状态（从配置加载后调用）"""
        if hasattr(self, 'category_panel') and hasattr(self.category_panel, 'sort_direction_button'):
            # 更新按钮图标
            self.category_panel.sort_direction_button.setText('↑' if self.config.sort_ascending else '↓')
            # 更新tooltip
            self._update_direction_button_tooltip()

    def _get_category_counts(self):
        """获取每个类别的分类数量统计

        Returns:
            dict: {category_name: count} 每个类别对应的图片数量
        """
        counts = {}
        for img_path, category in self.classified_images.items():
            if isinstance(category, list):
                # 多分类模式：一张图可能属于多个类别
                for cat in category:
                    counts[cat] = counts.get(cat, 0) + 1
            else:
                # 单分类模式
                counts[category] = counts.get(category, 0) + 1
        return counts

    def fit_to_window(self):
        """适应窗口大小"""
        if hasattr(self, 'image_label'):
            self.image_label.fit_to_window()
    
    def show_add_category_dialog(self):
        """显示添加类别对话框（UI操作方法）

        注意：此方法原名为 add_category()，为避免与 StateView 的
        add_category(category: str) 方法冲突而重命名。
        """
        if not self.current_dir:
            toast_warning(self,'请先选择图片目录')
            return

        # 记录添加前的类别数量
        original_count = len(self.categories)

        dialog = AddCategoriesDialog(self.categories, self)
        if dialog.exec():
            # 检查是否有新类别添加
            new_count = len(self.categories)
            added_count = new_count - original_count
            skipped_categories = getattr(dialog, 'skipped_categories', [])

            # 构建合并的 toast 消息
            if added_count > 0 and skipped_categories:
                # 有添加也有跳过
                skipped_text = ', '.join(skipped_categories[:3])
                if len(skipped_categories) > 3:
                    skipped_text += f' 等{len(skipped_categories)}个'
                toast_warning(self, f"已添加 {added_count} 个类别，{skipped_text} 已存在或已忽略")
            elif added_count > 0:
                # 只有添加
                if added_count == 1:
                    toast_success(self, "已添加 1 个新类别")
                else:
                    toast_success(self, f"已添加 {added_count} 个新类别")
            elif skipped_categories:
                # 只有跳过
                skipped_text = ', '.join(skipped_categories[:3])
                if len(skipped_categories) > 3:
                    skipped_text += f' 等{len(skipped_categories)}个'
                toast_warning(self, f"类别 {skipped_text} 已存在或已忽略，未添加新类别")
    
    # ===== 图像加载器回调方法 =====
    
    def on_image_loaded(self, image_path, image_data):
        """图像加载完成回调 - 只显示当前选中的图片"""
        try:
            # 关键修复：只显示最后一次请求的图片，其他都是预加载缓存
            # 使用保存的请求路径而不是当前索引，避免异步加载时索引变化导致判断错误
            normalized_loaded = str(Path(image_path).resolve())

            # 检查是否有记录的当前请求图片
            if hasattr(self, '_current_requested_image'):
                # 比较路径
                if self._current_requested_image != normalized_loaded:
                    # 不是当前图片，只是预加载缓存，不显示UI
                    self.logger.debug(f"预加载完成(不显示): {Path(image_path).name}")
                    return
            else:
                # 没有记录的请求路径（比如程序刚启动），也显示
                pass
            
            # 只有当前图片才进行UI显示
            # 转换图像数据为QPixmap
            if isinstance(image_data, QPixmap):
                # 这是QPixmap
                pixmap = image_data
            else:
                # 这可能是numpy数组或PIL图像，需要转换
                pixmap = self.convert_to_pixmap(image_data)
            
            # 显示图像
            if pixmap is not None and not pixmap.isNull():
                self.display_pixmap(pixmap, image_path)
                self.logger.info(f"当前图片显示完成: {Path(image_path).name}")
                self._on_ai_image_ready(image_data)
            else:
                self.logger.warning(f"无法显示图像: {image_path}")
                
        except Exception as e:
            self.logger.error(f"显示图像时出错: {e}")
            self.logger.error(traceback.format_exc())
    
    def on_thumbnail_loaded(self, image_path, thumbnail_data):
        """缩略图加载完成回调"""
        try:
            # 更新图片列表中的缩略图
            self.update_image_list_thumbnail(image_path, thumbnail_data)
        except Exception as e:
            self.logger.error(f"更新缩略图时出错: {e}")
    
    def on_loading_progress(self, message):
        """加载进度回调"""
        self.statusBar.showMessage(message)
    
    def convert_to_pixmap(self, image_data):
        """将图像数据转换为QPixmap"""
        try:
            if isinstance(image_data, np.ndarray):
                # numpy数组图像数据
                height, width, channel = image_data.shape
                bytes_per_line = 3 * width

                # 确保数组是C连续的，并复制数据避免生命周期问题
                # 这对于从磁盘缓存加载的图片非常重要
                if not image_data.flags['C_CONTIGUOUS']:
                    image_data = np.ascontiguousarray(image_data)

                # 创建QImage并立即复制数据，避免numpy数组被释放后QImage引用无效内存
                q_image = QImage(image_data.data, width, height, bytes_per_line, QImage.Format.Format_RGB888).copy()
                return QPixmap.fromImage(q_image)
                
            elif hasattr(image_data, 'mode'):
                # PIL图像
                if image_data.mode == 'RGB':
                    rgb_array = np.array(image_data)
                    height, width, channel = rgb_array.shape
                    bytes_per_line = 3 * width

                    # 确保数组是C连续的
                    if not rgb_array.flags['C_CONTIGUOUS']:
                        rgb_array = np.ascontiguousarray(rgb_array)

                    # 创建QImage并复制数据
                    q_image = QImage(rgb_array.data, width, height, bytes_per_line, QImage.Format.Format_RGB888).copy()
                    return QPixmap.fromImage(q_image)
                else:
                    # 转换为RGB
                    rgb_image = image_data.convert('RGB')
                    return self.convert_to_pixmap(rgb_image)
            
            return None
            
        except Exception as e:
            self.logger.error(f"转换图像格式时出错: {e}")
            return None
    
    def display_pixmap(self, pixmap, image_path):
        """显示QPixmap - 统一UI显示"""
        try:
            # 直接使用EnhancedImageLabel的set_image方法
            self.image_label.set_image(pixmap)
            
            # 使用统一的窗口标题更新方法
            self.update_window_title(image_path)
            
            # 更新状态栏显示当前图片名称
            filename = Path(image_path).name
            self.statusBar.showMessage(f"📷 {filename}")
            
            # 记录性能信息
            self.log_performance_info(
                "显示图片_完成",
                文件=filename,
                索引=f"{self.current_index + 1}/{len(self.image_files)}"
            )
            
        except Exception as e:
            self.logger.error(f"显示图像时出错: {e}")
    
    def update_image_list_thumbnail(self, image_path, thumbnail_data):
        """更新图片列表中的缩略图"""
        try:
            # Phase 1.1 Migration: 使用 Model API 高效更新缩略图 (O(1))
            # 不再遍历 ListWidget，直接通过 Model 的 Path 映射更新

            if not hasattr(self, 'image_list_model'):
                return

            icon = None

            # 1. 统一转换为 QIcon
            if hasattr(thumbnail_data, 'copy'):
                # 已经是 QPixmap
                icon = QIcon(thumbnail_data)
            else:
                # 可能是 numpy 数组，需要转换
                thumbnail_pixmap = self.convert_to_pixmap(thumbnail_data)
                if thumbnail_pixmap:
                    icon = QIcon(thumbnail_pixmap)

            # 2. 更新 Model (自动处理缓存和 View 刷新)
            if icon:
                self.image_list_model.set_thumbnail(str(image_path), icon)

        except Exception as e:
            self.logger.error(f"更新缩略图时出错: {e}")

    def show_help_dialog(self):
        """显示帮助对话框"""
        dialog = TabbedHelpDialog(self.version, self, config=getattr(self, 'config', None))
        # 确保对话框应用当前主题
        if hasattr(dialog, '_apply_theme'):
            dialog._apply_theme()
        dialog.exec()

    def show_settings_dialog(self):
        """显示设置对话框"""
        try:
            dialog = SettingsDialog(self)
            dialog.exec()
            # 注意：不再自动隐藏红点
            # 红点的显示/隐藏由更新检查逻辑控制
            # - 有待安装更新时：红点保持显示
            # - 用户选择立即安装：隐藏红点并退出
            # - 确认是最新版本：隐藏红点
        except Exception as e:
            self.logger.error(f"打开设置对话框失败: {e}")
            toast_error(self, f"打开设置失败: {str(e)}")

    def _check_and_show_tutorial(self):
        """检查是否需要显示教程"""
        try:
            if self.tutorial_manager and self.tutorial_manager.should_show_tutorial():
                self.logger.info("首次运行，显示教程引导")
                self.tutorial_manager.start_tutorial()
                # 教程显示时，不检查恢复目录（避免冲突）
            else:
                # 教程已完成或跳过，检查是否需要恢复上次目录
                QTimer.singleShot(500, self._check_and_restore_last_directory)
        except Exception as e:
            self.logger.error(f"检查教程显示状态失败: {e}")

    def _check_and_restore_last_directory(self):
        """检查并询问用户是否恢复上次打开的目录"""
        try:
            # 如果已经打开了目录，则不提示
            if self.current_dir is not None:
                return

            # 获取上次打开的目录
            app_config = get_app_config()
            last_dir = app_config.last_opened_directory

            # 如果没有记录或目录不存在，则不提示
            if not last_dir or not Path(last_dir).exists():
                return

            # 检测路径类型
            is_network = is_network_path(last_dir)
            path_type_icon = "🌐" if is_network else "💾"
            path_type_text = "网络路径" if is_network else "本地路径"

            # 提取盘符信息
            last_dir_path = Path(last_dir)
            drive_info = f" ({last_dir_path.drive})" if last_dir_path.drive else ""

            # 创建询问对话框
            msg_box = ThemedMessageBox(self)
            msg_box.setWindowTitle("恢复上次任务")
            msg_box.setText("检测到上次打开的工作目录")
            msg_box.setInformativeText(
                f"路径：{last_dir}\n"
                f"{path_type_icon} {path_type_text}{drive_info}\n\n"
                f"是否继续上次的任务？"
            )

            # 添加按钮
            yes_btn = msg_box.addButton("继续上次任务", QMessageBox.ButtonRole.YesRole)
            msg_box.addButton("重新选择", QMessageBox.ButtonRole.NoRole)
            msg_box.setDefaultButton(yes_btn)

            # 显示对话框并获取结果
            msg_box.exec()
            clicked_button = msg_box.clickedButton()

            if clicked_button == yes_btn:
                self.logger.info(f"用户选择恢复上次目录: {last_dir}")
                # 直接设置目录并打开
                self._open_directory_with_path(last_dir)
            else:
                self.logger.info("用户选择重新选择目录")

        except Exception as e:
            self.logger.error(f"检查恢复目录失败: {e}")

    def _open_directory_with_path(self, dir_path: str):
        """使用指定路径打开目录（内部方法，用于恢复上次目录）"""
        try:
            self.current_dir = Path(dir_path)

            # 检测微信目录并提醒（只提醒一次）
            if not hasattr(self, '_wechat_warned'):
                self._wechat_warned = False
            if not self._wechat_warned and self._check_wechat_directory(self.current_dir):
                self._wechat_warned = True
                self._show_wechat_warning()

            # 检查是否为网络路径
            path_str = str(self.current_dir)
            is_network_path = path_str.startswith('\\\\')

            # Task 1.1修复：通知图像加载器当前工作路径类型
            try:
                self.image_loader.set_working_path(self.current_dir)
            except Exception as e:
                self.logger.warning(f"设置图像加载器工作路径失败: {e}")

            if is_network_path:
                # 网络路径提醒
                toast_info(self, '🚀 SMB/NAS路径已启用专项优化')
                self.logger.info(f"恢复网络路径: {self.current_dir}")
            else:
                self.logger.info(f"恢复本地路径: {self.current_dir}")
                # 显示本地目录打开成功通知
                toast_success(self, f"已恢复目录：{self.current_dir.name}")

            # 设置配置文件路径为图片目录的父目录
            self.config.set_base_dir(str(self.current_dir.parent))

            # 先启动图片扫描，让UI立即响应
            self.load_images()

            # 延迟类别加载和同步操作
            QTimer.singleShot(100, self._delayed_load_categories)

        except Exception as e:
            self.logger.error(f"打开指定目录失败: {e}")
            toast_error(self, f"打开目录失败: {str(e)}")

    def start_tutorial(self):
        """开始或重新开始教程（提供给菜单调用）"""
        try:
            if self.tutorial_manager:
                self.tutorial_manager.reset_tutorial()
                toast_info(self, "教程已重新开始")
            else:
                toast_error(self, "教程系统不可用")
        except Exception as e:
            self.logger.error(f"启动教程失败: {e}")
            toast_error(self, f"启动教程失败: {str(e)}")

    def update_theme_button_state(self):
        """更新主题按钮状态（根据主题模式）"""
        if not hasattr(self, 'theme_button'):
            return

        try:
            app_config = get_app_config()
            theme_mode = app_config.theme_mode

            # 检查是否因为自动模式或系统模式需要禁用
            disabled_due_to_mode = theme_mode in ("auto", "system")

            if disabled_due_to_mode:
                # 因模式原因禁用
                self.theme_button.setEnabled(False)
                mode_name = "自动切换" if theme_mode == "auto" else "跟随系统"
                self.theme_button.setToolTip(f"已启用{mode_name}，点击查看提示")
            else:
                # 启用按钮
                self.theme_button.setEnabled(True)
                current_theme = default_theme.get_current_theme()
                theme_tooltip = '切换到暗色主题' if current_theme == "light" else '切换到亮色主题'
                self.theme_button.setToolTip(theme_tooltip)

        except Exception as e:
            self.logger.warning(f"更新主题按钮状态失败: {e}")

    def toggle_theme(self):
        """切换主题"""
        try:
            # 检查是否启用了自动切换
            app_config = get_app_config()

            # 重新加载配置确保使用最新值（防止缓存不一致）
            app_config.reload_config()

            # 如果启用了自动模式，禁止手动切换
            if app_config.theme_mode == "auto":
                toast_warning(self, "已启用自动切换，请先在设置中关闭")
                return

            # 如果启用了系统模式，也禁止手动切换
            if app_config.theme_mode == "system":
                toast_warning(self, "已启用跟随系统，请先在设置中关闭")
                return

            # 获取当前主题并切换
            current_theme = default_theme.get_current_theme()
            new_theme = "dark" if current_theme == "light" else "light"
            default_theme.set_theme(new_theme)

            # 保存到应用配置（app_config.json）
            app_config.theme = new_theme
            self.logger.info(f"主题已切换到: {new_theme}")

            # 应用主题到主窗口和所有组件
            self.apply_theme()

            # 更新按钮图标和提示
            if hasattr(self, 'theme_button'):
                theme_icon = '☾' if new_theme == "light" else '☼'  # ☾ 月亮(暗色) ☼ 太阳(亮色)
                theme_tooltip = '切换到暗色主题' if new_theme == "light" else '切换到亮色主题'
                self.theme_button.setText(theme_icon)
                self.theme_button.setToolTip(theme_tooltip)

            toast_success(self, f'已切换到{"暗色" if new_theme == "dark" else "亮色"}主题')
        except Exception as e:
            self.logger.error(f"切换主题失败: {e}", exc_info=True)
            toast_error(self, f'主题切换失败: {str(e)}')

    def apply_theme(self):
        """应用主题到主窗口和所有组件"""
        try:
            c = default_theme.colors

            # 应用主窗口样式
            self.setStyleSheet(MainWindowStyles.get_main_window_style())

            # 应用工具栏样式
            if hasattr(self, 'findChildren'):
                toolbars = self.findChildren(QToolBar)
                for toolbar in toolbars:
                    toolbar.setStyleSheet(ToolbarStyles.get_main_toolbar_style())

            # 更新右侧面板样式
            right_panel = self.findChild(QWidget, "right_panel")
            if right_panel:
                right_panel.setStyleSheet(f"""
                    QWidget#right_panel {{
                        background-color: {c.BACKGROUND_PRIMARY};
                        border: 1px solid {c.BORDER_MEDIUM};
                        border-radius: 6px;
                    }}
                """)

            # 更新图片列表区域
            # Phase 1.1: QListWidget → QListView
            if hasattr(self, 'image_list'):
                self.image_list.setStyleSheet(f"""
                    QListView {{
                        border: 1px solid {c.BORDER_MEDIUM};
                        border-radius: 4px;
                        background-color: {c.BACKGROUND_SECONDARY};
                        padding: 2px;
                    }}
                    QListView::item {{
                        border: 1px solid transparent;
                        border-radius: 3px;
                        padding: 4px 6px;
                        margin: 1px;
                        color: {c.TEXT_PRIMARY};
                    }}
                    QListView::item:hover {{
                        background-color: {c.BACKGROUND_HOVER};
                        border-color: {c.PRIMARY};
                    }}
                    QListView::item:selected {{
                        background-color: {c.PRIMARY};
                        color: white;
                        border-color: {c.PRIMARY_DARK};
                    }}
                    QScrollBar:vertical {{
                        border: 1px solid {c.BORDER_LIGHT};
                        background: {c.BACKGROUND_SECONDARY};
                        width: 14px;
                        border-radius: 4px;
                        margin: 0px;
                    }}
                    QScrollBar::handle:vertical {{
                        background: {c.PRIMARY};
                        border-radius: 3px;
                        min-height: 30px;
                        margin: 2px 2px 2px 2px;
                    }}
                    QScrollBar::handle:vertical:hover {{
                        background: {c.PRIMARY_DARK};
                        margin: 1px 1px 1px 1px;
                    }}
                    QScrollBar::handle:vertical:pressed {{
                        background: {c.PRIMARY_DARK};
                        margin: 0px 0px 0px 0px;
                    }}
                    QScrollBar:horizontal {{
                        border: 1px solid {c.BORDER_LIGHT};
                        background: {c.BACKGROUND_SECONDARY};
                        height: 10px;
                        border-radius: 3px;
                    }}
                    QScrollBar::handle:horizontal {{
                        background: {c.PRIMARY};
                        border-radius: 3px;
                        min-width: 15px;
                    }}
                    QScrollBar::handle:horizontal:hover {{
                        background: {c.PRIMARY_DARK};
                    }}
                    QScrollBar::handle:horizontal:pressed {{
                        background: {c.PRIMARY_DARK};
                    }}
                    QScrollBar::add-line:vertical,
                    QScrollBar::sub-line:vertical,
                    QScrollBar::add-line:horizontal,
                    QScrollBar::sub-line:horizontal {{
                        border: none;
                        background: none;
                    }}
                """)

            # 更新图片列表标题容器
            list_title_container = self.findChild(QWidget, "list_title_container")
            if list_title_container:
                list_title_container.setStyleSheet(f"""
                    QWidget#list_title_container {{
                        border-bottom: 1px solid {c.BORDER_MEDIUM};
                    }}
                """)
                delegate = self.image_list.itemDelegate()
                if hasattr(delegate, 'refresh_theme'):
                    delegate.refresh_theme()

            if hasattr(self, 'image_list_title_label'):
                self.image_list_title_label.setStyleSheet(f"""
                    QLabel {{
                        color: {c.PRIMARY};
                        font-size: 13px;
                        font-weight: bold;
                        border: none;
                        background: transparent;
                    }}
                """)

            # 更新搜索组件主题
            if hasattr(self, 'image_search_widget'):
                self.image_search_widget.apply_theme()

            if hasattr(self, 'category_focus_button'):
                self._apply_category_focus_button_style()

            # 更新CategoryPanel主题
            if hasattr(self, 'category_panel'):
                self.category_panel.apply_theme()

            # 更新工具栏展开条样式
            if hasattr(self, '_toolbar_expand_bar'):
                self._toolbar_expand_bar.setStyleSheet(f"""
                    QPushButton#toolbar_expand_bar {{
                        background-color: {c.BACKGROUND_SECONDARY};
                        color: {c.TEXT_SECONDARY};
                        border: 1px solid {c.BORDER_LIGHT};
                        border-radius: 3px;
                        font-size: 11px;
                        padding: 0px;
                    }}
                    QPushButton#toolbar_expand_bar:hover {{
                        background-color: {c.BACKGROUND_HOVER};
                        color: {c.PRIMARY};
                    }}
                """)

            # 更新ImageViewPanel主题
            if hasattr(self, 'image_view_panel'):
                self.image_view_panel.apply_theme()

            # 更新InfoPanel主题
            if hasattr(self, 'info_panel'):
                self.info_panel.apply_theme()

            # 更新统计面板
            if hasattr(self, 'statistics_panel') and hasattr(self.statistics_panel, 'apply_theme'):
                self.statistics_panel.apply_theme()

            tutorial_manager = getattr(self, 'tutorial_manager', None)
            if tutorial_manager and hasattr(tutorial_manager, 'bubble'):
                tutorial_manager.bubble.apply_theme()

            # 更新移除按钮（红色主题按钮）
            if hasattr(self, 'delete_button'):
                self.delete_button.setStyleSheet(f"""
                    QPushButton#remove_button {{
                        background-color: {c.ERROR};
                        color: white;
                        border: none;
                        border-radius: 4px;
                        font-size: 14px;
                        font-weight: normal;
                        text-align: center;
                    }}
                    QPushButton#remove_button:hover {{
                        background-color: {c.ERROR_DARK};
                    }}
                    QPushButton#remove_button:pressed {{
                        background-color: {c.ERROR_DARK};
                    }}
                """)

            # 更新EnhancedImageLabel背景
            if hasattr(self, 'image_label'):
                # 更新图像标签样式
                self.image_label.setStyleSheet(WidgetStyles.get_image_label_style())

                # 更新信息按钮样式
                if hasattr(self.image_label, 'info_button'):
                    self.image_label.info_button.setStyleSheet(WidgetStyles.get_info_button_style())

            # 更新所有QLabel的颜色
            if hasattr(self, 'findChildren'):
                for label in self.findChildren(QLabel):
                    current_style = label.styleSheet()
                    # 只更新没有特殊样式的标签
                    if "background-color: #FFF8E1" not in current_style and "color:" not in current_style:
                        label.setStyleSheet(f"QLabel {{ color: {c.TEXT_PRIMARY}; }}")

            # 更新状态栏样式
            if hasattr(self, 'statusBar'):
                self.statusBar.setStyleSheet(f"""
                    QStatusBar {{
                        background-color: {c.BACKGROUND_SECONDARY};
                        color: {c.TEXT_PRIMARY};
                        border-top: 1px solid {c.BORDER_MEDIUM};
                    }}
                    QStatusBar::item {{
                        border: none;
                    }}
                """)

            if hasattr(self, 'update_download_button'):
                self._apply_update_button_style(self._main_update_state)
            if hasattr(self, 'update_center_popover'):
                self.update_center_popover.apply_theme()

            # Phase 1.1 性能优化：只更新按钮样式，不重新创建（避免6秒卡顿）
            if hasattr(self, 'category_buttons') and self.category_buttons:
                # 不要调用 _update_category_buttons_internal()，那会重建所有按钮
                # 只更新现有按钮的标签颜色即可
                for button in self.category_buttons:
                    if hasattr(button, 'update_label_colors'):
                        button.update_label_colors()

            # 更新工具栏方形按钮样式（主题切换时需要刷新）
            toolbar_buttons = [
                ('mode_button', 'mode_button'),
                ('category_mode_button', 'category_mode_button'),
                ('refresh_button', 'refresh_button'),
                ('theme_button', 'theme_button'),
                ('settings_button', 'settings_button'),
                ('help_button', 'help_button'),
                ('filter_button', 'filter_button'),
                ('folder_button', 'folder_button'),
                ('_sidebar_toggle_btn', 'sidebar_toggle_button'),
                ('_toolbar_collapse_btn', 'toolbar_collapse_button'),
            ]
            for attr_name, object_name in toolbar_buttons:
                if hasattr(self, attr_name):
                    button = getattr(self, attr_name)
                    if button:
                        style_icon_button(button)

            self._update_ai_prediction_mode_button()

            # 强制重绘
            self.update()

        except Exception as e:
            self.logger.error(f"应用主题失败: {e}")

    def start_auto_theme_timer(self):
        """启动自动主题定时器"""
        if not self._auto_theme_timer.isActive():
            self._auto_theme_timer.start(60000)  # 每60秒（1分钟）检查一次
            self.logger.info("自动主题定时器已启动")

    def stop_auto_theme_timer(self):
        """停止自动主题定时器"""
        if self._auto_theme_timer.isActive():
            self._auto_theme_timer.stop()
            self.logger.info("自动主题定时器已停止")

    def _check_and_apply_auto_theme(self):
        """检查并应用自动主题（定时器回调）"""
        try:
            app_config = get_app_config()

            # 重新加载配置确保检测到模式变化（防止缓存不一致）
            app_config.reload_config()

            # 只有在自动模式下才进行检查
            if app_config.theme_mode != "auto":
                return

            # 根据时间获取应该使用的主题
            expected_theme = app_config.get_auto_theme_by_time()
            current_theme = app_config.theme

            # 如果主题需要切换
            if expected_theme != current_theme:
                self.logger.info(f"自动主题切换: {current_theme} -> {expected_theme}")

                # 更新配置
                app_config.theme = expected_theme
                default_theme.set_theme(expected_theme)

                # 应用主题到界面
                self.apply_theme()

                # 更新主题按钮图标
                if hasattr(self, 'theme_button'):
                    theme_icon = '☾' if expected_theme == "light" else '☼'
                    theme_tooltip = '切换到暗色主题' if expected_theme == "light" else '切换到亮色主题'
                    self.theme_button.setText(theme_icon)
                    self.theme_button.setToolTip(theme_tooltip)

        except Exception as e:
            self.logger.error(f"自动主题检查失败: {e}")

    def focusInEvent(self, event):
        """窗口获得焦点时的处理"""
        try:
            super().focusInEvent(event)
            self._shortcuts_active = True
            self._last_focus_time = time.time()
            self.logger.debug("窗口获得焦点，快捷键已激活")
        except Exception as e:
            self.logger.error(f"焦点获得事件处理失败: {e}")
    
    def focusOutEvent(self, event):
        """窗口失去焦点时的处理"""
        try:
            super().focusOutEvent(event)
            # 不立即禁用快捷键，给一个短暂的缓冲期
            QTimer.singleShot(500, self._check_focus_status)
            self.logger.debug("窗口失去焦点，延迟检查快捷键状态")
        except Exception as e:
            self.logger.error(f"焦点丢失事件处理失败: {e}")
    
    def _check_focus_status(self):
        """检查焦点状态"""
        try:
            if not self.isActiveWindow():
                self._shortcuts_active = False
                self.logger.debug("窗口非激活状态，快捷键已暂停")
            else:
                self._shortcuts_active = True
                self.logger.debug("窗口恢复激活状态，快捷键已重新激活")
        except Exception as e:
            self.logger.error(f"焦点状态检查失败: {e}")

    def _is_in_input_mode(self):
        """检测是否处于输入模式（有输入控件获得焦点）"""
        try:  
            focused_widget = self.focusWidget()
            if focused_widget:
                # 检查是否是输入控件
                input_widgets = (QLineEdit, QTextEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox, QComboBox)
                return isinstance(focused_widget, input_widgets)
            return False
        except Exception:
            return False

    # ===== 更新相关 =====
    def _connect_update_download_controller(self):
        """连接应用级下载状态，并恢复状态栏入口。"""
        controller = self.update_download_controller
        controller.progress_changed.connect(self._on_update_download_progress)
        controller.state_changed.connect(self._on_update_download_state)
        controller.download_completed.connect(self._on_update_download_completed)
        self._on_update_download_state(controller.state, controller.message)
        if controller.downloaded or controller.total:
            self._on_update_download_progress(
                controller.downloaded,
                controller.total,
            )

    def _on_update_download_completed(self, metadata: dict):
        """下载完成后提示用户通过统一入口决定何时重启。"""
        version = str(metadata.get("version", "")).strip()
        if version:
            self._latest_checked_version = version
        self._sync_update_popover()
        message = (
            f"更新 v{version} 已下载，可点击右下角版本按钮完成安装"
            if version
            else "更新包已下载，可点击右下角版本按钮完成安装"
        )
        toast_success(
            self,
            message,
        )

    def _on_update_download_progress(self, done: int, total: int):
        """更新状态栏中的后台下载进度入口。"""
        if not hasattr(self, 'update_download_button'):
            return
        state = self.update_download_controller.state
        if state == "ready":
            self._set_main_update_state(
                "ready",
                tooltip=self.update_download_controller.message,
            )
            return
        self._main_update_state = state
        if total > 0:
            percentage = min(100, int(done * 100 / total))
            if state == "verifying":
                prefix = "校验"
            elif state == "paused":
                prefix = "继续更新"
            elif state == "retrying":
                prefix = "重试更新"
            else:
                prefix = "更新"
            self.update_download_button.setText(
                f"v{__version__} · {prefix} {percentage}%"
            )
        else:
            self.update_download_button.setText(
                f"v{__version__} · "
                + ("已暂停" if state == "paused" else "下载中")
            )
        self._apply_update_button_style(state)
        self._sync_update_popover()

    def _on_update_download_state(self, state: str, message: str):
        """把下载控制器状态映射到右下角统一更新入口。"""
        if not hasattr(self, 'update_download_button'):
            return
        if state == "ready":
            self._clear_pending_update()
            self._set_main_update_state("ready", tooltip=message)
        elif state in {
            "downloading",
            "retrying",
            "verifying",
            "pausing",
            "cancelling",
        }:
            self._main_update_state = state
            self.update_download_button.setToolTip(message or "点击查看下载进度")
            if state == "verifying":
                self.update_download_button.setText(f"v{__version__} · 校验中")
            elif state == "retrying":
                self.update_download_button.setText(f"v{__version__} · 重试中")
            elif state == "pausing":
                self.update_download_button.setText(f"v{__version__} · 暂停中")
            elif state == "cancelling":
                self.update_download_button.setText(f"v{__version__} · 取消中")
            else:
                self.update_download_button.setText(f"v{__version__} · 下载中")
            self._apply_update_button_style(state)
        elif state == "paused":
            self._main_update_state = "paused"
            self.update_download_button.setText(f"v{__version__} · 已暂停")
            self.update_download_button.setToolTip(
                message or "更新已暂停，点击查看并继续下载"
            )
            self._apply_update_button_style("paused")
        elif state == "failed":
            self._set_main_update_state("failed", tooltip=message)
        elif state == "cancelled" and self._pending_update_manifest:
            self._set_main_update_state(
                "available",
                self._pending_update_version,
                "下载已取消，点击可重新下载",
            )
        else:
            self._set_main_update_state("idle")
        self._sync_update_popover()

    def _set_main_update_state(
        self,
        state: str,
        version: Optional[str] = None,
        tooltip: str = "",
    ):
        """设置右下角更新按钮的状态、文字和样式。"""
        self._main_update_state = state
        self._main_update_message = tooltip
        if version:
            self._latest_checked_version = str(version)
        if not hasattr(self, 'update_download_button'):
            return
        self.update_download_button.setEnabled(True)
        if state == "checking":
            self.update_download_button.setText(f"v{__version__} · 检查中")
            self.update_download_button.setToolTip("正在检查更新，点击查看")
        elif state == "available":
            label = f"v{__version__} · v{version} 可用" if version else f"v{__version__} · 有更新"
            self.update_download_button.setText(label)
            self.update_download_button.setToolTip(
                tooltip or "点击打开更新中心"
            )
        elif state == "ready":
            self.update_download_button.setText(f"v{__version__} · 重启更新")
            self.update_download_button.setToolTip(
                tooltip or "更新已下载，点击重启安装"
            )
        elif state == "failed":
            self.update_download_button.setText(f"v{__version__} · 更新异常")
            self.update_download_button.setToolTip(
                tooltip or "点击查看失败原因"
            )
        else:
            self.update_download_button.setText(f"v{__version__}")
            self.update_download_button.setToolTip("打开更新中心")
        self._apply_update_button_style(state)
        self._sync_update_popover()

    def _apply_update_button_style(self, state: str):
        """应用统一入口在不同更新状态下的颜色。"""
        c = default_theme.colors
        colors = {
            "idle": (c.BACKGROUND_CARD, c.BACKGROUND_HOVER, c.TEXT_PRIMARY, c.BORDER_MEDIUM),
            "checking": (c.PRIMARY, c.PRIMARY_DARK, c.TEXT_ON_PRIMARY, c.PRIMARY),
            "available": (c.WARNING, c.WARNING_DARK, c.TEXT_ON_PRIMARY, c.WARNING),
            "downloading": (c.PRIMARY, c.PRIMARY_DARK, c.TEXT_ON_PRIMARY, c.PRIMARY),
            "retrying": (c.PRIMARY, c.PRIMARY_DARK, c.TEXT_ON_PRIMARY, c.PRIMARY),
            "verifying": (c.PRIMARY, c.PRIMARY_DARK, c.TEXT_ON_PRIMARY, c.PRIMARY),
            "pausing": (c.WARNING, c.WARNING_DARK, c.TEXT_ON_PRIMARY, c.WARNING),
            "paused": (c.WARNING, c.WARNING_DARK, c.TEXT_ON_PRIMARY, c.WARNING),
            "cancelling": (c.BACKGROUND_CARD, c.BACKGROUND_HOVER, c.TEXT_PRIMARY, c.BORDER_MEDIUM),
            "ready": (c.SUCCESS, c.SUCCESS_DARK, c.TEXT_ON_PRIMARY, c.SUCCESS),
            "failed": (c.ERROR, c.ERROR_DARK, c.TEXT_ON_PRIMARY, c.ERROR),
        }
        normal, hover, text_color, border = colors.get(state, colors["idle"])
        self.update_download_button.setProperty("uiState", state)
        self.update_download_button.setFixedHeight(
            default_theme.sizes.BUTTON_HEIGHT_COMPACT_PX
        )
        self.update_download_button.setMinimumWidth(
            default_theme.sizes.BUTTON_MIN_WIDTH_PX
        )
        self.update_download_button.setStyleSheet(f"""
            QPushButton {{
                border: 1px solid {border};
                border-radius: 6px;
                padding: 0 12px;
                color: {text_color};
                background-color: {normal};
                font-size: 11px;
                font-weight: 500;
            }}
            QPushButton:hover {{
                background-color: {hover};
            }}
        """)
        self.update_download_button.show()

    def _sync_update_popover(self):
        """把唯一更新状态同步到版本按钮的锚定浮层。"""
        popover = getattr(self, "update_center_popover", None)
        if popover is None:
            return
        controller = self.update_download_controller
        active_states = set(
            getattr(
                controller,
                "ACTIVE_STATES",
                {"downloading", "retrying", "verifying", "pausing", "cancelling"},
            )
        )
        state = self._main_update_state
        if controller.state in active_states or controller.state in {
            "paused",
            "ready",
        }:
            state = controller.state
        latest_version = str(
            getattr(self, "_latest_checked_version", "")
            or getattr(controller, "_version", "")
            or getattr(self, "_pending_update_version", "")
            or ""
        )
        message = (
            controller.message
            if state in active_states | {"paused", "ready"}
            else getattr(self, "_main_update_message", "")
        )
        popover.set_state(
            state,
            __version__,
            latest_version,
            message,
            controller.downloaded,
            controller.total,
            can_download=bool(self._pending_update_manifest),
        )

    def _advance_update_check_animation(self):
        """只在更新中心内部显示手动检查动画。"""
        if self._main_update_state != "checking":
            self._update_check_animation_timer.stop()
            return
        self._update_check_animation_step = (
            self._update_check_animation_step + 1
        ) % 3
        popover = getattr(self, "update_center_popover", None)
        if popover is not None:
            popover.set_checking_step(self._update_check_animation_step)

    def _handle_update_button_clicked(self):
        """版本按钮只负责打开或收起唯一更新中心。"""
        popover = self.update_center_popover
        if popover.isVisible():
            popover.hide()
            return
        self._sync_update_popover()
        popover.show_anchored(self.update_download_button)

    def _start_pending_update_download(self):
        """从更新中心启动当前线上版本下载。"""
        manifest = self._pending_update_manifest
        version = self._pending_update_version
        if not manifest or not version:
            self._set_main_update_state("failed", tooltip="更新信息已失效，请重新检查")
            return
        if not self.update_download_controller.start(
            manifest,
            version,
        ):
            self._sync_update_popover()

    def _restart_and_install_update(self):
        """在浮层内确认后启动安装。"""
        if not self.update_download_controller.install_ready_update():
            self._set_main_update_state(
                "failed",
                tooltip=self.update_download_controller.message,
            )
    def _schedule_auto_update_check(self):
        """调度固定启用的自动检查更新。"""
        try:
            # 启动后5秒首次检查
            QTimer.singleShot(5000, self._auto_check_update_once)

            # 启动定期检查定时器（每1小时）
            self._start_periodic_update_check()
        except Exception as e:
            self.logger.debug(f"调度自动检查更新失败: {e}")

    def _start_periodic_update_check(self):
        """启动定期检查更新定时器（每1小时检查一次）"""
        try:
            # 创建定时器，每1小时（3600000毫秒）触发一次
            self.periodic_update_timer = QTimer(self)
            self.periodic_update_timer.timeout.connect(self._periodic_check_update)
            self.periodic_update_timer.start(3600000)  # 1小时 = 1 * 60 * 60 * 1000ms

            self.logger.info("定期检查更新已启动：每1小时检查一次")
        except Exception as e:
            self.logger.debug(f"启动定期检查更新定时器失败: {e}")

    def _periodic_check_update(self):
        """定期检查更新（每1小时触发）"""
        try:
            self.logger.info("定期检查更新：开始执行")
            self._auto_check_update_once()
        except Exception as e:
            self.logger.debug(f"定期检查更新失败: {e}")

    def _on_update_check_success(self, manifest, endpoint, token):
        """后台更新检查成功后更新状态栏，不主动弹出更新窗口。"""
        manual = self._update_check_manual
        self._finish_main_update_check()
        try:
            online_version = str(manifest.get('version', '')).strip()
            if not online_version:
                self._process_update_result(
                    None,
                    None,
                    endpoint,
                    token,
                    manual=manual,
                    error_message="更新信息格式错误",
                )
                return
            self.logger.info(f"检查线上更新：发现版本 v{online_version}")
            self._process_update_result(
                online_version,
                manifest,
                endpoint,
                token,
                manual=manual,
            )
        except Exception as e:
            self.logger.error(f"处理更新检查结果失败: {e}")
            if manual:
                self._set_main_update_state("failed", tooltip=str(e))
                toast_error(self, f"处理更新失败: {e}")

    def _on_update_check_failed(self, error_message):
        """后台更新失败时仍优先恢复已下载更新包的状态。"""
        manual = self._update_check_manual
        self._finish_main_update_check()
        self.logger.debug(f"检查线上更新失败: {error_message}")
        self._process_update_result(
            None,
            None,
            None,
            None,
            manual=manual,
            error_message=error_message,
        )

    def _auto_check_update_once(self):
        """执行一次静默检查；结果只进入右下角统一入口。"""
        self._start_main_update_check(manual=False)

    def _start_main_update_check(self, manual: bool = False):
        """启动主窗口更新检查，并记录本地完整更新包。"""

        if self.update_download_controller.is_active:
            self.logger.debug("更新包正在下载，跳过重复检查")
            if manual:
                self._sync_update_popover()
            return
        if self.update_download_controller.state == "paused":
            self._on_update_download_state(
                "paused",
                self.update_download_controller.message,
            )
            if manual:
                self._sync_update_popover()
            return

        if self.update_checker_thread and self.update_checker_thread.isRunning():
            if manual:
                self._update_check_manual = True
                self._set_main_update_state("checking")
                self._update_check_animation_step = 0
                self._update_check_animation_timer.start()
                toast_info(self, "正在检查更新...")
            return

        self._update_check_manual = manual
        if manual:
            self._set_main_update_state("checking")
            self._update_check_animation_step = 0
            self._update_check_animation_timer.start()
            toast_info(self, "正在检查更新...")

        local_pending_version = None
        local_download_path = None
        local_batch_path = None

        try:
            update_dir = get_update_dir()
            self.logger.debug(f"检查本地更新目录: {update_dir}")
            ready_update = load_ready_update(update_dir)
            if ready_update:
                local_pending_version = str(ready_update['version'])
                local_download_path = ready_update['path']
                batch_path = update_dir / 'update.bat'
                local_batch_path = batch_path if batch_path.exists() else None
                self.logger.info(
                    f"检测到已校验更新: version={local_pending_version}, "
                    f"exe={local_download_path.name}, batch={local_batch_path}"
                )
        except Exception as e:
            self.logger.debug(f"检查本地更新目录失败: {e}")

        self._local_update_info = {
            'version': local_pending_version,
            'path': local_download_path,
            'batch_path': local_batch_path,
        }

        try:
            self.logger.debug("检查线上更新：开始（后台线程）")
            endpoint = get_manifest_url(latest=True)
            proxy = get_app_config().update_proxy
            self.update_checker_thread = UpdateCheckerThread(
                endpoint,
                "",
                proxy,
            )
            self.update_checker_thread.check_success.connect(self._on_update_check_success)
            self.update_checker_thread.check_failed.connect(self._on_update_check_failed)
            self.update_checker_thread.start()
            self.logger.debug("后台更新检查线程已启动")
        except Exception as e:
            self.logger.debug(f"启动后台更新检查失败: {e}")
            self._on_update_check_failed(str(e))

    def _finish_main_update_check(self):
        """结束按钮检查动画并清理本次检查标记。"""
        self._update_check_animation_timer.stop()
        self._update_check_animation_step = 0
        self._update_check_manual = False

    def _clear_pending_update(self):
        """清理供右下角入口展示的线上更新信息。"""
        self._pending_update_version = None
        self._pending_update_manifest = None

    def _remember_pending_update(self, version, manifest, _token=None):
        """保存线上更新信息，等待用户点击状态栏入口。"""
        self._pending_update_version = str(version)
        self._pending_update_manifest = manifest

    def _discard_local_ready_update(self, local_batch_path=None):
        """删除不再适用的完整更新包及遗留安装脚本。"""
        self.update_download_controller.discard_ready_update()
        if local_batch_path and local_batch_path.exists():
            local_batch_path.unlink()

    def _process_update_result(
        self,
        online_version,
        manifest,
        endpoint,
        token,
        manual: Optional[bool] = None,
        error_message: str = "",
    ):
        """合并本地完整包和线上版本，更新右下角入口。"""
        if manual is None:
            manual = self._update_check_manual

        local_info = getattr(self, '_local_update_info', {})
        local_pending_version = local_info.get('version')
        local_download_path = local_info.get('path')
        local_batch_path = local_info.get('batch_path')

        if local_pending_version and local_download_path:
            try:
                local_vs_current = compare_version(
                    local_pending_version,
                    __version__,
                )
            except Exception as e:
                self.logger.warning(f"本地更新版本无效，清理更新包: {e}")
                local_vs_current = 0

            if local_vs_current <= 0:
                self.logger.info(
                    f"本地更新包v{local_pending_version}不高于当前版本"
                    f"v{__version__}，清理更新包"
                )
                try:
                    self._discard_local_ready_update(local_batch_path)
                except Exception as e:
                    self.logger.warning(f"清理本地更新包失败: {e}")
                local_pending_version = None
                local_download_path = None
            elif (
                online_version
                and compare_version(online_version, local_pending_version) > 0
            ):
                self.logger.info(
                    f"线上版本v{online_version}比本地v{local_pending_version}更新，"
                    "清理旧包并等待用户查看新版本"
                )
                try:
                    self._discard_local_ready_update(local_batch_path)
                except Exception as e:
                    self.logger.warning(f"清理旧更新包失败: {e}")
                local_pending_version = None
                local_download_path = None
            else:
                self._clear_pending_update()
                self._set_main_update_state(
                    "ready",
                    local_pending_version,
                    f"更新 v{local_pending_version} 已下载，点击重启安装",
                )
                if manual:
                    toast_info(
                        self,
                        f"更新 v{local_pending_version} 已下载，可重启更新",
                    )
                return

        if online_version:
            try:
                self._latest_checked_version = str(online_version)
                cmp_result = compare_version(online_version, __version__)
                if cmp_result > 0 and manifest:
                    already_pending = (
                        self._pending_update_version == str(online_version)
                        and self._pending_update_manifest is not None
                    )
                    self._remember_pending_update(
                        online_version,
                        manifest,
                        token,
                    )
                    self._set_main_update_state(
                        "available",
                        online_version,
                        "点击打开更新中心并下载",
                    )
                    if manual or not already_pending:
                        toast_info(self, f"发现新版本 v{online_version}")
                    return
                self.logger.debug(f"当前版本已是最新: v{__version__}")
                self._clear_pending_update()
                self._set_main_update_state("idle")
                if manual:
                    if cmp_result == 0:
                        toast_success(self, f"当前已是最新版本 v{__version__}")
                    else:
                        toast_info(
                            self,
                            f"当前版本 v{__version__} 高于线上版本 v{online_version}",
                        )
                return
            except Exception as e:
                error_message = str(e)
                self.logger.error(f"处理线上更新失败: {e}")

        if manual:
            failure_message = error_message or "无法获取更新信息"
            self._set_main_update_state("failed", tooltip=failure_message)
            toast_warning(self, failure_message)
    
    def keyPressEvent(self, event):
        """处理键盘事件 - 优化按键处理和日志记录"""
        try:
            # 检查快捷键是否激活
            if not self._shortcuts_active:
                self.logger.debug("快捷键未激活，尝试重新激活")
                self._shortcuts_active = True
                
            key = event.key()
            modifiers = event.modifiers()
            
            # 检查是否在输入模式
            in_input_mode = self._is_in_input_mode()
            
            # 过滤掉纯修饰键，避免日志噪音和冲突
            modifier_keys = {
                Qt.Key.Key_Control, Qt.Key.Key_Shift, Qt.Key.Key_Alt,
                Qt.Key.Key_Meta, Qt.Key.Key_AltGr, Qt.Key.Key_CapsLock,
                Qt.Key.Key_NumLock, Qt.Key.Key_ScrollLock
            }
            
            # 记录按键信息 - 分级别记录
            if not in_input_mode:  # 只在非输入模式下记录按键
                key_text = event.text() or f"Key_{key}"
                modifier_text = []
                if modifiers & Qt.KeyboardModifier.ControlModifier:
                    modifier_text.append("Ctrl")
                if modifiers & Qt.KeyboardModifier.ShiftModifier:
                    modifier_text.append("Shift")
                if modifiers & Qt.KeyboardModifier.AltModifier:
                    modifier_text.append("Alt")
                
                modifier_str = "+".join(modifier_text)
                full_key = f"{modifier_str}+{key_text}" if modifier_str else key_text
                
                # DEBUG级别：记录所有按键（包括修饰键和不支持的按键）
                if key in modifier_keys:
                    self.logger.debug(f"修饰键: {full_key} (code: {key})")
                else:
                    self.logger.debug(f"按键: {full_key} (code: {key}) - 快捷键状态: {'激活' if self._shortcuts_active else '禁用'}")

                # INFO级别：只记录已定义的快捷键
                if self._is_defined_shortcut(key, modifiers):
                    self.logger.info(f"快捷键触发: {full_key}")
            
            # 处理修饰键
            if key in modifier_keys:
                # 纯修饰键不处理，直接传递给父类
                super().keyPressEvent(event)
                return
            
            # 直接传递给父类让QAction系统处理所有快捷键
            super().keyPressEvent(event)
            
        except Exception as e:
            self.logger.error(f"键盘事件处理失败: {e}")
            super().keyPressEvent(event)

    def focusNextPrevChild(self, next_child: bool) -> bool:
        """Tab handling is centralized in AITabEventFilter."""
        return super().focusNextPrevChild(next_child)
    
    def select_previous_category(self):
        """选择上一个类别"""
        try:
            if not hasattr(self, 'current_category_index'):
                self.current_category_index = -1
            
            if self.category_buttons:
                self.current_category_index = (self.current_category_index - 1) % len(self.category_buttons)
                self.logger.debug(f"选择上一个类别: {self.current_category_index}/{len(self.category_buttons)}")
                self.highlight_selected_category()
                
        except Exception as e:
            self.logger.error(f"选择上一个类别失败: {e}")
    
    def select_next_category(self):
        """选择下一个类别"""
        try:
            if not hasattr(self, 'current_category_index'):
                self.current_category_index = -1
            
            if self.category_buttons:
                self.current_category_index = (self.current_category_index + 1) % len(self.category_buttons)
                self.logger.debug(f"选择下一个类别: {self.current_category_index}/{len(self.category_buttons)}")
                self.highlight_selected_category()
                
        except Exception as e:
            self.logger.error(f"选择下一个类别失败: {e}")
    
    def highlight_selected_category(self):
        """高亮选中的类别按钮"""
        try:
            self.logger.debug(f"高亮类别按钮: {self.current_category_index}/{len(self.category_buttons) if hasattr(self, 'category_buttons') else 0}")
            
            # 重置所有按钮样式
            for i, button in enumerate(self.category_buttons):
                if i == self.current_category_index:
                    # 高亮选中的按钮
                    button.setStyleSheet("""
                        QPushButton {
                            background-color: #0078d4;
                            color: white;
                            border: 2px solid #005a9e;
                            font-weight: bold;
                        }
                    """)
                    category_name = button.category_name
                    self.statusBar.showMessage(f"🎯 选中类别: {category_name} (按Enter确认)")
                    self.logger.debug(f"高亮类别按钮: {category_name}")
                else:
                    # 恢复正常样式
                    button.setStyleSheet("")
                        
        except Exception as e:
            self.logger.error(f"高亮类别失败: {e}")
    
    def rename_category(self, old_name, new_name):
        """重命名类别（通过CategoryManager）"""
        if not self._category_manager:
            self.logger.error("CategoryManager未初始化")
            return
        was_focused = self.focused_category == old_name
        previous_filter = self._category_focus_previous_classified_filter
        renamed = self._category_manager.rename_category(old_name, new_name)
        if renamed and was_focused:
            self.focused_category = new_name
            self._category_focus_previous_classified_filter = previous_filter
            self.filter_classified = True
            self._update_category_focus_button()
            self.apply_image_filter()
        # Manager会触发categories_changed信号，UI刷新在on_categories_changed中处理

    def ignore_category(self, category_name):
        """忽略类别 - 不删除目录，只是不显示

        注意：忽略类别时保留分类记录，这样恢复类别后分类状态自动恢复
        """
        try:
            if not self.current_dir:
                toast_error(self, "当前目录未设置")
                return

            # 添加到忽略列表
            if self.config.add_ignored_category(category_name):
                if self.focused_category == category_name:
                    self._deactivate_category_focus()

                # 从快捷键配置中移除该类别
                if category_name in self.config.category_shortcuts:
                    del self.config.category_shortcuts[category_name]

                # 保存配置（不保存分类状态，保留分类记录以便恢复）
                self.config.save_config()

                # 重新加载类别（会自动过滤掉忽略的类别）
                self.load_categories()

                # 刷新UI
                self._pending_reapply_filter = self.is_image_filter_active()
                self.schedule_ui_update('category_buttons', 'category_counts', 'image_list', 'statistics')

                # 显示成功提示
                toast_success(self, f"类别已忽略: {category_name}")
                self.logger.info(f"类别已忽略: {category_name}")
            else:
                toast_warning(self, f"类别 '{category_name}' 已经在忽略列表中")

        except Exception as e:
            self.logger.error(f"忽略类别失败: {e}", exc_info=True)
            toast_error(self, f"忽略类别失败: {str(e)}")

    def show_manage_ignored_dialog(self):
        """显示管理忽略列表对话框"""
        try:

            dialog = ManageIgnoredCategoriesDialog(self.config, self)
            dialog.exec()

        except Exception as e:
            self.logger.error(f"显示管理忽略对话框失败: {e}", exc_info=True)
            toast_error(self, f"显示对话框失败: {str(e)}")

    def delete_category(self, category_name):
        """删除类别（通过CategoryManager）"""
        if not self._category_manager:
            self.logger.error("CategoryManager未初始化")
            return
        was_focused = self.focused_category == category_name
        previous_filter = self._category_focus_previous_classified_filter
        if was_focused:
            # CategoryManager 删除成功后会立即重算过滤列表，先临时退出聚焦，
            # 避免它使用一个已不存在的类别得到空列表。
            self._deactivate_category_focus()

        deleted = self._category_manager.delete_category(category_name)
        if not deleted and was_focused:
            self.focused_category = category_name
            self._category_focus_previous_classified_filter = previous_filter
            self.filter_classified = True
            self._update_category_focus_button()
        # Manager会触发categories_changed信号，UI刷新在on_categories_changed中处理
    
    def save_state(self):
        """合并高频保存请求，在用户操作间隙写入一次最新状态。"""
        try:
            if not self.current_dir:
                return

            # 状态文件保存在图片目录的父目录（同级目录），而不是图片目录内
            parent_dir = self.current_dir.parent
            self._pending_state_file = parent_dir / 'classification_state.json'

            timer = getattr(self, '_state_save_timer', None)
            if timer is None:
                self._flush_pending_state_save()
            else:
                # 重复 start 会重启单次定时器，只保留最后一次请求。
                timer.start()
                max_timer = getattr(self, '_state_save_max_timer', None)
                if max_timer is not None and not max_timer.isActive():
                    max_timer.start()
        except Exception as e:
            self.logger.error(f"准备保存状态失败: {e}")

    def _build_state_data(self) -> dict:
        """在真正写盘前创建一次最新状态快照。"""
        return {
            'classified_images': dict(self.classified_images),
            'removed_images': list(self.removed_images),
            'last_index': self.current_index,
            'version': self.version,
            'is_copy_mode': self.is_copy_mode,
            'is_multi_category': self.is_multi_category,
            AI_PROJECT_STATE_KEY: getattr(
                self, "_ai_project_state", default_ai_project_state()
            ),
        }

    def _flush_pending_state_save(self) -> None:
        """写入合并后的最后一个状态保存请求。"""
        state_file = getattr(self, '_pending_state_file', None)
        if state_file is None:
            return
        timer = getattr(self, '_state_save_timer', None)
        if timer is not None:
            timer.stop()
        max_timer = getattr(self, '_state_save_max_timer', None)
        if max_timer is not None:
            max_timer.stop()
        self._pending_state_file = None
        self._async_save_state(state_file, self._build_state_data())

    def _save_state_sync(self):
        """立即同步保存当前状态到图片同级目录（用于重要状态变化）"""
        try:
            if not self.current_dir:
                return

            timer = getattr(self, '_state_save_timer', None)
            if timer is not None:
                timer.stop()
            max_timer = getattr(self, '_state_save_max_timer', None)
            if max_timer is not None:
                max_timer.stop()
            self._pending_state_file = None
            state_data = self._build_state_data()

            # 状态文件保存在图片目录的父目录（同级目录），而不是图片目录内
            parent_dir = self.current_dir.parent
            state_file = parent_dir / 'classification_state.json'

            # 立即同步保存
            with open(state_file, 'w', encoding='utf-8') as f:
                json.dump(state_data, f, ensure_ascii=False, indent=2)

            self.logger.info(f"模式状态已同步保存到: {state_file}")

        except Exception as e:
            self.logger.error(f"同步保存状态失败: {e}")

    def _async_save_state(self, state_file, state_data):
        """执行合并后的状态写入。"""
        try:
            with open(state_file, 'w', encoding='utf-8') as f:
                json.dump(state_data, f, ensure_ascii=False, indent=2)
                
            self.logger.debug(f"状态已异步保存到: {state_file}")
                
        except Exception as e:
            self.logger.error(f"异步保存状态失败: {e}")
            # 如果异步保存失败，尝试同步保存作为备份
            try:
                with open(state_file, 'w', encoding='utf-8') as f:
                    json.dump(state_data, f, ensure_ascii=False, indent=2)
                self.logger.info(f"备份同步保存成功: {state_file}")
            except Exception as backup_error:
                self.logger.error(f"备份保存也失败: {backup_error}")

    def closeEvent(self, event):
        """程序关闭前的清理和保存（简化版）"""
        try:
            self.logger.info("程序正在关闭，开始清理...")

            # 先停止更新下载并等待网络、文件句柄释放，避免窗口关闭后仍写文件。
            controller = getattr(self, 'update_download_controller', None)
            if controller and not controller.shutdown(timeout_ms=10000):
                self.logger.warning("更新下载尚未安全停止，暂缓关闭程序")
                event.ignore()
                return

            ai_downloads = getattr(
                self, '_ai_resource_download_manager', None
            )
            if ai_downloads and not ai_downloads.shutdown(timeout_ms=10000):
                self.logger.warning("AI 资源下载尚未安全停止，暂缓关闭程序")
                event.ignore()
                return

            # 保存窗口位置和大小
            self._save_window_geometry()

            # 停止定时器
            if hasattr(self, 'ui_update_timer'):
                self.ui_update_timer.stop()
            if hasattr(self, '_state_save_timer'):
                self._state_save_timer.stop()
            if hasattr(self, '_state_save_max_timer'):
                self._state_save_max_timer.stop()

            if getattr(self, '_ai_manager', None):
                self._ai_manager.shutdown()

            # 停止后台线程
            if hasattr(self, 'file_scanner') and self.file_scanner:
                try:
                    if hasattr(self.file_scanner, 'cancel_scan'):
                        self.file_scanner.cancel_scan()
                except Exception as e:
                    self.logger.warning(f"停止文件扫描器失败: {e}")

            # 最终保存（单次，失败只记录日志）
            try:
                if hasattr(self, 'config') and self.config:
                    self.config.save_config()
                if hasattr(self, 'current_dir') and self.current_dir:
                    self._save_state_sync()
                self.logger.info("最终状态保存完成")
            except Exception as e:
                self.logger.error(f"最终保存失败: {e}")

            self.logger.info("程序清理完成")

        except Exception as e:
            self.logger.error(f"closeEvent 处理失败: {e}")
        event.accept()

    def _check_wechat_directory(self, path: Path) -> bool:
        """检测是否为微信目录"""
        path_str = str(path).lower()
        return 'wechat' in path_str or '\\msg\\file\\' in path_str or '/msg/file/' in path_str

    def _show_wechat_warning(self):
        """显示微信目录警告"""
        msg_box = ThemedMessageBox(self)
        msg_box.setWindowTitle("⚠️ 微信目录警告")
        msg_box.setIcon(QMessageBox.Icon.Warning)
        msg_box.setText("检测到您正在微信的文件目录中工作")
        msg_box.setInformativeText(
            "微信会监控和锁定此目录中的文件，可能导致：\n"
            "• 配置保存失败\n"
            "• 状态保存失败\n"
            "• 文件操作权限错误\n\n"
            "建议：\n"
            "1. 将图片复制到其他目录（如 D:\\图片分类\\）\n"
            "2. 或关闭微信后使用\n\n"
            "继续使用可能会遇到权限错误，但不影响基本功能。"
        )

        msg_box.exec()

    def _create_styled_message_box(self, icon_type, title, text, buttons=None):
        """创建具有统一样式和中文按钮的消息框"""

        msgBox = ThemedMessageBox(self)
        msgBox.setIcon(icon_type)
        msgBox.setWindowTitle(title)
        msgBox.setText(text)

        # 设置程序图标
        try:
            icon_name = get_product_info()["icon_basename"]
            icon_path = self._get_resource_path(f'assets/{icon_name}.ico')
            if icon_path and icon_path.exists():
                msgBox.setWindowIcon(QIcon(str(icon_path)))
        except Exception:
            pass

        # 设置中文按钮
        if buttons is None:
            msgBox.setStandardButtons(QMessageBox.StandardButton.Ok)
            msgBox.button(QMessageBox.StandardButton.Ok).setText("确定")
        else:
            msgBox.setStandardButtons(buttons)
            # 中文化按钮文本
            if msgBox.button(QMessageBox.StandardButton.Ok):
                msgBox.button(QMessageBox.StandardButton.Ok).setText("确定")
            if msgBox.button(QMessageBox.StandardButton.Cancel):
                msgBox.button(QMessageBox.StandardButton.Cancel).setText("取消")
            if msgBox.button(QMessageBox.StandardButton.Yes):
                msgBox.button(QMessageBox.StandardButton.Yes).setText("是")
            if msgBox.button(QMessageBox.StandardButton.No):
                msgBox.button(QMessageBox.StandardButton.No).setText("否")
            if msgBox.button(QMessageBox.StandardButton.Apply):
                msgBox.button(QMessageBox.StandardButton.Apply).setText("应用")
            if msgBox.button(QMessageBox.StandardButton.Close):
                msgBox.button(QMessageBox.StandardButton.Close).setText("关闭")

        return msgBox
      
    def show_question_message(self, title, text):
        """显示询问消息框"""
        msgBox = self._create_styled_message_box(
            QMessageBox.Icon.Question,
            title,
            text,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        msgBox.setDefaultButton(QMessageBox.StandardButton.No)
        return msgBox.exec()

    def refresh_category_buttons_style(self):
        """强制刷新所有类别按钮的样式"""
        # 使用 CategoryPanel 刷新按钮样式
        if hasattr(self, 'category_panel'):
            self.category_panel.refresh_buttons_style()
            self.logger.debug("强制刷新类别按钮样式")
