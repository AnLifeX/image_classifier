<p align="right">
  <a href="./README.md">English</a> | <strong>简体中文</strong>
</p>

# 图像分类工具 (Image Classifier)

<div align="center">

![Version](https://img.shields.io/badge/version-7.0.0-blue.svg)
![Python](https://img.shields.io/badge/python-3.8+-green.svg)
![Platform](https://img.shields.io/badge/platform-Windows-lightgrey.svg)
![License](https://img.shields.io/badge/license-MIT-orange.svg)

一个高性能的图像分类整理工具，支持智能预加载、网络路径优化、多种图像格式、自定义快捷键等功能。

[快速下载](#快速下载) • [功能特性](#功能特性) • [安装使用](#安装使用) • [开发指南](#开发指南) • [构建发布](#构建发布)

</div>

## 项目简介

图像分类工具是一个基于 PyQt6 开发的桌面应用程序，专为需要快速整理和分类大量图片的用户设计。该工具具有直观的用户界面，支持多种图像格式，提供了高效的图像预加载和智能分类功能。

## 快速下载

**最新版本**: [访问 Releases 页面下载](https://github.com/AnLifeX/image_classifier/releases)

| 版本 | 平台 | 下载链接 | 大小 |
|------|------|----------|------|
| Latest | Windows | [Releases 页面](https://github.com/AnLifeX/image_classifier/releases/latest) | ~86MB |

> 下载 exe 文件后双击即可运行，无需安装 Python 环境。

### 适用场景

- 摄影师整理作品集
- 设计师管理素材库
- 个人照片分类整理
- 企业图片资产管理
- 大批量图片快速筛选

## 功能特性

### 核心功能

- **多格式支持**: 支持 JPG、JPEG、PNG、BMP、GIF、TIFF、WebP 等主流图像格式
- **智能预加载**: 高性能图像加载机制，支持大图片快速预览
- **快速分类**: 自定义分类按钮，支持快捷键操作
- **网络路径优化**: 针对网络共享路径进行性能优化
- **批量处理**: 支持文件夹扫描和批量图片操作
- **状态保存**: 自动保存工作进度和用户设置

### 用户界面

- 现代化设计，简洁美观，支持高DPI显示
- 双面板布局：图片列表 + 分类操作区域
- 高质量图片预览，支持缩放和全屏查看
- 实时显示扫描和处理进度
- 亮色/暗色主题切换

### 操作特性

- 自定义快捷键：为每个分类设置专属快捷键（1-9, A-Z）
- 键盘导航：支持方向键浏览图片和选择类别
- 多种分类模式：支持移动/复制模式，单分类/多分类模式切换
- 右键菜单：丰富的上下文菜单操作

## 技术栈

| 组件 | 技术 | 版本要求 | 用途 |
|------|------|----------|------|
| **GUI框架** | PyQt6 | >= 6.4.0 | 用户界面框架 |
| **图像处理** | OpenCV | >= 4.5.0 | 图像读取和处理 |
| **图像库** | Pillow | >= 9.0.0 | 图像格式支持 |
| **系统监控** | psutil | >= 5.8.0 | 系统资源监控 |
| **打包工具** | PyInstaller | >= 5.0.0 | 应用程序打包 |

## 系统要求

### 最低要求
- **操作系统**: Windows 10 或更高版本
- **Python版本**: Python 3.8+
- **内存**: 4GB RAM
- **存储空间**: 200MB 可用空间

### 推荐配置
- **操作系统**: Windows 10/11 64位
- **Python版本**: Python 3.10+
- **内存**: 8GB RAM 或更多
- **存储空间**: 1GB 可用空间
- **显示器**: 1920x1080 分辨率或更高

## 安装使用

### 方式一：直接运行（推荐）

下载已编译的可执行文件，无需安装Python环境：

1. 访问项目地址：https://github.com/AnLifeX/image_classifier
2. 进入 **Releases** 页面下载最新版本的 `ImageClassifier_vx.x.x.exe`
3. 双击运行即可使用

### 方式二：源码运行

适合开发者或需要自定义的用户：

```bash
# 1. 克隆项目
git clone https://github.com/AnLifeX/image_classifier.git
cd image_classifier

# 2. 安装依赖
pip install -r requirements.txt

# 3. 运行程序
python run.py
```

## 使用指南

### 基本操作流程

1. **启动程序**: 运行下载的 `ImageClassifier_vx.x.x.exe` 或 `python run.py`
2. **选择目录**: 点击工具栏的"打开文件夹"按钮，选择包含图片的目录
3. **创建分类**: 在分类区域添加需要的分类类别
4. **开始分类**:
   - 在图片列表中选择图片
   - 点击对应的分类按钮或使用快捷键
   - 图片将被移动到相应的分类文件夹

### 快捷键说明

| 快捷键 | 功能 | 说明 |
|--------|------|------|
| `←` `→` | 图片导航 | 在图片列表中前后浏览 |
| `↑` `↓` | 类别导航 | 在类别列表中上下选择 |
| `Enter` | 确认分类 | 将当前图片分类到选中类别 |
| `Tab` | AI 任务开关 | 开关“AI · 辅助”，或校验并启动/暂停“AI · 自动” |
| `1`-`9`, `A`-`Z` | 快速分类 | 使用数字或字母快捷键快速分类 |
| `Delete` | 移出图片 | 将图片移动到remove目录 |
| `F5` | 刷新 | 重新扫描当前目录 |
| `Ctrl+F` | 适应窗口 | 图片适应窗口大小 |
| `Ctrl +/-/0` | 缩放控制 | 放大/缩小/重置图片缩放 |

### 配置文件

程序会在当前目录生成以下配置文件：
- `logs/image_classifier.log`: 操作日志

程序会在图片目录的同级目录下生成以下配置文件：
- `config.json`: 类别记录和快捷键配置
- `classification_state.json`: 工作状态以及分类信息保存

### AI 辅助与全自动分类

当前分支支持“固定视觉编码器 + 空间颜色特征 + 类别均衡 KNN”的半自动分类：

- 每个项目第一次启用 AI 时，可选择从零开始、使用本目录已有人工标注、导入其他 `classification_state.json`，或直接复用旧项目的 AI `.npz` 特征库。
- `classification_state.json` 只保存分类记录；用它重新建库时仍需能读取原图片。原图片已迁移或删除时，请改选同目录下与基础模型匹配的 `ai_model_*_v1.npz`，复用过程不需要原图片。
- 提供速度优先（MobileNetV3）、均衡版本（ResNet18）和精度优先（DINOv2 ViT-S/14）三档可按需下载的基础模型。
- 辅助标注建议每类至少 5 张有效人工样本，全自动建议每类至少 20 张。不足时弹窗列出类别和数量，可取消或点击红色“强行开启”继续。至少两个类别各有 1 张有效样本才能预测；零样本类别无法识别，少样本结果需人工核对。
- 切换到未初始化的模型会重新建立该模型的项目特征库；已经初始化过的模型直接复用。
- 工具栏可切换 `AI · 辅助` 与 `AI · 自动`。两种模式都必须按 `Tab` 启动；辅助模式为当前图片提供建议，全自动模式仅在复制、单分类且至少两个类别各有 1 张有效样本时可用。
- 辅助模式开启后在停止翻页 200ms 后预测；推理期间图片预览区显示加载动画，完成后叠加建议结果，并在图片区域顶部显示提示。
- 每张图显示 Top-3 相似度结果；高可信结果只预选类别，仍需按 `Enter` 确认。
- “学习移除图片”默认关闭；开启后移除样本作为独立标签参与学习。辅助模式只显示移除建议，全自动模式可自动移除高可信结果。
- 全自动启动前会检查复制模式、单分类模式以及至少两个类别有有效样本，并显示风险确认；已处理图片默认跳过，不确定结果不会标注。
- 全自动运行期间锁定本窗口的翻页、分类和模式操作，只保留 `Tab` 暂停入口；窗口最小化不会停止后台任务。
- 任务完成后显示各类别、本次自动标注、移除、不确定和跳过数量；关闭结果窗口后自动定位到第一张不确定图片，并可发送系统通知。
- 人工确认或纠正后立即更新样本库，不需要逐张重新训练神经网络。
- 连续快速翻页时丢弃过期结果；项目训练结果与 `classification_state.json` 保存在同一目录。
- 模型配置页分别显示运行时和基础模型的下载进度；下载支持断点续传、大小校验和 SHA-256 校验。

模型包不会提交到仓库或塞进项目目录。开发机首次使用时执行：

```bash
# 仅导出/评估模型需要，正式 CPU 推理仍使用项目已有的 OpenCV
pip install torch torchvision onnx scikit-learn

# 分别导出三档外置 ONNX 基础模型包
python scripts/export_ai_model.py --profile speed
python scripts/export_ai_model.py --profile balanced
python scripts/export_ai_model.py --profile accuracy

# 可选：用已有分类状态做随机留出、连续帧分组和逐视频留出评估
python scripts/evaluate_ai_embeddings.py "D:\数据\classification_state.json"

# 把 removed_images 作为第四类一并评估
python scripts/evaluate_ai_embeddings.py "D:\数据\classification_state.json" --include-removed

# 启动后打开原图片目录，程序会自动读取现有标注并建立索引
python run.py
```

本地文件默认位于：

- 基础模型：`%USERPROFILE%\image_classifier\config\ai_models\<模型版本>\`
- 按需下载的推理运行时：`%USERPROFILE%\image_classifier\config\ai_runtimes\<cpu|gpu>\<版本>\`
- 项目训练结果：`classification_state.json` 同目录下的 `ai_model_speed_v1.npz`、`ai_model_balanced_v1.npz` 或 `ai_model_accuracy_v1.npz`

正式 EXE 不再捆绑 ONNX Runtime 或基础模型。推理设备默认为自动选择：首次启用时按所选设备下载 CPU 或 NVIDIA GPU Runtime；CUDA DLL、驱动或会话加载失败时会自动回退 CPU，不影响人工分类。项目样本库格式与推理设备无关，因此 CPU/GPU 切换不需要重新初始化；若当前进程已经加载 CPU Runtime，切换 GPU 后重启一次软件即可生效。

源码开发环境可按设备安装一种 ONNX Runtime，避免两个发行包长期共存：

```bash
# CPU
pip install -r requirements.txt
pip install onnxruntime==1.23.2

# NVIDIA GPU（CUDA 12 系列，安装前清理同名 CPU/GPU 发行包）
pip uninstall -y onnxruntime onnxruntime-gpu
pip install -r requirements-gpu.txt
```

本机 RTX 2060 SUPER、ONNX Runtime 1.21.1 的单张 720p 图片端到端实测（含缩放、归一化和空间颜色特征）：速度模型约 11.57ms、均衡模型约 10.46ms、精度模型约 15.88ms；相对同机 CPU 分别约快 2.5、3.7 和 6.5 倍。不同机器结果会随显卡占用和驱动变化。

该方案是项目内的增量助手，不是跨摄像机的通用分类器。新视角、新倍率或新环境应先由人工完成一批覆盖各类情况的样本，再参考“逐视频留出”结果决定是否采用建议。

## 项目结构

```
image_classifier/
├── assets/              # 资源文件
│   └── icon.ico         # 应用程序图标
├── core/                # 核心功能模块
│   ├── config.py        # 配置管理
│   ├── file_manager.py  # 文件操作管理
│   ├── image_loader.py  # 图像加载器
│   └── scanner.py       # 文件扫描器
├── ui/                  # 用户界面模块
│   ├── main_window.py   # 主窗口
│   ├── dialogs/         # 对话框
│   └── components/      # 自定义组件
├── utils/               # 工具模块
│   ├── exceptions.py    # 异常定义
│   ├── file_operations.py # 文件操作工具
│   └── performance.py   # 性能监控
├── main.py              # 主入口文件
├── run.py               # 启动脚本
├── build.py             # 构建脚本
├── requirements.txt     # 依赖列表
└── _version_.py         # 版本管理
```

## 开发指南

### 开发环境搭建

```bash
# 1. 克隆项目
git clone https://github.com/AnLifeX/image_classifier.git
cd image_classifier

# 2. 创建虚拟环境（推荐）
python -m venv venv
venv\Scripts\activate  # Windows

# 3. 安装开发依赖
pip install -r requirements-dev.txt

# 4. 运行开发版本
python run.py
```

### 运行测试

不要直接裸跑全量 `pytest`，统一通过带资源保护的测试入口：

```bash
# 全量测试：默认 2GB 内存上限、10 分钟总超时
python scripts/safe_pytest.py

# 定向测试，并自定义资源限制
python scripts/safe_pytest.py --max-memory-mb 1024 --timeout-seconds 120 -- tests/unit/test_main_window.py -q
```

Windows 下运行器使用 Job Object 设置进程树硬内存上限，同时监控工作集并在超时、超限或中断时清理全部测试进程。开发测试依赖统一维护在 `requirements-dev.txt`。

### 代码规范

- 使用 Python 3.8+ 语法特性
- 遵循 PEP 8 代码风格
- 使用类型注解提高代码可读性
- 完善的错误处理和日志记录
- 模块化设计，职责单一

## 构建发布

### 自动化发布流程

项目使用两个互不干扰的 GitHub Actions 发布通道：

| 版本 | 触发标签 | EXE | 自动更新 manifest |
| --- | --- | --- | --- |
| 基础版 | `vX.Y.Z` | `ImageClassifier_vX.Y.Z.exe` | `releases/latest/download/manifest.json` |
| AI 版 | `ai-vX.Y.Z` | `ImageClassifierAI_vX.Y.Z.exe` | `releases/download/ai-latest/manifest.json` |

AI 版本以 prerelease 发布，并同步覆盖 `ai-latest` 滚动资产，因此不会改变 GitHub 的基础版 `latest` Release。程序还会校验 manifest 中的 `edition` 和 `channel`，AI 版拒绝安装基础版更新；两个版本的断点下载分别保存在 `update` 和 `update/ai` 目录。

基础版继续使用蓝色 `assets/icon.*`；AI 版使用紫色并带 AI 标记的 `assets/icon-ai.*`。`python build.py --edition ai` 会自动选择 AI 图标，基础版构建不受影响。

**构建特性**：
- 基础版为单文件 exe，无需额外依赖
- 包含应用图标和资源文件
- 优化文件体积（约86MB）
- 支持 Windows 10+ 系统
- 自动版本号管理

AI CI 不再把 `onnxruntime`、`onnxruntime-gpu` 或基础模型收集进 EXE。用户首次使用对应设备或模型时在模型配置页按需下载，资源独立保存在用户配置目录，可以断点续传并单独更新。CUDA 12 / cuDNN 9 同样不打入 EXE；程序会优先读取 `IMAGE_CLASSIFIER_CUDA_DIR`，也会识别当前用户已有的兼容 PyTorch `torch\\lib`，仍找不到时显示原因并自动回退 CPU。

### 本地构建（开发用）

开发者可以使用本地构建脚本进行测试：

```bash
# 安装构建依赖
pip install pyinstaller

# 基础版
python build.py

# AI 版（内嵌 AI 产品标识和独立更新地址）
python build.py --edition ai
```

> 正式发布请使用 CI/CD 流程，确保构建一致性和版本管理。

## 问题排查

### 常见问题

**Q: 程序启动失败？**
A: 确保系统满足最低要求，Windows 10+ 和足够的内存空间。

**Q: 图片加载缓慢？**
A: 检查图片大小和格式，程序对超大图片（>50MB）可能需要更长加载时间。

**Q: 快捷键不响应？**
A: 确保程序窗口获得焦点，避免与其他程序的快捷键冲突。

**Q: 网络路径访问问题？**
A: 确保有足够的网络权限，程序对网络路径进行了特殊优化。

### 日志查看

程序运行时会生成详细日志：
- 位置：`logs/image_classifier.log`
- 包含：操作记录、错误信息、性能数据
- 用途：问题诊断和性能分析

## 贡献指南

欢迎提交问题报告和功能请求！

### 报告问题

1. 在 [Issues 页面](https://github.com/AnLifeX/image_classifier/issues) 创建新问题
2. 描述问题的重现步骤
3. 提供错误日志和系统信息
4. 如可能，提供截图说明

### 贡献代码

1. Fork 项目
2. 创建特性分支：`git checkout -b feature/amazing-feature`
3. 提交更改：`git commit -m 'Add amazing feature'`
4. 推送分支：`git push origin feature/amazing-feature`
5. 创建 Pull Request

## 开发文档

### 版本管理系统

本项目使用统一的版本管理系统，详细使用方法请参考：

**[版本管理系统说明 → VERSION_MANAGEMENT.md](VERSION_MANAGEMENT.md)**

## 更新日志

查看完整的更新历史和详细信息：

**[查看完整更新日志 → CHANGELOG.md](CHANGELOG.md)**

## 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 致谢

- [PyQt6](https://www.riverbankcomputing.com/software/pyqt/) - 强大的GUI框架
- [OpenCV](https://opencv.org/) - 计算机视觉库
- [Pillow](https://pillow.readthedocs.io/) - Python图像处理库
- [PyInstaller](https://pyinstaller.org/) - Python应用打包工具
