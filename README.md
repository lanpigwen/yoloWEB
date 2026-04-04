# yoloWEB

轻量说明（自动生成）：基于 Flask 的篮球动作检测与训练记录前端 + 后端（使用 YOLO/关键点检测）。

主要功能

- 实时或离线视频帧检测（使用 Ultralyics/YOLO）
- 投篮/运球/反应训练模式与计数
- 将训练结果保存到 Redis（作为简单数据存储）
- 支持将帧序列合成视频并与音频合并（依赖 ffmpeg CLI）

快速开始（开发）

1. 克隆仓库并切换到本分支（或 main）：

   git clone https://github.com/lanpigwen/yoloWEB.git
   cd yoloWEB

2. 建议使用虚拟环境并安装依赖（示例：CPU 环境）：

   python -m venv .venv
   .\.venv\Scripts\activate    # Windows
   # 或： source .venv/bin/activate  # Linux / macOS
   pip install -r requirements.txt

   # 若使用 GPU，请参见下方“依赖与安装”部分以安装适合的 torch 版本。

3. 配置模型文件与 Redis

- 本仓库原始结构在 pts/ 下包含多种模型权重（.engine/.onnx/.pt）。这些通常体积较大，建议将它们移到外部存储（GitHub Releases、云存储或使用 Git LFS）。
- 默认 Redis 配置：localhost:6379，db=1。你可以修改 app.py 或使用环境变量/配置文件（建议）。
- 需要在系统中安装 ffmpeg（命令行工具），用于音频/视频合并。

4. 运行（开发模式）：

   set FLASK_ENV=development  # Windows
   python app.py

   然后在浏览器打开 http://127.0.0.1:5000

依赖与安装建议

- requirements.txt 包含开发时的 Python 依赖。
- Torch/Ultralytics：CPU 或 GPU 安装方式不同。示例（CPU）：

   pip install torch --index-url https://download.pytorch.org/whl/cpu
   pip install ultralytics

- 如果打算在 headless 服务器上运行，建议安装 opencv-python-headless 而不是 opencv-python。

模型与发布注意事项

- 目前仓库中存在 pts/ 下的模型文件（.engine/.onnx/.pt）。这些文件不适合直接保存在 Git 普通仓库。建议：
  1. 将模型移到 Git LFS 或外部存储（Release/云）并在 README 中提供下载链接；
  2. 在 .gitignore 中排除模型文件；
  3. 在 README 中说明如何放置模型到 pts/ 目录以便运行。

改进建议（优先级）

1. 写入详细 README（本次基础版已生成）。
2. 固定依赖文件（requirements.txt）并在 README 提示 CPU/GPU 安装差异。 
3. 将大模型移出仓库历史（使用 Git LFS 或 Release）。
4. 用环境变量/配置文件管理 Redis、模型路径、FFmpeg 路径等。 
5. 提供 Dockerfile 和/或 gunicorn 启动示例用于生产部署。

许可证（License）

- 当前仓库未包含 LICENSE。请根据需要添加（MIT/Apache/专有等）。

---

如需，我可以：
- 把 README 扩展为中文/英文双语版；
- 创建 Dockerfile 示例；
- 把模型从仓库历史中移除并配置 Git LFS（需确认）。

