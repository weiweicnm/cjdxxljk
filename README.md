# 长大心理 - 校园心理健康平台 (Changda Psychology)

“长大心理”是一个温馨治愈的校园心理诊疗与预警平台，旨在通过数据分析与人工智能辅助，为校园心理老师和辅导员提供全方位的学生心理健康管理服务。

## 🌟 主要功能

### 1. 数据概览 (Dashboard)
- 实时监控全校学生心理健康状态
- 直观的预警分布统计（正常、轻度预警、重度预警）
- 自动分析近期高频心理因子（如抑郁、焦虑、人际关系敏感等）

### 2. 学生心理档案 (Student Data)
- 管理和分析学生 SCL-90 测评数据
- 支持导入 **CSV** 和 **Excel (.xlsx, .xls)** 格式的数据源
- 自动计算 SCL-90 总分、均分及阳性项目数，并智能判别预警级别
- 快速检索学生姓名、系别、专业或预警状态
- 提供一键跳转“AI 咨询建议”功能

### 3. 长大 AI 助手 (AI Assistant)
- 专属校园心理咨询辅助专家，性格温馨、治愈、充满生机
- 结合学生的具体 SCL-90 测试结果和家庭背景，提供个性化的心理状况分析
- 为心理老师提供专业的心理辅导建议和沟通话术支持

### 4. 情绪识别 (Emotion Recognition)
- 支持加载基于 PyTorch 训练并导出为 **ONNX** 格式的人脸情绪识别模型
- 可以自定义输入分辨率、图片通道（RGB/灰度) 及归一化方式
- 支持自定义分类标签
- 提供**静态图片上传**和**实时摄像头**两种面部情绪分析模式

## 🚀 快速开始

### 依赖安装
```bash
npm install
```

### 运行开发服务器
```bash
npm run dev
```

## 🛠️ 技术栈
- **前端框架**: React 19, Vite
- **UI & 样式**: Tailwind CSS, Lucide React (图标)
- **数据处理**: PapaParse (CSV 解析), SheetJS/xlsx (Excel 解析)
- **人工智能**: Google GenAI SDK (Gemini 大模型)
- **前端推理的机器学习模型**: ONNX Runtime Web (`onnxruntime-web`)

## 📝 情绪识别模型接入指南
由于浏览器不支持直接运行 `.pt` 或 `.pth` 文件，您需要将 PyTorch 模型导出为 ONNX 格式：

1. **在 Python 中导出模型 (PyTorch):**
```python
import torch

# 加载您的 PyTorch 模型
model = YourModel()
model.load_state_dict(torch.load('model.pth', map_location='cpu'))
model.eval()

# 创建虚拟输入 (BatchSize=1, Channels=3, Height=640, Width=640)
dummy_input = torch.randn(1, 3, 640, 640)

# 导出为 ONNX
torch.onnx.export(model, dummy_input, "emotion_model.onnx",
                  input_names=['input'], output_names=['output'])
```
2. 在平台的【情绪识别】页面上传导出的 `emotion_model.onnx` 文件即可调用。

## 🔒 隐私与安全性
本项目作为心理健康医疗辅助工具，不将最终的敏感心理数据储存在未经授权的第三方服务器。上传的测试文件以及接入的图像识别均在本地浏览器（或当前会话）中处理。
