# manga_trans

当前仓库已经按职责分区：

- `app/`：主程序、配置、文档
- `third_party/`：第三方模型和外部仓库资产
- `data/`：输入图片、输出图片、PDF 等运行数据
- `debug/`：人工调试、对比和临时产物
- `test/`：测试脚本和测试输出

## 启动

推荐在 `E:\tool\app` 下运行：

```bat
.\trans.cmd
```

或手动执行：

```bat
..\venv\Scripts\activate.bat
python trans.py
```

## CLI

```bat
python trans.py --input ../data/input_images --output ../data/output_images --config config.yaml --debug
```

默认路径都相对于 `app/`：

- `--input`：`../data/input_images`
- `--output`：`../data/output_images`
- `--config`：`config.yaml`

## 当前默认链路

项目现在只保留一条精简后的默认处理链路：

1. 输入图片
2. 文本检测
3. OCR
4. 翻译或直出
5. 修复原文区域
6. 嵌字
7. 输出图片

## 当前保留的核心能力

- 文本检测主后端：`comic-text-and-bubble-detector`
- 文本补框与整页兜底：`EasyOCR`
- 气泡检测主后端：`speech-bubble-segmentation`
- 气泡回退与召回增强：`detect_bubbles_by_edges()`
- 文本区域分割：`comic-text-detector` + 本地阈值融合
- OCR：`PaddleOCR-VL-1.5`，必要时回退 `EasyOCR`
- 修复：白底优先 `white_fill`，其余优先 `AOT`，不可用时回退 OpenCV
- 风格提示：`YuzuMarker`
- 渲染：Pillow 竖排 / 横排嵌字

## 气泡检测配置

`config.yaml` 新增了独立的 `bubble` 段：

```yaml
bubble:
  enabled: true
  backend: speech_bubble_segmentation
  model_path: ../third_party/speech-bubble-segmentation/model.pt
  min_confidence: 0.2
  fallback: edges
```

说明：

- 主检测优先使用 `speech-bubble-segmentation`
- 若模型缺失、运行时不可用或单张图推理失败，会自动回退到 edge bubbles
- 当主后端可用时，edge bubbles 仍会作为召回增强补充未重复区域

## 调试输出

传 `--debug` 时，JSON 会写到输出目录下的 `_debug/`。

如果使用默认输出目录，调试文件会出现在：

```text
E:\tool\data\output_images\_debug
```

## 检测可视化测试

从仓库根目录执行：

```bat
..\venv\Scripts\python.exe test\test_detection.py
```

默认读取 `data/input_images/`，产物输出到 `test/test_out/`：

- `test/test_out/images/`：带框预览图
- `test/test_out/json/`：单图检测摘要
- `test/test_out/summary.json`：整批汇总
