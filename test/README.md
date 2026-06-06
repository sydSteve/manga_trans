# test

`test/` 是一套独立镜像版流程，结构和 `app/` 对齐，用来在不影响主项目默认输入输出的前提下单独跑完整翻译链路。

## 目录说明

- `trans.py` / `trans.cmd`：测试版入口
- `manga_translator/`：主流程模块副本
- `manga_ocr/`：OCR 模块副本
- `input_images/`：测试输入目录
- `output_images/`：测试输出目录

第三方模型、字体和权重继续复用仓库根目录下的 `third_party/` 和系统字体，不在 `test/` 内重复拷贝。

## 启动

推荐在 `E:\tool\test` 下运行：

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
python trans.py --input .\input_images --output .\output_images --config .\config.yaml --debug
```

默认路径都相对于 `test/`：

- `--input`：`input_images`
- `--output`：`output_images`
- `--config`：`config.yaml`

## 默认处理链路

1. 读取输入图片
2. 文本检测与气泡检测
3. OCR 识别
4. 翻译或直出
5. 原文区域修复
6. 嵌字渲染
7. 写出结果图片

## 调试输出

使用 `--debug` 时，JSON 会写到输出目录下的 `_debug/`。如果使用默认输出目录，调试文件位置为：

```text
E:\tool\test\output_images\_debug
```

## 说明

- `app/` 的主流程逻辑没有被改动。
- `test/` 是当前时点的镜像副本，后续如果 `app/` 继续演进，需要按需同步到 `test/`。
