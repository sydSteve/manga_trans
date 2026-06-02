import img2pdf
import os
from pathlib import Path

# ===== 修改这两个变量 =====
image_folder = "output_images"      # 存放图片的文件夹
output_folder = "pdf"    # 存放 PDF 的文件夹
output_pdf_name = "合并后的漫画.pdf"     # PDF 文件名
# =========================

# 确保输出文件夹存在
Path(output_folder).mkdir(parents=True, exist_ok=True)

# 获取所有图片（按文件名排序）
image_ext = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
image_paths = sorted([
    p for p in Path(image_folder).iterdir()
    if p.suffix.lower() in image_ext
])

if not image_paths:
    print("错误：文件夹中没有图片")
    exit(1)

# 生成 PDF 完整路径
output_pdf_path = Path(output_folder) / output_pdf_name

# 合并图片为 PDF
with open(output_pdf_path, "wb") as f:
    f.write(img2pdf.convert(image_paths))

print(f"✅ PDF 已生成：{output_pdf_path}")