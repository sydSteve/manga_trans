import os
import torch
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

os.environ["OMP_NUM_THREADS"] = "4"        # 限制 OpenMP 线程
os.environ["MKL_NUM_THREADS"] = "4"        # 限制 MKL 线程
os.environ["OPENBLAS_NUM_THREADS"] = "4"   # 限制 OpenBLAS 线程
# 如果你有 GPU 并且想用，可以保留 CPU 线程数较少，让 GPU 干活
torch.set_num_threads(4)          # 限制 PyTorch 只用 4 个核心
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from manga_ocr import MangaOcr
import easyocr
import requests
import json
from pathlib import Path
from dotenv import load_dotenv

# 加载 .env 文件中的环境变量（如果存在）
load_dotenv()
#================对话泡检测====================
def detect_bubbles_by_edges(image_bgr, canny_low=50, canny_high=150,
                            min_area=3000, margin=5):
    """
    利用边缘检测 + 轮廓查找，检测封闭的气泡区域（不依赖颜色）。
    返回气泡矩形列表 [(x1,y1,x2,y2), ...]
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    # 边缘检测
    edges = cv2.Canny(gray, canny_low, canny_high)
    # 闭运算连接断裂边缘
    kernel = np.ones((3, 3), np.uint8)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    # 查找轮廓
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bubbles = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        # 外接矩形并向外扩展 margin
        x, y, w, h = cv2.boundingRect(cnt)
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = x + w + margin
        y2 = y + h + margin
        bubbles.append((x1, y1, x2, y2))
    return bubbles

def is_inside_any_bubble(bbox, bubbles):
    """判断文本框中心点是否在任一气泡矩形内"""
    xs = [p[0] for p in bbox]
    ys = [p[1] for p in bbox]
    cx = (min(xs) + max(xs)) / 2
    cy = (min(ys) + max(ys)) / 2
    for (x1, y1, x2, y2) in bubbles:
        if x1 <= cx <= x2 and y1 <= cy <= y2:
            return True
    return False
# ================== 翻译函数（根据开关决定行为） ==================
def translate_text(text, target_lang="zh"):
    """
    根据环境变量 USE_DEEPSEEK_API 决定使用真实翻译或模拟翻译。
    若为 'true' 且设置了 DEEPSEEK_API_KEY，则调用 DeepSeek API；
    否则返回  原文。
    """
    use_api = os.environ.get("USE_DEEPSEEK_API", "false").lower() == "true"
    if not use_api:
        return f"{text}"

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("USE_DEEPSEEK_API=true 但未设置 DEEPSEEK_API_KEY，使用模拟翻译")
        return f"{text}"

    url = "https://api.deepseek.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": f"将以下日文漫画对话翻译成{target_lang}，保持口语化。翻译就好，不要有任何解释。语言简洁干练。当有不明所以的文字出现时输出‘跳过’"},
            {"role": "user", "content": text}
        ],
        "temperature": 0.3,
        "max_tokens": 2000
    }
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"翻译失败: {e}")
        return text   # 失败时保留原文，避免中断流程


# ================== 嵌字函数（保持不变） ==================
def draw_text_vertical(draw, box, text, font_paths,
                       max_font_size=60, min_font_size=8,
                       width_fill_ratio=0.9, margin=5,
                       col_spacing=3, row_spacing=1):
    """
    在四边形框内以竖排方式绘制文本（从上到下，从右向左）。
    - 自动换列：当一列的文字高度超出框高时，自动移到左边新列。
    - 支持 \n 作为强制换列符。
    - 字号自动适应：通过二分查找找到能使所有列总宽度最接近框宽的字号。
    - width_fill_ratio: 总列宽希望占框宽的比例（0~1），默认0.9。
    - col_spacing: 列间距（像素）
    - row_spacing: 字符间垂直额外间距（像素），默认1
    """
    # 计算可用区域
    xs = [p[0] for p in box]
    ys = [p[1] for p in box]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    box_width = x_max - x_min - 2 * margin
    box_height = y_max - y_min - 2 * margin
    if box_width <= 0 or box_height <= 0:
        return False

    # 根据 \n 将文本拆成段落（每个段落会强制另起一列）
    paragraphs = text.split('\n')

    # 内部函数：给定字号，计算所有字符排版所需的总列宽
    def get_total_width_for_size(size):
        # 加载字体
        font = None
        for fp in font_paths:
            try:
                font = ImageFont.truetype(fp, size)
                break
            except:
                continue
        if font is None:
            font = ImageFont.load_default()

        # 测量单个全角字符的宽度和高度（以“测”字为参考）
        bbox = draw.textbbox((0, 0), '测', font=font)
        char_w = bbox[2] - bbox[0]
        char_h = bbox[3] - bbox[1]

        max_chars_per_col = max(1, int((box_height + row_spacing) // (char_h + row_spacing)))

        columns = 0
        for para in paragraphs:
            if not para:
                continue
            # 将当前段落文字按字符拆开
            chars = list(para)
            # 如果段落为空，跳过（columns 不增加额外列，除非有空格需求，这里忽略）
            idx = 0
            while idx < len(chars):
                # 本列还可以容纳的字符数
                remain = len(chars) - idx
                chars_in_col = min(max_chars_per_col, remain)
                idx += chars_in_col
                columns += 1
            # 每个段落结束后相当于一个强制换列（已经在段落之间隐含，因为下一段落从新列开始）
        # 如果没有字符，返回0
        if columns == 0:
            return 0
        total_width = columns * char_w + (columns - 1) * col_spacing
        return total_width

    # ---- 二分搜索最佳字号 ----
    # 先看最大字号能否放入框宽
    if get_total_width_for_size(max_font_size) > box_width:
        # 放不下，缩小字号
        lo, hi = min_font_size, max_font_size
        best_size = min_font_size
        while lo <= hi:
            mid = (lo + hi) // 2
            if get_total_width_for_size(mid) <= box_width:
                best_size = mid
                lo = mid + 1
            else:
                hi = mid - 1
        final_size = best_size
    else:
        # 放得下，尝试放大字号，直到总宽度接近 width_fill_ratio * box_width
        lo, hi = max_font_size, max_font_size * 2
        best_size = max_font_size
        while lo <= hi:
            mid = (lo + hi) // 2
            total_w = get_total_width_for_size(mid)
            if total_w <= box_width * width_fill_ratio:
                best_size = mid
                lo = mid + 1
            else:
                hi = mid - 1
        final_size = best_size

    # ---- 使用最终字号排版并绘制 ----
    # 加载最终字体
    font = None
    for fp in font_paths:
        try:
            font = ImageFont.truetype(fp, final_size)
            break
        except:
            continue
    if font is None:
        font = ImageFont.load_default()

    # 重新测量最终字号的尺寸
    bbox = draw.textbbox((0, 0), '测', font=font)
    char_w = bbox[2] - bbox[0]
    char_h = bbox[3] - bbox[1]

    max_chars_per_col = max(1, int((box_height + row_spacing) // (char_h + row_spacing)))

    # 将文本转化为字符列表，并记录列信息
    columns_chars = []   # 每列是一个字符列表
    for para in paragraphs:
        chars = list(para)
        idx = 0
        while idx < len(chars):
            remain = len(chars) - idx
            chars_in_col = min(max_chars_per_col, remain)
            columns_chars.append(chars[idx:idx + chars_in_col])
            idx += chars_in_col
        # 段落结束不自动增加空列，除非末尾有 \n，但 paragraph 分割已经处理了，
        # 如果两个段落之间需要强制换列，已经通过分段实现，这里不额外加

    if not columns_chars:
        return False

    total_cols = len(columns_chars)
    total_width = total_cols * char_w + (total_cols - 1) * col_spacing

    # 起始 x 坐标：从右向左排，最右列的 x 位置
    start_x_right = x_max - margin - char_w
    # 整个列块的水平偏移，使其在框内水平居中
    x_center_offset = (box_width - total_width) // 2
    start_x = start_x_right - x_center_offset

    # 逐列绘制
    for col_idx, chars in enumerate(columns_chars):
        # 计算本列的 x 坐标（列索引从右向左：col_idx=0 最右）
        col_x = start_x - col_idx * (char_w + col_spacing)
        # 计算本列文字的垂直起始 y 坐标（居中）
        col_text_height = len(chars) * (char_h + row_spacing) - row_spacing
        start_y = y_min + margin + (box_height - col_text_height) // 2

        # 绘制本列每个字符
        for row_idx, ch in enumerate(chars):
            char_y = start_y + row_idx * (char_h + row_spacing)
            # 字符水平居中在其单元格内（char_w宽度）
            # 直接使用 draw.text，传入字符左上角
            draw.text((col_x, char_y), ch, fill=(0, 0, 0), font=font)

    return True

# ================== 框合并函数（保持不变） ==================
def merge_same_line_boxes(boxes, x_gap=20, y_overlap_ratio=0.7):
    """仅合并水平方向紧邻、且垂直高度重叠高的框（同一行内的碎片）"""
    if len(boxes) <= 1:
        return boxes
    rects = []
    for box in boxes:
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        rects.append([min(xs), min(ys), max(xs), max(ys)])
    merged, used = [], [False] * len(boxes)
    for i in range(len(boxes)):
        if used[i]: continue
        group, used[i] = [i], True
        while True:
            changed = False
            for j in range(len(boxes)):
                if used[j]: continue
                for idx in group:
                    x1,y1,x2,y2 = rects[idx]
                    x3,y3,x4,y4 = rects[j]
                    overlap_y = max(0, min(y2,y4)-max(y1,y3))
                    h1,h2 = y2-y1, y4-y3
                    min_h = min(h1,h2)
                    if min_h == 0: continue
                    if overlap_y/min_h < y_overlap_ratio: continue
                    gap = max(x1,x3)-min(x2,x4) if (x2<x3 or x4<x1) else 0
                    if gap < x_gap:
                        group.append(j); used[j] = True
                        changed = True; break
                if changed: break
            if not changed: break
        if len(group) == 1:
            merged.append(boxes[group[0]])
        else:
            all_pts = []
            for idx in group: all_pts.extend(boxes[idx])
            xs = [p[0] for p in all_pts]; ys = [p[1] for p in all_pts]
            merged.append([[min(xs),min(ys)],[max(xs),min(ys)],
                           [max(xs),max(ys)],[min(xs),max(ys)]])
    return merged

# ================== 单张图片处理流水线 ==================
def process_single_image(image_path, output_path, manga_ocr, reader, font_paths,
                         use_filter=True, use_vertical=True):
    """
    处理单张漫画图片。

    use_filter: 是否启用文本框过滤（基于几何特征和背景亮度）
    use_vertical: True=竖排嵌字，False=横排
    """
    print(f"处理 {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        print(f"警告：无法读取 {image_path}，跳过")
        return False

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # ========== 1. 文字区域检测 ==========
    print("  检测文字区域...")
    ocr_result = reader.readtext(
        image_path,
        text_threshold=0.5,
        low_text=0.2,
        paragraph=False,
        canvas_size=1280
    )
    raw_boxes = [item[0] for item in ocr_result]

    # ========== 2. 同行合并 ==========
    bbox_list = merge_same_line_boxes(raw_boxes, x_gap=20, y_overlap_ratio=0.7)
    print(f"  合并后文本框数量: {len(bbox_list)}")

    # ========== 3. 过滤辅助函数 ==========
    def is_good_textbox(bbox, rgb_full, min_area=500, min_aspect=0.2, max_aspect=5,
                        edge_density_threshold=0.2, bright_threshold=120):
        """
        判断一个文本框是否可能是正常的对话文字（而非特效艺术字）。
        返回 True 表示通过过滤，应该处理。
        """
        pts = np.array(bbox, dtype=np.int32)
        x_min = int(np.min(pts[:,0]))
        y_min = int(np.min(pts[:,1]))
        x_max = int(np.max(pts[:,0]))
        y_max = int(np.max(pts[:,1]))
        w = x_max - x_min
        h = y_max - y_min
        area = w * h

        # 1. 面积太小，通常是单字碎片或噪声，跳过
        if area < min_area:
            return False

        # 2. 宽高比过于极端（如细长条），跳过
        if h == 0: return False
        aspect = w / h
        if aspect < min_aspect or aspect > max_aspect:
            return False

        # 3. 裁剪区域背景分析
        crop = rgb_full[y_min:y_max, x_min:x_max]
        if crop.size == 0:
            return False
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)

        # 背景亮度：采样四角平均灰度
        h_crop, w_crop = crop.shape[:2]
        sample_ratio = 0.2
        corners = [
            gray[0:int(h_crop*sample_ratio), 0:int(w_crop*sample_ratio)],
            gray[0:int(h_crop*sample_ratio), int(w_crop*(1-sample_ratio)):],
            gray[int(h_crop*(1-sample_ratio)):, 0:int(w_crop*sample_ratio)],
            gray[int(h_crop*(1-sample_ratio)):, int(w_crop*(1-sample_ratio)):]
        ]
        all_corners = np.concatenate([c.flatten() for c in corners])
        avg_brightness = np.mean(all_corners)

        # 如果背景太暗（如黑底），很可能是特效字，过滤掉
        if avg_brightness < bright_threshold:
            return False

        # 4. 边缘密度：使用 Canny 检测，边缘像素比例过高说明背景杂乱
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (h_crop * w_crop)
        if edge_density > edge_density_threshold:
            return False

        return True

    # ========== 4. 逐框处理 ==========
    pil_img = Image.fromarray(rgb_image)
    draw = ImageDraw.Draw(pil_img, 'RGBA')

    for i, bbox in enumerate(bbox_list):
        if use_filter and not is_good_textbox(bbox, rgb_image):
            continue  # 过滤掉艺术字/特效字

        pts = np.array(bbox, dtype=np.int32)
        x_min = int(np.min(pts[:,0]))
        y_min = int(np.min(pts[:,1]))
        x_max = int(np.max(pts[:,0]))
        y_max = int(np.max(pts[:,1]))
        if x_max <= x_min or y_max <= y_min:
            continue

        crop = rgb_image[y_min:y_max, x_min:x_max]
        if crop.size == 0:
            continue

        crop_pil = Image.fromarray(crop)
        # MangaOcr 加速：限制宽度
        if crop_pil.width > 400:
            ratio = 400.0 / crop_pil.width
            new_h = int(crop_pil.height * ratio)
            crop_pil = crop_pil.resize((400, new_h), Image.Resampling.LANCZOS)

        original_text = manga_ocr(crop_pil).strip()
        if not original_text:
            continue

        print(f"  区域{i}: {original_text}")
        translated = translate_text(original_text, target_lang="zh")
        print(f"         => {translated}")

        # 擦除原文
        overlay = Image.new('RGBA', pil_img.size, (255,255,255,0))
        overlay_draw = ImageDraw.Draw(overlay)
        overlay_draw.polygon([tuple(p) for p in pts], fill=(255,255,255,200))
        pil_img = Image.alpha_composite(pil_img.convert('RGBA'), overlay).convert('RGB')
        draw = ImageDraw.Draw(pil_img)

        # 嵌字
        if use_vertical:
            success = draw_text_vertical(
                draw, bbox, translated, font_paths,
                max_font_size=60, min_font_size=8,
                width_fill_ratio=0.9, margin=3,
                col_spacing=2, row_spacing=1
            )
        else:
            success = draw_text_in_box(
                draw, bbox, translated, font_paths,
                max_font_size=30, min_font_size=8, margin=3
            )
        if not success:
            print(f"  嵌字失败，框太小")

    pil_img.save(output_path, quality=95)
    print(f"  保存到 {output_path}")
    return True


# ================== 批量处理函数 ==================
def batch_process(input_dir, output_dir):
    """
    批量处理 input_dir 下所有支持格式的图片（包含子文件夹），结果保存到 output_dir，保持目录结构。
    """
    extensions = {".png", ".jpg", ".jpeg"}
    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"输入目录不存在: {input_dir}")

    output_path = Path(output_dir)

    # 递归收集所有图片文件
    image_files = []
    for ext in extensions:
        image_files.extend(input_path.rglob(f"*{ext}"))
        image_files.extend(input_path.rglob(f"*{ext.upper()}"))
    # 去重并排序
    image_files = sorted(list(set(image_files)))

    if not image_files:
        print("未找到任何图片文件")
        return

    print(f"共找到 {len(image_files)} 张图片")

    # 加载模型（只加载一次）
    print("加载 Manga OCR...")
    manga_ocr = MangaOcr()
    print("加载 EasyOCR 检测器 (日文)...")
    reader = easyocr.Reader(['ja'],gpu=True)

    # 字体路径
    font_paths = [
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/msgothic.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
    ]

    success_count = 0
    for img_file in image_files:
        # 计算相对路径，并在输出目录中创建对应的子文件夹
        relative_path = img_file.relative_to(input_path)
        out_file = output_path / relative_path
        out_file.parent.mkdir(parents=True, exist_ok=True)

        try:
            if process_single_image(str(img_file), str(out_file), manga_ocr, reader, font_paths):
                success_count += 1
        except Exception as e:
            print(f"处理 {relative_path} 时出错: {e}")

    print(f"批量处理完成：{success_count}/{len(image_files)} 张图片处理成功")


# ================== 主函数 ==================
if __name__ == "__main__":
    # 配置输入输出文件夹（可直接修改，或改为命令行参数）
    INPUT_DIR = "input_images"   # 放原图的文件夹
    OUTPUT_DIR = "output_images" # 结果输出文件夹

    batch_process(INPUT_DIR, OUTPUT_DIR)