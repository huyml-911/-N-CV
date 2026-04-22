"""
==============================================================================
  HỆ THỐNG TRUY XUẤT NGƯỜI ĐI LẠC - Giao diện Web UI (Gradio) v2
  Module: reid_app.py
==============================================================================

Nâng cấp v2:
  - 3 chế độ Input (Tabs): Đơn ảnh / Đa ảnh / Demo tự động
  - Light theme với tương phản màu cao, dễ đọc
  - Viền màu card kết quả rõ ràng (xanh lá / đỏ / xanh dương)

Cách chạy:
  python reid_app.py              # Local: http://localhost:7860
  python reid_app.py --share      # Public link: https://xxx.gradio.live
  python reid_app.py --port 8080  # Đổi cổng
"""

import os
import re
import time
import argparse
import tempfile
import warnings

import numpy as np
import gradio as gr
from PIL import Image, ImageDraw, ImageFont

warnings.filterwarnings("ignore")

# ── Import pipeline & search engine ──────────────────────────────────────────
try:
    from hog_reid_pipeline import Market1501Dataset, HOGFeatureExtractor
    from reid_search import GalleryIndex, ReIDSearchEngine
except ImportError as e:
    raise ImportError(
        f"Lỗi import: {e}\n"
        "Đảm bảo reid_app.py, hog_reid_pipeline.py và reid_search.py "
        "nằm cùng một thư mục."
    )

# ==============================================================================
# TRẠNG THÁI TOÀN CỤC
# ==============================================================================
_state = {
    "gallery_index": None,
    "engine":        None,
    "dataset_root":  None,
    "dataset_obj":   None,
}


# ==============================================================================
# HÀM TIỆN ÍCH DÙNG CHUNG
# ==============================================================================
def _parse_market1501_fname(fname: str):
    """Trích xuất (pid, cid) từ tên file Market-1501."""
    m = re.search(r"(-?\d+)_c(\d+)", os.path.basename(fname or ""))
    if m:
        return int(m.group(1)), int(m.group(2))
    return None, None


def _make_result_card(img_path: str, rank: int, sim: float, dist: float,
                      pid: int, cam_label: str, is_correct) -> Image.Image:
    """
    Tạo card kết quả bằng PIL thuần túy.

    Layout:
    +--[BORDER 8px]----------------------+
    | HEADER: Rank * Match * Sim * Info  |  <- nền màu theo trạng thái
    +------------------------------------+
    |                                    |
    |        ẢNH GALLERY                 |
    |                                    |
    | [=====SIM BAR=====..............] |  <- progress bar
    +------------------------------------+

    Viền dày 8px, không bị Gradio Gallery crop vì dùng object-fit:contain.
    Màu text tối trên nền sáng -> tương phản cao, dễ đọc.
    """
    BORDER   = 8
    HEADER_H = 64
    IMG_W    = 128
    IMG_H    = 224

    # -- Màu theo kết quả -----------------------------------------------
    if is_correct is True:
        border_rgb = (34, 160, 75)      # Xanh lá đậm
        header_bg  = (220, 248, 228)    # Xanh lá nhạt (nền sáng)
        text_dark  = (15, 90, 40)       # Chữ xanh lá rất đậm
        mark_str   = "[ĐÚNG]"
        mark_rgb   = (15, 100, 45)
    elif is_correct is False:
        border_rgb = (210, 50, 45)      # Đỏ đậm
        header_bg  = (252, 220, 218)    # Đỏ nhạt (nền sáng)
        text_dark  = (140, 20, 15)      # Chữ đỏ rất đậm
        mark_str   = "[SAI]"
        mark_rgb   = (150, 25, 20)
    else:
        border_rgb = (30, 110, 210)     # Xanh dương đậm
        header_bg  = (218, 232, 252)    # Xanh dương nhạt (nền sáng)
        text_dark  = (15, 70, 170)      # Chữ xanh dương rất đậm
        mark_str   = "[QUERY]"
        mark_rgb   = (20, 80, 185)

    canvas_w = IMG_W + BORDER * 2
    canvas_h = IMG_H + HEADER_H + BORDER * 2

    # -- Canvas nền trắng (Light theme) ---------------------------------
    card = Image.new("RGB", (canvas_w, canvas_h), (250, 251, 255))
    draw = ImageDraw.Draw(card)

    # -- Viền dày BORDER px ----------------------------------------------
    for i in range(BORDER):
        draw.rectangle(
            [i, i, canvas_w - 1 - i, canvas_h - 1 - i],
            outline=border_rgb,
        )

    # -- Header background -----------------------------------------------
    draw.rectangle(
        [BORDER, BORDER, canvas_w - BORDER - 1, BORDER + HEADER_H - 1],
        fill=header_bg,
    )

    # -- Ảnh gallery -----------------------------------------------------
    try:
        img_pil = Image.open(img_path).convert("RGB")
        img_pil = img_pil.resize((IMG_W, IMG_H), Image.LANCZOS)
    except Exception:
        img_pil = Image.new("RGB", (IMG_W, IMG_H), (210, 215, 225))
    card.paste(img_pil, (BORDER, BORDER + HEADER_H))

    # -- Similarity progress bar -----------------------------------------
    bar_y  = BORDER + HEADER_H + IMG_H - 14
    bar_x0 = BORDER + 4
    bar_x1 = BORDER + IMG_W - 4
    bar_w  = bar_x1 - bar_x0
    filled = int(sim * bar_w)
    draw.rectangle([bar_x0, bar_y, bar_x1, bar_y + 7],
                   fill=(210, 215, 225), outline=(170, 175, 190))
    if filled > 0:
        draw.rectangle([bar_x0, bar_y, bar_x0 + filled, bar_y + 7],
                       fill=border_rgb)

    # -- Text trên header (chữ TỐI trên nền SÁNG -> tương phản cao) ------
    font = ImageFont.load_default()
    tx   = BORDER + 6

    # Dòng 1: Rank + Mark
    draw.text((tx, BORDER + 5),
              f"#{rank:02d}  {mark_str}",
              font=font, fill=mark_rgb)

    # Dòng 2: Similarity
    draw.text((tx, BORDER + 19),
              f"Sim: {sim:.4f} ({int(sim*100)}%)",
              font=font, fill=(25, 30, 50))      # Gần đen, đọc rõ trên nền sáng

    # Dòng 3: Distance
    draw.text((tx, BORDER + 33),
              f"Dist: {dist:.5f}",
              font=font, fill=(60, 70, 100))     # Xám xanh đậm

    # Dòng 4: Camera + PID
    draw.text((tx, BORDER + 47),
              f"{cam_label} | PID {pid:04d}",
              font=font, fill=text_dark)

    return card


# ==============================================================================
# HÀM KHỞI TẠO ENGINE (dùng chung cho cả 3 tab)
# ==============================================================================
def _init_engine(dataset_root: str, rebuild_cache: bool):
    """
    Khởi tạo hoặc tái sử dụng Gallery Index.
    Trả về (engine, error_msg). Nếu thành công error_msg = None.
    """
    global _state

    if not dataset_root or not dataset_root.strip():
        return None, "Vui lòng nhập đường dẫn Dataset!"

    dataset_root = dataset_root.strip()
    if not os.path.isdir(dataset_root):
        return None, f"Không tìm thấy thư mục:\n{dataset_root}"

    gallery_dir = os.path.join(dataset_root, "bounding_box_test")
    if not os.path.isdir(gallery_dir):
        return None, (
            f"Không tìm thấy bounding_box_test/ trong:\n{dataset_root}\n"
            "Hãy kiểm tra lại đường dẫn Dataset."
        )

    need_rebuild = (
        _state["gallery_index"] is None
        or _state["dataset_root"] != dataset_root
        or rebuild_cache
    )

    if need_rebuild:
        try:
            dataset   = Market1501Dataset(dataset_root)
            extractor = HOGFeatureExtractor(
                img_height=128, img_width=64,
                orientations=9,
                pixels_per_cell=(8, 8),
                cells_per_block=(2, 2),
            )
            cache_dir   = os.path.join(dataset_root, "_reid_cache")
            gallery_idx = GalleryIndex(dataset, extractor, cache_dir=cache_dir)
            gallery_idx.build_or_load(force_rebuild=rebuild_cache)

            _state["gallery_index"] = gallery_idx
            _state["engine"]        = ReIDSearchEngine(gallery_idx, extractor, metric="cosine")
            _state["dataset_root"]  = dataset_root
            _state["dataset_obj"]   = dataset
        except Exception as e:
            return None, f"Lỗi khởi tạo Gallery Index:\n{e}"

    return _state["engine"], None


def _build_gallery_and_info(search_output: dict, top_k: int):
    """
    Chuyển đổi output ReIDSearchEngine thành (gallery_images, info_md).
    Dùng chung cho cả 3 chế độ.
    """
    results = search_output["results"]
    qi      = search_output["query_info"]
    elapsed = search_output["search_time"]

    # -- Gallery cards ---------------------------------------------------
    gallery_images = []
    for r in results:
        card = _make_result_card(
            img_path   = r["path"],
            rank       = r["rank"],
            sim        = r["similarity_score"],
            dist       = r["distance"],
            pid        = r["pid"],
            cam_label  = r["cam_label"],
            is_correct = r["is_correct_match"],
        )
        caption = (
            f"#{r['rank']} | Sim={r['similarity_score']:.3f} | "
            f"{r['cam_label']} | PID:{r['pid']:04d}"
        )
        gallery_images.append((card, caption))

    # -- Markdown info table ---------------------------------------------
    pid_str = f"`{qi['pid']:04d}`" if qi["pid"] else "`---`"
    cid_str = f"`Camera_{qi['cid']}`" if qi["cid"] else "`---`"

    lines = [
        f"### Query: `{qi['fname']}`",
        f"| Thuộc tính | Giá trị |",
        f"|---|---|",
        f"| Person ID | {pid_str} |",
        f"| Camera | {cid_str} |",
        f"| HOG dim | `{qi['feat_dim']}` chiều |",
        f"| Thời gian | `{elapsed:.4f}s` |",
        f"| Top-k | `{len(results)}` kết quả |",
        "",
        f"| Rank | Similarity | Distance | PID | Camera | Đúng? | File |",
        f"|:---:|:---:|:---:|:---:|:---:|:---:|:---|",
    ]
    for r in results:
        if r["is_correct_match"] is True:    m = "CÓ"
        elif r["is_correct_match"] is False: m = "KHÔNG"
        else:                                m = "---"
        bar = "=" * int(r["similarity_score"] * 8) + "." * (8 - int(r["similarity_score"] * 8))
        lines.append(
            f"| **#{r['rank']}** "
            f"| `{r['similarity_score']:.4f}` `[{bar}]` "
            f"| `{r['distance']:.5f}` "
            f"| `{r['pid']:04d}` "
            f"| {r['cam_label']} "
            f"| **{m}** "
            f"| `{r['fname']}` |"
        )

    has_gt = results[0]["is_correct_match"] is not None
    if has_gt:
        correct = sum(1 for r in results if r["is_correct_match"])
        r1 = results[0]["is_correct_match"]
        lines += [
            "", "---",
            "### Đánh giá (Ground Truth)",
            f"| Độ đo | Giá trị |",
            f"|---|---|",
            f"| Rank-1 | {'**ĐÚNG**' if r1 else '**SAI**'} |",
            f"| Đúng/Top-{top_k} | `{correct}/{len(results)}` |",
            f"| Precision@{top_k} | `{correct/len(results):.2%}` |",
        ]

    return gallery_images, "\n".join(lines)


# ==============================================================================
# CALLBACKS CHO 3 CHẾ ĐỘ
# ==============================================================================

# -- CHẾ ĐỘ 1: Đơn ảnh -------------------------------------------------------
def search_single(query_image, dataset_root: str, top_k: int, rebuild_cache: bool):
    """Tìm kiếm với 1 ảnh Query duy nhất."""
    if query_image is None:
        return [], "_Vui lòng upload ảnh Query trước!_", "Chưa có ảnh Query"

    engine, err = _init_engine(dataset_root, rebuild_cache)
    if err:
        return [], f"_Lỗi: {err}_", f"Lỗi: {err[:60]}"

    tmp = os.path.join(tempfile.gettempdir(), "reid_query_single.jpg")
    if isinstance(query_image, np.ndarray):
        query_image = Image.fromarray(query_image)
    query_image.save(tmp)

    q_pid, q_cid = _parse_market1501_fname(getattr(query_image, "name", "") or "")

    try:
        t0  = time.time()
        out = engine.search(tmp, top_k=int(top_k), query_pid=q_pid, query_cid=q_cid)
        elapsed = time.time() - t0
    except Exception as e:
        return [], f"_Lỗi tìm kiếm: {e}_", f"Lỗi: {e}"

    gallery, info = _build_gallery_and_info(out, int(top_k))
    r1 = out["results"][0]["is_correct_match"] if out["results"] else None
    gt_str = ""
    if r1 is True:  gt_str = "| Rank-1 ĐÚNG"
    elif r1 is False: gt_str = "| Rank-1 SAI"
    status = f"Xong! {len(out['results'])} kết quả {gt_str} | {elapsed:.3f}s"
    return gallery, info, status


# -- CHẾ ĐỘ 2: Nhiều ảnh ------------------------------------------------------
def search_multi(query_files, dataset_root: str, top_k: int, rebuild_cache: bool):
    """
    Tìm kiếm với nhiều ảnh Query cùng lúc.
    query_files: list of file paths từ gr.File(file_count="multiple")
    """
    if not query_files:
        return [], "_Vui lòng upload ít nhất 1 ảnh._", "Chưa có ảnh"

    engine, err = _init_engine(dataset_root, rebuild_cache)
    if err:
        return [], f"_Lỗi: {err}_", f"Lỗi: {err[:60]}"

    all_gallery = []
    all_info    = []
    total_start = time.time()

    for i, file_obj in enumerate(query_files):
        fpath = file_obj if isinstance(file_obj, str) else file_obj.name
        if not os.path.isfile(fpath):
            continue

        q_pid, q_cid = _parse_market1501_fname(fpath)

        try:
            out = engine.search(fpath, top_k=int(top_k),
                                query_pid=q_pid, query_cid=q_cid)
        except Exception as e:
            all_info.append(f"_Lỗi với ảnh {os.path.basename(fpath)}: {e}_")
            continue

        # -- Separator card: thumbnail ảnh query viền tím ---------------
        try:
            q_img = Image.open(fpath).convert("RGB")
            sep_w, sep_h = 144, 296
            sep = Image.new("RGB", (sep_w, sep_h), (240, 235, 255))
            d   = ImageDraw.Draw(sep)
            for k in range(6):
                d.rectangle([k, k, sep_w-1-k, sep_h-1-k], outline=(110, 50, 200))
            q_thumb = q_img.resize((130, 238), Image.LANCZOS)
            sep.paste(q_thumb, (7, 30))
            font = ImageFont.load_default()
            d.text((7, 8),   f"QUERY #{i+1}", font=font, fill=(90, 25, 175))
            fname_short = os.path.basename(fpath)[:17]
            d.text((7, 278), fname_short, font=font, fill=(90, 25, 175))
            all_gallery.append((sep, f"-- Query #{i+1}: {os.path.basename(fpath)} --"))
        except Exception:
            pass

        g, info_md = _build_gallery_and_info(out, int(top_k))
        all_gallery.extend(g)
        all_info.append(info_md)

    total_elapsed = time.time() - total_start
    combined_info = "\n\n---\n\n".join(all_info) if all_info else "_Không có kết quả._"
    status = f"Xong! {len(query_files)} ảnh Query | {total_elapsed:.2f}s"
    return all_gallery, combined_info, status


# -- CHẾ ĐỘ 3: Demo tự động ---------------------------------------------------
def search_demo(dataset_root: str, top_k: int, demo_n: int, rebuild_cache: bool):
    """Demo tự động: random N ảnh từ tập query Market-1501."""
    engine, err = _init_engine(dataset_root, rebuild_cache)
    if err:
        return [], f"_Lỗi: {err}_", f"Lỗi: {err[:60]}"

    dataset_obj = _state.get("dataset_obj")
    if dataset_obj is None or not dataset_obj.query:
        return [], "_Chưa load Dataset. Nhập đường dẫn và thử lại._", "Dataset chưa load"

    np.random.seed(int(time.time()) % 9999)
    n_sample = min(int(demo_n), len(dataset_obj.query))
    sample_records = np.random.choice(dataset_obj.query, size=n_sample, replace=False)

    all_gallery = []
    all_info    = []
    total_start = time.time()

    for i, rec in enumerate(sample_records):
        try:
            out = engine.search(
                rec["path"], top_k=int(top_k),
                query_pid=rec["pid"], query_cid=rec["cid"]
            )
        except Exception as e:
            all_info.append(f"_Lỗi query {rec['fname']}: {e}_")
            continue

        # -- Separator card: thumbnail query viền tím -------------------
        try:
            q_img = Image.open(rec["path"]).convert("RGB")
            sep_w, sep_h = 144, 296
            sep = Image.new("RGB", (sep_w, sep_h), (240, 235, 255))
            d   = ImageDraw.Draw(sep)
            for k in range(6):
                d.rectangle([k, k, sep_w-1-k, sep_h-1-k], outline=(110, 50, 200))
            q_thumb = q_img.resize((130, 238), Image.LANCZOS)
            sep.paste(q_thumb, (7, 30))
            font = ImageFont.load_default()
            d.text((7, 8),   f"DEMO #{i+1}", font=font, fill=(90, 25, 175))
            d.text((7, 278), f"PID:{rec['pid']:04d} C{rec['cid']}", font=font, fill=(90, 25, 175))
            all_gallery.append((sep, f"-- Demo Query #{i+1} | PID:{rec['pid']:04d} --"))
        except Exception:
            pass

        g, info_md = _build_gallery_and_info(out, int(top_k))
        all_gallery.extend(g)
        all_info.append(info_md)

    total_elapsed = time.time() - total_start
    combined_info = "\n\n---\n\n".join(all_info) if all_info else "_Không có kết quả._"
    status = f"Demo xong! {n_sample} query ngẫu nhiên | {total_elapsed:.2f}s"
    return all_gallery, combined_info, status


# ==============================================================================
# XÂY DỰNG GIAO DIỆN GRADIO
# ==============================================================================
def build_ui():
    """
    CSS theo nguyên tắc tương phản cao (WCAG AA):
      - Chữ tối (#1a1f2e, #0f172a) trên nền sáng (#ffffff, #f0f2f8)
      - Chữ sáng (#ffffff, #bfdbfe) trên nền tối (#1e3a8a, #1d4ed8)
      - Không dùng chữ xám nhạt trên nền tối hoặc chữ trắng trên nền xám nhạt
    """
    css = """
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600&family=DM+Mono:wght@400;500&display=swap');

    /* == Nền trắng toàn trang == */
    body, .gradio-container {
        background: #f0f2f8 !important;
        font-family: 'DM Sans', sans-serif !important;
        color: #1a1f2e !important;
    }

    /* == Header xanh đậm / chữ trắng == */
    .app-header {
        background: linear-gradient(135deg, #1e3a8a 0%, #1d4ed8 60%, #2563eb 100%);
        border-radius: 14px;
        padding: 22px 28px 18px;
        margin-bottom: 16px;
        box-shadow: 0 4px 18px rgba(30,58,138,0.3);
    }
    .app-header h1 {
        font-family: 'DM Mono', monospace !important;
        font-size: 1.4rem !important;
        font-weight: 600 !important;
        color: #ffffff !important;
        margin: 0 0 5px !important;
    }
    .app-header p {
        color: #bfdbfe !important;
        font-size: 0.82rem !important;
        margin: 0 !important;
        font-family: 'DM Mono', monospace !important;
    }

    /* == Nhãn panel == */
    .panel-label {
        font-size: 0.68rem !important;
        font-weight: 600 !important;
        color: #374151 !important;
        text-transform: uppercase;
        letter-spacing: 2px;
        padding-bottom: 10px;
        border-bottom: 2px solid #d1d5db;
        margin-bottom: 14px;
        font-family: 'DM Mono', monospace !important;
    }

    /* == Label Gradio component (Ảnh Query, Slider...) == */
    label > span, .label-wrap span, .svelte-1gfkn6j {
        font-family: 'DM Sans', sans-serif !important;
        font-size: 0.83rem !important;
        font-weight: 600 !important;
        color: #111827 !important;
    }

    /* == Info text nhỏ dưới component == */
    .info-text, span.text-sm {
        color: #4b5563 !important;
        font-size: 0.77rem !important;
    }

    /* == Textbox / Input == */
    input[type="text"], textarea {
        background: #ffffff !important;
        border: 1.5px solid #d1d5db !important;
        border-radius: 8px !important;
        color: #111827 !important;
        font-family: 'DM Mono', monospace !important;
        font-size: 0.84rem !important;
        padding: 8px 12px !important;
    }
    input[type="text"]:focus, textarea:focus {
        border-color: #2563eb !important;
        box-shadow: 0 0 0 3px rgba(37,99,235,0.15) !important;
        outline: none !important;
    }
    input[type="text"]::placeholder, textarea::placeholder {
        color: #9ca3af !important;
    }

    /* == Tab buttons (3 chế độ) == */
    button[role="tab"] {
        font-family: 'DM Mono', monospace !important;
        font-size: 0.78rem !important;
        font-weight: 500 !important;
        color: #374151 !important;
        background: #f3f4f6 !important;
        border: 1px solid #d1d5db !important;
        border-radius: 8px 8px 0 0 !important;
        padding: 8px 16px !important;
        margin-right: 3px !important;
        transition: all 0.15s !important;
    }
    button[role="tab"]:hover {
        background: #dbeafe !important;
        color: #1d4ed8 !important;
        border-color: #93c5fd !important;
    }
    button[role="tab"][aria-selected="true"] {
        background: #1d4ed8 !important;
        color: #ffffff !important;
        border-color: #1d4ed8 !important;
        font-weight: 600 !important;
    }

    /* == Nút Search (primary) == */
    button.primary, .search-btn {
        background: linear-gradient(135deg, #1d4ed8, #2563eb) !important;
        border: none !important;
        border-radius: 10px !important;
        color: #ffffff !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 0.93rem !important;
        font-weight: 600 !important;
        padding: 11px 22px !important;
        box-shadow: 0 3px 10px rgba(37,99,235,0.4) !important;
        transition: all 0.2s !important;
        cursor: pointer !important;
        letter-spacing: 0.2px !important;
    }
    button.primary:hover, .search-btn:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 5px 18px rgba(37,99,235,0.5) !important;
    }
    button.primary:active { transform: translateY(0) !important; }

    /* == Nút Demo (secondary) == */
    button.secondary, .demo-btn {
        background: #eff6ff !important;
        border: 2px solid #3b82f6 !important;
        border-radius: 10px !important;
        color: #1d4ed8 !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 0.93rem !important;
        font-weight: 600 !important;
        padding: 11px 22px !important;
        transition: all 0.2s !important;
        cursor: pointer !important;
    }
    button.secondary:hover, .demo-btn:hover {
        background: #dbeafe !important;
        border-color: #1d4ed8 !important;
        transform: translateY(-1px) !important;
    }

    /* == Slider == */
    input[type="range"] { accent-color: #2563eb !important; }

    /* == Status bar == */
    .status-box textarea {
        background: #f0fdf4 !important;
        border: 1.5px solid #6ee7b7 !important;
        color: #064e3b !important;
        font-family: 'DM Mono', monospace !important;
        font-size: 0.82rem !important;
        border-radius: 8px !important;
        font-weight: 500 !important;
    }

    /* == Accordion == */
    .gradio-accordion {
        background: #f9fafb !important;
        border: 1px solid #e5e7eb !important;
        border-radius: 10px !important;
    }
    .gradio-accordion summary {
        color: #1f2937 !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
        font-family: 'DM Sans', sans-serif !important;
    }

    /* == Upload zone == */
    .upload-zone {
        border: 2px dashed #93c5fd !important;
        border-radius: 10px !important;
        background: #eff6ff !important;
    }
    .upload-zone:hover {
        border-color: #2563eb !important;
        background: #dbeafe !important;
    }

    /* == Legend box == */
    .legend-box {
        background: #eff6ff;
        border: 1px solid #bfdbfe;
        border-radius: 8px;
        padding: 10px 14px;
        font-size: 0.82rem;
        color: #1e3a8a;
        margin-bottom: 10px;
        line-height: 1.5;
    }

    /* == Markdown output == */
    .gradio-markdown {
        color: #111827 !important;
        background: #ffffff !important;
        border: 1px solid #e5e7eb !important;
        border-radius: 12px !important;
        padding: 18px !important;
        font-family: 'DM Sans', sans-serif !important;
    }
    .gradio-markdown h3 {
        color: #1e3a8a !important;
        font-family: 'DM Mono', monospace !important;
        font-size: 0.9rem !important;
        border-bottom: 1px solid #e5e7eb;
        padding-bottom: 6px;
        margin-top: 16px !important;
    }
    .gradio-markdown table {
        width: 100% !important;
        border-collapse: collapse !important;
        font-size: 0.79rem !important;
        font-family: 'DM Mono', monospace !important;
        color: #111827 !important;
    }
    .gradio-markdown th {
        background: #1e3a8a !important;
        color: #ffffff !important;
        padding: 7px 10px !important;
        border: 1px solid #1e3a8a !important;
        text-align: center !important;
    }
    .gradio-markdown td {
        padding: 5px 10px !important;
        border: 1px solid #e5e7eb !important;
        color: #111827 !important;
        text-align: center !important;
    }
    .gradio-markdown td:last-child { text-align: left !important; }
    .gradio-markdown tr:nth-child(even) td { background: #f9fafb !important; }
    .gradio-markdown code {
        background: #eff6ff !important;
        color: #1d4ed8 !important;
        padding: 1px 5px !important;
        border-radius: 4px !important;
        font-family: 'DM Mono', monospace !important;
        font-size: 0.8rem !important;
    }
    .gradio-markdown strong { color: #111827 !important; font-weight: 700 !important; }
    .gradio-markdown hr { border-color: #e5e7eb !important; margin: 16px 0 !important; }

    /* == Gallery == */
    .gradio-gallery {
        background: #f9fafb !important;
        border: 1px solid #e5e7eb !important;
        border-radius: 12px !important;
    }
    .gradio-gallery img {
        border-radius: 4px !important;
        transition: transform 0.18s !important;
    }
    .gradio-gallery img:hover { transform: scale(1.06) !important; }

    /* == Checkbox == */
    input[type="checkbox"] { accent-color: #2563eb !important; }
    """

    with gr.Blocks(
        css=css,
        title="Re-ID Search Engine",
        theme=gr.themes.Default(
            primary_hue="blue",
            neutral_hue="slate",
            font=gr.themes.GoogleFont("DM Sans"),
            font_mono=gr.themes.GoogleFont("DM Mono"),
        ),
    ) as demo:

        # ============================================================
        # HEADER
        # ============================================================
        gr.HTML("""
        <div class="app-header">
            <h1>&#128269; Re-ID Search Engine</h1>
            <p>Hệ thống truy xuất người đi lạc qua mạng lưới Camera giám sát
               &nbsp;&middot;&nbsp; HOG Features + Cosine Distance
               &nbsp;&middot;&nbsp; Market-1501 Dataset</p>
        </div>
        """)

        # ============================================================
        # LAYOUT CHÍNH: 2 CỘT
        # ============================================================
        with gr.Row(equal_height=False):

            # ────────────────────────────────────────────
            # CỘT TRÁI: INPUT (3 Tabs chế độ)
            # ────────────────────────────────────────────
            with gr.Column(scale=1, min_width=340):
                gr.HTML('<div class="panel-label">&#x2B06; Input &mdash; Chọn chế độ tìm kiếm</div>')

                # -- Cấu hình chung (trên cả 3 tab) --------------------
                dataset_root_box = gr.Textbox(
                    label="Đường dẫn Dataset (dataset_root)",
                    placeholder="VD: C:/Users/UIT/Market-1501-v15.09.15",
                    lines=1,
                    info="Thư mục chứa bounding_box_test/ và query/",
                )
                top_k_slider = gr.Slider(
                    minimum=1, maximum=50, value=10, step=1,
                    label="Số kết quả Top-k",
                    info="Số ảnh Gallery trả về cho mỗi Query (1 đến 50)",
                )

                # ══ 3 TABS CHẾ ĐỘ INPUT ══════════════════════════════
                with gr.Tabs():

                    # -- TAB 1: Đơn ảnh ---------------------------------
                    with gr.Tab("Đơn ảnh"):
                        gr.HTML("""<div class="legend-box">
                            <b>Chế độ 1:</b> Upload <b>1 ảnh</b> của người cần tìm.
                            Kéo thả ảnh vào ô dưới hoặc click để chọn file.
                        </div>""")
                        single_query_img = gr.Image(
                            label="Ảnh Query",
                            type="pil",
                            elem_classes=["upload-zone"],
                            height=220,
                            sources=["upload"],
                        )
                        single_search_btn = gr.Button(
                            "Tìm kiếm",
                            variant="primary",
                            elem_classes=["search-btn"],
                        )

                    # -- TAB 2: Nhiều ảnh --------------------------------
                    with gr.Tab("Nhiều ảnh"):
                        gr.HTML("""<div class="legend-box">
                            <b>Chế độ 2:</b> Upload <b>nhiều ảnh</b> cùng lúc
                            (Ctrl+Click để chọn nhiều file).
                            Kết quả Top-k hiển thị riêng cho từng ảnh Query.
                        </div>""")
                        multi_query_files = gr.File(
                            label="Danh sách ảnh Query (chọn nhiều file)",
                            file_count="multiple",
                            file_types=["image"],
                            height=180,
                        )
                        multi_search_btn = gr.Button(
                            "Tìm kiếm tất cả",
                            variant="primary",
                            elem_classes=["search-btn"],
                        )

                    # -- TAB 3: Demo tự động ----------------------------
                    with gr.Tab("Demo tự động"):
                        gr.HTML("""<div class="legend-box">
                            <b>Chế độ 3:</b> Không cần upload ảnh.
                            Hệ thống tự chọn ngẫu nhiên ảnh Query từ tập Market-1501
                            và hiển thị kết quả tự động.
                        </div>""")
                        demo_n_slider = gr.Slider(
                            minimum=1, maximum=10, value=5, step=1,
                            label="Số ảnh Query ngẫu nhiên",
                            info="Hệ thống chọn ngẫu nhiên N ảnh từ tập query",
                        )
                        demo_run_btn = gr.Button(
                            "Chạy Demo",
                            variant="secondary",
                            elem_classes=["demo-btn"],
                        )
                        gr.Markdown(
                            "_Mỗi lần nhấn sẽ chọn bộ ảnh ngẫu nhiên khác nhau._"
                        )

                # -- Tùy chọn nâng cao (chung cho cả 3 tab) ------------
                with gr.Accordion("Tùy chọn nâng cao", open=False):
                    rebuild_cache_chk = gr.Checkbox(
                        label="Rebuild Gallery Cache",
                        value=False,
                        info=(
                            "Tích nếu Dataset thay đổi. "
                            "Lần đầu chạy mất ~5-10 phút. "
                            "Cache lưu tại <dataset_root>/_reid_cache/"
                        ),
                    )

                # -- Thanh trạng thái ----------------------------------
                status_bar = gr.Textbox(
                    label="Trạng thái",
                    value="Sẵn sàng - Chọn chế độ và nhấn Tìm kiếm.",
                    interactive=False,
                    max_lines=2,
                    elem_classes=["status-box"],
                )

            # ────────────────────────────────────────────
            # CỘT PHẢI: OUTPUT
            # ────────────────────────────────────────────
            with gr.Column(scale=2, min_width=520):
                gr.HTML('<div class="panel-label">&#x2B07; Output &mdash; Kết quả tìm kiếm</div>')

                gr.HTML("""<div class="legend-box">
                    <b>Chú thích viền card:</b>&nbsp;
                    <span style="color:#22a04b;font-weight:700;">&#9632; Xanh lá</span>
                    = Đúng người (MATCH) &nbsp;&middot;&nbsp;
                    <span style="color:#d23228;font-weight:700;">&#9632; Đỏ</span>
                    = Sai người (WRONG) &nbsp;&middot;&nbsp;
                    <span style="color:#1e6ed2;font-weight:700;">&#9632; Xanh dương</span>
                    = Không biết Ground Truth (QUERY)
                </div>""")

                with gr.Tabs():
                    with gr.Tab("Thư viện ảnh (Gallery)"):
                        result_gallery = gr.Gallery(
                            label="Top-k Kết quả",
                            show_label=False,
                            columns=5,
                            rows=3,
                            height=580,
                            object_fit="contain",
                            allow_preview=True,
                            preview=True,
                        )

                    with gr.Tab("Bảng Metadata"):
                        result_info = gr.Markdown(
                            value="_Kết quả sẽ hiển thị ở đây sau khi tìm kiếm._"
                        )

        # ============================================================
        # HƯỚNG DẪN SỬ DỤNG
        # ============================================================
        with gr.Accordion("Hướng dẫn sử dụng", open=False):
            gr.Markdown("""
### Ba chế độ tìm kiếm

| Chế độ | Khi nào dùng | Thao tác |
|---|---|---|
| **Đơn ảnh** | Có 1 ảnh người cần tìm | Upload 1 ảnh -> Tìm kiếm |
| **Nhiều ảnh** | Có nhiều ảnh từ các góc khác nhau | Upload nhiều file -> Tìm kiếm tất cả |
| **Demo** | Muốn xem hệ thống hoạt động | Nhập Dataset path -> Chạy Demo |

### Đọc kết quả Gallery

| Thông số | Ý nghĩa |
|---|---|
| **Sim** | Similarity Score 0.0-1.0. Càng gần 1.0 càng giống Query |
| **Dist** | Cosine Distance. Càng gần 0.0 càng giống |
| **PID** | Person ID - danh tính người trong ảnh Gallery |
| **Camera_N** | Số hiệu camera đã ghi hình ảnh đó |

### Lưu ý kỹ thuật
- **Lần đầu chạy**: ~5-10 phút để build HOG Cache cho ~19.000 ảnh Gallery.
- **Lần sau**: ~2 giây (load từ cache).
- Ảnh query tốt nhất là ảnh **toàn thân, nền đơn giản**.
- Upload ảnh từ tập Market-1501: hệ thống tự nhận PID/CID và tô màu viền Đúng/Sai.
            """)

        # ============================================================
        # KẾT NỐI SỰ KIỆN
        # ============================================================

        # Tab 1: Đơn ảnh
        single_search_btn.click(
            fn=search_single,
            inputs=[single_query_img, dataset_root_box, top_k_slider, rebuild_cache_chk],
            outputs=[result_gallery, result_info, status_bar],
            show_progress="full",
        )

        # Tab 2: Nhiều ảnh
        multi_search_btn.click(
            fn=search_multi,
            inputs=[multi_query_files, dataset_root_box, top_k_slider, rebuild_cache_chk],
            outputs=[result_gallery, result_info, status_bar],
            show_progress="full",
        )

        # Tab 3: Demo tự động
        demo_run_btn.click(
            fn=search_demo,
            inputs=[dataset_root_box, top_k_slider, demo_n_slider, rebuild_cache_chk],
            outputs=[result_gallery, result_info, status_bar],
            show_progress="full",
        )

    return demo


# ==============================================================================
# ENTRY POINT
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Re-ID Search Engine - Web UI v2")
    parser.add_argument(
        "--share", action="store_true",
        help="Tạo public link qua Gradio tunnel (https://xxx.gradio.live, miễn phí 72h)",
    )
    parser.add_argument(
        "--port", type=int, default=7860,
        help="Cổng server local (mặc định: 7860)",
    )
    args = parser.parse_args()

    print("+==========================================================+")
    print("|   Re-ID Search Engine v2  -  Web UI (Gradio)             |")
    print("|==========================================================|")
    if args.share:
        print("|  Chế độ: PUBLIC - Đang tạo Gradio public link...         |")
        print("|  Link sẽ hiện ra bên dưới khi server khởi động xong.     |")
    else:
        print(f"|  Chế độ: LOCAL  ->  http://localhost:{args.port:<24}|")
        print("|  Để tạo public link: python reid_app.py --share          |")
    print("+==========================================================+")

    demo = build_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
        show_error=True,
        inbrowser=not args.share,
    )