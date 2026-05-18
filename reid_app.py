"""
==============================================================================
  HỆ THỐNG TRUY XUẤT NGƯỜI ĐI LẠC - Web UI (Gradio)
  Module: reid_app.py
==============================================================================

Phiên bản FULL tương thích RGB-HOG pipeline mới

Tính năng:
✅ RGB-HOG Search Engine
✅ Upload ảnh query trực tiếp
✅ Demo random query
✅ Top-k retrieval
✅ Gallery preview
✅ Similarity score
✅ Correct / Incorrect match
✅ Multi-image visualization
✅ Cache loading nhanh
✅ Gradio Web UI đẹp
"""

import os
import cv2
import time
import random
import warnings
import numpy as np
import gradio as gr
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ==============================================================================
# IMPORT BACKEND
# ==============================================================================
try:

    from hog_reid_pipeline import (
        Market1501Dataset,
        RGBHOGExtractor,
    )

    from reid_search import (
        GalleryIndex,
        ReIDSearchEngine,
    )

except ImportError as e:

    raise ImportError(
        f"""
Lỗi import backend:
{e}

Đảm bảo:
- hog_reid_pipeline.py
- reid_search.py
- reid_app.py
nằm cùng thư mục.
"""
    )


# ==============================================================================
# CONFIG
# ==============================================================================
DATASET_ROOT = "./Market-1501-v15.09.15"

TOP_K_DEFAULT = 5

CACHE_DIR = "./reid_cache"


# ==============================================================================
# LOAD SYSTEM
# ==============================================================================
print("=" * 70)
print("LOADING RGB-HOG RE-ID SYSTEM")
print("=" * 70)

dataset = Market1501Dataset(DATASET_ROOT)

extractor = RGBHOGExtractor(

    img_h=128,
    img_w=64,

    orientations=12,

    ppc=(8, 8),

    cpb=(2, 2),
)

gallery_index = GalleryIndex(

    dataset=dataset,
    extractor=extractor,
    cache_dir=CACHE_DIR,

)

gallery_index.build_or_load(rebuild=False)

engine = ReIDSearchEngine(
    gallery_index,
    extractor,
)

print("\nSYSTEM READY.")
print("=" * 70)


# ==============================================================================
# UTIL
# ==============================================================================
def load_rgb(path):

    img = cv2.imread(path)

    if img is None:
        return None

    img = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2RGB,
    )

    return img


def make_result_figure(
    query_img,
    results,
):

    n = len(results)

    fig = plt.figure(
        figsize=(3 * (n + 1), 4)
    )

    # ------------------------------------------------------------------
    # QUERY
    # ------------------------------------------------------------------
    plt.subplot(1, n + 1, 1)

    plt.imshow(query_img)

    plt.title("QUERY")

    plt.axis("off")

    # ------------------------------------------------------------------
    # RESULTS
    # ------------------------------------------------------------------
    for i, r in enumerate(results):

        img = load_rgb(r["path"])

        plt.subplot(1, n + 1, i + 2)

        plt.imshow(img)

        sim = r["similarity_score"]

        pid = r["pid"]

        cid = r["cid"]

        correct = r["is_correct_match"]

        if correct is None:
            status = "UNKNOWN"
        else:
            status = "TRUE" if correct else "FALSE"

        title = (
            f"Top-{i+1}\n"
            f"PID={pid} C{cid}\n"
            f"SIM={sim:.3f}\n"
            f"{status}"
        )

        plt.title(
            title,
            fontsize=9,
        )

        plt.axis("off")

    plt.tight_layout()

    return fig


# ==============================================================================
# SEARCH FUNCTION
# ==============================================================================
def search_person(
    query_image,
    top_k,
):

    if query_image is None:

        return (
            None,
            "❌ Hãy upload ảnh query.",
            None,
        )

    # ------------------------------------------------------------------
    # SAVE TEMP IMAGE
    # ------------------------------------------------------------------
    temp_path = "temp_query.jpg"

    query_bgr = cv2.cvtColor(
        query_image,
        cv2.COLOR_RGB2BGR,
    )

    cv2.imwrite(
        temp_path,
        query_bgr,
    )

    # ------------------------------------------------------------------
    # SEARCH
    # ------------------------------------------------------------------
    start = time.time()

    result = engine.search(

        query_image_path=temp_path,

        top_k=int(top_k),

        query_pid=None,
        query_cid=None,

    )

    elapsed = time.time() - start

    # ------------------------------------------------------------------
    # RESULT TEXT
    # ------------------------------------------------------------------
    text = []

    text.append("RGB-HOG PERSON RE-ID RESULTS")
    text.append("=" * 50)

    text.append(
        f"Search Time: {elapsed:.4f}s"
    )

    text.append(
        f"Top-K: {top_k}"
    )

    text.append("")

    for r in result["results"]:

        text.append(
            f"[Top-{r['rank']}] "
            f"PID={r['pid']} | "
            f"Camera={r['cid']} | "
            f"Similarity={r['similarity_score']:.4f}"
        )

    text = "\n".join(text)

    # ------------------------------------------------------------------
    # FIGURE
    # ------------------------------------------------------------------
    fig = make_result_figure(
        query_image,
        result["results"],
    )

    # ------------------------------------------------------------------
    # GALLERY
    # ------------------------------------------------------------------
    gallery = []

    for r in result["results"]:

        gallery.append((
            r["path"],
            f"Top-{r['rank']} | "
            f"SIM={r['similarity_score']:.4f}"
        ))

    return (
        fig,
        text,
        gallery,
    )


# ==============================================================================
# DEMO FUNCTION
# ==============================================================================
def random_demo(
    top_k,
):

    rec = random.choice(
        dataset.query
    )

    query_path = rec["path"]

    query_img = load_rgb(query_path)

    result = engine.search(

        query_image_path=query_path,

        top_k=int(top_k),

        query_pid=rec["pid"],
        query_cid=rec["cid"],

    )

    # ------------------------------------------------------------------
    # TEXT
    # ------------------------------------------------------------------
    text = []

    text.append("DEMO MODE")
    text.append("=" * 50)

    text.append(
        f"Query PID={rec['pid']} | "
        f"Camera={rec['cid']}"
    )

    text.append("")

    for r in result["results"]:

        correct = (
            "TRUE"
            if r["is_correct_match"]
            else "FALSE"
        )

        text.append(
            f"[Top-{r['rank']}] "
            f"PID={r['pid']} | "
            f"SIM={r['similarity_score']:.4f} | "
            f"{correct}"
        )

    text = "\n".join(text)

    # ------------------------------------------------------------------
    # FIG
    # ------------------------------------------------------------------
    fig = make_result_figure(
        query_img,
        result["results"],
    )

    # ------------------------------------------------------------------
    # GALLERY
    # ------------------------------------------------------------------
    gallery = []

    for r in result["results"]:

        correct = (
            "TRUE"
            if r["is_correct_match"]
            else "FALSE"
        )

        gallery.append((
            r["path"],
            f"Top-{r['rank']} | "
            f"{correct} | "
            f"SIM={r['similarity_score']:.4f}"
        ))

    return (
        query_img,
        fig,
        text,
        gallery,
    )


# ==============================================================================
# THEME
# ==============================================================================
css = """

.gradio-container {
    background: #f5f5f5;
}

h1 {
    text-align: center;
    color: #111;
}

.result-text textarea {
    font-size: 15px !important;
    font-family: Consolas !important;
}

"""


# ==============================================================================
# UI
# ==============================================================================
with gr.Blocks(
    css=css,
    theme=gr.themes.Soft(),
    title="RGB-HOG Person Re-ID",
) as demo:

    gr.Markdown(
        """
# RGB-HOG PERSON RE-ID SYSTEM

Person Re-Identification sử dụng:
- RGB HOG Feature
- Cosine Similarity
- Market1501 Dataset
"""
    )

    with gr.Row():

        with gr.Column(scale=1):

            query_input = gr.Image(
                label="Upload Query Image",
                type="numpy",
            )

            top_k_slider = gr.Slider(
                minimum=1,
                maximum=20,
                value=TOP_K_DEFAULT,
                step=1,
                label="Top-K",
            )

            search_btn = gr.Button(
                "SEARCH",
                variant="primary",
            )

            demo_btn = gr.Button(
                "RANDOM DEMO",
            )

        with gr.Column(scale=2):

            result_plot = gr.Plot(
                label="Visualization"
            )

            result_text = gr.Textbox(
                label="Search Results",
                lines=15,
                elem_classes=["result-text"],
            )

    gallery_output = gr.Gallery(

        label="Retrieved Images",

        columns=5,

        height="auto",

        object_fit="contain",
    )

    # ------------------------------------------------------------------
    # SEARCH EVENT
    # ------------------------------------------------------------------
    search_btn.click(

        fn=search_person,

        inputs=[
            query_input,
            top_k_slider,
        ],

        outputs=[
            result_plot,
            result_text,
            gallery_output,
        ],
    )

    # ------------------------------------------------------------------
    # DEMO EVENT
    # ------------------------------------------------------------------
    demo_btn.click(

        fn=random_demo,

        inputs=[
            top_k_slider,
        ],

        outputs=[
            query_input,
            result_plot,
            result_text,
            gallery_output,
        ],
    )


# ==============================================================================
# MAIN
# ==============================================================================
if __name__ == "__main__":

    demo.launch(

    server_name="127.0.0.1",

    server_port=7860,

    inbrowser=True,

    share=True,

    debug=True,
)