import os
import cv2
import numpy as np
import gradio as gr

from skimage.feature import hog

# ============================================================
# CONFIG
# ============================================================

IMG_H = 128
IMG_W = 64

RGB_BINS = 8

TOP_K = 5

DATASET_DIR = 'Market-1501-v15.09.15'

GALLERY_DIR = os.path.join(
    DATASET_DIR,
    'bounding_box_test'
)

# ============================================================
# LOAD FEATURES
# ============================================================

print("🔵 Loading gallery features...")

data = np.load(
    'gallery_features.npz',
    allow_pickle=True
)

g_feats = data['g_feats']

g_pids = data['g_pids']

g_camids = data['g_camids']

g_fnames = data['g_fnames']

print("✅ Gallery loaded!")

print("Gallery shape:", g_feats.shape)

# ============================================================
# PREPROCESS
# ============================================================

def preprocess(img):

    img = cv2.resize(
        img,
        (IMG_W, IMG_H),
        interpolation=cv2.INTER_LINEAR
    )

    # ========================================================
    # CLAHE ON Y CHANNEL
    # ========================================================

    ycrcb = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2YCrCb
    )

    y, cr, cb = cv2.split(ycrcb)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    y = clahe.apply(y)

    ycrcb = cv2.merge([y, cr, cb])

    img = cv2.cvtColor(
        ycrcb,
        cv2.COLOR_YCrCb2BGR
    )

    # ========================================================
    # FLOAT NORMALIZE
    # ========================================================

    img = img.astype(np.float32) / 255.0

    img = np.clip(img, 0, 1)

    return img

# ============================================================
# RGB HISTOGRAM
# ============================================================

def extract_rgb_hist(img):

    hist = cv2.calcHist(
        [img],
        [0, 1, 2],
        None,
        [RGB_BINS, RGB_BINS, RGB_BINS],
        [0, 1, 0, 1, 0, 1]
    )

    hist = hist.flatten().astype(np.float32)

    # ========================================================
    # L1 NORMALIZE
    # ========================================================

    hist /= (hist.sum() + 1e-8)

    return hist

# ============================================================
# HOG FEATURE
# ============================================================

def extract_hog_feature(gray):

    feat = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm='L2-Hys',
        transform_sqrt=True,
        feature_vector=True
    )

    return feat.astype(np.float32)

# ============================================================
# FINAL FEATURE
# ============================================================

def extract_feature(img):

    img = preprocess(img)

    rgb_feat = extract_rgb_hist(img)

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    hog_feat = extract_hog_feature(gray)

    feat = np.concatenate([
        rgb_feat,
        hog_feat
    ]).astype(np.float32)

    # ========================================================
    # GLOBAL L2 NORMALIZE
    # ========================================================

    norm = np.linalg.norm(feat)

    if norm > 0:
        feat /= norm

    return feat

# ============================================================
# SEARCH
# ============================================================

def search_person(query_img):

    # ========================================================
    # GRADIO IMAGE = RGB
    # CONVERT TO BGR FOR OPENCV PIPELINE
    # ========================================================

    query_img = cv2.cvtColor(
        query_img,
        cv2.COLOR_RGB2BGR
    )

    # ========================================================
    # EXTRACT FEATURE
    # ========================================================

    q_feat = extract_feature(query_img)

    # ========================================================
    # COSINE SIMILARITY
    # ========================================================

    sims = np.dot(
        g_feats,
        q_feat
    )

    # ========================================================
    # SORT DESCENDING
    # ========================================================

    top_idx = np.argsort(
        -sims
    )[:TOP_K]

    # ========================================================
    # DEBUG
    # ========================================================

    best_idx = top_idx[0]

    print()
    print("================================================")

    print("TOP-1:")
    print(g_fnames[best_idx])

    print()

    print("SIMILARITY:")
    print(float(sims[best_idx]))

    print("================================================")

    results = []

    # ========================================================
    # BUILD RESULTS
    # ========================================================

    for rank, idx in enumerate(top_idx):

        gallery_path = os.path.join(
            GALLERY_DIR,
            str(g_fnames[idx])
        )

        gallery_img = cv2.cvtColor(
            cv2.imread(gallery_path),
            cv2.COLOR_BGR2RGB
        )

        sim = float(sims[idx])

        caption = (
            f"Rank #{rank+1}\n"
            f"ID: {g_pids[idx]}\n"
            f"Cam: {g_camids[idx]}\n"
            f"Sim: {sim:.6f}"
        )

        results.append(
            (gallery_img, caption)
        )

    return results

# ============================================================
# UI
# ============================================================

title = "🔍 HOG-RGB Person Re-ID"

description = """
Upload ảnh từ Market-1501 gallery để test retrieval.
"""

demo = gr.Interface(
    fn=search_person,

    inputs=gr.Image(
        type='numpy',
        label='Upload Query Image'
    ),

    outputs=gr.Gallery(
        label='Top-K Results',
        columns=5,
        height='auto'
    ),

    title=title,

    description=description
)

demo.launch(share=True)