"""
==============================================================================
  HỆ THỐNG TRUY XUẤT NGƯỜI ĐI LẠC - RGB HOG Re-ID Pipeline
  Phiên bản tối ưu theo notebook hog(1).ipynb
==============================================================================

Đặc điểm phiên bản này:
✅ RGB-HOG (3 channel)
✅ Feature normalization
✅ Cosine distance bằng np.matmul
✅ Cache feature .npy
✅ Pre-sort ranking
✅ Rank-1 / mAP / F1 / CMC
✅ Visualization Top-k
✅ Tối ưu giống notebook
"""

import os
import re
import cv2
import time
import argparse
import warnings

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from skimage.feature import hog

warnings.filterwarnings("ignore")


# ==============================================================================
# DATASET
# ==============================================================================
class Market1501Dataset:

    _FNAME_PATTERN = re.compile(r"(-?\d+)_c(\d+)")

    def __init__(self, dataset_root):

        self.root = dataset_root

        self._validate_structure()

        print("[Dataset] Loading QUERY...")
        self.query = self._load_split("query")

        print("[Dataset] Loading GALLERY...")
        self.gallery = self._load_split("bounding_box_test")

        print(f"[Dataset] Query  : {len(self.query)}")
        print(f"[Dataset] Gallery: {len(self.gallery)}")

    def _validate_structure(self):

        required = [
            "query",
            "bounding_box_test",
        ]

        for d in required:

            path = os.path.join(self.root, d)

            if not os.path.isdir(path):
                raise FileNotFoundError(path)

    def _parse_filename(self, fname):

        match = self._FNAME_PATTERN.search(fname)

        if not match:
            return None, None

        pid = int(match.group(1))
        cid = int(match.group(2))

        return pid, cid

    def _load_split(self, split_name):

        split_dir = os.path.join(self.root, split_name)

        records = []

        for fname in sorted(os.listdir(split_dir)):

            if not fname.lower().endswith(".jpg"):
                continue

            pid, cid = self._parse_filename(fname)

            if pid is None:
                continue

            # remove junk images
            if pid <= 0:
                continue

            records.append({
                "path": os.path.join(split_dir, fname),
                "pid": pid,
                "cid": cid,
                "fname": fname,
            })

        return records

    def get_paths(self, split):

        data = self.query if split == "query" else self.gallery

        return [x["path"] for x in data]

    def get_pids(self, split):

        data = self.query if split == "query" else self.gallery

        return np.array([x["pid"] for x in data])

    def get_cids(self, split):

        data = self.query if split == "query" else self.gallery

        return np.array([x["cid"] for x in data])


# ==============================================================================
# RGB HOG EXTRACTOR
# ==============================================================================
class RGBHOGExtractor:

    def __init__(
        self,
        img_h=128,
        img_w=64,
        orientations=12,
        ppc=(8, 8),
        cpb=(2, 2),
        cache_dir="hog_cache",
    ):

        self.img_h = img_h
        self.img_w = img_w

        self.orientations = orientations
        self.ppc = ppc
        self.cpb = cpb

        self.cache_dir = cache_dir

        os.makedirs(cache_dir, exist_ok=True)

        dummy = np.zeros((img_h, img_w), dtype=np.float32)

        feat_single = hog(
            dummy,
            orientations=self.orientations,
            pixels_per_cell=self.ppc,
            cells_per_block=self.cpb,
            block_norm="L2-Hys",
            feature_vector=True,
        )

        # RGB => x3 dimension
        self.feature_dim = len(feat_single) * 3

        print(f"[RGB-HOG] Feature dimension: {self.feature_dim}")

    def preprocess(self, img_bgr):

        img = cv2.resize(
            img_bgr,
            (self.img_w, self.img_h),
            interpolation=cv2.INTER_LINEAR,
        )

        # BGR -> RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img_rgb = img_rgb.astype(np.float32) / 255.0

        return img_rgb

    def extract_single(self, img_path):

        img = cv2.imread(img_path)

        if img is None:
            return np.zeros(self.feature_dim, dtype=np.float32)

        img = self.preprocess(img)

        # HOG trên từng RGB channel
        hog_r = hog(
            img[:, :, 0],
            orientations=self.orientations,
            pixels_per_cell=self.ppc,
            cells_per_block=self.cpb,
            block_norm="L2-Hys",
            feature_vector=True,
        )

        hog_g = hog(
            img[:, :, 1],
            orientations=self.orientations,
            pixels_per_cell=self.ppc,
            cells_per_block=self.cpb,
            block_norm="L2-Hys",
            feature_vector=True,
        )

        hog_b = hog(
            img[:, :, 2],
            orientations=self.orientations,
            pixels_per_cell=self.ppc,
            cells_per_block=self.cpb,
            block_norm="L2-Hys",
            feature_vector=True,
        )

        feat = np.concatenate([
            hog_r,
            hog_g,
            hog_b,
        ])

        feat = feat.astype(np.float32)

        # Normalize feature vector
        norm = np.linalg.norm(feat)

        if norm > 0:
            feat = feat / norm

        return feat

    def extract_batch(self, img_paths, cache_name=None):

        cache_path = None

        if cache_name is not None:

            cache_path = os.path.join(
                self.cache_dir,
                cache_name,
            )

            if os.path.exists(cache_path):

                print(f"[CACHE] Loading: {cache_path}")

                return np.load(cache_path)

        feats = np.zeros(
            (len(img_paths), self.feature_dim),
            dtype=np.float32,
        )

        for i, path in enumerate(
            tqdm(img_paths, desc="Extract RGB-HOG", unit="img")
        ):

            feats[i] = self.extract_single(path)

        if cache_path is not None:

            np.save(cache_path, feats)

            print(f"[CACHE] Saved: {cache_path}")

        return feats


# ==============================================================================
# EVALUATOR
# ==============================================================================
class ReIDEvaluator:

    def __init__(self, top_k=10):

        self.top_k = top_k

    def get_matches(
        self,
        q_pid,
        q_cid,
        g_pids,
        g_cids,
        ranked_indices,
    ):

        is_junk = (g_pids == q_pid) & (g_cids == q_cid)

        is_match = (g_pids == q_pid) & (g_cids != q_cid)

        num_gt = is_match.sum()

        if num_gt == 0:
            return None, 0

        matches = []

        for idx in ranked_indices:

            if is_junk[idx]:
                continue

            matches.append(1 if is_match[idx] else 0)

        return np.array(matches), int(num_gt)

    def compute_rank1(
        self,
        sorted_indices,
        q_pids,
        g_pids,
        q_cids,
        g_cids,
    ):

        correct = 0
        valid = 0

        for q_idx in range(len(q_pids)):

            ranked_indices = sorted_indices[q_idx]

            matches, _ = self.get_matches(
                q_pids[q_idx],
                q_cids[q_idx],
                g_pids,
                g_cids,
                ranked_indices,
            )

            if matches is None:
                continue

            valid += 1

            if matches[0] == 1:
                correct += 1

        return correct / valid

    def compute_map(
        self,
        sorted_indices,
        q_pids,
        g_pids,
        q_cids,
        g_cids,
    ):

        ap_list = []

        for q_idx in range(len(q_pids)):

            ranked_indices = sorted_indices[q_idx]

            matches, num_gt = self.get_matches(
                q_pids[q_idx],
                q_cids[q_idx],
                g_pids,
                g_cids,
                ranked_indices,
            )

            if matches is None:
                continue

            correct = 0
            ap = 0.0

            for k, rel in enumerate(matches, start=1):

                if rel == 1:

                    correct += 1

                    precision = correct / k

                    ap += precision

            ap /= num_gt

            ap_list.append(ap)

        return np.mean(ap_list)

    def compute_f1_at_k(
        self,
        sorted_indices,
        q_pids,
        g_pids,
        q_cids,
        g_cids,
    ):

        k = self.top_k

        f1s = []
        precisions = []
        recalls = []

        for q_idx in range(len(q_pids)):

            ranked_indices = sorted_indices[q_idx]

            matches, num_gt = self.get_matches(
                q_pids[q_idx],
                q_cids[q_idx],
                g_pids,
                g_cids,
                ranked_indices,
            )

            if matches is None:
                continue

            topk = matches[:k]

            tp = topk.sum()

            precision = tp / len(topk)

            recall = tp / num_gt

            if precision + recall > 0:
                f1 = 2 * precision * recall / (precision + recall)
            else:
                f1 = 0

            f1s.append(f1)
            precisions.append(precision)
            recalls.append(recall)

        return {
            "f1": np.mean(f1s),
            "precision": np.mean(precisions),
            "recall": np.mean(recalls),
        }

    def compute_cmc(
        self,
        sorted_indices,
        q_pids,
        g_pids,
        q_cids,
        g_cids,
        ranks=[1, 5, 10],
    ):

        cmc = {r: [] for r in ranks}

        for q_idx in range(len(q_pids)):

            ranked_indices = sorted_indices[q_idx]

            matches, _ = self.get_matches(
                q_pids[q_idx],
                q_cids[q_idx],
                g_pids,
                g_cids,
                ranked_indices,
            )

            if matches is None:
                continue

            for r in ranks:

                if np.any(matches[:r] == 1):
                    cmc[r].append(1)
                else:
                    cmc[r].append(0)

        return {
            f"Rank-{r}": np.mean(cmc[r])
            for r in ranks
        }


# ==============================================================================
# VISUALIZATION
# ==============================================================================
def show_topk_results(
    query_idx,
    query_paths,
    gallery_paths,
    sorted_indices,
    q_pids,
    g_pids,
    top_k=5,
):

    q_path = query_paths[query_idx]

    ranked = sorted_indices[query_idx]

    plt.figure(figsize=(15, 4))

    q_img = cv2.cvtColor(
        cv2.imread(q_path),
        cv2.COLOR_BGR2RGB,
    )

    plt.subplot(1, top_k + 1, 1)
    plt.imshow(q_img)
    plt.title("QUERY")
    plt.axis("off")

    count = 0

    for idx in ranked:

        g_path = gallery_paths[idx]

        g_img = cv2.cvtColor(
            cv2.imread(g_path),
            cv2.COLOR_BGR2RGB,
        )

        correct = g_pids[idx] == q_pids[query_idx]

        count += 1

        plt.subplot(1, top_k + 1, count + 1)

        plt.imshow(g_img)

        plt.title(
            f"Top-{count}\n"
            f"{'TRUE' if correct else 'FALSE'}"
        )

        plt.axis("off")

        if count >= top_k:
            break

    plt.tight_layout()
    plt.show()


# ==============================================================================
# REPORT
# ==============================================================================
def print_report(results, top_k):

    print("\n" + "=" * 60)
    print("RGB-HOG PERSON RE-ID RESULTS")
    print("=" * 60)

    print(f"Rank-1 : {results['rank1']:.4f}")
    print(f"mAP    : {results['map']:.4f}")
    print(f"F1@{top_k}  : {results['f1']:.4f}")
    print(f"P@{top_k}   : {results['precision']:.4f}")
    print(f"R@{top_k}   : {results['recall']:.4f}")

    print("\nCMC:")

    for k, v in results['cmc'].items():
        print(f"{k}: {v:.4f}")

    print("=" * 60)


# ==============================================================================
# MAIN
# ==============================================================================
def main(args):

    total_start = time.time()

    # ------------------------------------------------------------------
    # LOAD DATASET
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 1 - LOAD DATASET")
    print("=" * 60)

    dataset = Market1501Dataset(args.dataset_root)

    query_paths = dataset.get_paths("query")
    gallery_paths = dataset.get_paths("gallery")

    q_pids = dataset.get_pids("query")
    g_pids = dataset.get_pids("gallery")

    q_cids = dataset.get_cids("query")
    g_cids = dataset.get_cids("gallery")

    # ------------------------------------------------------------------
    # FEATURE EXTRACTION
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 2 - RGB-HOG FEATURE EXTRACTION")
    print("=" * 60)

    extractor = RGBHOGExtractor(
        img_h=args.img_h,
        img_w=args.img_w,
        orientations=12,
        ppc=(8, 8),
        cpb=(2, 2),
    )

    print("\nExtract QUERY features...")

    query_feats = extractor.extract_batch(
        query_paths,
        cache_name="query_feats_rgb.npy",
    )

    print(query_feats.shape)

    print("\nExtract GALLERY features...")

    gallery_feats = extractor.extract_batch(
        gallery_paths,
        cache_name="gallery_feats_rgb.npy",
    )

    print(gallery_feats.shape)

    # ------------------------------------------------------------------
    # COSINE DISTANCE
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 3 - COSINE DISTANCE")
    print("=" * 60)

    start = time.time()

    dist_mat = 1.0 - np.matmul(
        query_feats,
        gallery_feats.T,
    )

    print(f"Distance matrix shape: {dist_mat.shape}")
    print(f"Time: {time.time() - start:.2f}s")

    # ------------------------------------------------------------------
    # SORT
    # ------------------------------------------------------------------
    print("\nSorting gallery indices...")

    sorted_indices = np.argsort(dist_mat, axis=1)

    print(sorted_indices.shape)

    # ------------------------------------------------------------------
    # EVALUATION
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 4 - EVALUATION")
    print("=" * 60)

    evaluator = ReIDEvaluator(top_k=args.top_k)

    rank1 = evaluator.compute_rank1(
        sorted_indices,
        q_pids,
        g_pids,
        q_cids,
        g_cids,
    )

    map_score = evaluator.compute_map(
        sorted_indices,
        q_pids,
        g_pids,
        q_cids,
        g_cids,
    )

    f1_results = evaluator.compute_f1_at_k(
        sorted_indices,
        q_pids,
        g_pids,
        q_cids,
        g_cids,
    )

    cmc = evaluator.compute_cmc(
        sorted_indices,
        q_pids,
        g_pids,
        q_cids,
        g_cids,
    )

    results = {
        "rank1": rank1,
        "map": map_score,
        "f1": f1_results["f1"],
        "precision": f1_results["precision"],
        "recall": f1_results["recall"],
        "cmc": cmc,
    }

    print_report(results, args.top_k)

    # ------------------------------------------------------------------
    # VISUALIZATION
    # ------------------------------------------------------------------
    if args.visualize:

        print("\nVisualization...")

        show_topk_results(
            query_idx=args.query_idx,
            query_paths=query_paths,
            gallery_paths=gallery_paths,
            sorted_indices=sorted_indices,
            q_pids=q_pids,
            g_pids=g_pids,
            top_k=args.vis_topk,
        )

    total_time = time.time() - total_start

    print(f"\nTotal time: {total_time:.2f}s")


# ==============================================================================
# ENTRY POINT
# ==============================================================================
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset_root",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--img_h",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--img_w",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--visualize",
        action="store_true",
    )

    parser.add_argument(
        "--query_idx",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--vis_topk",
        type=int,
        default=5,
    )

    args = parser.parse_args()

    main(args)