"""
==============================================================================
  HỆ THỐNG TRUY XUẤT NGƯỜI ĐI LẠC - RGB HOG Re-ID Search Engine
  Module: reid_search.py
==============================================================================

Phiên bản đã đồng bộ hoàn toàn với RGB-HOG pipeline mới:

✅ RGB-HOG 3 channels
✅ Cosine similarity bằng np.matmul
✅ Feature normalization
✅ Gallery cache .npy
✅ Query search thực tế
✅ Top-k retrieval
✅ CSV export
✅ Visualization-friendly
✅ Tối ưu tốc độ giống notebook
"""

import os
import re
import json
import time
import argparse
import warnings

import numpy as np

warnings.filterwarnings("ignore")


# ==============================================================================
# IMPORT PIPELINE
# ==============================================================================
try:

    from hog_reid_pipeline import (
        Market1501Dataset,
        RGBHOGExtractor,
    )

except ImportError as e:

    raise ImportError(
        "\n[ERROR] Không import được hog_reid_pipeline.py\n"
        "Hãy đảm bảo:\n"
        "  1. reid_search.py cùng thư mục với hog_reid_pipeline.py\n"
        "  2. Trong pipeline phải có:\n"
        "       - Market1501Dataset\n"
        "       - RGBHOGExtractor\n\n"
        f"Chi tiết lỗi:\n{e}"
    )


# ==============================================================================
# PARSE MARKET1501 FILENAME
# ==============================================================================
_FNAME_RE = re.compile(r"(-?\d+)_c(\d+)")

def parse_market1501_fname(fname):

    m = _FNAME_RE.search(os.path.basename(fname))

    if m:
        return int(m.group(1)), int(m.group(2))

    return None, None


# ==============================================================================
# GALLERY INDEX
# ==============================================================================
class GalleryIndex:

    def __init__(
        self,
        dataset,
        extractor,
        cache_dir="./reid_cache",
    ):

        self.dataset = dataset
        self.extractor = extractor

        self.cache_dir = cache_dir

        os.makedirs(cache_dir, exist_ok=True)

        self.feature_cache = os.path.join(
            cache_dir,
            "gallery_rgbhog.npy",
        )

        self.meta_cache = os.path.join(
            cache_dir,
            "gallery_metadata.json",
        )

        self.features = None
        self.metadata = None

    def build_or_load(self, rebuild=False):

        if rebuild:

            if os.path.exists(self.feature_cache):
                os.remove(self.feature_cache)

            if os.path.exists(self.meta_cache):
                os.remove(self.meta_cache)

            print("[GalleryIndex] Old cache removed.")

        if (
            os.path.exists(self.feature_cache)
            and os.path.exists(self.meta_cache)
        ):

            self.load()

        else:

            self.build()

    def build(self):

        print("\n[GalleryIndex] Building RGB-HOG gallery index...")

        t0 = time.time()

        gallery_records = self.dataset.gallery

        gallery_paths = [
            x["path"]
            for x in gallery_records
        ]

        self.features = self.extractor.extract_batch(
            gallery_paths,
            cache_name=None,
        )

        self.metadata = []

        for idx, rec in enumerate(gallery_records):

            self.metadata.append({

                "index": idx,
                "path": rec["path"],
                "fname": rec["fname"],
                "pid": int(rec["pid"]),
                "cid": int(rec["cid"]),

            })

        np.save(
            self.feature_cache,
            self.features,
        )

        with open(
            self.meta_cache,
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                self.metadata,
                f,
                indent=2,
                ensure_ascii=False,
            )

        elapsed = time.time() - t0

        print(f"[GalleryIndex] DONE in {elapsed:.2f}s")
        print(f"[GalleryIndex] Shape: {self.features.shape}")

    def load(self):

        print("\n[GalleryIndex] Loading cache...")

        t0 = time.time()

        self.features = np.load(
            self.feature_cache
        )

        with open(
            self.meta_cache,
            "r",
            encoding="utf-8",
        ) as f:

            self.metadata = json.load(f)

        elapsed = time.time() - t0

        print(f"[GalleryIndex] DONE in {elapsed:.2f}s")
        print(f"[GalleryIndex] Shape: {self.features.shape}")

    @property
    def size(self):

        return len(self.metadata)


# ==============================================================================
# SEARCH ENGINE
# ==============================================================================
class ReIDSearchEngine:

    def __init__(
        self,
        gallery_index,
        extractor,
    ):

        self.gallery_index = gallery_index
        self.extractor = extractor

    def search(
        self,
        query_image_path,
        top_k=10,
        query_pid=None,
        query_cid=None,
    ):

        if not os.path.isfile(query_image_path):

            raise FileNotFoundError(query_image_path)

        t0 = time.time()

        # ------------------------------------------------------------------
        # EXTRACT QUERY FEATURE
        # ------------------------------------------------------------------
        query_feat = self.extractor.extract_single(
            query_image_path
        )

        # ------------------------------------------------------------------
        # FAST COSINE DISTANCE
        # normalized feature -> cosine = dot product
        # ------------------------------------------------------------------
        similarities = np.matmul(
            self.gallery_index.features,
            query_feat,
        )

        distances = 1.0 - similarities

        ranked_indices = np.argsort(distances)

        # ------------------------------------------------------------------
        # FILTER JUNK
        # ------------------------------------------------------------------
        filtered = []

        for idx in ranked_indices:

            meta = self.gallery_index.metadata[idx]

            if (
                query_pid is not None
                and query_cid is not None
                and meta["pid"] == query_pid
                and meta["cid"] == query_cid
            ):
                continue

            filtered.append(idx)

            if len(filtered) >= top_k:
                break

        # ------------------------------------------------------------------
        # BUILD RESULTS
        # ------------------------------------------------------------------
        results = []

        for rank, idx in enumerate(filtered, start=1):

            meta = self.gallery_index.metadata[idx]

            sim = float(similarities[idx])

            dist = float(distances[idx])

            is_correct = None

            if query_pid is not None:

                is_correct = (
                    meta["pid"] == query_pid
                )

            results.append({

                "rank": rank,

                "similarity_score": round(sim, 4),

                "distance": round(dist, 6),

                "fname": meta["fname"],

                "path": meta["path"],

                "pid": meta["pid"],

                "cid": meta["cid"],

                "cam_label": f"Camera_{meta['cid']}",

                "is_correct_match": is_correct,

            })

        elapsed = time.time() - t0

        return {

            "query_info": {

                "path": query_image_path,

                "fname": os.path.basename(
                    query_image_path
                ),

                "pid": query_pid,

                "cid": query_cid,

            },

            "results": results,

            "search_time": round(elapsed, 4),

            "top_k": top_k,

        }

    def search_batch(
        self,
        query_paths,
        top_k=10,
        query_pids=None,
        query_cids=None,
    ):

        outputs = []

        n = len(query_paths)

        print(f"\n[Search] Batch size: {n}")

        for i, qpath in enumerate(query_paths):

            print(
                f"[{i+1}/{n}] "
                f"{os.path.basename(qpath)}"
            )

            qpid = (
                query_pids[i]
                if query_pids else None
            )

            qcid = (
                query_cids[i]
                if query_cids else None
            )

            out = self.search(
                qpath,
                top_k=top_k,
                query_pid=qpid,
                query_cid=qcid,
            )

            outputs.append(out)

        return outputs


# ==============================================================================
# RESULT FORMATTER
# ==============================================================================
class ResultFormatter:

    @staticmethod
    def print_single_result(result):

        qi = result["query_info"]

        print("\n" + "=" * 70)

        print(f"QUERY: {qi['fname']}")

        if qi["pid"] is not None:

            print(
                f"PID={qi['pid']} | "
                f"Camera_{qi['cid']}"
            )

        print(f"Search Time: {result['search_time']}s")

        print("=" * 70)

        print(
            f"{'Rank':<6}"
            f"{'Sim':<10}"
            f"{'PID':<8}"
            f"{'CID':<8}"
            f"{'Correct':<10}"
            f"{'Filename'}"
        )

        print("-" * 70)

        for r in result["results"]:

            if r["is_correct_match"] is None:
                correct = "?"
            else:
                correct = "TRUE" if r["is_correct_match"] else "FALSE"

            print(
                f"{r['rank']:<6}"
                f"{r['similarity_score']:<10.4f}"
                f"{r['pid']:<8}"
                f"{r['cid']:<8}"
                f"{correct:<10}"
                f"{r['fname']}"
            )

        print("=" * 70)

    @staticmethod
    def save_csv(results, csv_path):

        import csv

        with open(
            csv_path,
            "w",
            newline="",
            encoding="utf-8-sig",
        ) as f:

            writer = csv.writer(f)

            writer.writerow([

                "query_fname",
                "query_pid",
                "query_cid",

                "rank",

                "similarity_score",

                "distance",

                "gallery_fname",

                "gallery_pid",

                "gallery_cid",

                "is_correct",

            ])

            for res in results:

                qi = res["query_info"]

                for r in res["results"]:

                    writer.writerow([

                        qi["fname"],
                        qi["pid"],
                        qi["cid"],

                        r["rank"],

                        r["similarity_score"],

                        r["distance"],

                        r["fname"],

                        r["pid"],

                        r["cid"],

                        r["is_correct_match"],

                    ])

        print(f"\n[CSV] Saved: {csv_path}")


# ==============================================================================
# MAIN
# ==============================================================================
def main(args):

    total_start = time.time()

    print("=" * 70)
    print("RGB-HOG PERSON RE-ID SEARCH ENGINE")
    print("=" * 70)

    # ------------------------------------------------------------------
    # DATASET
    # ------------------------------------------------------------------
    dataset = Market1501Dataset(
        args.dataset_root
    )

    # ------------------------------------------------------------------
    # EXTRACTOR
    # ------------------------------------------------------------------
    extractor = RGBHOGExtractor(

        img_h=128,
        img_w=64,

        orientations=12,

        ppc=(8, 8),

        cpb=(2, 2),

    )

    # ------------------------------------------------------------------
    # GALLERY INDEX
    # ------------------------------------------------------------------
    gallery_index = GalleryIndex(
        dataset=dataset,
        extractor=extractor,
        cache_dir=args.cache_dir,
    )

    gallery_index.build_or_load(
        rebuild=args.rebuild_cache
    )

    print(
        f"[Gallery] {gallery_index.size} images ready."
    )

    # ------------------------------------------------------------------
    # SEARCH ENGINE
    # ------------------------------------------------------------------
    engine = ReIDSearchEngine(
        gallery_index,
        extractor,
    )

    formatter = ResultFormatter()

    # ------------------------------------------------------------------
    # QUERY
    # ------------------------------------------------------------------
    query_paths = []
    query_pids = []
    query_cids = []

    if args.demo:

        print("\n[DEMO MODE]")

        np.random.seed(42)

        samples = np.random.choice(
            dataset.query,
            size=min(
                args.demo_n,
                len(dataset.query),
            ),
            replace=False,
        )

        for rec in samples:

            query_paths.append(rec["path"])
            query_pids.append(rec["pid"])
            query_cids.append(rec["cid"])

            print(
                f"  + {rec['fname']} "
                f"(PID={rec['pid']})"
            )

    else:

        if args.query_image is None:

            raise ValueError(
                "Bạn phải dùng --query_image hoặc --demo"
            )

        for qpath in args.query_image:

            if not os.path.isfile(qpath):

                print(f"Missing: {qpath}")

                continue

            pid, cid = parse_market1501_fname(
                qpath
            )

            query_paths.append(qpath)
            query_pids.append(pid)
            query_cids.append(cid)

    # ------------------------------------------------------------------
    # SEARCH
    # ------------------------------------------------------------------
    results = engine.search_batch(

        query_paths=query_paths,

        top_k=args.top_k,

        query_pids=query_pids,

        query_cids=query_cids,

    )

    # ------------------------------------------------------------------
    # PRINT
    # ------------------------------------------------------------------
    for res in results:

        formatter.print_single_result(res)

    # ------------------------------------------------------------------
    # CSV
    # ------------------------------------------------------------------
    if args.save_csv:

        formatter.save_csv(
            results,
            args.save_csv,
        )

    total_time = time.time() - total_start

    print(f"\nTOTAL TIME: {total_time:.2f}s")


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
        "--query_image",
        type=str,
        nargs="+",
        default=None,
    )

    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--cache_dir",
        type=str,
        default="./reid_cache",
    )

    parser.add_argument(
        "--rebuild_cache",
        action="store_true",
    )

    parser.add_argument(
        "--save_csv",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--demo",
        action="store_true",
    )

    parser.add_argument(
        "--demo_n",
        type=int,
        default=3,
    )

    args = parser.parse_args()

    main(args)