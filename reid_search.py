"""
==============================================================================
  HỆ THỐNG TRUY XUẤT NGƯỜI ĐI LẠC - Re-ID Search Engine
  Module: reid_search.py
==============================================================================

Module này bổ sung phần INPUT → OUTPUT thực tế cho pipeline HOG Re-ID:

  Luồng xử lý:
  ┌─────────────────────────────────────────────────────────────────┐
  │  [INPUT]  Ảnh Query của người dùng (1 hoặc nhiều file ảnh)     │
  │      ↓                                                          │
  │  [STEP 1] Build Gallery Index (trích HOG toàn bộ gallery)       │
  │      ↓  (lần đầu chậm, lần sau load từ cache .npy)             │
  │  [STEP 2] Trích xuất HOG feature cho ảnh Query                 │
  │      ↓                                                          │
  │  [STEP 3] Tính Cosine Distance → Rank kết quả                  │
  │      ↓                                                          │
  │  [OUTPUT] Top-k kết quả: Similarity Score + Metadata + ảnh     │
  └─────────────────────────────────────────────────────────────────┘

Kiến trúc class:
  ├── GalleryIndex     : Build & cache HOG features của toàn Gallery
  ├── ReIDSearchEngine : Nhận Query → Trả về Top-k kết quả
  └── ResultFormatter  : Format & hiển thị kết quả ra terminal / CSV

Cách chạy (2 chế độ):

  # Chế độ 1: Tìm kiếm với 1 ảnh query bất kỳ từ người dùng
  python reid_search.py \\
      --dataset_root ./Market-1501-v15.09.15 \\
      --query_image  /path/to/your_photo.jpg \\
      --top_k 10

  # Chế độ 2: Tìm kiếm với nhiều ảnh query (batch)
  python reid_search.py \\
      --dataset_root ./Market-1501-v15.09.15 \\
      --query_image  photo1.jpg photo2.jpg photo3.jpg \\
      --top_k 10 \\
      --save_csv results.csv

  # Chế độ 3: Demo tự động (lấy ngẫu nhiên 5 query từ tập Market-1501 query)
  python reid_search.py \\
      --dataset_root ./Market-1501-v15.09.15 \\
      --demo --demo_n 5 --top_k 10
"""

import os
import re
import json
import time
import argparse
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import cv2
from skimage.feature import hog
from scipy.spatial.distance import cdist
from tqdm import tqdm

# Import các class từ pipeline gốc
# (Đảm bảo hog_reid_pipeline.py nằm cùng thư mục)
try:
    from hog_reid_pipeline import (
        Market1501Dataset,
        HOGFeatureExtractor,
    )
except ImportError:
    raise ImportError(
        "Không tìm thấy hog_reid_pipeline.py!\n"
        "Hãy đảm bảo reid_search.py và hog_reid_pipeline.py "
        "nằm cùng một thư mục."
    )

warnings.filterwarnings("ignore")


# ==============================================================================
# CLASS 1: GalleryIndex
# Nhiệm vụ: Build, lưu cache và tải lại Gallery HOG feature index.
#
# Vì Gallery có ~19.000 ảnh, việc trích xuất HOG mỗi lần chạy rất chậm.
# GalleryIndex lưu kết quả vào file .npy để tái sử dụng (cache).
# ==============================================================================
class GalleryIndex:
    """
    Quản lý Gallery feature index với cơ chế cache thông minh.

    Lần đầu chạy: Trích xuất HOG cho toàn Gallery → lưu vào cache .npy
    Các lần sau : Load trực tiếp từ cache → nhanh hơn ~50x

    Cấu trúc cache:
        <cache_dir>/gallery_features.npy   → Ma trận đặc trưng (N, D)
        <cache_dir>/gallery_metadata.json  → Metadata của từng ảnh
    """

    def __init__(
        self,
        dataset: Market1501Dataset,
        extractor: HOGFeatureExtractor,
        cache_dir: str = "./reid_cache",
    ):
        """
        Args:
            dataset   : Market1501Dataset đã được khởi tạo
            extractor : HOGFeatureExtractor đã được khởi tạo
            cache_dir : Thư mục lưu cache (tự tạo nếu chưa có)
        """
        self.dataset   = dataset
        self.extractor = extractor
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

        # Đường dẫn file cache
        self._feat_cache  = os.path.join(cache_dir, "gallery_features.npy")
        self._meta_cache  = os.path.join(cache_dir, "gallery_metadata.json")

        # Dữ liệu Gallery (được load khi gọi build_or_load)
        self.features  = None   # np.ndarray shape (N_gallery, D)
        self.metadata  = None   # list of dict (Metadata từng ảnh)

    def build_or_load(self, force_rebuild: bool = False) -> None:
        """
        Load Gallery index từ cache hoặc build mới nếu chưa có.

        Args:
            force_rebuild: True → Xóa cache và build lại từ đầu
        """
        # Xóa cache nếu yêu cầu rebuild
        if force_rebuild:
            print("[GalleryIndex] force_rebuild=True → Xóa cache cũ...")
            for f in [self._feat_cache, self._meta_cache]:
                if os.path.exists(f):
                    os.remove(f)

        # Kiểm tra cache có tồn tại không
        if os.path.exists(self._feat_cache) and os.path.exists(self._meta_cache):
            print("[GalleryIndex] Phát hiện cache → Đang load...")
            self._load_from_cache()
        else:
            print("[GalleryIndex] Chưa có cache → Đang build Gallery Index...")
            self._build_index()

    def _build_index(self) -> None:
        """
        Trích xuất HOG feature cho toàn bộ Gallery và lưu cache.

        Metadata của mỗi ảnh bao gồm:
            - path   : Đường dẫn tuyệt đối tới file ảnh
            - fname  : Tên file (ví dụ: 0001_c2s3_000451_03.jpg)
            - pid    : Person ID (danh tính người)
            - cid    : Camera ID (số hiệu camera)
            - cam_label : Nhãn camera thân thiện (Camera_2)
            - index  : Vị trí trong mảng feature (để look-up nhanh)
        """
        gallery_records = self.dataset.gallery
        gallery_paths   = [r["path"] for r in gallery_records]

        print(f"[GalleryIndex] Trích xuất HOG cho {len(gallery_paths)} ảnh Gallery...")
        t0 = time.time()

        # Trích xuất batch HOG (có progress bar)
        self.features = self.extractor.extract_batch(
            gallery_paths, desc="  Building Gallery Index"
        )

        # Xây dựng danh sách metadata cho từng ảnh trong gallery
        self.metadata = []
        for idx, record in enumerate(gallery_records):
            self.metadata.append({
                "index"    : idx,
                "path"     : record["path"],
                "fname"    : record["fname"],
                "pid"      : int(record["pid"]),
                "cid"      : int(record["cid"]),
                # Nhãn camera thân thiện để hiển thị trong kết quả
                "cam_label": f"Camera_{record['cid']}",
            })

        elapsed = time.time() - t0
        print(f"[GalleryIndex] ✓ Build xong trong {elapsed:.1f}s "
              f"| Feature shape: {self.features.shape}")

        # Lưu cache
        print(f"[GalleryIndex] Lưu cache vào: {self.cache_dir}/")
        np.save(self._feat_cache, self.features)
        with open(self._meta_cache, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)
        print("[GalleryIndex] ✓ Cache đã lưu thành công!")

    def _load_from_cache(self) -> None:
        """Load Gallery features và metadata từ file cache đã lưu."""
        t0 = time.time()
        self.features = np.load(self._feat_cache)
        with open(self._meta_cache, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)
        print(
            f"[GalleryIndex] ✓ Load cache xong trong {time.time()-t0:.2f}s "
            f"| {len(self.metadata)} ảnh | Feature shape: {self.features.shape}"
        )

    @property
    def size(self) -> int:
        """Trả về số lượng ảnh trong Gallery index."""
        return len(self.metadata) if self.metadata else 0


# ==============================================================================
# CLASS 2: ReIDSearchEngine
# Nhiệm vụ: Nhận ảnh Query đầu vào → Trả về Top-k kết quả có đầy đủ
#           Similarity Score và Metadata.
# ==============================================================================
class ReIDSearchEngine:
    """
    Engine truy xuất ảnh người trong Gallery dựa trên ảnh Query đầu vào.

    Luồng xử lý cho MỖI ảnh Query:
        1. Tiền xử lý + trích xuất HOG feature vector
        2. Tính Cosine Distance với toàn bộ Gallery
        3. Sắp xếp theo khoảng cách tăng dần
        4. Chuyển đổi Distance → Similarity Score (0~1)
        5. Đóng gói kết quả kèm Metadata
    """

    def __init__(
        self,
        gallery_index: GalleryIndex,
        extractor: HOGFeatureExtractor,
        metric: str = "cosine",
    ):
        """
        Args:
            gallery_index: GalleryIndex đã được build/load
            extractor    : HOGFeatureExtractor dùng chung với gallery
            metric       : Độ đo khoảng cách ('cosine' hoặc 'euclidean')
        """
        self.gallery_index = gallery_index
        self.extractor     = extractor
        self.metric        = metric

    def _distance_to_similarity(
        self, distances: np.ndarray, metric: str
    ) -> np.ndarray:
        """
        Chuyển đổi khoảng cách thành Similarity Score (càng cao càng giống).

        Với Cosine Distance:
            cosine_distance ∈ [0, 2]
            similarity = 1 - cosine_distance / 2  → ∈ [0, 1]
            (0.0 = hoàn toàn khác, 1.0 = giống hệt nhau)

        Với Euclidean Distance:
            similarity = 1 / (1 + distance)       → ∈ (0, 1]

        Args:
            distances: Mảng khoảng cách từ query đến các gallery images
            metric   : Tên độ đo ('cosine' hoặc 'euclidean')

        Returns:
            Mảng Similarity Score tương ứng, cùng shape với distances
        """
        if metric == "cosine":
            # Cosine distance ∈ [0, 2]: 0 = identical, 2 = opposite
            # Normalize về [0, 1]: similarity = 1 - dist/2
            return np.clip(1.0 - distances / 2.0, 0.0, 1.0)
        else:
            # Euclidean: dùng công thức decay
            return 1.0 / (1.0 + distances)

    def search(
        self,
        query_image_path: str,
        top_k: int = 10,
        query_pid: int = None,
        query_cid: int = None,
    ) -> dict:
        """
        Thực hiện tìm kiếm cho một ảnh Query.

        Args:
            query_image_path: Đường dẫn tới ảnh Query của người dùng
            top_k           : Số lượng kết quả trả về
            query_pid       : Person ID của query (None nếu không biết)
            query_cid       : Camera ID của query (None nếu không biết)

        Returns:
            dict chứa:
                'query_info'  : Thông tin ảnh Query
                'results'     : Danh sách Top-k kết quả (list of dict)
                'search_time' : Thời gian tìm kiếm (giây)

            Mỗi kết quả trong 'results' bao gồm:
                'rank'            : Thứ hạng (1 = gần nhất)
                'similarity_score': Điểm tương đồng [0.0 ~ 1.0]
                'distance'        : Khoảng cách gốc (cosine/euclidean)
                'fname'           : Tên file ảnh trong Gallery
                'path'            : Đường dẫn ảnh trong Gallery
                'pid'             : Person ID trong Gallery
                'cid'             : Camera ID trong Gallery
                'cam_label'       : Nhãn camera (Camera_1, Camera_2, ...)
                'is_correct_match': True nếu PID trùng với Query (khi biết GT)
        """
        t0 = time.time()

        # ---- Bước 1: Kiểm tra file ảnh Query có tồn tại không ----
        if not os.path.isfile(query_image_path):
            raise FileNotFoundError(
                f"[SearchEngine] Không tìm thấy ảnh Query: {query_image_path}"
            )

        # ---- Bước 2: Trích xuất HOG feature cho ảnh Query ----
        query_feat = self.extractor.extract_single(query_image_path)
        # Reshape về (1, D) để dùng cdist
        query_feat_2d = query_feat.reshape(1, -1)

        # ---- Bước 3: Tính khoảng cách tới toàn bộ Gallery ----
        # dist_vec shape: (1, N_gallery) → squeeze → (N_gallery,)
        dist_vec = cdist(
            query_feat_2d,
            self.gallery_index.features,
            metric=self.metric
        ).squeeze()

        # ---- Bước 4: Sắp xếp theo khoảng cách tăng dần ----
        # argsort trả về indices sắp xếp từ nhỏ đến lớn (gần → xa)
        sorted_indices = np.argsort(dist_vec)

        # ---- Bước 5: Chuyển đổi distance → Similarity Score ----
        similarity_scores = self._distance_to_similarity(dist_vec, self.metric)

        # ---- Bước 6: Lọc junk nếu biết query CID ----
        # Theo chuẩn Market-1501: loại bỏ ảnh cùng PID & CID với query
        # (nếu không biết pid/cid thì bỏ qua bước lọc này)
        filtered_indices = []
        for idx in sorted_indices:
            meta = self.gallery_index.metadata[idx]
            # Loại bỏ junk: cùng PID và cùng CID với query
            if (query_pid is not None and query_cid is not None
                    and meta["pid"] == query_pid
                    and meta["cid"] == query_cid):
                continue  # Bỏ qua ảnh junk
            filtered_indices.append(idx)
            if len(filtered_indices) >= top_k:
                break  # Đủ Top-k rồi thì dừng

        # ---- Bước 7: Đóng gói kết quả Top-k kèm đầy đủ Metadata ----
        results = []
        for rank, idx in enumerate(filtered_indices, start=1):
            meta   = self.gallery_index.metadata[idx]
            dist   = float(dist_vec[idx])
            sim    = float(similarity_scores[idx])

            # Kiểm tra có phải là match đúng không (khi biết ground truth)
            is_correct = None
            if query_pid is not None:
                is_correct = (meta["pid"] == query_pid)

            results.append({
                # ---- Thông tin xếp hạng ----
                "rank"             : rank,
                "similarity_score" : round(sim, 4),   # [0.0 ~ 1.0]
                "distance"         : round(dist, 6),  # Khoảng cách gốc
                # ---- Metadata của ảnh Gallery ----
                "fname"            : meta["fname"],
                "path"             : meta["path"],
                "pid"              : meta["pid"],
                "cid"              : meta["cid"],
                "cam_label"        : meta["cam_label"],
                # ---- Đánh giá kết quả (nếu có ground truth) ----
                "is_correct_match" : is_correct,
            })

        search_time = time.time() - t0

        # ---- Tổng hợp kết quả trả về ----
        return {
            "query_info": {
                "path"     : query_image_path,
                "fname"    : os.path.basename(query_image_path),
                "pid"      : query_pid,
                "cid"      : query_cid,
                "cam_label": f"Camera_{query_cid}" if query_cid else "Unknown",
                "feat_dim" : int(query_feat.shape[0]),
            },
            "results"     : results,
            "search_time" : round(search_time, 4),
            "top_k"       : top_k,
            "metric"      : self.metric,
        }

    def search_batch(
        self,
        query_paths: list,
        top_k: int = 10,
        query_pids: list = None,
        query_cids: list = None,
    ) -> list:
        """
        Tìm kiếm batch cho nhiều ảnh Query cùng lúc.

        Args:
            query_paths: Danh sách đường dẫn ảnh Query
            top_k      : Số kết quả trả về cho mỗi query
            query_pids : Danh sách PID tương ứng (None nếu không biết)
            query_cids : Danh sách CID tương ứng (None nếu không biết)

        Returns:
            Danh sách kết quả, mỗi phần tử là output của search()
        """
        all_results = []
        n = len(query_paths)

        print(f"\n[SearchEngine] Tìm kiếm batch cho {n} ảnh Query...")
        for i, qpath in enumerate(query_paths):
            qpid = query_pids[i] if query_pids else None
            qcid = query_cids[i] if query_cids else None

            print(f"\n[SearchEngine] Query [{i+1}/{n}]: {os.path.basename(qpath)}")
            result = self.search(qpath, top_k=top_k,
                                 query_pid=qpid, query_cid=qcid)
            all_results.append(result)

        return all_results


# ==============================================================================
# CLASS 3: ResultFormatter
# Nhiệm vụ: Hiển thị kết quả ra terminal và xuất file CSV.
# ==============================================================================
class ResultFormatter:
    """
    Format và hiển thị kết quả tìm kiếm Re-ID ra terminal hoặc CSV.

    Hỗ trợ 2 dạng output:
        - Terminal : In bảng kết quả trực tiếp ra console (dễ đọc)
        - CSV file : Lưu kết quả dạng bảng để phân tích thêm
    """

    @staticmethod
    def print_single_result(search_output: dict, show_path: bool = False):
        """
        In kết quả tìm kiếm của MỘT ảnh Query ra terminal.

        Args:
            search_output: Output dict từ ReIDSearchEngine.search()
            show_path    : True → hiển thị đường dẫn đầy đủ thay vì chỉ fname
        """
        qi       = search_output["query_info"]
        results  = search_output["results"]
        top_k    = search_output["top_k"]
        metric   = search_output["metric"]
        elapsed  = search_output["search_time"]

        # ── Header Query ────────────────────────────────────────────
        border = "═" * 72
        print(f"\n{border}")
        print(f"  🔍  KẾT QUẢ TÌM KIẾM - Re-ID Search Engine")
        print(border)
        print(f"  Ảnh Query    : {qi['fname']}")
        if qi['pid']:
            print(f"  Person ID    : {qi['pid']:04d}   |  {qi['cam_label']}")
        print(f"  Feat. Dim    : {qi['feat_dim']} chiều HOG")
        print(f"  Metric       : {metric.capitalize()} Distance")
        print(f"  Thời gian    : {elapsed:.4f}s")
        print(f"  Top-k        : {top_k} kết quả")
        print(f"{'─'*72}")

        # ── Bảng kết quả ────────────────────────────────────────────
        # Tiêu đề cột
        col_name = "Tên file ảnh Gallery" if not show_path else "Đường dẫn ảnh Gallery"
        print(
            f"  {'Rank':<5} {'Similarity':>11} {'Distance':>10}  "
            f"{'PID':>5}  {'Camera':<10}  {col_name}"
        )
        print(f"  {'─'*5} {'─'*11} {'─'*10}  {'─'*5}  {'─'*10}  {'─'*30}")

        for r in results:
            # Biểu tượng đánh dấu kết quả đúng/sai (nếu có ground truth)
            if r["is_correct_match"] is None:
                marker = "  "   # Không biết ground truth
            elif r["is_correct_match"]:
                marker = "✓ "   # Đúng person
            else:
                marker = "✗ "   # Sai person

            # Tên file hiển thị (rút ngắn nếu cần)
            display_name = r["path"] if show_path else r["fname"]
            if len(display_name) > 35:
                display_name = "..." + display_name[-32:]

            # Thanh trực quan hóa Similarity Score (█ bar)
            bar_len  = int(r["similarity_score"] * 10)
            sim_bar  = "█" * bar_len + "░" * (10 - bar_len)

            print(
                f"  {marker}#{r['rank']:<3}  "
                f"{r['similarity_score']:>8.4f}  "
                f"({sim_bar})  "
                f"{r['distance']:>9.6f}  "
                f"{r['pid']:>5}  "
                f"{r['cam_label']:<10}  "
                f"{display_name}"
            )

        print(f"{'─'*72}")

        # ── Tóm tắt kết quả (nếu có ground truth) ───────────────────
        if results and results[0]["is_correct_match"] is not None:
            correct_in_topk = sum(1 for r in results if r["is_correct_match"])
            rank1_correct   = results[0]["is_correct_match"]
            print(f"\n  📊 Tóm tắt:")
            print(f"     Rank-1 Match : {'✓ ĐÚNG' if rank1_correct else '✗ SAI'}")
            print(f"     Đúng / Top-{top_k}: {correct_in_topk} / {top_k} ảnh")
            print(f"     Precision@{top_k}: {correct_in_topk/top_k:.2%}")

        print(f"{border}\n")

    @staticmethod
    def print_batch_summary(all_results: list):
        """
        In tóm tắt kết quả của cả batch Query.

        Args:
            all_results: Danh sách output từ search_batch()
        """
        n = len(all_results)
        border = "═" * 60
        print(f"\n{border}")
        print(f"  📋  TỔNG KẾT BATCH - {n} ảnh Query")
        print(border)
        print(f"  {'STT':<5} {'Query File':<30} {'Rank-1':>8} {'Sim@1':>8}")
        print(f"  {'─'*5} {'─'*30} {'─'*8} {'─'*8}")

        rank1_count = 0
        for i, res in enumerate(all_results, 1):
            qi      = res["query_info"]
            top1    = res["results"][0] if res["results"] else None
            is_r1   = top1["is_correct_match"] if top1 else None
            sim1    = top1["similarity_score"] if top1 else 0.0

            # Chỉ đếm rank-1 nếu biết ground truth
            if is_r1 is True:
                rank1_count += 1
                mark = "✓"
            elif is_r1 is False:
                mark = "✗"
            else:
                mark = "?"

            fname_short = qi["fname"][:28] + ".." if len(qi["fname"]) > 30 else qi["fname"]
            print(
                f"  {i:<5} {fname_short:<30} "
                f"{'  ' + mark:>8} {sim1:>8.4f}"
            )

        print(f"  {'─'*5} {'─'*30} {'─'*8} {'─'*8}")

        # Tổng kết chỉ khi có ground truth
        queries_with_gt = [r for r in all_results
                           if r["results"] and r["results"][0]["is_correct_match"] is not None]
        if queries_with_gt:
            rank1_acc = rank1_count / len(queries_with_gt)
            avg_time  = np.mean([r["search_time"] for r in all_results])
            print(f"\n  Rank-1 Accuracy : {rank1_acc:.2%} "
                  f"({rank1_count}/{len(queries_with_gt)} queries)")
            print(f"  Thời gian TB/query: {avg_time:.4f}s")
        print(f"{border}\n")

    @staticmethod
    def save_csv(all_results: list, csv_path: str):
        """
        Lưu tất cả kết quả Top-k ra file CSV.

        Mỗi hàng trong CSV = 1 kết quả gallery cho 1 query.
        Cấu trúc CSV:
            query_fname, query_pid, query_cid, rank, similarity_score,
            distance, gallery_fname, gallery_pid, gallery_cid,
            cam_label, is_correct_match, search_time

        Args:
            all_results: List kết quả từ search_batch()
            csv_path   : Đường dẫn file CSV đầu ra
        """
        import csv

        fieldnames = [
            "query_fname", "query_pid", "query_cid",
            "rank", "similarity_score", "distance",
            "gallery_fname", "gallery_pid", "gallery_cid",
            "cam_label", "is_correct_match", "search_time",
        ]

        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for res in all_results:
                qi = res["query_info"]
                for r in res["results"]:
                    writer.writerow({
                        "query_fname"      : qi["fname"],
                        "query_pid"        : qi["pid"] or "",
                        "query_cid"        : qi["cid"] or "",
                        "rank"             : r["rank"],
                        "similarity_score" : r["similarity_score"],
                        "distance"         : r["distance"],
                        "gallery_fname"    : r["fname"],
                        "gallery_pid"      : r["pid"],
                        "gallery_cid"      : r["cid"],
                        "cam_label"        : r["cam_label"],
                        "is_correct_match" : r["is_correct_match"],
                        "search_time"      : res["search_time"],
                    })

        print(f"[ResultFormatter] ✓ Đã lưu kết quả CSV: {csv_path}")
        print(f"[ResultFormatter]   ({len(all_results)} queries × "
              f"{all_results[0]['top_k']} results = "
              f"{len(all_results) * all_results[0]['top_k']} hàng)")


# ==============================================================================
# HÀM PARSE METADATA TỪ TÊN FILE MARKET-1501
# Dùng để tự động lấy PID/CID từ tên file query (khi chạy demo)
# ==============================================================================
_FNAME_RE = re.compile(r"(-?\d+)_c(\d+)")

def parse_market1501_fname(fname: str):
    """Trích xuất (pid, cid) từ tên file Market-1501. Trả về (None, None) nếu không parse được."""
    m = _FNAME_RE.search(os.path.basename(fname))
    if m:
        return int(m.group(1)), int(m.group(2))
    return None, None


# ==============================================================================
# HÀM CHÍNH (MAIN)
# ==============================================================================
def main(args):
    total_start = time.time()

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║   Re-ID Search Engine  |  HOG + Cosine Distance             ║")
    print("║   Hệ thống truy xuất người đi lạc qua camera giám sát       ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    # ── BƯỚC 1: Khởi tạo Dataset & Extractor ─────────────────────────
    print("\n" + "═" * 60)
    print("BƯỚC 1: KHỞI TẠO HỆ THỐNG")
    print("═" * 60)
    dataset   = Market1501Dataset(args.dataset_root)
    extractor = HOGFeatureExtractor(
        img_height=128, img_width=64,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
    )

    # ── BƯỚC 2: Build/Load Gallery Index ─────────────────────────────
    print("\n" + "═" * 60)
    print("BƯỚC 2: GALLERY INDEX (HOG Feature Cache)")
    print("═" * 60)
    gallery_idx = GalleryIndex(
        dataset=dataset,
        extractor=extractor,
        cache_dir=args.cache_dir,
    )
    gallery_idx.build_or_load(force_rebuild=args.rebuild_cache)
    print(f"[Main] Gallery index sẵn sàng: {gallery_idx.size} ảnh")

    # ── BƯỚC 3: Khởi tạo Search Engine ───────────────────────────────
    engine    = ReIDSearchEngine(gallery_idx, extractor, metric=args.metric)
    formatter = ResultFormatter()

    # ── BƯỚC 4: Xác định danh sách Query ─────────────────────────────
    print("\n" + "═" * 60)
    print("BƯỚC 3: XÁC ĐỊNH ẢNH QUERY")
    print("═" * 60)

    query_paths = []
    query_pids  = []
    query_cids  = []

    if args.demo:
        # ── Chế độ Demo: Lấy ngẫu nhiên N ảnh từ tập Market-1501 query ──
        print(f"[Main] Chế độ DEMO: Chọn ngẫu nhiên {args.demo_n} ảnh từ tập Query Market-1501")
        np.random.seed(42)  # Fix seed để kết quả có thể tái hiện
        sample_records = np.random.choice(
            dataset.query, size=min(args.demo_n, len(dataset.query)), replace=False
        )
        for rec in sample_records:
            query_paths.append(rec["path"])
            query_pids.append(rec["pid"])
            query_cids.append(rec["cid"])
            print(f"  + {rec['fname']}  (PID={rec['pid']:04d}, Camera_{rec['cid']})")

    else:
        # ── Chế độ thủ công: Dùng ảnh người dùng cung cấp ──────────────
        if not args.query_image:
            raise ValueError(
                "Bạn phải cung cấp --query_image hoặc dùng --demo!\n"
                "Ví dụ: python reid_search.py --dataset_root ./Market-1501 "
                "--query_image my_photo.jpg"
            )

        for qpath in args.query_image:
            if not os.path.isfile(qpath):
                print(f"[Main] ⚠ Không tìm thấy file: {qpath} → Bỏ qua")
                continue

            # Thử parse PID/CID từ tên file (nếu là ảnh Market-1501)
            pid, cid = parse_market1501_fname(qpath)
            query_paths.append(qpath)
            query_pids.append(pid)   # None nếu không parse được
            query_cids.append(cid)   # None nếu không parse được
            pid_str = f"PID={pid:04d}" if pid else "PID=Unknown"
            cid_str = f"Camera_{cid}" if cid else "Camera=Unknown"
            print(f"  + {os.path.basename(qpath)}  ({pid_str}, {cid_str})")

    if not query_paths:
        print("[Main] Không có ảnh Query hợp lệ nào! Thoát.")
        return

    # ── BƯỚC 5: Thực hiện tìm kiếm ───────────────────────────────────
    print("\n" + "═" * 60)
    print(f"BƯỚC 4: TÌM KIẾM TOP-{args.top_k} KẾT QUẢ")
    print("═" * 60)

    all_results = engine.search_batch(
        query_paths=query_paths,
        top_k=args.top_k,
        query_pids=query_pids,
        query_cids=query_cids,
    )

    # ── BƯỚC 6: Hiển thị kết quả ─────────────────────────────────────
    print("\n" + "═" * 60)
    print("BƯỚC 5: KẾT QUẢ TÌM KIẾM")
    print("═" * 60)

    for res in all_results:
        formatter.print_single_result(res, show_path=args.show_path)

    # In tóm tắt batch nếu có nhiều hơn 1 query
    if len(all_results) > 1:
        formatter.print_batch_summary(all_results)

    # ── BƯỚC 7: Lưu CSV (nếu được yêu cầu) ──────────────────────────
    if args.save_csv:
        print("═" * 60)
        print("BƯỚC 6: XUẤT KẾT QUẢ CSV")
        print("═" * 60)
        formatter.save_csv(all_results, args.save_csv)

    # ── Tổng kết thời gian ────────────────────────────────────────────
    total_time = time.time() - total_start
    print(f"[Main] ✓ Hoàn thành! Tổng thời gian: {total_time:.2f}s")


# ==============================================================================
# ENTRY POINT
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Re-ID Search Engine: Nhận ảnh Query → Truy xuất Top-k từ Gallery",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # ── Dataset ───────────────────────────────────────────────────────
    parser.add_argument(
        "--dataset_root", type=str, required=True,
        help="Đường dẫn thư mục Market-1501-v15.09.15/",
    )

    # ── Ảnh Query (chế độ thủ công) ──────────────────────────────────
    parser.add_argument(
        "--query_image", type=str, nargs="+", default=None,
        metavar="PATH",
        help=(
            "Đường dẫn tới ảnh Query (có thể nhiều ảnh).\n"
            "Ví dụ: --query_image photo.jpg  hoặc  --query_image a.jpg b.jpg"
        ),
    )

    # ── Chế độ Demo ───────────────────────────────────────────────────
    parser.add_argument(
        "--demo", action="store_true",
        help="Chạy demo: tự động lấy ngẫu nhiên ảnh từ tập Market-1501 query",
    )
    parser.add_argument(
        "--demo_n", type=int, default=3,
        help="Số ảnh Query lấy ngẫu nhiên trong chế độ demo (default: 3)",
    )

    # ── Tham số tìm kiếm ─────────────────────────────────────────────
    parser.add_argument(
        "--top_k", type=int, default=10,
        help="Số lượng kết quả trả về (default: 10)",
    )
    parser.add_argument(
        "--metric", type=str, default="cosine",
        choices=["cosine", "euclidean"],
        help="Độ đo khoảng cách (default: cosine)",
    )

    # ── Cache Gallery Index ───────────────────────────────────────────
    parser.add_argument(
        "--cache_dir", type=str, default="./reid_cache",
        help="Thư mục lưu cache Gallery Index (default: ./reid_cache)",
    )
    parser.add_argument(
        "--rebuild_cache", action="store_true",
        help="Xóa cache cũ và build Gallery Index lại từ đầu",
    )

    # ── Output ────────────────────────────────────────────────────────
    parser.add_argument(
        "--save_csv", type=str, default=None,
        metavar="results.csv",
        help="Lưu kết quả Top-k ra file CSV (default: không lưu)",
    )
    parser.add_argument(
        "--show_path", action="store_true",
        help="Hiển thị đường dẫn đầy đủ thay vì chỉ tên file trong bảng kết quả",
    )

    args = parser.parse_args()
    main(args)
