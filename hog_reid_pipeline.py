"""
==============================================================================
  HỆ THỐNG TRUY XUẤT NGƯỜI ĐI LẠC - HOG Re-ID Pipeline
  Đồ án môn học: Person Re-Identification với đặc trưng HOG truyền thống
  Tập dữ liệu: Market-1501
==============================================================================

Kiến trúc pipeline (OOP):
  ├── Market1501Dataset  : Đọc & phân tích dữ liệu chuẩn Market-1501
  ├── HOGFeatureExtractor: Tiền xử lý ảnh + trích xuất vector HOG
  └── ReIDEvaluator      : So khớp + tính Rank-1, mAP, F1-Score

Cách chạy:
  python hog_reid_pipeline.py --dataset_root /path/to/Market-1501-v15.09.15
"""

import os
import re
import argparse
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import cv2
from skimage.feature import hog
from scipy.spatial.distance import cdist
from tqdm import tqdm

warnings.filterwarnings("ignore")


# ==============================================================================
# CLASS 1: Market1501Dataset
# Nhiệm vụ: Đọc và phân tích cấu trúc thư mục của tập dữ liệu Market-1501.
# ==============================================================================
class Market1501Dataset:
    """
    Đọc tập dữ liệu Market-1501 theo chuẩn benchmark Re-ID.

    Quy ước đặt tên file ảnh của Market-1501:
        <PID>_<CID>s<SID>_<Frame>_<Det>.jpg
        Ví dụ: 0001_c1s1_000151_01.jpg
            PID  = 0001  -> Person ID (danh tính người)
            CID  = c1    -> Camera ID (số hiệu camera, bắt đầu từ 1)
            SID  = s1    -> Sequence ID
            Frame= 000151-> Số frame trong video
            Det  = 01    -> Số thứ tự detection

    Lưu ý quan trọng:
        - Ảnh có PID = -1 là ảnh rác (junk images), phải loại bỏ.
        - Ảnh có PID = 0  là ảnh distractor (nhiễu), phải loại bỏ.
    """

    # Pattern regex để parse tên file ảnh Market-1501
    _FNAME_PATTERN = re.compile(r"(-?\d+)_c(\d+)")

    def __init__(self, dataset_root: str):
        """
        Khởi tạo Dataset với đường dẫn gốc tới thư mục Market-1501.

        Args:
            dataset_root: Đường dẫn tới thư mục gốc chứa
                          bounding_box_train/, query/, bounding_box_test/
        """
        self.root = dataset_root
        self._validate_structure()

        # Đọc dữ liệu cho cả 3 tập
        print("[Dataset] Đang đọc tập Query...")
        self.query = self._load_split("query")

        print("[Dataset] Đang đọc tập Gallery (bounding_box_test)...")
        self.gallery = self._load_split("bounding_box_test")

        print(f"[Dataset] ✓ Query : {len(self.query)} ảnh hợp lệ")
        print(f"[Dataset] ✓ Gallery: {len(self.gallery)} ảnh hợp lệ")

    def _validate_structure(self):
        """Kiểm tra cấu trúc thư mục có đúng chuẩn Market-1501 không."""
        required = ["query", "bounding_box_test"]
        for d in required:
            path = os.path.join(self.root, d)
            if not os.path.isdir(path):
                raise FileNotFoundError(
                    f"[Dataset] Không tìm thấy thư mục: {path}\n"
                    "Vui lòng kiểm tra lại đường dẫn dataset_root."
                )

    def _parse_filename(self, fname: str):
        """
        Trích xuất Person ID (PID) và Camera ID (CID) từ tên file.

        Args:
            fname: Tên file, ví dụ '0001_c2s3_000451_03.jpg'

        Returns:
            (pid, cid) nếu parse thành công, ngược lại trả về (None, None)
        """
        match = self._FNAME_PATTERN.search(fname)
        if not match:
            return None, None
        pid = int(match.group(1))  # Person ID (có thể âm nếu là ảnh rác)
        cid = int(match.group(2))  # Camera ID (1-indexed)
        return pid, cid

    def _load_split(self, split_name: str):
        """
        Tải một tập dữ liệu (query hoặc gallery), loại bỏ ảnh rác.

        Args:
            split_name: Tên thư mục con ('query' hoặc 'bounding_box_test')

        Returns:
            List of dict: [{'path': str, 'pid': int, 'cid': int}, ...]
        """
        split_dir = os.path.join(self.root, split_name)
        records = []

        for fname in sorted(os.listdir(split_dir)):
            # Chỉ xử lý file ảnh .jpg
            if not fname.lower().endswith(".jpg"):
                continue

            pid, cid = self._parse_filename(fname)
            if pid is None:
                continue  # Bỏ qua file không parse được

            # *** QUAN TRỌNG: Loại bỏ ảnh rác theo chuẩn Market-1501 ***
            # PID = -1 : Ảnh background / junk không có người (phải loại bỏ)
            # PID = 0  : Một số phiên bản dataset dùng 0 cho distractor
            if pid <= 0:
                continue

            records.append({
                "path": os.path.join(split_dir, fname),
                "pid":  pid,
                "cid":  cid,
                "fname": fname,
            })

        return records

    def get_pids(self, split: str):
        """Trả về mảng numpy chứa PID của từng ảnh trong split."""
        data = self.query if split == "query" else self.gallery
        return np.array([r["pid"] for r in data], dtype=np.int32)

    def get_cids(self, split: str):
        """Trả về mảng numpy chứa CID của từng ảnh trong split."""
        data = self.query if split == "query" else self.gallery
        return np.array([r["cid"] for r in data], dtype=np.int32)

    def get_paths(self, split: str):
        """Trả về danh sách đường dẫn ảnh của split."""
        data = self.query if split == "query" else self.gallery
        return [r["path"] for r in data]


# ==============================================================================
# CLASS 2: HOGFeatureExtractor
# Nhiệm vụ: Tiền xử lý ảnh và trích xuất vector đặc trưng HOG.
# ==============================================================================
class HOGFeatureExtractor:
    """
    Trích xuất đặc trưng HOG (Histogram of Oriented Gradients) từ ảnh người.

    HOG là phương pháp trích xuất đặc trưng dựa trên phân phối hướng gradient
    cục bộ. Trong Re-ID, HOG bắt được thông tin về hình dạng cơ thể và
    đường viền trang phục - hai yếu tố quan trọng để phân biệt người.

    Tham số HOG được lựa chọn cho bài toán Re-ID:
        - Image size  : 128x64 (H x W) - chuẩn cho pedestrian detection
        - Orientations: 9   - phân chia gradient thành 9 hướng (0°~180°)
        - Pixels/cell : 8x8 - kích thước ô cơ sở để tính histogram
        - Cells/block : 2x2 - nhóm ô để chuẩn hóa (block normalization)
        - Block stride: 1   - bước dịch chuyển block (overlap = 50%)
    """

    def __init__(
        self,
        img_height: int = 128,
        img_width: int = 64,
        orientations: int = 9,
        pixels_per_cell: tuple = (8, 8),
        cells_per_block: tuple = (2, 2),
        n_workers: int = 4,
    ):
        """
        Khởi tạo HOG Feature Extractor.

        Args:
            img_height   : Chiều cao ảnh sau khi resize (pixel)
            img_width    : Chiều rộng ảnh sau khi resize (pixel)
            orientations : Số lượng bin hướng gradient
            pixels_per_cell: Kích thước ô (height, width)
            cells_per_block : Số ô trong mỗi block (h, w)
            n_workers    : Số luồng song song cho multiprocessing
        """
        self.img_h = img_height
        self.img_w = img_width
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.n_workers = n_workers

        # Tính kích thước vector HOG để pre-allocate bộ nhớ
        # Công thức: n_cells_h = img_h / ppc_h ; n_cells_w = img_w / ppc_w
        # n_blocks_h = n_cells_h - cpb_h + 1 ; tương tự cho w
        # feature_dim = n_blocks_h * n_blocks_w * cpb_h * cpb_w * orientations
        n_cells_h = img_height // pixels_per_cell[0]
        n_cells_w = img_width  // pixels_per_cell[1]
        n_blocks_h = n_cells_h - cells_per_block[0] + 1
        n_blocks_w = n_cells_w - cells_per_block[1] + 1
        self.feature_dim = (
            n_blocks_h * n_blocks_w
            * cells_per_block[0] * cells_per_block[1]
            * orientations
        )
        print(f"[HOG] Kích thước vector HOG: {self.feature_dim} chiều")

    def _preprocess(self, img_bgr: np.ndarray) -> np.ndarray:
        """
        Tiền xử lý ảnh trước khi trích xuất HOG.

        Các bước:
            1. Resize về kích thước chuẩn (128x64) bằng bilinear interpolation.
            2. Chuyển sang không gian màu grayscale để giảm nhiễu màu sắc
               và tập trung vào cấu trúc hình dạng/trang phục.
            3. Chuẩn hóa cường độ pixel về [0, 1] để ổn định gradient.

        Args:
            img_bgr: Ảnh BGR đọc từ OpenCV

        Returns:
            Ảnh grayscale float32 đã được chuẩn hóa, shape (H, W)
        """
        # Bước 1: Resize về kích thước chuẩn
        img_resized = cv2.resize(
            img_bgr, (self.img_w, self.img_h),
            interpolation=cv2.INTER_LINEAR
        )

        # Bước 2: Chuyển sang grayscale
        # HOG hoạt động trên gradient cường độ sáng, không cần thông tin màu
        img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

        # Bước 3: Chuẩn hóa về [0.0, 1.0]
        img_norm = img_gray.astype(np.float32) / 255.0

        return img_norm

    def extract_single(self, img_path: str) -> np.ndarray:
        """
        Trích xuất HOG feature vector từ một ảnh.

        Args:
            img_path: Đường dẫn tới file ảnh

        Returns:
            Feature vector HOG dạng numpy array, shape (feature_dim,)
            Trả về vector 0 nếu không đọc được ảnh.
        """
        # Đọc ảnh bằng OpenCV (BGR format)
        img = cv2.imread(img_path)
        if img is None:
            print(f"[HOG] Cảnh báo: Không đọc được ảnh: {img_path}")
            return np.zeros(self.feature_dim, dtype=np.float32)

        # Tiền xử lý ảnh
        img_preprocessed = self._preprocess(img)

        # Trích xuất HOG feature bằng skimage
        # - feature_vector=True: trả về 1D array thay vì multi-dim tensor
        feat = hog(
            img_preprocessed,
            orientations=self.orientations,
            pixels_per_cell=self.pixels_per_cell,
            cells_per_block=self.cells_per_block,
            block_norm="L2-Hys",  # Chuẩn hóa Lowe's L2-Hys (ổn định nhất)
            feature_vector=True,
        )

        return feat.astype(np.float32)

    def extract_batch(self, img_paths: list, desc: str = "Extracting HOG") -> np.ndarray:
        """
        Trích xuất HOG feature cho toàn bộ danh sách ảnh (có progress bar).

        Sử dụng vòng lặp tuần tự với tqdm để hiển thị tiến trình.
        (Multiprocessing có thể dùng nếu muốn nhưng cần serialize function)

        Args:
            img_paths: Danh sách đường dẫn ảnh
            desc     : Nhãn hiển thị trên progress bar

        Returns:
            Ma trận đặc trưng, shape (N, feature_dim), dtype float32
        """
        n = len(img_paths)
        # Pre-allocate bộ nhớ để tránh dynamic resize
        features = np.zeros((n, self.feature_dim), dtype=np.float32)

        for i, path in enumerate(tqdm(img_paths, desc=desc, unit="img")):
            features[i] = self.extract_single(path)

        return features


# ==============================================================================
# CLASS 3: ReIDEvaluator
# Nhiệm vụ: So khớp đặc trưng và tính toán các độ đo đánh giá chuẩn.
# ==============================================================================
class ReIDEvaluator:
    """
    Đánh giá hệ thống Re-ID theo 3 độ đo chuẩn trong báo cáo:
        1. Rank-1 Accuracy
        2. mean Average Precision (mAP)
        3. F1-Score tại Top-k

    Quy tắc đánh giá chuẩn Market-1501:
        - Với mỗi query (pid_q, cid_q), ta loại bỏ khỏi gallery
          các ảnh CÙNG PID VÀ CÙNG CID (same-camera same-ID).
          Đây là "junk images" vì chúng quá giống ảnh query (same view).
        - Giữ lại các ảnh cùng PID nhưng khác CID (true matches).
        - Giữ lại tất cả ảnh khác PID (distractors).
    """

    def __init__(self, top_k: int = 10):
        """
        Args:
            top_k: Ngưỡng k để tính F1-Score (mặc định 10)
        """
        self.top_k = top_k

    def compute_distance_matrix(
        self,
        query_feats: np.ndarray,
        gallery_feats: np.ndarray,
        metric: str = "cosine",
    ) -> np.ndarray:
        """
        Tính ma trận khoảng cách giữa query và gallery.

        Sử dụng scipy.spatial.distance.cdist - được tối ưu hóa bằng C,
        nhanh hơn nhiều so với vòng lặp Python thông thường.

        Args:
            query_feats  : Shape (Q, D) - Q query, D chiều đặc trưng
            gallery_feats: Shape (G, D) - G gallery images
            metric       : 'cosine' hoặc 'euclidean'

        Returns:
            dist_mat: Shape (Q, G), dist_mat[i,j] = khoảng cách query_i -> gallery_j
        """
        print(f"[Evaluator] Tính distance matrix ({metric})... "
              f"Query: {query_feats.shape[0]}, Gallery: {gallery_feats.shape[0]}")
        t0 = time.time()
        dist_mat = cdist(query_feats, gallery_feats, metric=metric)
        print(f"[Evaluator] ✓ Distance matrix shape: {dist_mat.shape} "
              f"| Thời gian: {time.time()-t0:.2f}s")
        return dist_mat.astype(np.float32)

    def _get_matches(
        self,
        q_pid: int,
        q_cid: int,
        g_pids: np.ndarray,
        g_cids: np.ndarray,
        sorted_indices: np.ndarray,
    ):
        """
        Lấy danh sách nhãn đúng/sai cho một query theo thứ tự đã rank.

        Áp dụng quy tắc loại bỏ chuẩn Market-1501:
            - "Junk" (loại bỏ): cùng PID, cùng CID với query
            - "Match" (correct): cùng PID, khác CID
            - "Distractor" (incorrect): khác PID

        Args:
            q_pid        : Person ID của query
            q_cid        : Camera ID của query
            g_pids       : Mảng PID của toàn gallery
            g_cids       : Mảng CID của toàn gallery
            sorted_indices: Chỉ số gallery đã được sắp xếp theo khoảng cách tăng dần

        Returns:
            matches     : Mảng nhị phân (1=đúng, 0=sai) đã lọc bỏ junk
            num_valid_gt: Số lượng ảnh đúng thực tế trong gallery (N_q trong công thức mAP)
        """
        # Xây dựng mask cho các loại ảnh
        is_junk = (g_pids == q_pid) & (g_cids == q_cid)   # Loại bỏ
        is_match = (g_pids == q_pid) & (g_cids != q_cid)   # True match

        # Tổng số ảnh đúng trong gallery (dùng cho mẫu số của mAP)
        num_valid_gt = is_match.sum()

        if num_valid_gt == 0:
            # Query không có ảnh match nào trong gallery -> bỏ qua
            return None, 0

        # Lọc danh sách đã rank: loại bỏ junk images
        matches = []
        for idx in sorted_indices:
            if is_junk[idx]:
                continue  # Bỏ qua junk, không đếm vào kết quả
            matches.append(1 if is_match[idx] else 0)

        return np.array(matches, dtype=np.int32), int(num_valid_gt)

    # --------------------------------------------------------------------------
    # ĐỘ ĐO 1: Rank-1 Accuracy
    # Công thức: Rank-1 = (1/Q) * Σ r(q)
    # r(q) = 1 nếu ảnh top-1 của gallery trùng PID với query
    # --------------------------------------------------------------------------
    def compute_rank1(
        self,
        dist_mat: np.ndarray,
        q_pids: np.ndarray,
        g_pids: np.ndarray,
        q_cids: np.ndarray,
        g_cids: np.ndarray,
    ) -> float:
        """
        Tính Rank-1 Accuracy.

        Với mỗi query:
            - Sắp xếp gallery theo khoảng cách tăng dần.
            - Loại bỏ junk images (cùng PID & CID).
            - Kiểm tra xem ảnh đứng đầu (rank-1) có cùng PID không.
            - r(q) = 1 nếu đúng, r(q) = 0 nếu sai.

        Returns:
            Rank-1 accuracy trong khoảng [0, 1]
        """
        num_query = dist_mat.shape[0]
        num_correct = 0
        num_valid_queries = 0

        for q_idx in range(num_query):
            # Sắp xếp gallery theo khoảng cách từ nhỏ đến lớn
            sorted_indices = np.argsort(dist_mat[q_idx])

            matches, num_gt = self._get_matches(
                q_pids[q_idx], q_cids[q_idx],
                g_pids, g_cids, sorted_indices
            )

            if matches is None:
                continue  # Query không có match trong gallery

            num_valid_queries += 1

            # r(q) = 1 nếu ảnh rank-1 (matches[0]) là đúng
            if matches[0] == 1:
                num_correct += 1

        rank1 = num_correct / num_valid_queries if num_valid_queries > 0 else 0.0
        return rank1

    # --------------------------------------------------------------------------
    # ĐỘ ĐO 2: mean Average Precision (mAP)
    # Công thức: mAP = (1/Q) * Σ_q [ (1/N_q) * Σ_k P(k)*rel(k) ]
    # --------------------------------------------------------------------------
    def compute_map(
        self,
        dist_mat: np.ndarray,
        q_pids: np.ndarray,
        g_pids: np.ndarray,
        q_cids: np.ndarray,
        g_cids: np.ndarray,
    ) -> float:
        """
        Tính mean Average Precision (mAP).

        Với mỗi query q:
            Average Precision (AP) = (1/N_q) * Σ_{k=1}^{n} P(k) * rel(k)

            - N_q: số ảnh đúng trong gallery (mẫu số, như trong công thức LaTeX)
            - P(k): Precision tại vị trí k = (số ảnh đúng trong top-k) / k
            - rel(k): 1 nếu ảnh hạng k là đúng, 0 nếu sai

        mAP = trung bình AP của tất cả query.

        Returns:
            mAP score trong khoảng [0, 1]
        """
        num_query = dist_mat.shape[0]
        ap_list = []

        for q_idx in range(num_query):
            # Sắp xếp gallery theo khoảng cách tăng dần
            sorted_indices = np.argsort(dist_mat[q_idx])

            matches, num_gt = self._get_matches(
                q_pids[q_idx], q_cids[q_idx],
                g_pids, g_cids, sorted_indices
            )

            if matches is None:
                continue  # Query này không có match -> bỏ qua

            # Tính Average Precision cho query này
            # Duyệt qua danh sách kết quả đã lọc
            num_correct_so_far = 0
            ap = 0.0

            for k, rel in enumerate(matches, start=1):
                if rel == 1:
                    num_correct_so_far += 1
                    # P(k) = số đúng trong top-k / k
                    precision_at_k = num_correct_so_far / k
                    # Cộng dồn: P(k) * rel(k) = precision_at_k * 1
                    ap += precision_at_k

            # Chia cho N_q (tổng ảnh đúng trong gallery) - theo công thức LaTeX
            ap = ap / num_gt
            ap_list.append(ap)

        mean_ap = np.mean(ap_list) if ap_list else 0.0
        return float(mean_ap)

    # --------------------------------------------------------------------------
    # ĐỘ ĐO 3: F1-Score tại Top-k
    # Công thức: F1 = 2 * (Precision * Recall) / (Precision + Recall)
    # Trong ngữ cảnh retrieval tại Top-k:
    #   Precision@k = TP_k / k
    #   Recall@k    = TP_k / N_q
    # --------------------------------------------------------------------------
    def compute_f1_at_k(
        self,
        dist_mat: np.ndarray,
        q_pids: np.ndarray,
        g_pids: np.ndarray,
        q_cids: np.ndarray,
        g_cids: np.ndarray,
    ) -> dict:
        """
        Tính F1-Score trung bình tại ngưỡng Top-k.

        Định nghĩa trong ngữ cảnh truy xuất (Retrieval) tại Top-k:
            - TP_k    : Số ảnh đúng trong top-k kết quả
            - FP_k    : Số ảnh sai trong top-k   = k - TP_k
            - FN      : Ảnh đúng bị bỏ lỡ (ngoài top-k) = N_q - TP_k

            Precision@k = TP_k / (TP_k + FP_k) = TP_k / k
            Recall@k    = TP_k / (TP_k + FN)   = TP_k / N_q

            F1@k = 2 * Precision@k * Recall@k / (Precision@k + Recall@k)

        Returns:
            dict với các key: 'f1', 'precision', 'recall', 'k'
        """
        k = self.top_k
        num_query = dist_mat.shape[0]
        f1_list, prec_list, rec_list = [], [], []

        for q_idx in range(num_query):
            sorted_indices = np.argsort(dist_mat[q_idx])

            matches, num_gt = self._get_matches(
                q_pids[q_idx], q_cids[q_idx],
                g_pids, g_cids, sorted_indices
            )

            if matches is None:
                continue

            # Lấy top-k kết quả (sau khi đã loại junk)
            top_k_matches = matches[:k]

            # TP_k: số ảnh đúng trong top-k
            tp_k = int(top_k_matches.sum())
            # Số ảnh thực sự được lấy (có thể ít hơn k nếu gallery nhỏ)
            actual_k = len(top_k_matches)

            # Precision@k = TP_k / k
            precision = tp_k / actual_k if actual_k > 0 else 0.0

            # Recall@k = TP_k / N_q (N_q = num_gt)
            recall = tp_k / num_gt if num_gt > 0 else 0.0

            # F1-Score = 2 * P * R / (P + R)
            if precision + recall > 0:
                f1 = 2 * precision * recall / (precision + recall)
            else:
                f1 = 0.0

            f1_list.append(f1)
            prec_list.append(precision)
            rec_list.append(recall)

        return {
            "f1":        float(np.mean(f1_list))   if f1_list   else 0.0,
            "precision": float(np.mean(prec_list)) if prec_list else 0.0,
            "recall":    float(np.mean(rec_list))  if rec_list  else 0.0,
            "k":         k,
        }

    def evaluate(
        self,
        dist_mat: np.ndarray,
        q_pids: np.ndarray,
        g_pids: np.ndarray,
        q_cids: np.ndarray,
        g_cids: np.ndarray,
    ) -> dict:
        """
        Chạy toàn bộ pipeline đánh giá và trả về kết quả 3 độ đo.

        Args:
            dist_mat: Ma trận khoảng cách (Q, G)
            q_pids  : PID của query, shape (Q,)
            g_pids  : PID của gallery, shape (G,)
            q_cids  : CID của query, shape (Q,)
            g_cids  : CID của gallery, shape (G,)

        Returns:
            dict chứa rank1, map, f1, precision, recall
        """
        print("\n[Evaluator] ========== Bắt đầu đánh giá ==========")

        print("[Evaluator] Đang tính Rank-1 Accuracy...")
        t0 = time.time()
        rank1 = self.compute_rank1(dist_mat, q_pids, g_pids, q_cids, g_cids)
        print(f"[Evaluator] ✓ Rank-1: {rank1:.4f} | {time.time()-t0:.2f}s")

        print("[Evaluator] Đang tính mAP...")
        t0 = time.time()
        map_score = self.compute_map(dist_mat, q_pids, g_pids, q_cids, g_cids)
        print(f"[Evaluator] ✓ mAP   : {map_score:.4f} | {time.time()-t0:.2f}s")

        print(f"[Evaluator] Đang tính F1-Score@{self.top_k}...")
        t0 = time.time()
        f1_results = self.compute_f1_at_k(dist_mat, q_pids, g_pids, q_cids, g_cids)
        print(
            f"[Evaluator] ✓ F1@{self.top_k} : {f1_results['f1']:.4f} "
            f"(P={f1_results['precision']:.4f}, R={f1_results['recall']:.4f}) "
            f"| {time.time()-t0:.2f}s"
        )

        return {
            "rank1":     rank1,
            "map":       map_score,
            "f1":        f1_results["f1"],
            "precision": f1_results["precision"],
            "recall":    f1_results["recall"],
            "top_k":     self.top_k,
        }


# ==============================================================================
# HÀM IN BÁO CÁO KẾT QUẢ
# ==============================================================================
def print_report(results: dict):
    """
    In báo cáo kết quả theo dạng bảng chuẩn để đưa vào báo cáo đồ án.
    """
    border = "=" * 62
    print(f"\n{border}")
    print("         KẾT QUẢ ĐÁNH GIÁ - HOG Re-ID Pipeline")
    print(f"         Tập dữ liệu: Market-1501")
    print(f"         Phương pháp: HOG + Cosine Distance")
    print(border)
    print(f"  {'Độ đo':<30} {'Giá trị':>10}  {'Ghi chú':<18}")
    print("-" * 62)
    print(
        f"  {'Rank-1 Accuracy':<30} "
        f"{results['rank1']*100:>9.2f}%  "
        f"{'(Chỉ số chính)':<18}"
    )
    print(
        f"  {'mean Average Precision (mAP)':<30} "
        f"{results['map']*100:>9.2f}%  "
        f"{'(Truy xuất toàn diện)':<18}"
    )
    print(
        f"  {'F1-Score @ Top-' + str(results['top_k']):<30} "
        f"{results['f1']*100:>9.2f}%  "
        f"{'(Cân bằng P & R)':<18}"
    )
    print("-" * 62)
    print(
        f"  {'  ├─ Precision @ Top-' + str(results['top_k']):<30} "
        f"{results['precision']*100:>9.2f}%"
    )
    print(
        f"  {'  └─ Recall @ Top-' + str(results['top_k']):<30} "
        f"{results['recall']*100:>9.2f}%"
    )
    print(border)
    print("\n  Chú thích:")
    print(f"  - Rank-1: Tỷ lệ query có ảnh TOP-1 đúng danh tính")
    print(f"  - mAP   : Độ chính xác trung bình toàn diện trên gallery")
    print(f"  - F1    : Trung bình điều hòa Precision & Recall @ Top-{results['top_k']}")
    print(border)


# ==============================================================================
# HÀM CHÍNH (MAIN PIPELINE)
# ==============================================================================
def main(args):
    """
    Pipeline chính: Dataset -> Feature Extraction -> Matching -> Evaluation
    """
    total_start = time.time()

    # ------------------------------------------------------------------
    # BƯỚC 1: Tải dữ liệu
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("BƯỚC 1: TẢI DỮ LIỆU MARKET-1501")
    print("=" * 60)
    dataset = Market1501Dataset(args.dataset_root)

    query_paths   = dataset.get_paths("query")
    gallery_paths = dataset.get_paths("gallery")
    q_pids = dataset.get_pids("query")
    g_pids = dataset.get_pids("gallery")
    q_cids = dataset.get_cids("query")
    g_cids = dataset.get_cids("gallery")

    # ------------------------------------------------------------------
    # BƯỚC 2: Trích xuất đặc trưng HOG
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("BƯỚC 2: TRÍCH XUẤT ĐẶC TRƯNG HOG")
    print("=" * 60)
    extractor = HOGFeatureExtractor(
        img_height=args.img_h,
        img_width=args.img_w,
        orientations=args.orientations,
        pixels_per_cell=(args.ppc, args.ppc),
        cells_per_block=(args.cpb, args.cpb),
    )

    print("\n[HOG] Trích xuất đặc trưng cho tập QUERY...")
    t0 = time.time()
    query_feats = extractor.extract_batch(query_paths, desc="  Query HOG")
    print(f"[HOG] ✓ Query features: {query_feats.shape} | {time.time()-t0:.1f}s")

    print("\n[HOG] Trích xuất đặc trưng cho tập GALLERY...")
    t0 = time.time()
    gallery_feats = extractor.extract_batch(gallery_paths, desc="  Gallery HOG")
    print(f"[HOG] ✓ Gallery features: {gallery_feats.shape} | {time.time()-t0:.1f}s")

    # ------------------------------------------------------------------
    # BƯỚC 3: Tính Distance Matrix
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("BƯỚC 3: TÍNH MA TRẬN KHOẢNG CÁCH (COSINE DISTANCE)")
    print("=" * 60)
    evaluator = ReIDEvaluator(top_k=args.top_k)
    dist_mat = evaluator.compute_distance_matrix(
        query_feats, gallery_feats, metric=args.metric
    )

    # ------------------------------------------------------------------
    # BƯỚC 4: Đánh giá với 3 độ đo chuẩn
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("BƯỚC 4: ĐÁNH GIÁ VỚI 3 ĐỘ ĐO (Rank-1, mAP, F1)")
    print("=" * 60)
    results = evaluator.evaluate(dist_mat, q_pids, g_pids, q_cids, g_cids)

    # ------------------------------------------------------------------
    # BƯỚC 5: In kết quả
    # ------------------------------------------------------------------
    print_report(results)

    total_time = time.time() - total_start
    print(f"\n[Pipeline] Tổng thời gian chạy: {total_time:.1f}s "
          f"({total_time/60:.1f} phút)")

    return results


# ==============================================================================
# ENTRY POINT
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="HOG-based Person Re-ID Pipeline cho Market-1501"
    )

    # --- Đường dẫn dataset ---
    parser.add_argument(
        "--dataset_root",
        type=str,
        required=True,
        help="Đường dẫn tới thư mục Market-1501-v15.09.15/",
    )

    # --- Tham số HOG ---
    parser.add_argument("--img_h",       type=int, default=128,
                        help="Chiều cao ảnh sau resize (default: 128)")
    parser.add_argument("--img_w",       type=int, default=64,
                        help="Chiều rộng ảnh sau resize (default: 64)")
    parser.add_argument("--orientations", type=int, default=9,
                        help="Số bin hướng HOG (default: 9)")
    parser.add_argument("--ppc",          type=int, default=8,
                        help="Pixels per cell (default: 8)")
    parser.add_argument("--cpb",          type=int, default=2,
                        help="Cells per block (default: 2)")

    # --- Tham số matching ---
    parser.add_argument(
        "--metric", type=str, default="cosine",
        choices=["cosine", "euclidean"],
        help="Độ đo khoảng cách: 'cosine' hoặc 'euclidean' (default: cosine)"
    )

    # --- Tham số đánh giá ---
    parser.add_argument(
        "--top_k", type=int, default=10,
        help="Ngưỡng k cho tính F1-Score (default: 10)"
    )

    args = parser.parse_args()

    print("╔══════════════════════════════════════════════════════════╗")
    print("║  HOG-based Person Re-ID Pipeline - Market-1501 Dataset  ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"  Dataset root : {args.dataset_root}")
    print(f"  Image size   : {args.img_h}x{args.img_w}")
    print(f"  HOG params   : orientations={args.orientations}, "
          f"ppc={args.ppc}, cpb={args.cpb}")
    print(f"  Distance     : {args.metric}")
    print(f"  Top-k (F1)   : {args.top_k}")

    main(args)
