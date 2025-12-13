from typing import Any, Dict, List, Optional, Tuple
from collections import Counter

from .node import TreeNode
from .utils import (
    calculate_entropy,
    calculate_information_gain,
    calculate_split_info,
    calculate_gain_ratio,
    find_best_threshold_continuous,
    is_continuous,
    majority_label,
)
import numpy as np
import pandas as pd


class DecisionTreeC45:
    """
    Decision Tree C4.5 Implementation từ đầu
    
    Cải tiến so với ID3:
    1. Dùng Gain Ratio thay vì Information Gain → Khắc phục bias
    2. Xử lý continuous features (tìm threshold tối ưu)
    3. Pruning để tránh overfitting (optional)
    4. Xử lý missing values (optional)
    
    Parameters:
        max_depth: Độ sâu tối đa của cây (None = không giới hạn)
        min_samples_split: Số mẫu tối thiểu để split node
        min_samples_leaf: Số mẫu tối thiểu tại mỗi leaf
        min_gain_ratio: Gain Ratio tối thiểu để thực hiện split
        criterion: 'gain_ratio' (C4.5) hoặc 'information_gain' (ID3)
    """

    def __init__(
        self,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        min_gain_ratio: float = 1e-7,
        criterion: str = "gain_ratio",
    ) -> None:
        # Khởi tạo siêu tham số
        self.max_depth = max_depth
        self.min_samples_split = int(min_samples_split)
        self.min_samples_leaf = int(min_samples_leaf)
        self.min_gain_ratio = float(min_gain_ratio)
        if criterion not in ("gain_ratio", "information_gain"):
            raise ValueError("criterion phải là 'gain_ratio' hoặc 'information_gain'")
        self.criterion = criterion

        # Thuộc tính trong quá trình học
        self.root: Optional[TreeNode] = None
        self.feature_names: List[str] = []
        self.class_labels: List[Any] = []
        self.feature_types: List[str] = []  # 'categorical' | 'continuous'
        self.feature_importance_: Dict[str, float] = {}

    def fit(self, X: Any, y: Any, feature_names: Optional[List[str]] = None):
        """
        Train Decision Tree trên dataset
        
        Args:
            X: Feature matrix (numpy array hoặc pandas DataFrame)
            y: Labels (numpy array hoặc pandas Series)
            feature_names: List tên các features (tự động lấy nếu X là DataFrame)
            
        Returns:
            self
            
        Process:
            1. Chuyển X, y về numpy array
            2. Lưu feature names và class names
            3. Phát hiện feature types (categorical/continuous)
            4. Build tree đệ quy
        """
        # Chuẩn hóa dữ liệu đầu vào
        if isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns) if feature_names is None else feature_names
            X_np = X.values
        else:
            X_np = np.asarray(X)
            if feature_names is None:
                self.feature_names = [f"f{i}" for i in range(X_np.shape[1])]
            else:
                self.feature_names = feature_names

        y_np = np.asarray(y)
        if y_np.ndim != 1:
            y_np = y_np.ravel()

        # Lưu danh sách nhãn lớp
        self.class_labels = list(sorted(set(y_np)))

        # Phát hiện loại đặc trưng dựa trên dữ liệu
        self.feature_types = []
        for j in range(X_np.shape[1]):
            col = X_np[:, j]
            self.feature_types.append("continuous" if is_continuous(col) else "categorical")

        # Reset importance
        self.feature_importance_ = {name: 0.0 for name in self.feature_names}

        # Xây dựng cây
        self.root = self._build_tree(X_np, y_np, depth=0)
        return self

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int = 0) -> TreeNode:
        """
        Xây dựng cây đệ quy - TRÁI TIM của thuật toán
        
        Stopping Criteria:
            1. Depth >= max_depth
            2. Số samples < min_samples_split
            3. All samples cùng class (pure node)
            4. Không tìm được split tốt (gain_ratio < threshold)
            5. Không còn features để split (được xử lý qua best split)
            
        Returns:
            TreeNode
        """
        n_samples, n_features = X.shape

        # Tạo node và điền thông tin thống kê
        class_counts = Counter(y)
        node = TreeNode(
            samples=n_samples,
            class_distribution=dict(class_counts),
        )

        # Điều kiện dừng: node thuần khất hoặc quá nhỏ/đủ sâu
        if len(class_counts) == 1:
            node.value = next(iter(class_counts))
            return node
        if self.max_depth is not None and depth >= self.max_depth:
            node.value = majority_label(y)
            return node
        if n_samples < self.min_samples_split:
            node.value = majority_label(y)
            return node

        # Tìm split tốt nhất
        best_feature_idx, best_threshold, best_score = self._find_best_split(X, y)
        if best_feature_idx is None or (self.criterion == "gain_ratio" and best_score < self.min_gain_ratio):
            node.value = majority_label(y)
            return node

        # Gán thông tin node split
        node.feature_idx = best_feature_idx
        node.feature_name = self.feature_names[best_feature_idx]
        node.threshold = best_threshold

        # Cập nhật importance (cộng dồn điểm split ở node này)
        self.feature_importance_[node.feature_name] += float(best_score)

        # Chia tập dữ liệu thành các nhánh con
        subsets = self._split_dataset(X, y, best_feature_idx, best_threshold)

        # Với continuous: 2 nhánh 'left' và 'right'. Với categorical: nhiều nhánh.
        for key, (X_sub, y_sub) in subsets.items():
            # Leaf constraint: đảm bảo mỗi lá có tối thiểu min_samples_leaf
            if len(y_sub) < self.min_samples_leaf:
                child = TreeNode(value=majority_label(y))
            else:
                child = self._build_tree(X_sub, y_sub, depth + 1)
            node.add_child(key, child)

        return node

    def _find_best_split(self, X: np.ndarray, y: np.ndarray) -> Tuple[Optional[int], Optional[float], float]:
        """
        Tìm feature và threshold tốt nhất để split
        
        ĐÂY LÀ ĐIỂM KHÁC BIỆT C4.5 vs ID3!
        
        For mỗi feature:
            - Nếu categorical: Tính Gain Ratio trực tiếp
            - Nếu continuous: Tìm best threshold, sau đó tính Gain Ratio
            
        Returns:
            best_feature_idx: int
            best_threshold: float (hoặc None nếu categorical)
            best_gain_ratio: float
            
        C4.5 chọn feature có GAIN RATIO cao nhất (không phải IG!)
        """
        n_features = X.shape[1]
        best_idx: Optional[int] = None
        best_threshold: Optional[float] = None
        best_score: float = 0.0

        for j in range(n_features):
            col = X[:, j]
            if self.feature_types[j] == "continuous":
                # Tìm ngưỡng tốt nhất theo Gain Ratio
                t, gr = find_best_threshold_continuous(col, y)
                score = gr if self.criterion == "gain_ratio" else calculate_information_gain([[v] for v in col], y, 0, t)
                if score > best_score and t is not None:
                    best_idx, best_threshold, best_score = j, t, score
            else:
                # Rời rạc: tính trực tiếp
                score = (
                    calculate_gain_ratio(X, y, j)
                    if self.criterion == "gain_ratio"
                    else calculate_information_gain(X, y, j)
                )
                if score > best_score:
                    best_idx, best_threshold, best_score = j, None, score

        return best_idx, best_threshold, best_score

    def _split_dataset(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_idx: int,
        threshold: Optional[float] = None,
    ) -> Dict[Any, Tuple[np.ndarray, np.ndarray]]:
        """
        Chia dataset thành các subsets
        
        Categorical: {value1: (X1, y1), value2: (X2, y2), ...}
        Continuous: {'<=': (X_left, y_left), '>': (X_right, y_right)}
        
        Returns:
            dict: subsets
        """
        subsets: Dict[Any, Tuple[np.ndarray, np.ndarray]] = {}
        col = X[:, feature_idx]

        if threshold is not None:
            left_mask = col <= threshold
            right_mask = ~left_mask
            subsets["<="] = (X[left_mask], y[left_mask])
            subsets[">"] = (X[right_mask], y[right_mask])
        else:
            # Nhóm theo giá trị rời rạc
            values = set(col)
            for v in values:
                mask = col == v
                subsets[v] = (X[mask], y[mask])
        return subsets

    def predict(self, X: Any) -> np.ndarray:
        """
        Dự đoán cho nhiều samples
        
        Args:
            X: Feature matrix
            
        Returns:
            numpy array: Predicted labels
        """
        if isinstance(X, pd.DataFrame):
            X_np = X.values
        else:
            X_np = np.asarray(X)
        preds = [self._predict_single(X_np[i], self.root) for i in range(X_np.shape[0])]
        return np.asarray(preds)

    def _predict_single(self, x: np.ndarray, node: Optional[TreeNode]) -> Any:
        """
        Dự đoán cho 1 sample đệ quy
        
        Logic:
            1. Nếu node là leaf → return value
            2. Ngược lại, xem giá trị feature → đi xuống child tương ứng
            3. Đệ quy cho đến khi gặp leaf
        """
        # Xử lý an toàn
        if node is None:
            return None
        if node.is_leaf:
            return node.value

        j = node.feature_idx
        v = x[j]
        if node.threshold is not None:
            key = "<=" if v <= node.threshold else ">"
            child = node.children.get(key)
        else:
            child = node.children.get(v)
            if child is None:
                # Nếu giá trị danh mục chưa từng thấy, fallback về đa số tại node hiện tại
                # (node.class_distribution có thể dùng, ở đây trả về nhãn đa số toàn cục)
                return majority_label(list(node.class_distribution.keys())) if node.class_distribution else None
        return self._predict_single(x, child)

    def score(self, X: Any, y: Any) -> float:
        """Tính accuracy"""
        y_pred = self.predict(X)
        y_true = np.asarray(y)
        return float((y_pred == y_true).mean())

    def get_params(self) -> Dict[str, Any]:
        """Return hyperparameters (để compatible với sklearn API)"""
        return {
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
            "min_gain_ratio": self.min_gain_ratio,
            "criterion": self.criterion,
        }

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Tính feature importance dựa trên Gain Ratio
        
        Features được dùng nhiều và có GR cao → Important
        
        Returns:
            dict: {feature_name: importance_score}
        """
        # Chuẩn hóa tổng về 1 để dễ so sánh
        total = sum(self.feature_importance_.values())
        if total == 0:
            return self.feature_importance_
        return {k: v / total for k, v in self.feature_importance_.items()}

    def get_tree_info(self) -> Dict[str, int]:
        """
        Lấy thông tin về cây
        
        Returns:
            dict: {
                'depth': max depth,
                'n_nodes': total nodes,
                'n_leaves': số leaf nodes
            }
        """
        return {
            "depth": self._get_depth(self.root),
            "n_nodes": self._count_nodes(self.root),
            "n_leaves": self._count_leaves(self.root),
        }

    def _count_nodes(self, node: Optional[TreeNode]) -> int:
        """Helper: Đếm nodes đệ quy"""
        if node is None:
            return 0
        return 1 + sum(self._count_nodes(child) for child in node.children.values())

    def _count_leaves(self, node: Optional[TreeNode]) -> int:
        if node is None:
            return 0
        if node.is_leaf:
            return 1
        return sum(self._count_leaves(child) for child in node.children.values())

    def _get_depth(self, node: Optional[TreeNode]) -> int:
        """Helper: Tính depth đệ quy"""
        if node is None or node.is_leaf:
            return 0
        return 1 + max((self._get_depth(child) for child in node.children.values()), default=0)


class DecisionTreeID3(DecisionTreeC45):
    """
    ID3 implementation - chỉ khác C4.5 ở việc dùng IG thay vì GR
    
    Inherit từ C4.5 nhưng override _find_best_split để dùng IG
    """

    def __init__(self, **kwargs):
        super().__init__(criterion="information_gain", **kwargs)

    def _find_best_split(self, X: np.ndarray, y: np.ndarray) -> Tuple[Optional[int], Optional[float], float]:
        n_features = X.shape[1]
        best_idx: Optional[int] = None
        best_threshold: Optional[float] = None
        best_score: float = 0.0

        for j in range(n_features):
            col = X[:, j]
            if self.feature_types[j] == "continuous":
                # ID3 không định nghĩa continuous; ta hỗ trợ bằng cách tìm threshold theo IG
                # tái dùng candidate từ midpoints giống C4.5
                # Lấy candidates bằng find_best_threshold_continuous nhưng tính GR; ở đây tự tạo lại để dùng IG
                pairs = sorted(zip(col, y), key=lambda t: t[0])
                values = [v for v, _ in pairs]
                candidates = []
                for i in range(1, len(values)):
                    if values[i] != values[i - 1]:
                        candidates.append((values[i] + values[i - 1]) / 2.0)
                for t in candidates:
                    ig = calculate_information_gain([[v] for v in col], y, 0, t)
                    if ig > best_score:
                        best_idx, best_threshold, best_score = j, t, ig
            else:
                ig = calculate_information_gain(X, y, j)
                if ig > best_score:
                    best_idx, best_threshold, best_score = j, None, ig

        return best_idx, best_threshold, best_score
