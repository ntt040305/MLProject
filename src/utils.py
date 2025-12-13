import math
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


# Các hàm tính toán cho ID3/C4.5 – có chú thích tiếng Việt chi tiết

def calculate_entropy(y: Sequence[Any]) -> float:
    """
    Tính Entropy của tập dữ liệu
    
    Công thức: H(S) = -Σ(p_i × log₂(p_i))
    - Trong đó p_i là tỷ lệ xuất hiện của lớp i trong tập S.
    - Ý nghĩa: đo độ hỗn loạn (không đồng nhất) của nhãn.
      + H = 0: tất cả cùng một lớp (thuần khiết)
      + H cao: phân bố lớp cân bằng, khó dự đoán.

    Args:
        y: numpy array hoặc list các labels
        
    Returns:
        float: Entropy value (0 = hoàn toàn thuần khất, cao = lộn xộn)
        
    Example:
        >>> y = ['e', 'e', 'p', 'p']
        >>> round(calculate_entropy(y), 6)
        1.0

    Lưu ý xử lý biên:
    - Nếu tổng số mẫu = 0 → entropy = 0.0 (không thông tin)
    - Nếu có xác suất p_i = 0 → bỏ qua hạng tử đó vì p*log(p) = 0.
    """
    total = len(y)
    if total == 0:
        return 0.0
    counts = Counter(y)
    entropy = 0.0
    for c in counts.values():
        p = c / total
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy


def _split_by_threshold(values: Sequence[float], y: Sequence[Any], threshold: float) -> Tuple[List[Any], List[Any]]:
    """Chia dữ liệu theo ngưỡng: trả về nhãn bên trái (<=) và bên phải (>)"""
    left = [y[i] for i, v in enumerate(values) if v <= threshold]
    right = [y[i] for i, v in enumerate(values) if v > threshold]
    return left, right


def calculate_information_gain(
    X: Sequence[Sequence[Any]],
    y: Sequence[Any],
    feature_idx: int,
    threshold: Optional[float] = None,
) -> float:
    """
    Tính Information Gain - Dùng cho ID3
    
    Công thức: IG(S, A) = H(S) - Σ(|S_v|/|S| × H(S_v))
    
    Args:
        X: Feature matrix (numpy array or pandas DataFrame or list-of-lists)
        y: Labels
        feature_idx: Index của feature để tính
        threshold: Ngưỡng nếu feature là continuous (None nếu categorical)
        
    Returns:
        float: Information Gain value
        
    Giải thích:
        - H(S): Entropy của toàn bộ dataset
        - |S_v|: Số samples có giá trị v
        - H(S_v): Entropy của subset có giá trị v

    Xử lý biên:
        - Nếu dữ liệu rỗng → IG = 0.0
        - Nếu tất cả y cùng lớp → IG = 0.0 (không cần split)
        - Nếu threshold được cung cấp → coi là bài toán liên tục 2 nhánh (<= và >)
    """
    n = len(y)
    if n == 0:
        return 0.0
    # Nếu tập hiện tại đã thuần khất → không còn thông tin để tăng
    if len(set(y)) == 1:
        return 0.0

    base_entropy = calculate_entropy(y)

    # Lấy cột đặc trưng theo feature_idx từ X
    # Hỗ trợ cả list-of-lists và pandas DataFrame
    if hasattr(X, "iloc"):
        col = list(X.iloc[:, feature_idx])
    else:
        col = [row[feature_idx] for row in X]

    if threshold is not None:
        # Continuous: chia thành hai nhánh theo ngưỡng
        left_labels, right_labels = _split_by_threshold(col, y, threshold)
        w_left = len(left_labels) / n
        w_right = len(right_labels) / n
        new_entropy = w_left * calculate_entropy(left_labels) + w_right * calculate_entropy(right_labels)
        info_gain = base_entropy - new_entropy
        return info_gain
    else:
        # Categorical: nhóm theo từng giá trị
        groups: Dict[Any, List[Any]] = {}
        for i, v in enumerate(col):
            groups.setdefault(v, []).append(y[i])
        new_entropy = 0.0
        for g in groups.values():
            w = len(g) / n
            new_entropy += w * calculate_entropy(g)
        info_gain = base_entropy - new_entropy
        return info_gain


def calculate_split_info(
    X: Sequence[Sequence[Any]],
    feature_idx: int,
    threshold: Optional[float] = None,
) -> float:
    """
    Tính Split Information - KEY của C4.5!
    
    Công thức: SplitInfo(S, A) = -Σ(|S_v|/|S| × log₂(|S_v|/|S|))
    
    ĐÂY LÀ ĐIỂM KHÁC BIỆT giữa C4.5 và ID3!
    
    Mục đích: Penalize các attribute có nhiều giá trị (như ID)
    
    Args:
        X: Feature matrix
        feature_idx: Index của feature
        threshold: Ngưỡng cho continuous
        
    Returns:
        float: Split Info value
        
    Lưu ý:
        - SplitInfo càng cao → Feature có nhiều nhánh → Bị penalize
        - Nếu feature có N giá trị unique → SplitInfo cao
        - Nếu chia mà một nhánh rỗng → tỷ trọng 0 → bỏ qua hạng tử đó.
    """
    # Lấy cột đặc trưng
    if hasattr(X, "iloc"):
        col = list(X.iloc[:, feature_idx])
    else:
        col = [row[feature_idx] for row in X]
    n = len(col)
    if n == 0:
        return 0.0

    if threshold is not None:
        # Continuous: chia 2 nhánh
        left_count = sum(1 for v in col if v <= threshold)
        right_count = n - left_count
        split_info = 0.0
        for cnt in (left_count, right_count):
            if cnt > 0:
                w = cnt / n
                split_info -= w * math.log2(w)
        return split_info
    else:
        # Categorical: nhiều nhánh
        counts: Dict[Any, int] = Counter(col)
        split_info = 0.0
        for cnt in counts.values():
            if cnt > 0:
                w = cnt / n
                split_info -= w * math.log2(w)
        return split_info


def calculate_gain_ratio(
    X: Sequence[Sequence[Any]],
    y: Sequence[Any],
    feature_idx: int,
    threshold: Optional[float] = None,
) -> float:
    """
    Tính Gain Ratio - TRỌNG TÂM của C4.5!
    
    Công thức: GR(S, A) = IG(S, A) / SplitInfo(S, A)
    
    ĐÂY LÀ LÝ DO C4.5 KHẮC PHỤC BIAS của ID3!
    
    Args:
        X, y, feature_idx, threshold: giống các hàm trên
        
    Returns:
        float: Gain Ratio value (0 nếu SplitInfo = 0)
        
    So sánh:
        ID3: Chỉ xem IG → Chọn feature có nhiều values
        C4.5: Xem IG/SplitInfo → Không bị lừa bởi nhiều values
        
    Example:
        Feature "ID" (8000 unique values):
            - IG rất cao (vì mỗi ID = 1 sample = pure)
            - SplitInfo cũng rất cao (nhiều nhánh)
            - GR = IG/SplitInfo = Thấp → Không chọn!
    """
    ig = calculate_information_gain(X, y, feature_idx, threshold)
    split_info = calculate_split_info(X, feature_idx, threshold)
    if split_info == 0.0:
        # Tránh chia cho 0: nếu SplitInfo = 0 (ví dụ: tất cả cùng một giá trị) → GR = 0
        return 0.0
    return ig / split_info


def find_best_threshold_continuous(
    X_column: Sequence[float],
    y: Sequence[Any],
) -> Tuple[Optional[float], float]:
    """
    Tìm ngưỡng tốt nhất cho continuous feature
    
    C4.5 xử lý continuous bằng cách:
    1. Sort các giá trị unique
    2. Thử các midpoint giữa 2 giá trị liên tiếp
    3. Chọn threshold có Gain Ratio cao nhất
    
    Args:
        X_column: numpy array của 1 feature (continuous)
        y: Labels
        
    Returns:
        best_threshold: float hoặc None nếu không có candidate
        best_gain_ratio: float (0 nếu không có candidate)

    Xử lý biên:
        - Nếu số mẫu < 2 hoặc tất cả giá trị giống nhau → không có ngưỡng.
        - Nếu y thuần khất → mọi ngưỡng đều vô nghĩa → trả về (None, 0.0).
    """
    n = len(X_column)
    if n == 0 or len(set(y)) <= 1:
        return None, 0.0

    # Sắp xếp theo giá trị feature
    pairs = sorted(zip(X_column, y), key=lambda t: t[0])
    values = [v for v, _ in pairs]
    labels = [lbl for _, lbl in pairs]

    candidates: List[float] = []
    for i in range(1, len(values)):
        if values[i] != values[i - 1]:
            # midpoint giữa hai giá trị liên tiếp
            candidates.append((values[i] + values[i - 1]) / 2.0)

    best_t: Optional[float] = None
    best_gr: float = 0.0
    if not candidates:
        return None, 0.0

    # Đánh giá Gain Ratio cho từng ngưỡng
    # Để tái sử dụng calculate_gain_ratio, ta coi X là ma trận một cột
    X_single_col = [[v] for v in values]
    for t in candidates:
        gr = calculate_gain_ratio(X_single_col, labels, feature_idx=0, threshold=t)
        if gr > best_gr:
            best_gr, best_t = gr, t
    return best_t, best_gr


def is_continuous(X_column: Sequence[Any], threshold: int = 10) -> bool:
    """
    Tự động phát hiện feature là continuous hay categorical
    
    Logic:
        - Nếu dtype = float → continuous
        - Nếu số unique values > threshold → continuous
        - Ngược lại → categorical

    Lưu ý:
        - Một số cột int nhưng có nhiều giá trị (ví dụ: năm, mileage) nên coi là continuous.
        - Ngưỡng 10 chỉ là heuristic, có thể điều chỉnh.
    """
    # Kiểm tra kiểu số float trực tiếp
    try:
        # Nếu phần lớn phần tử là float, xem như continuous
        floats = sum(1 for v in X_column if isinstance(v, float))
        if floats / max(1, len(X_column)) > 0.5:
            return True
    except Exception:
        pass

    # Kiểm tra số lượng giá trị duy nhất
    unique_count = len(set(X_column))
    return unique_count > threshold


def majority_label(labels: Sequence[Any]) -> Any:
    """Trả về nhãn phổ biến nhất (phục vụ xây dựng lá)."""
    return Counter(labels).most_common(1)[0][0]
