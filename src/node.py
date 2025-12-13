from typing import Any, Dict, Optional


class TreeNode:
    """
    Node của cây quyết định
    
    Attributes:
        feature_idx: Index của feature để split (None nếu là leaf)
        feature_name: Tên feature (để dễ đọc)
        threshold: Ngưỡng cho continuous feature (None nếu categorical)
        children: Dict {value: child_node} cho categorical hoặc {<='threshold': left, >'threshold': right}
        value: Class label nếu là leaf node
        is_leaf: Boolean - node này có phải leaf không
        samples: Số samples tại node này
        class_distribution: Dict {class: count} - phân bố class tại node
    """

    def __init__(
        self,
        feature_idx: Optional[int] = None,
        feature_name: Optional[str] = None,
        threshold: Optional[float] = None,
        children: Optional[Dict[Any, "TreeNode"]] = None,
        value: Optional[Any] = None,
        samples: int = 0,
        class_distribution: Optional[Dict[Any, int]] = None,
    ) -> None:
        # Chỉ số và tên thuộc tính dùng để chia node (nếu không phải lá)
        self.feature_idx: Optional[int] = feature_idx
        self.feature_name: Optional[str] = feature_name

        # Nếu thuộc tính là liên tục, threshold xác định nhánh trái (<=) và phải (>)
        self.threshold: Optional[float] = threshold

        # Cấu trúc con: với categorical là {giá trị: child}; với continuous là {'<=': left, '>': right}
        self.children: Dict[Any, "TreeNode"] = children or {}

        # Giá trị lớp nếu node là lá
        self.value: Optional[Any] = value

        # Số lượng mẫu đi qua node này (để báo cáo)
        self.samples: int = samples

        # Phân bố lớp tại node: {nhãn: số lượng}
        self.class_distribution: Dict[Any, int] = class_distribution or {}

    @property
    def is_leaf(self) -> bool:
        """Trả về True nếu node là lá (có value)."""
        return self.value is not None

    def add_child(self, key: Any, node: "TreeNode") -> None:
        """
        Thêm nhánh con.
        - key: giá trị thuộc tính (categorical) hoặc ký hiệu ('<=' hoặc '>') cho continuous.
        - node: nút con tương ứng.
        """
        self.children[key] = node
