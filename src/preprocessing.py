import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# Module tiền xử lý cho dataset xe hơi – chú thích tiếng Việt đầy đủ

def load_dataset(filepath: str) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
    """
    Load dataset từ CSV, tự động xử lý cho cấu trúc dataset của tôi
    
    Returns:
        X: Features (DataFrame)
        y: Labels (Series)
        feature_names: List tên features
        class_names: List tên classes
    """
    df = pd.read_csv(filepath)
    # Mặc định mục tiêu: 'Condition' (phân loại)
    target = "Condition"
    # Bỏ cột gây bias/không hữu ích: ID và Model (nhiều giá trị duy nhất)
    drop_cols = ["Car ID", "Model"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    df = df.dropna()
    X = df.drop(columns=[target])
    y = df[target]
    feature_names = list(X.columns)
    class_names = sorted(list(set(y)))
    return X, y, feature_names, class_names


def preprocess_data(
    df: pd.DataFrame, target_column: str
) -> Tuple[pd.DataFrame, pd.Series, List[str], Dict[int, str], Optional[LabelEncoder]]:
    """
    Preprocessing cho dataset của tôi:
    1. Separate features và target
    2. Handle missing values (nếu có)
    3. Encode categorical target nếu cần
    4. Detect feature types
    
    Returns:
        X, y, feature_names, feature_types, label_encoder
    """
    df2 = df.copy()
    df2 = df2.dropna()
    X = df2.drop(columns=[target_column])
    y = df2[target_column]

    # Nếu target là chuỗi → encode để tiện chấm điểm, nhưng vẫn giữ y dạng Series chuỗi cho C4.5
    le: Optional[LabelEncoder] = None
    if not pd.api.types.is_numeric_dtype(y):
        le = LabelEncoder()
        # Không thay thế y ở đây để thuật toán hoạt động với nhãn gốc (chuỗi)
        # Nếu cần dùng số, có thể dùng le.transform(y)

    feature_names = list(X.columns)
    feature_types = detect_feature_types(X)
    return X, y, feature_names, feature_types, le


def detect_feature_types(X: pd.DataFrame, threshold: int = 10) -> Dict[int, str]:
    """
    Tự động phát hiện loại feature
    
    Returns:
        dict: {feature_idx: 'categorical' hoặc 'continuous'}
    """
    types: Dict[int, str] = {}
    for i, c in enumerate(X.columns):
        col = X[c]
        if pd.api.types.is_float_dtype(col):
            types[i] = "continuous"
        else:
            # Heuristic: nhiều giá trị unique thì coi là continuous (vd: Year, Mileage)
            unique_count = col.nunique(dropna=True)
            types[i] = "continuous" if unique_count > threshold else "categorical"
    return types


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Chia data thành train/val/test
    
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(test_size + val_size), random_state=random_state, stratify=y
    )
    # Tỉ lệ val trong phần temp
    val_ratio = val_size / (test_size + val_size) if (test_size + val_size) > 0 else 0.0
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=(1 - val_ratio), random_state=random_state, stratify=y_temp
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def create_bias_demo_dataset() -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Tạo dataset nhỏ để chứng minh vấn đề bias của ID3
    
    Dataset có:
    - Feature hữu ích: 'weather' (3 values: sunny, rainy, cloudy)
    - Feature vô dụng: 'id' (8 unique values)
    
    ID3 sẽ chọn 'id' vì IG cao
    C4.5 sẽ chọn 'weather' vì GR cao
    
    Returns:
        X, y, feature_names
    """
    data = {
        "weather": [
            "sunny", "sunny", "rainy", "cloudy", "sunny", "rainy", "cloudy", "sunny",
        ],
        "id": [1, 2, 3, 4, 5, 6, 7, 8],
    }
    # Nhãn phụ thuộc mạnh vào weather, không phụ thuộc id
    y = pd.Series(["go", "go", "stay", "stay", "go", "stay", "stay", "go"])
    X = pd.DataFrame(data)
    feature_names = ["weather", "id"]
    return X, y, feature_names
