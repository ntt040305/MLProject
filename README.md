# Decision Tree C4.5 - Triển khai từ đầu

Triển khai **C4.5 Decision Tree** đầy đủ từ đầu, kèm so sánh chi tiết với **ID3** để chứng minh ưu điểm của **Gain Ratio**.

## Tạo lập nhanh

```bash
# 1. Cài đặt dependencies
pip install -r requirements.txt

# 2. Training & Evaluation
python main.py --train
python main.py --evaluate

# 3. Demo calculations
python main.py --demo

# 4. Compare ID3 vs C4.5
python main.py --compare

# 5. Interactive prediction
python main.py --predict

# 6. Run experiments
python experiments/compare_id3_c45.py

# 7. Open notebook
jupyter notebook notebooks/demo_interactive.ipynb
```

## Cấu trúc Dự án

```
decision_tree_c45/
├── src/
│   ├── node.py              # TreeNode class
│   ├── utils.py             # Entropy, IG, GR calculations
│   ├── decision_tree.py     # DecisionTreeC45 & DecisionTreeID3
│   ├── preprocessing.py     # Data loading & preprocessing
│   └── visualization.py     # Plotting & tree visualization
├── experiments/
│   ├── compare_id3_c45.py   # Comprehensive experiments
│   └── results/             # Plots & results CSV
├── notebooks/
│   └── demo_interactive.ipynb  # Interactive demo with calculations
├── tests/
│   └── test_decision_tree.py
├── main.py                  # CLI entry point
├── requirements.txt
├── README.md
└── .gitignore
```

## Điểm Highlight

### 1. **ID3 vs C4.5: Vấn đề BIAS**
- ID3 dùng **Information Gain** → bị lừa bởi features có nhiều unique values
- C4.5 dùng **Gain Ratio** → khắc phục bias bằng cách chia IG cho SplitInfo

**Demo dataset:**
```
weather: sunny, rainy, cloudy (3 values) → HỮUU ÍCH
id: 1, 2, 3, 4, 5, 6, 7, 8 (8 values) → VÔ DỤNG

ID3: IG(id) > IG(weather) → chọn id (SAI)
C4.5: GR(weather) > GR(id) → chọn weather (ĐÚNG)
```

### 2. **Xử lý Continuous Features**
- C4.5 tự động tìm threshold tối ưu cho features liên tục
- Thử midpoints giữa các giá trị unique
- Đánh giá bằng Gain Ratio, không cần rời rạc hóa sẵn

### 3. **Ngôn ngữ & Comments**
- Tất cả mã nguồn + công thức đều có chú thích **tiếng Việt chi tiết**
- Mỗi hàm đều giải thích công thức toán học
- Ví dụ cụ thể trong docstrings

## Công thức

### Entropy
$$H(S) = -\sum_{i} p_i \log_2(p_i)$$

### Information Gain
$$IG(S, A) = H(S) - \sum_v \frac{|S_v|}{|S|} H(S_v)$$

### Split Info (C4.5)
$$SplitInfo(S, A) = -\sum_v \frac{|S_v|}{|S|} \log_2\left(\frac{|S_v|}{|S|}\right)$$

### Gain Ratio (C4.5)
$$GR(S, A) = \frac{IG(S, A)}{SplitInfo(S, A)}$$

## Quick Start Examples

### Training
```python
from src.decision_tree import DecisionTreeC45
from src.preprocessing import load_dataset, split_data

X, y, feature_names, _ = load_dataset('data/car_price_prediction_.csv')
X_train, _, X_test, y_train, _, y_test = split_data(X, y)

tree = DecisionTreeC45(max_depth=10)
tree.fit(X_train, y_train)
accuracy = tree.score(X_test, y_test)
print(f"Accuracy: {accuracy:.3f}")
```

### Visualization
```python
from src.visualization import print_tree, plot_feature_importance

print_tree(tree.root)
plot_feature_importance(tree, save_path='results/importance.png')
```

### Bias Demo
```python
from src.decision_tree import DecisionTreeID3, DecisionTreeC45
from src.preprocessing import create_bias_demo_dataset

X, y, features = create_bias_demo_dataset()
id3 = DecisionTreeID3().fit(X, y)
c45 = DecisionTreeC45().fit(X, y)

print(f"ID3 chọn: {id3.root.feature_name}")   # 'id' (SAI)
print(f"C4.5 chọn: {c45.root.feature_name}") # 'weather' (ĐÚNG)
```

## File Quan trọng

| File | Mô tả |
|------|-------|
| `main.py` | CLI entry point với 5 commands chính |
| `experiments/compare_id3_c45.py` | 5 experiments chứng minh C4.5 vượt trội |
| `notebooks/demo_interactive.ipynb` | 8 cells: manual calculations → real training |
| `src/utils.py` | Entropy, IG, GR calculations với Vietnamese comments |
| `src/decision_tree.py` | DecisionTreeC45 & DecisionTreeID3 classes |

## Kết quả & Outputs

Tất cả results được lưu vào `experiments/results/`:

- `bias_demo.png` - Minh họa IG vs GR
- `performance_results.csv` - Accuracy/precision/recall so sánh
- `comparison_metrics.png` - Bar chart so sánh models
- `feature_importance.png` - Top features theo Gain Ratio
- `robustness_boxplot.png` - Độ ổn định qua splits
- `report.txt` - Tóm tắt tất cả experiments

## Chạy Experiments

```bash
python experiments/compare_id3_c45.py
```

Experiments:
1. **Bias Demonstration** - ID3 vs C4.5 trên small dataset
2. **Performance Comparison** - Accuracy, precision, recall, F1
3. **Hyperparameter Tuning** - Effect of max_depth
4. **Feature Importance** - Top important attributes
5. **Robustness Testing** - Stability across random splits

## Kiểm thử

```bash
pytest -q
```

## Yêu cầu

- Python 3.7+
- pandas, numpy, scikit-learn, matplotlib, seaborn, graphviz, jupyter, pytest

## Lưu ý

- Dataset mặc định: `data/car_price_prediction_.csv`
- Target: `Condition` (phân loại)
- Dropped cols: `Car ID`, `Model` (high-cardinality)
- Models được lưu tại `models/c45_model.pkl`
