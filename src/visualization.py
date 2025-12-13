import matplotlib.pyplot as plt
import seaborn as sns
from graphviz import Digraph
import numpy as np
from typing import Any, Dict, Optional

from .node import TreeNode

# Set Vietnamese font và style đẹp
plt.rcParams['font.family'] = 'DejaVu Sans'
sns.set_style('whitegrid')
sns.set_palette('husl')


def plot_tree(
    tree: TreeNode,
    feature_names: Optional[list] = None,
    class_names: Optional[list] = None,
    max_depth: Optional[int] = None,
    save_path: Optional[str] = None,
):
    """
    Vẽ decision tree bằng graphviz
    
    Hiển thị:
    - Feature names và thresholds tại internal nodes
    - Class distribution tại leaf nodes  
    - Gain Ratio tại mỗi split (nếu có trong feature_importance ở cây)
    """
    dot = Digraph(comment='C4.5 Decision Tree', format='png')

    def node_label(n: TreeNode) -> str:
        if n.is_leaf:
            dist = ", ".join(f"{k}: {v}" for k, v in n.class_distribution.items())
            return f"Leaf\nclass={n.value}\nN={n.samples}\n[{dist}]"
        else:
            if n.threshold is not None:
                return f"{n.feature_name} <= {round(n.threshold, 4)}\nN={n.samples}"
            return f"{n.feature_name}\nN={n.samples}"

    counter = 0
    def add_nodes(n: Optional[TreeNode], parent_id: Optional[str] = None, edge_label: Optional[str] = None, depth: int = 0):
        nonlocal counter
        if n is None:
            return
        if max_depth is not None and depth > max_depth:
            return
        node_id = f"n{counter}"
        counter += 1
        dot.node(node_id, node_label(n))
        if parent_id is not None:
            dot.edge(parent_id, node_id, label=str(edge_label) if edge_label is not None else "")
        for k, child in n.children.items():
            add_nodes(child, node_id, k, depth + 1)

    add_nodes(tree)

    if save_path is None:
        save_path = 'decision_tree_c45/experiments/results/tree'
    dot.render(save_path, cleanup=True)


def plot_feature_importance(tree, feature_names=None, top_n: int = 10, save_path: Optional[str] = None):
    """
    Bar chart feature importance
    
    Vietnamese labels, đẹp mắt
    """
    # Giả định cây có thuộc tính feature_importance_
    importances: Dict[str, float] = getattr(tree, 'feature_importance_', {})
    if not importances:
        return
    items = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:top_n]
    labels = [k for k, _ in items]
    values = [v for _, v in items]

    plt.figure(figsize=(8, 5))
    sns.barplot(x=values, y=labels)
    plt.xlabel('Độ quan trọng (chuẩn hóa)')
    plt.ylabel('Tên thuộc tính')
    plt.title('Mức độ quan trọng của thuộc tính (C4.5)')
    plt.tight_layout()
    if save_path is None:
        save_path = 'decision_tree_c45/experiments/results/feature_importance.png'
    plt.savefig(save_path)
    plt.close()


def plot_confusion_matrix(y_true, y_pred, class_names=None, save_path: Optional[str] = None):
    """
    Heatmap confusion matrix với counts và percentages
    
    Vietnamese labels
    """
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_percent = cm / cm.sum() * 100
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Ma trận nhầm lẫn (số lượng)')
    plt.xlabel('Dự đoán')
    plt.ylabel('Thực tế')
    plt.tight_layout()
    if save_path is None:
        save_path = 'decision_tree_c45/experiments/results/confusion_counts.png'
    plt.savefig(save_path)
    plt.close()

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Greens')
    plt.title('Ma trận nhầm lẫn (phần trăm)')
    plt.xlabel('Dự đoán')
    plt.ylabel('Thực tế')
    plt.tight_layout()
    save_path2 = (save_path.replace('.png', '_percent.png') if save_path else 'decision_tree_c45/experiments/results/confusion_percent.png')
    plt.savefig(save_path2)
    plt.close()


def plot_comparison_metrics(metrics_dict: Dict[str, Dict[str, float]], save_path: Optional[str] = None):
    """
    So sánh metrics của ID3 vs C4.5 vs sklearn
    
    Bar charts cho accuracy, precision, recall, f1
    """
    # metrics_dict: {'C4.5': {'accuracy': 0.9, 'precision': ...}, 'ID3': {...}}
    models = list(metrics_dict.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    data = {m: [metrics_dict[model].get(m, np.nan) for model in models] for m in metrics}

    plt.figure(figsize=(10, 6))
    for i, m in enumerate(metrics):
        plt.subplot(2, 2, i + 1)
        sns.barplot(x=models, y=data[m])
        plt.title(f'So sánh {m}')
        plt.ylabel(m)
        plt.xticks(rotation=15)
    plt.tight_layout()
    if save_path is None:
        save_path = 'decision_tree_c45/experiments/results/comparison_metrics.png'
    plt.savefig(save_path)
    plt.close()


def plot_learning_curves(train_sizes, train_scores, val_scores, save_path: Optional[str] = None):
    """
    Learning curves để detect overfitting
    
    Vietnamese labels
    """
    plt.figure(figsize=(8, 5))
    plt.plot(train_sizes, train_scores, marker='o', label='Huấn luyện')
    plt.plot(train_sizes, val_scores, marker='s', label='Xác thực')
    plt.xlabel('Số lượng mẫu huấn luyện')
    plt.ylabel('Điểm số')
    plt.title('Learning Curves')
    plt.legend()
    plt.tight_layout()
    if save_path is None:
        save_path = 'decision_tree_c45/experiments/results/learning_curves.png'
    plt.savefig(save_path)
    plt.close()


def plot_tree_depth_analysis(depths, metrics, save_path: Optional[str] = None):
    """
    Ảnh hưởng của max_depth đến performance
    
    Line plot: depth vs accuracy/f1
    """
    plt.figure(figsize=(8, 5))
    for k, v in metrics.items():
        plt.plot(depths, v, marker='o', label=k)
    plt.xlabel('Độ sâu tối đa của cây')
    plt.ylabel('Điểm số')
    plt.title('Ảnh hưởng của độ sâu đến hiệu năng')
    plt.legend()
    plt.tight_layout()
    if save_path is None:
        save_path = 'decision_tree_c45/experiments/results/depth_analysis.png'
    plt.savefig(save_path)
    plt.close()


def visualize_bias_problem(id3_results: Dict[str, float], c45_results: Dict[str, float], save_path: Optional[str] = None):
    """
    Visualize vấn đề bias:
    - Side-by-side comparison
    - Show IG vs GR cho mỗi feature
    - Highlight sự khác biệt
    """
    # id3_results: {'weather': IG, 'id': IG}
    # c45_results: {'weather': GR, 'id': GR}
    features = sorted(set(list(id3_results.keys()) + list(c45_results.keys())))
    ig_vals = [id3_results.get(f, np.nan) for f in features]
    gr_vals = [c45_results.get(f, np.nan) for f in features]

    x = np.arange(len(features))
    width = 0.35
    plt.figure(figsize=(8, 5))
    plt.bar(x - width / 2, ig_vals, width, label='ID3 - IG')
    plt.bar(x + width / 2, gr_vals, width, label='C4.5 - GR')
    plt.xticks(x, features)
    plt.ylabel('Giá trị')
    plt.title('Minh họa bias: IG vs GR')
    plt.legend()
    plt.tight_layout()
    if save_path is None:
        save_path = 'decision_tree_c45/experiments/results/bias_demo.png'
    plt.savefig(save_path)
    plt.close()


def print_tree(node: TreeNode, indent: str = "") -> None:
    """In cây quyết định theo dạng text với nhãn tiếng Việt."""
    if node is None:
        print(indent + "[Rỗng]")
        return
    if node.is_leaf:
        print(indent + f"[Lá] lớp = {node.value}, N = {node.samples}")
        return
    if node.threshold is not None:
        print(indent + f"{node.feature_name} <= {node.threshold}")
        print_tree(node.children.get('<='), indent + "  ")
        print(indent + f"{node.feature_name} > {node.threshold}")
        print_tree(node.children.get('>'), indent + "  ")
    else:
        for v, child in node.children.items():
            print(indent + f"{node.feature_name} == {v}")
            print_tree(child, indent + "  ")