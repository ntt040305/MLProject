"""
Main script để train, evaluate, và demo Decision Tree C4.5

Usage:
    python main.py --dataset data/car_price_prediction_.csv --train
    python main.py --dataset data/car_price_prediction_.csv --evaluate
    python main.py --demo
    python main.py --compare
    python main.py --predict
"""

import argparse
import sys
import pickle
import os
from pathlib import Path

from src.decision_tree import DecisionTreeC45, DecisionTreeID3
from src.preprocessing import load_dataset, split_data, create_bias_demo_dataset
from src.visualization import (
    print_tree,
    plot_feature_importance,
    plot_confusion_matrix,
    plot_comparison_metrics,
)
from src.utils import calculate_entropy, calculate_information_gain, calculate_gain_ratio
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import pandas as pd
import numpy as np


def train_model(args):
    """Train C4.5 model"""
    print("[*] Starting training...")
    
    # Load data
    X, y, feature_names, class_names = load_dataset(args.dataset)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    print(f"[OK] Data loaded: {len(X)} samples, {len(feature_names)} features")
    
    # Train C4.5
    tree = DecisionTreeC45(
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
    )
    tree.fit(X_train, y_train)
    
    # Evaluate
    y_pred = tree.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted', zero_division=0)
    
    print(f"\n[RESULTS]")
    print(f"  Accuracy:  {acc:.3f}")
    print(f"  Precision: {prec:.3f}")
    print(f"  Recall:    {rec:.3f}")
    print(f"  F1-score:  {f1:.3f}")
    
    info = tree.get_tree_info()
    print(f"\n[TREE INFO]")
    print(f"  Depth:     {info['depth']}")
    print(f"  Nodes:     {info['n_nodes']}")
    print(f"  Leaves:    {info['n_leaves']}")
    
    # Save model
    model_path = 'models/c45_model.pkl'
    os.makedirs('models', exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump(tree, f)
    print(f"\n[OK] Model saved: {model_path}")
    
    # Visualize tree (text)
    print("\n[TREE STRUCTURE]")
    print_tree(tree.root)
    
    # Save plots
    os.makedirs('experiments/results', exist_ok=True)
    plot_feature_importance(tree, save_path='experiments/results/main_feature_importance.png')
    plot_confusion_matrix(y_test, y_pred, save_path='experiments/results/main_confusion.png')
    print("\n[OK] Plots saved: experiments/results/")


def evaluate_model(args):
    """Evaluate trained model"""
    print("[*] Evaluating model...")
    
    model_path = 'models/c45_model.pkl'
    if not os.path.exists(model_path):
        print(f"[ERROR] Model not found: {model_path}")
        return
    
    with open(model_path, 'rb') as f:
        tree = pickle.load(f)
    
    X, y, _, _ = load_dataset(args.dataset)
    _, _, X_test, _, _, y_test = split_data(X, y)
    
    y_pred = tree.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted', zero_division=0)
    
    print(f"Accuracy:  {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall:    {rec:.3f}")
    print(f"F1-score:  {f1:.3f}")


def demo_manual_calculation():
    """
    Demo tính toán Entropy, IG, GR bằng tay
    với dataset nhỏ
    """
    print("[*] DEMO - Manual calculations")
    print("="*60)
    
    # Dataset nhỏ: [A, B, A, B, B]
    y = np.array(['A', 'B', 'A', 'B', 'B'])
    print(f"\nDataset: {y}")
    print(f"Distribution: A=2, B=3")
    
    # Tính Entropy
    from src.utils import calculate_entropy
    h = calculate_entropy(y)
    p_a = 2 / 5
    p_b = 3 / 5
    import math
    h_manual = -(p_a * math.log2(p_a) + p_b * math.log2(p_b))
    
    print(f"\n[1] Entropy H(S) = -sum(p_i * log2(p_i))")
    print(f"  p(A) = {p_a:.1f}, p(B) = {p_b:.1f}")
    print(f"  H = -({p_a:.1f} * log2({p_a:.1f}) + {p_b:.1f} * log2({p_b:.1f}))")
    print(f"  H = {h_manual:.4f}")
    print(f"  [OK] Function: {h:.4f}")
    
    # Feature categorical: color [red, red, blue, red, blue]
    X_color = np.array(['red', 'red', 'blue', 'red', 'blue'])
    print(f"\n[2] Feature 'color': {X_color}")
    
    # Chia theo red/blue
    red_y = y[X_color == 'red']  # ['A', 'B', 'B']
    blue_y = y[X_color == 'blue']  # ['A', 'B']
    h_red = calculate_entropy(red_y)
    h_blue = calculate_entropy(blue_y)
    w_red = len(red_y) / len(y)
    w_blue = len(blue_y) / len(y)
    ig = h - (w_red * h_red + w_blue * h_blue)
    
    print(f"  Red (3 samples): {red_y} -> H_red = {h_red:.4f}")
    print(f"  Blue (2 samples): {blue_y} -> H_blue = {h_blue:.4f}")
    print(f"  IG = H - (3/5 * {h_red:.4f} + 2/5 * {h_blue:.4f}) = {ig:.4f}")
    
    # Split Info
    split_info = -(w_red * math.log2(w_red) + w_blue * math.log2(w_blue))
    gr = ig / split_info if split_info > 0 else 0
    
    print(f"\n[3] Split Info = -(3/5 * log2(3/5) + 2/5 * log2(2/5)) = {split_info:.4f}")
    print(f"[4] Gain Ratio = IG / SplitInfo = {ig:.4f} / {split_info:.4f} = {gr:.4f}")
    print(f"\n[*] C4.5 uses Gain Ratio (dividing by SplitInfo) to avoid bias!")


def compare_id3_c45(args):
    """Quick comparison ID3 vs C4.5 trên dataset bias"""
    print("[*] Comparing ID3 vs C4.5 - Bias Demonstration")
    print("="*60)
    
    X, y, feature_names = create_bias_demo_dataset()
    
    # Tính IG và GR
    Xm = X.values
    ig_per_feature = {}
    gr_per_feature = {}
    for idx, fname in enumerate(feature_names):
        ig = calculate_information_gain(Xm, list(y), idx)
        gr = calculate_gain_ratio(Xm, list(y), idx)
        ig_per_feature[fname] = ig
        gr_per_feature[fname] = gr
    
    print(f"\nDataset: weather vs id")
    print(f"Target depends on: WEATHER (useful)")
    print(f"Target does NOT depend on: ID (useless but many values)")
    
    print(f"\n[Information Gain - ID3]:")
    for k, v in ig_per_feature.items():
        print(f"  {k}: IG = {v:.4f}")
    best_ig = max(ig_per_feature, key=ig_per_feature.get)
    print(f"  => ID3 chooses: {best_ig}")
    
    print(f"\n[Gain Ratio - C4.5]:")
    for k, v in gr_per_feature.items():
        print(f"  {k}: GR = {v:.4f}")
    best_gr = max(gr_per_feature, key=gr_per_feature.get)
    print(f"  => C4.5 chooses: {best_gr}")
    
    # Train
    id3 = DecisionTreeID3(max_depth=3)
    id3.fit(X, y)
    c45 = DecisionTreeC45(max_depth=3)
    c45.fit(X, y)
    
    print(f"\n[ID3 Tree]:")
    print(f"  Root: {id3.root.feature_name}")
    print_tree(id3.root)
    
    print(f"\n[C4.5 Tree]:")
    print(f"  Root: {c45.root.feature_name}")
    print_tree(c45.root)


def interactive_predict(args):
    """Interactive prediction"""
    print("[*] Interactive Prediction")
    print("="*60)
    
    model_path = 'models/c45_model.pkl'
    if not os.path.exists(model_path):
        print(f"[ERROR] Model not found. Run --train first")
        return
    
    with open(model_path, 'rb') as f:
        tree = pickle.load(f)
    
    X, y, feature_names, _ = load_dataset(args.dataset)
    
    print(f"\nFeatures: {feature_names}")
    print("Select a random sample from dataset:")
    
    # Chọn ngẫu nhiên 1 sample
    idx = np.random.randint(0, len(X))
    sample = X.iloc[idx]
    
    print(f"\n[Sample {idx}]:")
    for fname, val in sample.items():
        print(f"  {fname} = {val}")
    
    pred = tree.predict(X.iloc[[idx]])[0]
    true = y.iloc[idx]
    
    print(f"\n[Prediction]: {pred}")
    print(f"[Actual]:     {true}")
    print(f"[Result]:     {'CORRECT' if pred == true else 'WRONG'}")


def main():
    parser = argparse.ArgumentParser(description='Decision Tree C4.5 - Main Entry Point')
    parser.add_argument('--dataset', type=str, default='data/car_price_prediction_.csv', help='Dataset path')
    parser.add_argument('--train', action='store_true', help='Train model')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate model')
    parser.add_argument('--demo', action='store_true', help='Demo calculations')
    parser.add_argument('--compare', action='store_true', help='Compare ID3 vs C4.5')
    parser.add_argument('--predict', action='store_true', help='Interactive prediction')
    parser.add_argument('--max_depth', type=int, default=None, help='Max tree depth')
    parser.add_argument('--min_samples_split', type=int, default=2, help='Min samples to split')
    
    args = parser.parse_args()
    
    if args.train:
        train_model(args)
    elif args.evaluate:
        evaluate_model(args)
    elif args.demo:
        demo_manual_calculation()
    elif args.compare:
        compare_id3_c45(args)
    elif args.predict:
        interactive_predict(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
