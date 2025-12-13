import sys
sys.path.append('..')

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import time

from decision_tree_c45.src.decision_tree import DecisionTreeC45, DecisionTreeID3
from decision_tree_c45.src.preprocessing import (
    load_dataset,
    split_data,
    create_bias_demo_dataset,
)
from decision_tree_c45.src.visualization import (
    visualize_bias_problem,
    plot_comparison_metrics,
    plot_confusion_matrix,
    plot_feature_importance,
)


def experiment_1_bias_demonstration():
    """
    EXPERIMENT 1: Chứng minh vấn đề BIAS của Information Gain
    
    Mục tiêu:
        Chứng minh ID3 (dùng IG) bị lừa bởi features có nhiều values
        Chứng minh C4.5 (dùng GR) không bị lừa
    """
    print("="*60)
    print("EXPERIMENT 1: CHỨNG MINH VẤN ĐỀ BIAS CỦA INFORMATION GAIN")
    print("="*60)

    # Tạo dataset minh họa bias
    X, y, feature_names = create_bias_demo_dataset()

    # Tính IG và GR cho từng feature
    from decision_tree_c45.src.utils import calculate_information_gain, calculate_gain_ratio
    Xm = X.values
    ig_per_feature = {}
    gr_per_feature = {}
    for idx, fname in enumerate(feature_names):
        ig = calculate_information_gain(Xm, list(y), idx)
        gr = calculate_gain_ratio(Xm, list(y), idx)
        ig_per_feature[fname] = ig
        gr_per_feature[fname] = gr

    print("\n➤ Information Gain theo feature:")
    for k, v in ig_per_feature.items():
        print(f"  - {k}: IG = {v:.4f}")
    print("\n➤ Gain Ratio theo feature:")
    for k, v in gr_per_feature.items():
        print(f"  - {k}: GR = {v:.4f}")

    # Train ID3 và C4.5 để xem feature đầu tiên được chọn
    id3 = DecisionTreeID3(max_depth=3)
    id3.fit(X, y)
    c45 = DecisionTreeC45(max_depth=3)
    c45.fit(X, y)

    id3_first = id3.root.feature_name
    c45_first = c45.root.feature_name
    print(f"\n🌪️ ID3 chọn feature đầu tiên: {id3_first}")
    print(f"✅ C4.5 chọn feature đầu tiên: {c45_first}")

    # Vẽ IG vs GR
    visualize_bias_problem(ig_per_feature, gr_per_feature, save_path='decision_tree_c45/experiments/results/bias_demo.png')

    # Lưu bảng kết quả
    df_res = pd.DataFrame({
        'feature': feature_names,
        'IG': [ig_per_feature[f] for f in feature_names],
        'GR': [gr_per_feature[f] for f in feature_names],
    })
    df_res.to_csv('decision_tree_c45/experiments/results/bias_demo_table.csv', index=False)

    return {
        'ig': ig_per_feature,
        'gr': gr_per_feature,
        'id3_first': id3_first,
        'c45_first': c45_first,
    }


def experiment_2_performance_comparison(dataset_path: str):
    """
    EXPERIMENT 2: So sánh Performance toàn diện
    """
    print("\n" + "="*60)
    print("EXPERIMENT 2: SO SÁNH PERFORMANCE")
    print("="*60)

    X, y, feature_names, class_names = load_dataset(dataset_path)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    # C4.5
    t0 = time.time()
    c45 = DecisionTreeC45(max_depth=10)
    c45.fit(X_train, y_train)
    t_c45 = time.time() - t0
    y_pred_c45 = c45.predict(X_test)
    acc_c45 = accuracy_score(y_test, y_pred_c45)
    prec_c45, rec_c45, f1_c45, _ = precision_recall_fscore_support(y_test, y_pred_c45, average='weighted', zero_division=0)

    # ID3
    t0 = time.time()
    id3 = DecisionTreeID3(max_depth=10)
    id3.fit(X_train, y_train)
    t_id3 = time.time() - t0
    y_pred_id3 = id3.predict(X_test)
    acc_id3 = accuracy_score(y_test, y_pred_id3)
    prec_id3, rec_id3, f1_id3, _ = precision_recall_fscore_support(y_test, y_pred_id3, average='weighted', zero_division=0)

    # sklearn baseline (entropy)
    X_enc = X.copy()
    from sklearn.preprocessing import LabelEncoder
    encoders = {}
    for c in X_enc.columns:
        if not pd.api.types.is_numeric_dtype(X_enc[c]):
            enc = LabelEncoder()
            X_enc[c] = enc.fit_transform(X_enc[c])
            encoders[c] = enc
    Xtr, Xva, Xte = X_enc.loc[X_train.index], X_enc.loc[X_val.index], X_enc.loc[X_test.index]
    t0 = time.time()
    cart = DecisionTreeClassifier(criterion='entropy', random_state=42)
    cart.fit(Xtr, y_train)
    t_cart = time.time() - t0
    y_pred_cart = cart.predict(Xte)
    acc_cart = accuracy_score(y_test, y_pred_cart)
    prec_cart, rec_cart, f1_cart, _ = precision_recall_fscore_support(y_test, y_pred_cart, average='weighted', zero_division=0)

    # Kết quả DataFrame
    results_df = pd.DataFrame([
        {"model": "C4.5", "accuracy": acc_c45, "precision": prec_c45, "recall": rec_c45, "f1": f1_c45, "train_time": t_c45, "depth": c45.get_tree_info()["depth"]},
        {"model": "ID3", "accuracy": acc_id3, "precision": prec_id3, "recall": rec_id3, "f1": f1_id3, "train_time": t_id3, "depth": id3.get_tree_info()["depth"]},
        {"model": "sklearn", "accuracy": acc_cart, "precision": prec_cart, "recall": rec_cart, "f1": f1_cart, "train_time": t_cart, "depth": cart.tree_.max_depth},
    ])
    results_df.to_csv('decision_tree_c45/experiments/results/performance_results.csv', index=False)

    # Biểu đồ so sánh
    metrics_dict = {
        'C4.5': {"accuracy": acc_c45, "precision": prec_c45, "recall": rec_c45, "f1": f1_c45},
        'ID3': {"accuracy": acc_id3, "precision": prec_id3, "recall": rec_id3, "f1": f1_id3},
        'sklearn': {"accuracy": acc_cart, "precision": prec_cart, "recall": rec_cart, "f1": f1_cart},
    }
    plot_comparison_metrics(metrics_dict, save_path='decision_tree_c45/experiments/results/comparison_metrics.png')

    # Ma trận nhầm lẫn
    plot_confusion_matrix(y_test, y_pred_c45, save_path='decision_tree_c45/experiments/results/c45_confusion.png')
    plot_confusion_matrix(y_test, y_pred_id3, save_path='decision_tree_c45/experiments/results/id3_confusion.png')
    plot_confusion_matrix(y_test, y_pred_cart, save_path='decision_tree_c45/experiments/results/sklearn_confusion.png')

    # Tầm quan trọng thuộc tính
    plot_feature_importance(c45, save_path='decision_tree_c45/experiments/results/c45_feature_importance.png')

    return {
        'results': results_df,
        'metrics': metrics_dict,
    }


def experiment_3_hyperparameter_analysis(dataset_path: str):
    print("\n" + "="*60)
    print("EXPERIMENT 3: HYPERPARAMETER TUNING")
    print("="*60)
    X, y, _, _ = load_dataset(dataset_path)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    depths = [3, 5, 10, 15]
    accs = []
    for d in depths:
        model = DecisionTreeC45(max_depth=d)
        model.fit(X_train, y_train)
        accs.append(accuracy_score(y_val, model.predict(X_val)))
    from decision_tree_c45.src.visualization import plot_tree_depth_analysis
    plot_tree_depth_analysis(depths, {"accuracy": accs}, save_path='decision_tree_c45/experiments/results/depth_analysis.png')
    return {"depths": depths, "accuracy": accs}


def experiment_4_feature_importance(dataset_path: str):
    print("\n" + "="*60)
    print("EXPERIMENT 4: FEATURE IMPORTANCE")
    print("="*60)
    X, y, feature_names, _ = load_dataset(dataset_path)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    c45 = DecisionTreeC45(max_depth=10).fit(X_train, y_train)
    plot_feature_importance(c45, save_path='decision_tree_c45/experiments/results/feature_importance.png')
    return c45.get_feature_importance()


def experiment_5_robustness_test(dataset_path: str):
    print("\n" + "="*60)
    print("EXPERIMENT 5: ROBUSTNESS TESTING")
    print("="*60)
    X, y, _, _ = load_dataset(dataset_path)
    accs_c45 = []
    accs_id3 = []
    for seed in [0, 1, 2, 3, 4]:
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, random_state=seed)
        c45 = DecisionTreeC45(max_depth=10).fit(X_train, y_train)
        id3 = DecisionTreeID3(max_depth=10).fit(X_train, y_train)
        accs_c45.append(accuracy_score(y_test, c45.predict(X_test)))
        accs_id3.append(accuracy_score(y_test, id3.predict(X_test)))
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    df = pd.DataFrame({"C4.5": accs_c45, "ID3": accs_id3})
    plt.figure(figsize=(6,4))
    sns.boxplot(data=df)
    plt.title('Độ ổn định qua các lần split')
    plt.ylabel('Accuracy')
    plt.tight_layout()
    plt.savefig('decision_tree_c45/experiments/results/robustness_boxplot.png')
    plt.close()
    return {"accs_c45": accs_c45, "accs_id3": accs_id3}


def generate_report(results: dict):
    """Tạo báo cáo tổng hợp từ tất cả experiments"""
    import os
    os.makedirs('decision_tree_c45/experiments/results', exist_ok=True)
    with open('decision_tree_c45/experiments/results/report.txt', 'w', encoding='utf-8') as f:
        f.write("BÁO CÁO EXPERIMENTS\n")
        f.write("\n[Experiment 1 - Bias]\n")
        bias = results.get('bias', {})
        if bias:
            f.write(f"ID3 chọn: {bias.get('id3_first')} | C4.5 chọn: {bias.get('c45_first')}\n")
        f.write("\n[Experiment 2 - Performance]\n")
        perf = results.get('performance', {})
        if perf and 'results' in perf:
            f.write(perf['results'].to_string(index=False))
            f.write("\n")
    perf = results.get('performance', {})
    if perf and 'results' in perf:
        perf['results'].to_csv('decision_tree_c45/experiments/results/summary.csv', index=False)


def main():
    """
    Chạy tất cả experiments
    """
    import os
    os.makedirs('decision_tree_c45/experiments/results', exist_ok=True)
    dataset_path = sys.argv[1] if len(sys.argv) > 1 else 'data/car_price_prediction_.csv'
    print(f"\n🚀 BẮT ĐẦU EXPERIMENTS với dataset: {dataset_path}\n")
    results = {}
    results['bias'] = experiment_1_bias_demonstration()
    results['performance'] = experiment_2_performance_comparison(dataset_path)
    results['hyperparams'] = experiment_3_hyperparameter_analysis(dataset_path)
    results['importance'] = experiment_4_feature_importance(dataset_path)
    results['robustness'] = experiment_5_robustness_test(dataset_path)
    generate_report(results)
    print("\n✅ TẤT CẢ EXPERIMENTS HOÀN TẤT!")
    print("📊 Kết quả đã được lưu tại: experiments/results/")


if __name__ == "__main__":
    main()
