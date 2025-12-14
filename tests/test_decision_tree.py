from src.preprocessing import load_dataset, detect_feature_types
from src.decision_tree import DecisionTreeC45


def test_fit_predict_runs():
    # Đọc đúng dataset car_price từ thư mục data
    X, y, feature_names, _ = load_dataset("data/car_price_prediction_.csv")
    feature_types = detect_feature_types(X)

    tree = DecisionTreeC45(max_depth=5)
    tree.fit(X, y, feature_names=feature_names)

    preds = tree.predict(X.head(10))
    assert len(preds) == 10
