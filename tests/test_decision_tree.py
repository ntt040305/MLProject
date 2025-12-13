import pandas as pd
from decision_tree_c45.src.preprocessing import preprocess, detect_types
from decision_tree_c45.src.decision_tree import DecisionTreeC45

# Kiểm thử đơn giản cho cây C4.5

def test_fit_predict_runs():
    df = pd.read_csv("data/car_price_prediction_.csv")
    target = "Condition"
    df_clean, features, types = preprocess(df, target, drop_cols=["Car ID", "Model"])  # bỏ cột ID/model
    X = df_clean[features]
    y = df_clean[target]

    tree = DecisionTreeC45(feature_types=types, max_depth=5)
    tree.fit(X, y)
    preds = tree.predict(X.head(10))
    assert len(preds) == 10
