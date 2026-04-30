# -*- coding: utf-8 -*-
"""
第四步：SHAP 可解释性分析
==================================================
输入：
- output_preprocessed/clean_train.csv
- output_preprocessed/clean_test.csv
- modeling_outputs/xgb_model.pkl
- modeling_outputs/lgb_model.pkl
- modeling_outputs/best_model_info.json

输出：
- modeling_outputs/shap_top20_importance.csv
- modeling_outputs/shap_bar_top20.png
- modeling_outputs/shap_summary_beeswarm.png

说明：
1. 默认对“最佳模型”做 SHAP
2. 为控制开销，只抽样部分训练集（默认 5000 条）
3. 用于论文“可解释性分析”部分
"""

# -*- coding: utf-8 -*-
"""
第四步：SHAP 可解释性分析（修正版）
==================================================
修复重点：
1. 不再使用 shap.Explainer(model, X_sample)
2. 显式使用 shap.TreeExplainer
3. 自动优先读取最佳模型；若最佳模型失败，则自动回退到另一个模型
4. 严格按 encoded_feature_columns.json 对齐特征列
5. 强制 X_sample 为纯 float32 且无缺失/无非法值
6. 对不同 SHAP 版本做兼容处理

输入：
- output_preprocessed/clean_train.csv
- output_preprocessed/clean_test.csv
- modeling_outputs/xgb_model.pkl
- modeling_outputs/lgb_model.pkl
- modeling_outputs/best_model_info.json
- modeling_outputs/encoded_feature_columns.json

输出：
- modeling_outputs/shap_top20_importance.csv
- modeling_outputs/shap_bar_top20.png
- modeling_outputs/shap_summary_beeswarm.png
- modeling_outputs/shap_used_model.json
"""

from pathlib import Path
import json
import re
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

TRAIN_PATH = Path("output_preprocessed/clean_train.csv")
TEST_PATH = Path("output_preprocessed/clean_test.csv")
OUTPUT_DIR = Path("modeling_outputs")
LABEL_COL = "isDefault"


# =========================
# 1. 工具函数
# =========================
def select_enhanced_features(df: pd.DataFrame):
    """
    与 03_xgb_lgbm_models_fixed.py 保持一致
    """
    banned_cols = {"id", "policyCode", "n3", LABEL_COL}
    banned_prefixes = ["employmentTitle", "title", "postCode"]

    selected = []
    for c in df.columns:
        if c in banned_cols:
            continue
        if any(c == bp or c.startswith(bp + "_") for bp in banned_prefixes):
            continue
        selected.append(c)

    return selected


def sanitize_feature_names(columns):
    """
    与树模型脚本保持一致，避免列名不一致。
    """
    new_cols = []
    for c in columns:
        c = str(c)

        c = c.replace("[", "_")
        c = c.replace("]", "_")
        c = c.replace("<", "lt_")
        c = c.replace(">", "gt_")

        c = c.replace(" ", "_")
        c = c.replace(",", "_")
        c = c.replace("(", "_")
        c = c.replace(")", "_")
        c = c.replace("/", "_")
        c = c.replace("-", "_")
        c = c.replace(".", "_")
        c = c.replace(":", "_")
        c = c.replace(";", "_")
        c = c.replace("=", "_")

        c = re.sub(r"_+", "_", c)
        c = c.strip("_")
        if c == "":
            c = "feature"

        new_cols.append(c)

    # 防止重复列名
    seen = {}
    final_cols = []
    for c in new_cols:
        if c not in seen:
            seen[c] = 0
            final_cols.append(c)
        else:
            seen[c] += 1
            final_cols.append(f"{c}_{seen[c]}")

    return final_cols


def force_numeric_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    强制所有列变成纯 float32，防止 SHAP / 树模型接口异常。
    """
    out = df.copy()

    bool_cols = out.select_dtypes(include=["bool"]).columns.tolist()
    if bool_cols:
        out[bool_cols] = out[bool_cols].astype(np.int8)

    obj_cols = out.select_dtypes(include=["object"]).columns.tolist()
    for c in obj_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.fillna(0)
    out = out.astype(np.float32)

    return out


def one_hot_align(train_df: pd.DataFrame, test_df: pd.DataFrame, label_col="isDefault"):
    """
    重新生成与树模型一致的编码矩阵。
    """
    train_x = train_df.drop(columns=[label_col], errors="ignore").copy()
    test_x = test_df.copy()

    train_rows = len(train_x)
    combined = pd.concat([train_x, test_x], axis=0, ignore_index=True)

    cat_cols = combined.select_dtypes(exclude=[np.number]).columns.tolist()

    combined_encoded = pd.get_dummies(
        combined,
        columns=cat_cols,
        drop_first=False,
        dtype=np.int8
    )

    combined_encoded.columns = sanitize_feature_names(combined_encoded.columns)
    combined_encoded = force_numeric_df(combined_encoded)

    encoded_train = combined_encoded.iloc[:train_rows, :].copy()
    encoded_test = combined_encoded.iloc[train_rows:, :].copy()
    encoded_train[label_col] = train_df[label_col].astype(np.int8).values

    return encoded_train, encoded_test


def align_to_saved_feature_columns(X: pd.DataFrame, feature_columns: list) -> pd.DataFrame:
    """
    严格按训练时保存的列顺序对齐，缺什么补 0，多余的删掉。
    这是最关键的一步，避免 SHAP 输入矩阵和模型训练矩阵不一致。
    """
    X = X.copy()

    # 补缺失列
    for c in feature_columns:
        if c not in X.columns:
            X[c] = 0

    # 删多余列
    extra_cols = [c for c in X.columns if c not in feature_columns]
    if extra_cols:
        X = X.drop(columns=extra_cols, errors="ignore")

    # 调整顺序
    X = X[feature_columns].copy()

    # 最后再强制数值化
    X = force_numeric_df(X)

    return X


def get_shap_values_tree(model, X_sample: pd.DataFrame):
    """
    显式使用 TreeExplainer，兼容不同版本 shap。
    返回：
    - shap_values_2d: (n_samples, n_features)
    """
    # 优先最稳妥的 TreeExplainer
    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(X_sample)

    # 二分类常见两种情况：
    # 1) 返回 ndarray(shape=(n_samples, n_features))
    # 2) 返回 list [class0, class1]
    if isinstance(shap_values, list):
        # 二分类时取正类
        if len(shap_values) == 2:
            shap_values_2d = np.array(shap_values[1])
        else:
            shap_values_2d = np.array(shap_values[0])
    else:
        shap_values_2d = np.array(shap_values)

    return explainer, shap_values_2d


def plot_beeswarm_compat(explainer, shap_values_2d, X_sample, out_path: Path):
    """
    兼容不同 SHAP 版本的 beeswarm 绘制。
    """
    try:
        # 新版 SHAP 可以构造 Explanation
        explanation = shap.Explanation(
            values=shap_values_2d,
            base_values=np.repeat(
                explainer.expected_value if np.isscalar(explainer.expected_value) else np.mean(explainer.expected_value),
                X_sample.shape[0]
            ),
            data=X_sample.values,
            feature_names=X_sample.columns.tolist()
        )
        plt.figure()
        shap.plots.beeswarm(explanation, max_display=20, show=False)
        plt.tight_layout()
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        return
    except Exception:
        pass

    # 老版本 fallback
    plt.figure()
    shap.summary_plot(shap_values_2d, X_sample, show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


# =========================
# 2. 主流程
# =========================
def main():
    print("========== 第1步：读取数据 ==========")
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    selected_cols = select_enhanced_features(train_df)
    train_df = train_df[selected_cols + [LABEL_COL]].copy()
    test_df = test_df[selected_cols].copy()

    print("========== 第2步：重新编码并对齐 ==========")
    encoded_train, _ = one_hot_align(train_df, test_df, label_col=LABEL_COL)

    X_full = encoded_train.drop(columns=[LABEL_COL]).copy()

    # 读取训练时保存的列顺序
    with open(OUTPUT_DIR / "encoded_feature_columns.json", "r", encoding="utf-8") as f:
        feature_columns = json.load(f)

    X_full = align_to_saved_feature_columns(X_full, feature_columns)

    # 抽样，控制开销
    sample_size = min(5000, len(X_full))
    X_sample = X_full.sample(sample_size, random_state=42).reset_index(drop=True)

    print(f"X_sample shape: {X_sample.shape}")
    print("X_sample dtype 统计：")
    print(X_sample.dtypes.value_counts())

    # 读取最佳模型信息
    with open(OUTPUT_DIR / "best_model_info.json", "r", encoding="utf-8") as f:
        best_info = json.load(f)

    best_name = best_info["best_model_by_auc"]

    # 优先使用最佳模型，失败则回退到另一个模型
    model_candidates = []
    if best_name == "XGBoost":
        model_candidates = [
            ("XGBoost", OUTPUT_DIR / "xgb_model.pkl"),
            ("LightGBM", OUTPUT_DIR / "lgb_model.pkl")
        ]
    else:
        model_candidates = [
            ("LightGBM", OUTPUT_DIR / "lgb_model.pkl"),
            ("XGBoost", OUTPUT_DIR / "xgb_model.pkl")
        ]

    used_model_name = None
    used_model_path = None
    explainer = None
    shap_values_2d = None

    print("========== 第3步：尝试 SHAP 解释 ==========")
    last_error = None
    for model_name, model_path in model_candidates:
        try:
            print(f"尝试模型：{model_name}")
            model = joblib.load(model_path)

            explainer, shap_values_2d = get_shap_values_tree(model, X_sample)

            # 基本合法性检查
            if shap_values_2d is None:
                raise ValueError("shap_values is None")
            if len(shap_values_2d) == 0:
                raise ValueError("shap_values is empty")
            if shap_values_2d.shape[1] != X_sample.shape[1]:
                raise ValueError(
                    f"shap_values 维度不匹配: {shap_values_2d.shape} vs X_sample {X_sample.shape}"
                )

            used_model_name = model_name
            used_model_path = str(model_path)
            print(f"SHAP 成功，使用模型：{used_model_name}")
            break

        except Exception as e:
            last_error = e
            print(f"{model_name} 失败：{repr(e)}")

    if used_model_name is None:
        raise RuntimeError(f"SHAP 对两个模型都失败，最后错误：{repr(last_error)}")

    print("========== 第4步：计算 Top20 重要性 ==========")
    mean_abs = np.abs(shap_values_2d).mean(axis=0)
    imp = pd.DataFrame({
        "feature": X_sample.columns,
        "mean_abs_shap": mean_abs
    }).sort_values("mean_abs_shap", ascending=False)

    top20 = imp.head(20)
    top20.to_csv(OUTPUT_DIR / "shap_top20_importance.csv", index=False, encoding="utf-8-sig")

    print("========== 第5步：绘图 ==========")
    # 条形图
    plt.figure(figsize=(10, 8))
    plot_df = top20.iloc[::-1]
    plt.barh(plot_df["feature"], plot_df["mean_abs_shap"])
    plt.xlabel("Mean |SHAP value|")
    plt.ylabel("Feature")
    plt.title(f"Top 20 SHAP Importance ({used_model_name})")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "shap_bar_top20.png", dpi=300)
    plt.close()

    # beeswarm
    plot_beeswarm_compat(
        explainer=explainer,
        shap_values_2d=shap_values_2d,
        X_sample=X_sample,
        out_path=OUTPUT_DIR / "shap_summary_beeswarm.png"
    )

    with open(OUTPUT_DIR / "shap_used_model.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "used_model": used_model_name,
                "used_model_path": used_model_path,
                "sample_size": int(sample_size),
                "n_features": int(X_sample.shape[1])
            },
            f,
            ensure_ascii=False,
            indent=2
        )

    print("========== SHAP 分析完成 ==========")
    print(top20.head(10))


if __name__ == "__main__":
    main()