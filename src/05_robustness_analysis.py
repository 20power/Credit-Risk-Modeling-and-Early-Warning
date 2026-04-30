# -*- coding: utf-8 -*-
"""
第6章：XGBoost阈值敏感性曲线 + 按贷款期限分组稳健性检验
========================================================

功能：
1. 重新训练 XGBoost 模型，并在验证集上输出预测概率；
2. 计算不同阈值下 Precision、Recall、F1、Accuracy；
3. 绘制 XGBoost 阈值敏感性曲线；
4. 按贷款期限 term 分组，检验模型在不同期限样本中的稳健性。

输入文件：
- output_preprocessed/clean_train.csv
- output_preprocessed/clean_test.csv

输出文件：
- robustness_outputs/xgb_threshold_sensitivity_table.csv
- robustness_outputs/图6-1_XGBoost阈值敏感性曲线.png
- robustness_outputs/图6-1_XGBoost阈值敏感性曲线.pdf
- robustness_outputs/xgb_term_group_robustness.csv

建议放置位置：
- 表6-3 XGBoost不同阈值下模型表现
- 图6-1 XGBoost模型阈值敏感性曲线
- 表6-4 按贷款期限分组的稳健性检验结果
"""

from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    roc_curve
)

from xgboost import XGBClassifier


# =========================
# 1. 路径设置
# =========================
TRAIN_PATH = Path("output_preprocessed/clean_train.csv")
TEST_PATH = Path("output_preprocessed/clean_test.csv")

OUTPUT_DIR = Path("robustness_outputs")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

LABEL_COL = "isDefault"


# =========================
# 2. 工具函数
# =========================
def ks_score(y_true, y_prob):
    """计算 KS 值。"""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return float(np.max(tpr - fpr))


def calc_metrics(y_true, y_prob, threshold=0.55):
    """
    根据给定阈值计算模型指标。
    """
    y_pred = (y_prob >= threshold).astype(int)

    return {
        "AUC": float(roc_auc_score(y_true, y_prob)),
        "KS": ks_score(y_true, y_prob),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
        "Accuracy": float(accuracy_score(y_true, y_pred)),
    }


def select_enhanced_features(df: pd.DataFrame):
    """
    选择增强模型变量：
    - 保留基础变量
    - 加入匿名变量 n0~n14
    - 删除 id、policyCode、n3
    - 不使用高基数字段 employmentTitle、title、postCode
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
    清洗列名，避免 XGBoost 因特殊字符报错。
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

    # 防止清洗后重名
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
    强制所有特征列为纯数值，避免 object / bool 混入模型。
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
    训练集和预测集统一独热编码，保证字段完全一致。
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


def normalize_term_value(x):
    """
    规范化贷款期限取值。
    兼容：
    - 3 / 5
    - '3'
    - '36 months'
    - '60 months'
    """
    if pd.isna(x):
        return "Unknown"

    s = str(x).strip().lower()

    if s in ["3", "3.0"]:
        return "3年期"
    if s in ["5", "5.0"]:
        return "5年期"
    if "36" in s:
        return "3年期"
    if "60" in s:
        return "5年期"

    return str(x)


# =========================
# 3. 主流程
# =========================
def main():
    print("========== 第1步：读取数据 ==========")
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    print(f"clean_train shape: {train_df.shape}")
    print(f"clean_test  shape: {test_df.shape}")

    # 保存原始 term，用于后续分组检验
    if "term" not in train_df.columns:
        raise ValueError("数据中未找到 term 字段，无法进行按贷款期限分组稳健性检验。")

    term_original = train_df["term"].copy()

    print("\n========== 第2步：筛选增强模型变量 ==========")
    selected_cols = select_enhanced_features(train_df)

    train_model_df = train_df[selected_cols + [LABEL_COL]].copy()
    test_model_df = test_df[selected_cols].copy()

    print(f"selected feature count(before encoding): {len(selected_cols)}")

    print("\n========== 第3步：独热编码并对齐 ==========")
    encoded_train, encoded_test = one_hot_align(
        train_model_df,
        test_model_df,
        label_col=LABEL_COL
    )

    X = encoded_train.drop(columns=[LABEL_COL]).copy()
    y = encoded_train[LABEL_COL].astype(np.int8).copy()

    print(f"encoded feature count: {X.shape[1]}")

    print("\n========== 第4步：训练/验证集划分 ==========")
    indices = np.arange(len(X))

    X_train, X_valid, y_train, y_valid, idx_train, idx_valid = train_test_split(
        X,
        y,
        indices,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    term_valid = term_original.iloc[idx_valid].reset_index(drop=True)
    term_valid_group = term_valid.apply(normalize_term_value)

    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())
    scale_pos_weight = neg / pos if pos > 0 else 1.0

    print(f"train rows: {len(X_train)}")
    print(f"valid rows: {len(X_valid)}")
    print(f"scale_pos_weight: {scale_pos_weight:.4f}")

    print("\n========== 第5步：训练 XGBoost ==========")
    xgb_model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="auc",
        random_state=42,
        n_jobs=-1,
        scale_pos_weight=scale_pos_weight
    )

    xgb_model.fit(X_train, y_train)

    print("\n========== 第6步：验证集预测 ==========")
    xgb_prob = xgb_model.predict_proba(X_valid)[:, 1]

    # =========================
    # 4. 阈值敏感性检验
    # =========================
    print("\n========== 第7步：阈值敏感性检验 ==========")

    threshold_rows = []
    for threshold in np.arange(0.10, 0.91, 0.05):
        threshold = float(np.round(threshold, 2))
        metrics = calc_metrics(y_valid, xgb_prob, threshold=threshold)

        threshold_rows.append({
            "threshold": threshold,
            **metrics
        })

    threshold_df = pd.DataFrame(threshold_rows)
    threshold_df.to_csv(
        OUTPUT_DIR / "xgb_threshold_sensitivity_table.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # 论文表格建议只筛几个代表阈值
    selected_thresholds = [0.30, 0.40, 0.50, 0.55, 0.60, 0.70]
    threshold_table_for_paper = threshold_df[
        threshold_df["threshold"].isin(selected_thresholds)
    ].copy()

    threshold_table_for_paper.to_csv(
        OUTPUT_DIR / "表6-3_XGBoost不同阈值下模型表现.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("阈值敏感性结果：")
    print(threshold_table_for_paper)

    # =========================
    # 5. 绘制阈值敏感性曲线
    # =========================
    print("\n========== 第8步：绘制阈值敏感性曲线 ==========")

    plt.figure(figsize=(7.2, 5.2))

    plt.plot(
        threshold_df["threshold"],
        threshold_df["Precision"],
        marker="o",
        linewidth=2.0,
        label="Precision"
    )
    plt.plot(
        threshold_df["threshold"],
        threshold_df["Recall"],
        marker="s",
        linewidth=2.0,
        label="Recall"
    )
    plt.plot(
        threshold_df["threshold"],
        threshold_df["F1"],
        marker="^",
        linewidth=2.0,
        label="F1-score"
    )

    plt.axvline(
        x=0.55,
        linestyle="--",
        linewidth=1.2,
        color="gray",
        label="Selected threshold = 0.55"
    )

    plt.xlabel("Classification threshold", fontsize=11)
    plt.ylabel("Metric value", fontsize=11)
    plt.title("Threshold Sensitivity of XGBoost", fontsize=13)

    plt.grid(alpha=0.25, linestyle="--", linewidth=0.7)
    plt.legend(loc="best", fontsize=9, frameon=True)

    plt.xlim(0.10, 0.90)
    plt.ylim(0.0, 1.0)

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "图6-1_XGBoost阈值敏感性曲线.png",
        dpi=600,
        bbox_inches="tight"
    )
    plt.savefig(
        OUTPUT_DIR / "图6-1_XGBoost阈值敏感性曲线.pdf",
        bbox_inches="tight"
    )

    plt.close()

    # =========================
    # 6. 按贷款期限分组稳健性检验
    # =========================
    print("\n========== 第9步：按贷款期限分组稳健性检验 ==========")

    valid_result_df = pd.DataFrame({
        "y_true": y_valid.values,
        "xgb_prob": xgb_prob,
        "term_group": term_valid_group.values
    })

    group_rows = []

    for group_name, group_df in valid_result_df.groupby("term_group"):
        y_g = group_df["y_true"].values
        p_g = group_df["xgb_prob"].values

        # 如果某个组只有单一类别，AUC无法计算
        if len(np.unique(y_g)) < 2:
            auc = np.nan
            ks = np.nan
        else:
            auc = roc_auc_score(y_g, p_g)
            ks = ks_score(y_g, p_g)

        metrics_g = calc_metrics(y_g, p_g, threshold=0.55)

        group_rows.append({
            "group_variable": "贷款期限",
            "sample_group": group_name,
            "sample_count": int(len(group_df)),
            "default_count": int(group_df["y_true"].sum()),
            "default_rate": float(group_df["y_true"].mean()),
            "AUC": float(auc) if not pd.isna(auc) else np.nan,
            "KS": float(ks) if not pd.isna(ks) else np.nan,
            "Precision": metrics_g["Precision"],
            "Recall": metrics_g["Recall"],
            "F1": metrics_g["F1"],
            "Accuracy": metrics_g["Accuracy"],
            "threshold": 0.55
        })

    group_result_df = pd.DataFrame(group_rows)

    # 排序：3年期在前，5年期在后
    order_map = {"3年期": 0, "5年期": 1}
    group_result_df["order"] = group_result_df["sample_group"].map(order_map).fillna(99)
    group_result_df = group_result_df.sort_values("order").drop(columns=["order"])

    group_result_df.to_csv(
        OUTPUT_DIR / "xgb_term_group_robustness.csv",
        index=False,
        encoding="utf-8-sig"
    )

    group_result_df.to_csv(
        OUTPUT_DIR / "表6-4_按贷款期限分组稳健性检验结果.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("分组稳健性检验结果：")
    print(group_result_df)

    print("\n========== 全部完成 ==========")
    print(f"输出目录：{OUTPUT_DIR.resolve()}")
    print("主要输出文件：")
    print("- 表6-3_XGBoost不同阈值下模型表现.csv")
    print("- 图6-1_XGBoost阈值敏感性曲线.png")
    print("- 图6-1_XGBoost阈值敏感性曲线.pdf")
    print("- 表6-4_按贷款期限分组稳健性检验结果.csv")


if __name__ == "__main__":
    main()