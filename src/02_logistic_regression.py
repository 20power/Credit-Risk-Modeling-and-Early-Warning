# -*- coding: utf-8 -*-
"""
第二步：Logistic 回归（影响因素识别主模型）
==================================================
输入：
- output_preprocessed/logistic_train.csv

输出：
- modeling_outputs/logistic_metrics.json
- modeling_outputs/logistic_coef_table.csv
- modeling_outputs/logistic_selected_features.txt
- modeling_outputs/logistic_valid_predictions.csv

说明：
1. 本脚本只使用“基础模型变量”，不纳入匿名变量 n0~n14
2. 适合作为论文中“影响因素识别”的核心统计模型
3. 使用 statsmodels GLM(Binomial) 输出系数、显著性、OR 值
"""
# -*- coding: utf-8 -*-

from pathlib import Path
import json
import re
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score,
    f1_score, accuracy_score, roc_curve
)

DATA_PATH = Path("output_preprocessed/logistic_train.csv")
OUTPUT_DIR = Path("modeling_outputs")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

LABEL_COL = "isDefault"


def ks_score(y_true, y_prob):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return float(np.max(tpr - fpr))


def select_base_features(df: pd.DataFrame):
    """
    只选基础模型变量：
    - 不纳入匿名变量 n0~n14
    - 不纳入高基数变量 employmentTitle/title/postCode
    - 避免明显重复变量一起进模型
    """
    # 保留的连续变量
    exact_numeric = {
        "loanAmnt",
        "interestRate",
        "installment",
        "annualIncome",
        "dti",
        "delinquency_2years",
        "openAcc",
        "pubRec",
        "pubRecBankruptcies",
        "revolBal",
        "revolUtil",
        "totalAcc",
        "employmentLength_num",
        "employmentLength_missing_flag",
        "credit_history_months",
        "loan_income_ratio",
        "installment_income_ratio",
        "fico_mid",
        "revol_loan_ratio",
        "issue_year",
        "issue_month",
        "issue_quarter",
    }

    # 保留的类别变量前缀（dummy 后）
    categorical_prefixes = [
        "term_",
        "grade_",
        "homeOwnership_",
        "verificationStatus_",
        "purpose_",
        "regionCode_",
        "initialListStatus_",
        "applicationType_",
    ]

    banned_prefixes = [
        "employmentTitle_",
        "title_",
        "postCode_",
        "subGrade_",   # 和 grade 高度重合，先不用
        "employmentLength_",  # 已用 employmentLength_num，不再重复用 raw dummy
    ]

    selected = []

    for c in df.columns:
        if c == LABEL_COL:
            continue

        # 跳过匿名变量
        if re.fullmatch(r"n\d+", c):
            continue

        # 跳过不想要的高基数或冗余哑变量
        if any(c.startswith(bp) for bp in banned_prefixes):
            continue

        # 精确数值变量
        if c in exact_numeric:
            selected.append(c)
            continue

        # 合法 dummy 变量
        if any(c.startswith(p) for p in categorical_prefixes):
            selected.append(c)

    # 每组 dummy 丢掉一个基准类，避免完全共线
    to_drop = []
    for prefix in categorical_prefixes:
        group = sorted([c for c in selected if c.startswith(prefix)])
        if len(group) > 1:
            to_drop.append(group[0])

    selected = [c for c in selected if c not in to_drop]

    return selected


def force_numeric_matrix(X: pd.DataFrame) -> pd.DataFrame:
    """
    强制把 DataFrame 压成纯数值矩阵
    """
    X = X.copy()

    print("原始 X dtype 统计：")
    print(X.dtypes.value_counts())

    # bool -> int8
    bool_cols = X.select_dtypes(include=["bool"]).columns.tolist()
    if bool_cols:
        X[bool_cols] = X[bool_cols].astype(np.int8)

    # object -> numeric
    obj_cols = X.select_dtypes(include=["object"]).columns.tolist()
    for c in obj_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    # 全部清洗为纯浮点
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)
    X = X.astype(np.float64)

    # 去掉常数列
    nunique = X.nunique(dropna=False)
    constant_cols = nunique[nunique <= 1].index.tolist()
    if constant_cols:
        print(f"删除常数列数量：{len(constant_cols)}")
        X = X.drop(columns=constant_cols, errors="ignore")

    print("修正后 X dtype 统计：")
    print(X.dtypes.value_counts())

    return X


def main():
    df = pd.read_csv(DATA_PATH)

    selected_features = select_base_features(df)

    X = df[selected_features].copy()
    y = df[LABEL_COL].astype(np.int8).copy()

    # 关键修复：强制纯数值矩阵
    X = force_numeric_matrix(X)

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    X_train_sm = sm.add_constant(X_train, has_constant="add")
    X_valid_sm = sm.add_constant(X_valid, has_constant="add")

    # 再保险：显式转 numpy float64
    X_train_np = np.asarray(X_train_sm, dtype=np.float64)
    X_valid_np = np.asarray(X_valid_sm, dtype=np.float64)
    y_train_np = np.asarray(y_train, dtype=np.float64)
    y_valid_np = np.asarray(y_valid, dtype=np.float64)

    model = sm.GLM(y_train_np, X_train_np, family=sm.families.Binomial())
    result = model.fit(maxiter=200, disp=0)

    valid_prob = result.predict(X_valid_np)
    valid_pred = (valid_prob >= 0.5).astype(int)

    metrics = {
        "auc": float(roc_auc_score(y_valid_np, valid_prob)),
        "ks": ks_score(y_valid_np, valid_prob),
        "precision": float(precision_score(y_valid_np, valid_pred, zero_division=0)),
        "recall": float(recall_score(y_valid_np, valid_pred, zero_division=0)),
        "f1": float(f1_score(y_valid_np, valid_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_valid_np, valid_pred)),
        "n_features": int(X.shape[1]),
        "n_train": int(len(X_train)),
        "n_valid": int(len(X_valid)),
    }

    param_names = ["const"] + X.columns.tolist()
    conf_int = result.conf_int()

    coef_table = pd.DataFrame({
        "variable": param_names,
        "coef": result.params,
        "odds_ratio": np.exp(result.params),
        "std_err": result.bse,
        "z_value": result.tvalues,
        "p_value": result.pvalues,
        "ci_lower": conf_int[:, 0],
        "ci_upper": conf_int[:, 1],
    }).sort_values(by="p_value", ascending=True)

    pred_df = pd.DataFrame({
        "y_true": y_valid_np,
        "y_prob": valid_prob,
        "y_pred": valid_pred
    })

    with open(OUTPUT_DIR / "logistic_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    coef_table.to_csv(OUTPUT_DIR / "logistic_coef_table.csv", index=False, encoding="utf-8-sig")
    pred_df.to_csv(OUTPUT_DIR / "logistic_valid_predictions.csv", index=False, encoding="utf-8-sig")

    with open(OUTPUT_DIR / "logistic_selected_features.txt", "w", encoding="utf-8") as f:
        for col in X.columns.tolist():
            f.write(col + "\n")

    print("Logistic 回归完成。")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()