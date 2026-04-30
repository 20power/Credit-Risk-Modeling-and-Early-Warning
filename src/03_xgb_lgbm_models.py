# -*- coding: utf-8 -*-
"""
第三步：XGBoost + LightGBM（风险预警增强模型）
==================================================
输入：
- output_preprocessed/clean_train.csv
- output_preprocessed/clean_test.csv

输出：
- modeling_outputs/tree_model_comparison.csv
- modeling_outputs/xgb_feature_importance.csv
- modeling_outputs/lgb_feature_importance.csv
- modeling_outputs/best_model_info.json
- modeling_outputs/xgb_test_predictions.csv
- modeling_outputs/lgb_test_predictions.csv
- modeling_outputs/encoded_feature_columns.json

说明：
1. 本脚本使用“基础变量 + 匿名变量”做增强模型
2. 高基数字段 employmentTitle/title/postCode 默认不纳入
3. 训练/验证集分层划分
"""

# -*- coding: utf-8 -*-
"""
第三步：XGBoost + LightGBM（修正版）
==================================================
修复重点：
1. 对 get_dummies 后的列名做统一清洗，去掉 [, ], < 等非法字符
2. 强制所有特征列为纯数值类型
3. 保留基础变量 + 匿名变量的增强模型路线
4. 输出模型比较、特征重要性、测试集预测结果

输入：
- output_preprocessed/clean_train.csv
- output_preprocessed/clean_test.csv

输出：
- modeling_outputs/tree_model_comparison.csv
- modeling_outputs/xgb_feature_importance.csv
- modeling_outputs/lgb_feature_importance.csv
- modeling_outputs/best_model_info.json
- modeling_outputs/xgb_test_predictions.csv
- modeling_outputs/lgb_test_predictions.csv
- modeling_outputs/encoded_feature_columns.json
"""

from pathlib import Path
import json
import re
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score,
    f1_score, accuracy_score, roc_curve
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

TRAIN_PATH = Path("output_preprocessed/clean_train.csv")
TEST_PATH = Path("output_preprocessed/clean_test.csv")
OUTPUT_DIR = Path("modeling_outputs_gb")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
LABEL_COL = "isDefault"


def ks_score(y_true, y_prob):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return float(np.max(tpr - fpr))


def select_enhanced_features(df: pd.DataFrame):
    """
    选择增强模型变量：
    - 保留基础解释变量
    - 加入匿名变量 n0~n14（不含 n3，因为前面已删除）
    - 不纳入高基数字段 employmentTitle/title/postCode
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
    清洗列名，避免 XGBoost 报错：
    - 不能有 [, ], <
    - 也顺手把空格、逗号、括号等替换掉
    """
    new_cols = []
    for c in columns:
        c = str(c)

        # 重点修复 XGBoost 禁止字符
        c = c.replace("[", "_")
        c = c.replace("]", "_")
        c = c.replace("<", "lt_")
        c = c.replace(">", "gt_")

        # 常见杂项字符也一起清理
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

        # 连续下划线压缩
        c = re.sub(r"_+", "_", c)
        c = c.strip("_")

        # 防止空字符串
        if c == "":
            c = "feature"

        new_cols.append(c)

    # 防止清洗后重复列名
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
    强制所有列转为纯数值，避免 object/bool 混入树模型。
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
    训练集和测试集统一做独热编码，并保证列完全对齐。
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

    # 关键修复1：清洗列名
    combined_encoded.columns = sanitize_feature_names(combined_encoded.columns)

    # 关键修复2：强制纯数值
    combined_encoded = force_numeric_df(combined_encoded)

    encoded_train = combined_encoded.iloc[:train_rows, :].copy()
    encoded_test = combined_encoded.iloc[train_rows:, :].copy()
    encoded_train[label_col] = train_df[label_col].astype(np.int8).values

    return encoded_train, encoded_test


def build_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "auc": float(roc_auc_score(y_true, y_prob)),
        "ks": ks_score(y_true, y_prob),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }


def main():
    print("========== 第1步：读取清洗数据 ==========")
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    print(f"clean_train shape: {train_df.shape}")
    print(f"clean_test  shape: {test_df.shape}")

    print("\n========== 第2步：筛选增强模型变量 ==========")
    selected_cols = select_enhanced_features(train_df)
    train_df = train_df[selected_cols + [LABEL_COL]].copy()
    test_df = test_df[selected_cols].copy()
    print(f"selected feature count(before encoding): {len(selected_cols)}")

    print("\n========== 第3步：统一独热编码 ==========")
    encoded_train, encoded_test = one_hot_align(train_df, test_df, label_col=LABEL_COL)

    X = encoded_train.drop(columns=[LABEL_COL]).copy()
    y = encoded_train[LABEL_COL].astype(np.int8).copy()
    X_test = encoded_test.copy()

    print("X dtype 统计：")
    print(X.dtypes.value_counts())

    print("\n========== 第4步：训练/验证集划分 ==========")
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())
    scale_pos_weight = neg / pos if pos > 0 else 1.0

    print(f"train rows: {len(X_train)}, valid rows: {len(X_valid)}")
    print(f"scale_pos_weight: {scale_pos_weight:.4f}")

    print("\n========== 第5步：训练 XGBoost / LightGBM ==========")
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

    lgb_model = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary",
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    )

    xgb_model.fit(X_train, y_train)
    lgb_model.fit(X_train, y_train)

    print("\n========== 第6步：验证集评估 ==========")
    xgb_prob = xgb_model.predict_proba(X_valid)[:, 1]
    lgb_prob = lgb_model.predict_proba(X_valid)[:, 1]

    # --------------------------------------------------
    # 新增：保存验证集预测概率，用于后续绘制 ROC 曲线
    # --------------------------------------------------
    # 该文件至少包含：
    # y_true   : 验证集真实标签
    # xgb_prob : XGBoost 对验证集输出的违约概率
    # lgb_prob : LightGBM 对验证集输出的违约概率
    #
    # 后续绘制 ROC 曲线时，可直接读取该文件。
    tree_valid_pred_df = pd.DataFrame({
        "y_true": y_valid.values,
        "xgb_prob": xgb_prob,
        "lgb_prob": lgb_prob
    })

    tree_valid_pred_df.to_csv(
        OUTPUT_DIR / "tree_valid_predictions.csv",
        index=False,
        encoding="utf-8-sig"
    )

    xgb_metrics = build_metrics(y_valid, xgb_prob)
    lgb_metrics = build_metrics(y_valid, lgb_prob)

    comp = pd.DataFrame([
        {"model": "XGBoost", **xgb_metrics},
        {"model": "LightGBM", **lgb_metrics},
    ])
    comp.to_csv(OUTPUT_DIR / "tree_model_comparison.csv", index=False, encoding="utf-8-sig")

    print(comp)

    print("\n========== 第7步：特征重要性 ==========")
    xgb_imp = pd.DataFrame({
        "feature": X.columns,
        "importance": xgb_model.feature_importances_
    }).sort_values("importance", ascending=False)
    xgb_imp.to_csv(OUTPUT_DIR / "xgb_feature_importance.csv", index=False, encoding="utf-8-sig")

    lgb_imp = pd.DataFrame({
        "feature": X.columns,
        "importance": lgb_model.feature_importances_
    }).sort_values("importance", ascending=False)
    lgb_imp.to_csv(OUTPUT_DIR / "lgb_feature_importance.csv", index=False, encoding="utf-8-sig")

    print("\n========== 第8步：选最佳模型 ==========")
    if xgb_metrics["auc"] >= lgb_metrics["auc"]:
        best_name = "XGBoost"
    else:
        best_name = "LightGBM"

    print(f"best model by auc: {best_name}")

    print("\n========== 第9步：全量重训 ==========")
    full_scale_pos_weight = (y == 0).sum() / (y == 1).sum()

    xgb_full = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="auc",
        random_state=42,
        n_jobs=-1,
        scale_pos_weight=full_scale_pos_weight
    )

    lgb_full = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary",
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    )

    xgb_full.fit(X, y)
    lgb_full.fit(X, y)

    print("\n========== 第10步：测试集预测 ==========")
    xgb_test_prob = xgb_full.predict_proba(X_test)[:, 1]
    lgb_test_prob = lgb_full.predict_proba(X_test)[:, 1]

    pd.DataFrame({"default_prob": xgb_test_prob}).to_csv(
        OUTPUT_DIR / "xgb_test_predictions.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame({"default_prob": lgb_test_prob}).to_csv(
        OUTPUT_DIR / "lgb_test_predictions.csv", index=False, encoding="utf-8-sig"
    )

    print("\n========== 第11步：保存模型和元信息 ==========")
    joblib.dump(xgb_full, OUTPUT_DIR / "xgb_model.pkl")
    joblib.dump(lgb_full, OUTPUT_DIR / "lgb_model.pkl")

    with open(OUTPUT_DIR / "encoded_feature_columns.json", "w", encoding="utf-8") as f:
        json.dump(list(X.columns), f, ensure_ascii=False, indent=2)

    best_info = {
        "best_model_by_auc": best_name,
        "xgb_metrics": xgb_metrics,
        "lgb_metrics": lgb_metrics,
        "n_features": int(X.shape[1]),
        "n_train_rows": int(X.shape[0])
    }
    with open(OUTPUT_DIR / "best_model_info.json", "w", encoding="utf-8") as f:
        json.dump(best_info, f, ensure_ascii=False, indent=2)

    print("全部完成。")


if __name__ == "__main__":
    main()