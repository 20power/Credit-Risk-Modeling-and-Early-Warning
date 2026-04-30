# -*- coding: utf-8 -*-
"""
贷款违约论文：数据清洗与预处理
==================================================
修复重点：
1. 独热编码时强制输出为数值 dtype，避免 bool/object 混入
2. 输出 logistic_train / logistic_test 前，强制全部转为纯数值矩阵
3. 删除无效变量：id, policyCode, n3
4. 构造信用历史长度、收入负担比、fico_mid 等衍生变量
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd

# =========================
# 0. 路径配置
# =========================
TRAIN_PATH = Path("train.csv")
TEST_PATH = Path("testA.csv")

OUTPUT_DIR = Path("output_preprocessed")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

META_PATH = OUTPUT_DIR / "preprocess_metadata.json"
CLEAN_TRAIN_PATH = OUTPUT_DIR / "clean_train.csv"
CLEAN_TEST_PATH = OUTPUT_DIR / "clean_test.csv"
LOGIT_TRAIN_PATH = OUTPUT_DIR / "logistic_train.csv"
LOGIT_TEST_PATH = OUTPUT_DIR / "logistic_test.csv"


# =========================
# 1. 基础函数
# =========================
def load_data(train_path: Path, test_path: Path):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df


def safe_month_diff(late_date: pd.Series, early_date: pd.Series) -> pd.Series:
    late = pd.to_datetime(late_date, errors="coerce")
    early = pd.to_datetime(early_date, errors="coerce")
    diff = (late.dt.year - early.dt.year) * 12 + (late.dt.month - early.dt.month)
    return diff


def map_employment_length(series: pd.Series) -> pd.Series:
    mapping = {
        "10+ years": 10,
        "9 years": 9,
        "8 years": 8,
        "7 years": 7,
        "6 years": 6,
        "5 years": 5,
        "4 years": 4,
        "3 years": 3,
        "2 years": 2,
        "1 year": 1,
        "< 1 year": 0.5
    }
    return series.map(mapping)


def identify_column_types(df: pd.DataFrame, label_col: str = "isDefault"):
    exclude_cols = {label_col}
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude_cols]
    object_cols = [c for c in df.select_dtypes(exclude=[np.number]).columns if c not in exclude_cols]
    return numeric_cols, object_cols


# =========================
# 2. 特征工程
# =========================
def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # 2.1 删除无效/冗余变量
    out = out.drop(columns=["id", "policyCode", "n3"], errors="ignore")

    # 2.2 时间变量处理
    if "issueDate" in out.columns:
        out["issueDate"] = pd.to_datetime(out["issueDate"], errors="coerce")
        out["issue_year"] = out["issueDate"].dt.year
        out["issue_month"] = out["issueDate"].dt.month
        out["issue_quarter"] = out["issueDate"].dt.quarter

    if "earliesCreditLine" in out.columns:
        out["earliesCreditLine"] = pd.to_datetime(
            out["earliesCreditLine"], format="%b-%Y", errors="coerce"
        )
        out["earliest_year"] = out["earliesCreditLine"].dt.year
        out["earliest_month"] = out["earliesCreditLine"].dt.month

    # 2.3 构造信用历史长度（月）
    if "issueDate" in out.columns and "earliesCreditLine" in out.columns:
        out["credit_history_months"] = safe_month_diff(out["issueDate"], out["earliesCreditLine"])

    # 2.4 employmentLength 数值化
    if "employmentLength" in out.columns:
        out["employmentLength_num"] = map_employment_length(out["employmentLength"])
        out["employmentLength_missing_flag"] = out["employmentLength"].isna().astype(np.int8)

    # 2.5 收入与负担衍生变量
    if "annualIncome" in out.columns:
        annual_income_nonzero = out["annualIncome"].replace(0, np.nan)

        if "loanAmnt" in out.columns:
            out["loan_income_ratio"] = out["loanAmnt"] / annual_income_nonzero

        if "installment" in out.columns:
            out["installment_income_ratio"] = out["installment"] / annual_income_nonzero

    # 2.6 fico 中位值
    if "ficoRangeLow" in out.columns and "ficoRangeHigh" in out.columns:
        out["fico_mid"] = (out["ficoRangeLow"] + out["ficoRangeHigh"]) / 2

    # 2.7 revolving 负担变量
    if "revolBal" in out.columns and "loanAmnt" in out.columns:
        out["revol_loan_ratio"] = out["revolBal"] / out["loanAmnt"].replace(0, np.nan)

    # 2.8 删除原始日期列
    out = out.drop(columns=["issueDate", "earliesCreditLine"], errors="ignore")

    return out


# =========================
# 3. 缺失值填补
# =========================
def fit_imputation_rules(train_df: pd.DataFrame, label_col: str = "isDefault"):
    num_cols, obj_cols = identify_column_types(train_df, label_col=label_col)

    num_impute = {}
    for c in num_cols:
        if train_df[c].notna().any():
            num_impute[c] = float(train_df[c].median())
        else:
            num_impute[c] = 0.0

    obj_impute = {}
    for c in obj_cols:
        non_na = train_df[c].dropna()
        if len(non_na) == 0:
            obj_impute[c] = "Unknown"
        else:
            obj_impute[c] = non_na.mode().iloc[0]

    return {"num_impute": num_impute, "obj_impute": obj_impute}


def apply_imputation_rules(df: pd.DataFrame, rules: dict, label_col: str = "isDefault") -> pd.DataFrame:
    out = df.copy()
    num_cols, obj_cols = identify_column_types(out, label_col=label_col)

    for c in num_cols:
        if c in rules["num_impute"]:
            out[c] = out[c].fillna(rules["num_impute"][c])

    for c in obj_cols:
        if c in rules["obj_impute"]:
            out[c] = out[c].fillna(rules["obj_impute"][c])

    return out


# =========================
# 4. 构造 Logistic 数据集
# =========================
def build_logistic_dataset(train_df: pd.DataFrame, test_df: pd.DataFrame, label_col: str = "isDefault"):
    train_x = train_df.drop(columns=[label_col], errors="ignore").copy()
    test_x = test_df.copy()

    train_rows = len(train_x)
    combined = pd.concat([train_x, test_x], axis=0, ignore_index=True)

    cat_cols = combined.select_dtypes(exclude=[np.number]).columns.tolist()

    # 关键修复1：显式指定哑变量 dtype，避免 bool 混入
    combined_encoded = pd.get_dummies(
        combined,
        columns=cat_cols,
        drop_first=False,
        dtype=np.int8
    )

    # 关键修复2：强制所有列转为纯数值
    combined_encoded = combined_encoded.apply(pd.to_numeric, errors="coerce")
    combined_encoded = combined_encoded.replace([np.inf, -np.inf], np.nan).fillna(0)
    combined_encoded = combined_encoded.astype(np.float32)

    encoded_train = combined_encoded.iloc[:train_rows, :].copy()
    encoded_test = combined_encoded.iloc[train_rows:, :].copy()

    encoded_train[label_col] = train_df[label_col].astype(np.int8).values

    return encoded_train, encoded_test


# =========================
# 5. 主流程
# =========================
def main():
    print("========== 第1步：读取数据 ==========")
    train_df, test_df = load_data(TRAIN_PATH, TEST_PATH)
    print(f"train shape: {train_df.shape}")
    print(f"test  shape: {test_df.shape}")

    print("\n========== 第2步：特征工程 ==========")
    train_fe = feature_engineering(train_df)
    test_fe = feature_engineering(test_df)
    print(f"train_fe shape: {train_fe.shape}")
    print(f"test_fe  shape: {test_fe.shape}")

    print("\n========== 第3步：拟合缺失值规则 ==========")
    rules = fit_imputation_rules(train_fe, label_col="isDefault")

    print("\n========== 第4步：应用缺失值规则 ==========")
    clean_train = apply_imputation_rules(train_fe, rules, label_col="isDefault")
    clean_test = apply_imputation_rules(test_fe, rules, label_col="isDefault")

    print("\n========== 第5步：生成 Logistic 数据集 ==========")
    logistic_train, logistic_test = build_logistic_dataset(clean_train, clean_test, label_col="isDefault")

    print("logistic_train dtype 统计：")
    print(logistic_train.dtypes.value_counts())

    print("\n========== 第6步：保存输出 ==========")
    clean_train.to_csv(CLEAN_TRAIN_PATH, index=False, encoding="utf-8-sig")
    clean_test.to_csv(CLEAN_TEST_PATH, index=False, encoding="utf-8-sig")
    logistic_train.to_csv(LOGIT_TRAIN_PATH, index=False, encoding="utf-8-sig")
    logistic_test.to_csv(LOGIT_TEST_PATH, index=False, encoding="utf-8-sig")

    meta = {
        "deleted_columns": ["id", "policyCode", "n3"],
        "created_features": [
            "issue_year", "issue_month", "issue_quarter",
            "earliest_year", "earliest_month",
            "credit_history_months",
            "employmentLength_num", "employmentLength_missing_flag",
            "loan_income_ratio", "installment_income_ratio",
            "fico_mid", "revol_loan_ratio"
        ],
        "clean_train_shape": list(clean_train.shape),
        "clean_test_shape": list(clean_test.shape),
        "logistic_train_shape": list(logistic_train.shape),
        "logistic_test_shape": list(logistic_test.shape)
    }

    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("保存完成：")
    print(f"- {CLEAN_TRAIN_PATH}")
    print(f"- {CLEAN_TEST_PATH}")
    print(f"- {LOGIT_TRAIN_PATH}")
    print(f"- {LOGIT_TEST_PATH}")
    print(f"- {META_PATH}")
    print("\n========== 预处理完成 ==========")


if __name__ == "__main__":
    main()