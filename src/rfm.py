# -*- coding: utf-8 -*-
"""RFM 打分与八分层

打分规则：R / F / M 各自按五分位切 1~5 分。
  · R（最近一次消费距今天数）越小越好 → 反向打分
  · F（消费频次）、M（消费金额）越大越好 → 正向打分

八分层：R / F / M 各按「是否高于均值」二分，组合成 8 类（经典 RFM 模型）。
"""

import pandas as pd

# R / F / M 二分组合 -> 客户类型
SEGMENT_RULES = {
    (1, 1, 1): "重要价值客户",   # 近、频、高
    (1, 0, 1): "重要发展客户",   # 近、低频、高
    (0, 1, 1): "重要保持客户",   # 远、频、高
    (0, 0, 1): "重要挽留客户",   # 远、低频、高
    (1, 1, 0): "一般价值客户",
    (1, 0, 0): "一般发展客户",
    (0, 1, 0): "一般保持客户",
    (0, 0, 0): "一般挽留客户",
}


def score_rfm(df):
    """给 R / F / M 打 1~5 分。返回带 R_score/F_score/M_score 的 DataFrame。"""
    out = df.copy()

    # R 越小越好 -> 反向：用 -recency 排名
    out["R_score"] = pd.qcut(out["recency"].rank(method="first", ascending=True),
                             5, labels=[5, 4, 3, 2, 1]).astype(int)
    # F / M 越大越好
    out["F_score"] = pd.qcut(out["frequency"].rank(method="first"),
                             5, labels=[1, 2, 3, 4, 5]).astype(int)
    out["M_score"] = pd.qcut(out["monetary"].rank(method="first"),
                             5, labels=[1, 2, 3, 4, 5]).astype(int)
    out["RFM_total"] = out["R_score"] + out["F_score"] + out["M_score"]
    return out


def segment(df):
    """按 R/F/M 是否高于均值二分，映射到 8 类客户。"""
    out = df.copy()
    r_hi = (out["R_score"] >= out["R_score"].mean()).astype(int)
    f_hi = (out["F_score"] >= out["F_score"].mean()).astype(int)
    m_hi = (out["M_score"] >= out["M_score"].mean()).astype(int)
    out["segment"] = [
        SEGMENT_RULES[(r, f, m)] for r, f, m in zip(r_hi, f_hi, m_hi)
    ]
    return out


def segment_summary(df):
    """分层汇总：客户数、营收、营收占比、平均 R/F/M。"""
    total = df["monetary"].sum()
    g = df.groupby("segment").agg(
        customers=("customer_id", "count"),
        revenue=("monetary", "sum"),
        avg_recency=("recency", "mean"),
        avg_frequency=("frequency", "mean"),
        avg_monetary=("monetary", "mean"),
    ).reset_index()
    g["revenue_pct"] = g["revenue"] / total * 100
    g = g.sort_values("revenue", ascending=False).reset_index(drop=True)
    return g


if __name__ == "__main__":
    from extract import fetch_customer_rfm
    d = segment(score_rfm(fetch_customer_rfm()))
    print(segment_summary(d))
