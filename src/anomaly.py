# -*- coding: utf-8 -*-
"""经营异常检测

核心发现：月度**订单量**基本恒定，但**单均金额**剧烈波动 ——
订单量不变而金额剧变，说明异常出在「金额维度」，而非「交易活跃度」。

检测方法：稳健 z-score（中位数 + MAD），比均值+标准差更抗离群值污染。
阈值取 |z| >= 2.5（经验值，等同于「明显偏离主体分布」）。
"""

import numpy as np
import pandas as pd


def detect_anomaly(monthly, threshold=2.5):
    """检测单均金额异常月份。返回 (带标记和 z 值的 DataFrame, 异常行 DataFrame)。"""
    df = monthly.copy()
    v = df["avg_order_value"].to_numpy(dtype=float)

    med = np.median(v)
    mad = np.median(np.abs(v - med))
    # MAD 为 0 时退化到标准差，避免除零
    scale = mad * 1.4826 if mad > 0 else (v.std(ddof=1) or 1.0)

    df["z_score"] = (v - med) / scale
    df["is_anomaly"] = df["z_score"].abs() >= threshold
    df["deviation_pct"] = (v / med - 1) * 100

    anomalies = df[df["is_anomaly"]].copy()
    return df, anomalies


def order_volume_stability(monthly):
    """评估订单量稳定性：变异系数 CV（标准差/均值）。

    CV 越小越稳定；用于佐证「订单量恒定、异常在金额端」这一结论。
    """
    o = monthly["orders"].to_numpy(dtype=float)
    return {
        "mean": float(o.mean()),
        "std": float(o.std(ddof=1)),
        "cv": float(o.std(ddof=1) / o.mean()) if o.mean() else 0.0,
    }


def pareto(customer_df, top_n=10):
    """帕累托：营收 Top N 客户占总营收比例。"""
    d = customer_df.sort_values("monetary", ascending=False).reset_index(drop=True)
    total = d["monetary"].sum()
    top_n_share = d["monetary"].head(top_n).sum() / total * 100
    top20_cnt = max(1, int(len(d) * 0.2))
    top20_share = d["monetary"].head(top20_cnt).sum() / total * 100

    d["cum_pct"] = d["monetary"].cumsum() / total * 100
    return {
        "top_n": top_n,
        "top_n_share": float(top_n_share),
        "top20_count": top20_cnt,
        "top20_share": float(top20_share),
        "detail": d.head(top_n),
        "cum_curve": d[["customer_name", "monetary", "cum_pct"]],
    }


if __name__ == "__main__":
    from extract import fetch_monthly, fetch_customer_rfm
    m = fetch_monthly()
    md, an = detect_anomaly(m)
    print(md[["month", "revenue", "orders", "avg_order_value", "z_score", "deviation_pct", "is_anomaly"]].to_string(index=False))
    print()
    print("订单量稳定性:", order_volume_stability(m))
    print()
    p = pareto(fetch_customer_rfm())
    print("Top%d 占比 %.2f%% | Top20%%(%d家) 占比 %.2f%%" % (p["top_n"], p["top_n_share"], p["top20_count"], p["top20_share"]))
    print()
    print("异常月份:")
    print(an[["month", "avg_order_value", "deviation_pct"]].to_string(index=False) if len(an) else "  无")
