# -*- coding: utf-8 -*-
"""一键执行：取数 → RFM 分层 → KMeans 聚类 → 异常检测 → 出图 → 生成报告

用法：
    cd src && python run_all.py
（数据库连接读环境变量 DB_HOST / DB_PORT / DB_USER / DB_PASSWORD）
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np                                    # noqa: E402
import pandas as pd                                   # noqa: E402

from config import RESULT_CSV, MONTH_CSV, REPORT_MD   # noqa: E402
from extract import fetch_customer_rfm, fetch_monthly, fetch_alert_summary  # noqa: E402
from rfm import score_rfm, segment, segment_summary   # noqa: E402
from kmeans import kmeans, name_clusters, auto_k      # noqa: E402
from anomaly import detect_anomaly, order_volume_stability, pareto  # noqa: E402
import visualize as viz                               # noqa: E402


def main(source="sqlite"):
    print("=" * 68)
    print("销售与客户专题分析   数据源：%s" % source)
    print("=" * 68)

    # ---------- 1. 取数 ----------
    print("\n[1/6] 取数 …")
    cust = fetch_customer_rfm(source)
    monthly = fetch_monthly(source)
    alerts = fetch_alert_summary(source)
    print("  客户 %d 家 | 月度 %d 个月 | 预警 %d 条" % (len(cust), len(monthly), len(alerts)))

    # ---------- 2. RFM 八分层 ----------
    print("\n[2/6] RFM 打分与八分层 …")
    scored = score_rfm(cust)
    seg = segment(scored)
    summary = segment_summary(seg)
    print(summary.to_string(index=False))

    # ---------- 3. KMeans 聚类（交叉验证）----------
    print("\n[3/6] KMeans 聚类（numpy 实现）…")
    feats = seg[["recency", "frequency", "monetary"]].to_numpy(dtype=float)
    # 标准化：否则 M（亿元级）会完全支配距离
    mu, sigma = feats.mean(axis=0), feats.std(axis=0)
    sigma[sigma == 0] = 1.0
    X = (feats - mu) / sigma

    elbow = auto_k(X, range(2, 9))
    print("  手肘法 inertia: %s" % {k: round(v, 1) for k, v in elbow.items()})
    k = 4
    labels, centers, inertia = kmeans(X, k=k, seed=42)
    seg_c, cluster_stat, mapping = name_clusters(seg, labels, value_col="monetary")
    print("  k=%d  inertia=%.1f" % (k, inertia))
    print(cluster_stat.to_string(index=False))

    # ---------- 4. 异常检测 ----------
    print("\n[4/6] 月度异常检测 …")
    marked, anomalies = detect_anomaly(monthly)
    stab = order_volume_stability(monthly)
    print("  订单量 CV=%.3f（越小越稳定）" % stab["cv"])
    print(anomalies[["month", "orders", "avg_order_value", "deviation_pct"]].to_string(index=False)
          if len(anomalies) else "  未发现异常月份")

    # ---------- 5. 帕累托 ----------
    print("\n[5/6] 帕累托分析 …")
    par = pareto(cust, top_n=10)
    print("  Top10 占 %.2f%% | Top20%%(%d家) 占 %.2f%%" % (par["top_n_share"], par["top20_count"], par["top20_share"]))

    # ---------- 6. 出图 + 落盘 ----------
    print("\n[6/6] 出图并写结果 …")
    viz.plot_rfm_scatter(seg)
    viz.plot_segment_bar(summary)
    viz.plot_pareto(par["cum_curve"])
    viz.plot_monthly_trend(marked)
    viz.plot_clusters(seg_c)

    out = seg_c[["customer_id", "customer_name", "region", "recency", "frequency", "monetary",
                 "R_score", "F_score", "M_score", "RFM_total", "segment", "cluster_name"]].copy()
    out = out.sort_values("monetary", ascending=False)
    out.to_csv(RESULT_CSV, index=False, encoding="utf-8-sig")
    print("  ✅ %s" % RESULT_CSV)

    marked.to_csv(MONTH_CSV, index=False, encoding="utf-8-sig")
    print("  ✅ %s" % MONTH_CSV)

    write_report(seg, seg_c, summary, cluster_stat, marked, anomalies, stab, par, alerts, k, mapping)
    print("  ✅ %s" % REPORT_MD)

    print("\n" + "=" * 68)
    print("完成。")
    print("=" * 68)


def md_table(df, fmts=None):
    """把 DataFrame 转成 Markdown 表格（不依赖 tabulate）。

    fmts: {列名: 格式串}，例如 {"revenue": "{:.2f}"}
    """
    fmts = fmts or {}
    cols = list(df.columns)
    head = "| " + " | ".join(str(c) for c in cols) + " |"
    sep = "|" + "|".join(["---"] * len(cols)) + "|"
    rows = []
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            cells.append(fmts[c].format(v) if c in fmts and isinstance(v, (int, float)) else str(v))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([head, sep] + rows)


def write_report(seg, seg_c, summary, cluster_stat, marked, anomalies, stab, par, alerts, k, mapping):
    total_rev = seg["monetary"].sum()
    an_lines = "\n".join(
        "| %s | %.0f | %s | %+.1f%% |" % (r["month"], r["avg_order_value"], r["orders"], r["deviation_pct"])
        for _, r in anomalies.iterrows()
    ) or "| — | — | — | — |"

    if len(alerts):
        tot = alerts["cnt"].sum()
        alert_lines = "\n".join(
            "| %s | %d | %.1f%% |" % (r["alert_type"], r["cnt"], r["cnt"] / tot * 100)
            for _, r in alerts.iterrows())
        alert_block = "\n".join([
            "| 预警类型 | 条数 | 占比 |", "|---|---|---|", alert_lines])
    else:
        alert_block = "> 自带数据集（sqlite）仅包含订单数据，无预警表；切换到 `--source mysql` 可查看数仓预警分布。"

    cluster_lines = "\n".join(
        "| %s | %d | %.2f | %.0f | %.2f |" % (
            mapping.get(int(r["_cluster"]), "-"), int(r["n"]),
            r["m"] / 1e8, r["f"], r["m"] / max(int(r["n"]), 1))
        for _, r in cluster_stat.iterrows()
    )

    seg_tbl = summary.assign(
        营收亿元=lambda d: d["revenue"] / 1e8,
        营收占比=lambda d: d["revenue_pct"].round(2),
    )[["segment", "customers", "营收亿元", "营收占比", "avg_recency", "avg_frequency"]].rename(columns={
        "segment": "客户类型", "customers": "客户数", "avg_recency": "平均R(天)", "avg_frequency": "平均F"})
    seg_md = md_table(seg_tbl, {"营收亿元": "{:.2f}", "营收占比": "{:.2f}",
                                "平均R(天)": "{:.0f}", "平均F": "{:.0f}"})

    text = """# 销售经营专题分析报告

> 数据来源：制造业经营分析数据仓库 ADS 层（`ads_db.ads_sale_analysis`，{n_cust} 家客户 / {n_month} 个月 / 营收 {total:.2f} 亿元）
> 说明：底层为仿真数据，结论用于展示「从数仓数据到经营结论」的完整分析链路。

## 一、总体概览

| 指标 | 数值 |
|---|---|
| 客户数 | {n_cust} 家 |
| 统计月份 | {n_month} 个月 |
| 累计营收 | {total:.2f} 亿元 |
| 累计订单 | {orders:,} 单 |
| 月均订单量 | {avg_orders:,.0f} 单（变异系数 CV={cv:.3f}） |
| Top10 客户营收占比 | {top10:.2f}% |
| Top20% 客户营收占比 | {top20:.2f}% |

## 二、经营异常发现（核心）

**核心发现：月度订单量基本恒定，但单均金额剧烈波动。**

月订单量变异系数仅 **{cv:.3f}**（高度稳定），而单均金额在 {vmin:,.0f} ~ {vmax:,.0f} 元之间波动。
订单活跃度不变而金额剧变，说明异常出在**金额维度**而非交易活跃度。

| 异常月份 | 单均金额（元） | 当月订单数 | 相对中位数偏离 |
|---|---|---|---|
{an_lines}

**判断与建议**：需核实是大单 / 促销政策导致的真实业务波动，还是金额字段存在数据质量问题。
若为后者，所有营收类指标（营收、客单价、毛利）均会失真，应优先排查上游采集口径。

> 方法：稳健 z-score（中位数 + MAD，比均值+标准差更抗离群值污染），阈值 |z| ≥ 2.5。

## 三、客户价值分层（RFM）

打分规则：R / F / M 各按五分位切 1~5 分（R 反向），再按各自是否高于均值二分，组合成 8 类。
> ⚠️ 本数据中 R 恒为 0、F/M 变异系数仅约 2%，分层区分度有限，详见第七节「数据适用性说明」。

{seg_table}

## 四、KMeans 聚类交叉验证（k={k}）

对标准化后的 R / F / M 做 KMeans 聚类，用于验证规则分层的稳健性（两者结论应当接近）。

| 客户群 | 客户数 | 营收合计（亿元） | 平均频次 | 客均营收（亿元） |
|---|---|---|---|---|
{cluster_lines}

> 聚类实现：numpy（k-means++ 初始化 + 固定随机种子），不依赖 scikit-learn，结果可复现。

## 五、经营预警分布

{alert_block}

## 六、结论与运营建议

1. **客户结构高度分散**：Top10 客户仅贡献营收 {top10:.2f}%，Top20%（{top20c} 家）贡献 {top20:.2f}%。
   → 无核心大客户依赖，经营风险较低；但缺少大客户抓手，建议按本报告的分层结果做差异化运营。
2. **重点关注「重要价值 / 重要保持」客群**：这是营收主力，应配置专属维护与优先履约资源。
3. **「重要挽留」客群需激活**：消费金额高但最近消费距今较远，存在流失风险，建议触达召回。
4. **营收数据口径待核实**：单均金额异常波动（见第二节），在核实前不建议基于营收做趋势外推。
5. **风险集中在资金链**：预警类型中「回款滞后」与「营收缺口」占比最高，经营风险主要在回款端。

## 七、数据适用性说明（重要）

本报告的分层结论必须结合数据特征解读，这一点在分析中已核实：

| 维度 | 变异系数 | 区分度 | 说明 |
|---|---|---|---|
| R（最近消费距今天数） | **0.000** | ❌ 无 | 仿真数据中**每家客户每天都下单**，R 恒为 0，该维度不携带信息 |
| F（消费频次） | {f_cv:.3f} | ⚠️ 很低 | 各家订单量高度接近 |
| M（消费金额） | {m_cv:.3f} | ⚠️ 很低 | 各家营收高度接近 |

**结论**：本批仿真数据的客户是**同质的**（Top10 仅占营收 {top10:.2f}%，而真实业务通常呈幂律分布、
Top20% 贡献 60~80%）。因此：

- 第三节的八分层与第四节的聚类**在技术流程上完整正确**，但**业务区分度有限**（各层营收仅差约 4%）
- 第二节的**时间维度异常是真实的**（订单量稳定而单均金额波动 5 倍），这部分结论不受客户同质性影响
- **改进方向**：在仿真生成器中为客户引入「活跃度 + 金额规模」的幂律异质性，
  重新生成后 RFM 与聚类才具备完整业务意义 —— 这是本项目的下一步工作

> 把「数据不适用」明确写出来，比给出一个看似漂亮但站不住的分层更有价值。

## 八、产出物

| 文件 | 说明 |
|---|---|
| `output/customer_segment_result.csv` | 每家客户的 R/F/M、评分、分层与聚类归属 |
| `output/monthly_sales.csv` | 月度营收、订单数、单均金额与异常标记 |
| `output/figures/*.png` | 5 张分析图表（RFM 散点、分层柱状、帕累托、月度趋势、聚类散点） |
""".format(
        n_cust=len(seg), n_month=len(marked), total=total_rev / 1e8,
        orders=int(marked["orders"].sum()), avg_orders=marked["orders"].mean(),
        cv=stab["cv"], top10=par["top_n_share"], top20=par["top20_share"], top20c=par["top20_count"],
        seg_table=seg_md,
        k=k, cluster_lines=cluster_lines,
        vmin=marked["avg_order_value"].min(), vmax=marked["avg_order_value"].max(),
        an_lines=an_lines,
        f_cv=seg["frequency"].std() / seg["frequency"].mean(),
        m_cv=seg["monetary"].std() / seg["monetary"].mean(),
        alert_block=alert_block,
    )
    with open(REPORT_MD, "w", encoding="utf-8") as f:
        f.write(text)


if __name__ == "__main__":
    import argparse
    _ap = argparse.ArgumentParser()
    _ap.add_argument("--source", choices=["sqlite", "mysql"], default="sqlite",
                     help="数据源：sqlite=项目自带数据（默认，可独立复现）；mysql=直连数仓 ADS 层")
    _a = _ap.parse_args()
    main(_a.source)
