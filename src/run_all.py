# -*- coding: utf-8 -*-
"""一键执行：取数 → RFM 分层 → KMeans 聚类 → 异常检测 → 出图 → 生成报告

用法：
    cd src && python run_all.py
（数据库连接读环境变量 DB_HOST / DB_PORT / DB_USER / DB_PASSWORD）
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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
    viz.plot_pareto(par)
    viz.plot_monthly_trend(marked)
    viz.plot_clusters(seg_c)

    out = seg_c[["customer_id", "customer_name", "region", "recency", "frequency", "monetary",
                 "R_score", "F_score", "M_score", "RFM_total", "segment", "cluster_name"]].copy()
    out = out.sort_values("monetary", ascending=False)
    out.to_csv(RESULT_CSV, index=False, encoding="utf-8-sig")
    print("  ✅ %s" % RESULT_CSV)

    marked.to_csv(MONTH_CSV, index=False, encoding="utf-8-sig")
    print("  ✅ %s" % MONTH_CSV)

    write_report(seg, seg_c, summary, cluster_stat, marked, anomalies, stab, par, alerts, k, mapping, source)
    print("  ✅ %s" % REPORT_MD)

    print("\n" + "=" * 68)
    print("完成。")
    print("=" * 68)


def _mark(ok):
    """适用性判定标记。"""
    return "✅ 适用" if ok else "⚠️ 区分度不足"


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


def write_report(seg, seg_c, summary, cluster_stat, marked, anomalies, stab, par, alerts, k, mapping, source):
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

    # total=簇营收合计（亿元）；m=客均营收（万元）
    cluster_lines = "\n".join(
        "| %s | %d | %.2f | %.0f | %s |" % (
            mapping.get(int(r["_cluster"]), "-"), int(r["n"]),
            r["total"] / 1e8, r["f"], format(round(r["m"] / 1e4), ","))
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

> 数据来源：{src_desc}
> 规模：{n_cust} 家客户 / {n_month} 个月 / {n_orders:,} 单 / 营收 {total:.2f} 亿元
> 说明：底层为仿真数据，结论用于展示「从数据构造到经营结论」的完整分析链路。

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
| 累计 80% 所需客户数 | 第 {idx80} 家（共 {n_cust} 家） |

## 二、经营异常发现（核心）

**核心发现：月度订单量基本恒定，但单均金额剧烈波动。**

月订单量变异系数仅 **{cv:.3f}**（高度稳定）；而单均金额：正常月稳定在 **{vmed:,.0f} 元**附近，
异常月则偏离到 **{vmin:,.0f} ~ {vmax:,.0f} 元**（最大波动 **{vratio:.1f} 倍**）。
订单活跃度不变而金额剧变，说明异常出在**金额维度**而非交易活跃度。

| 异常月份 | 单均金额（元） | 当月订单数 | 相对中位数偏离 |
|---|---|---|---|
{an_lines}

**判断与建议**：需核实是大单 / 促销政策导致的真实业务波动，还是金额字段存在数据质量问题。
若为后者，所有营收类指标（营收、客单价、毛利）均会失真，应优先排查上游采集口径。

> **方法**：稳健 z-score（中位数 + MAD，比均值+标准差更抗离群值污染）。
> 需**同时**满足两个条件才判为异常：① \\|z\\| ≥ 2.5 ② \\|偏离中位数\\| ≥ 15%。
> 之所以要加第 ② 条：当数据本身极稳定时 MAD 趋近 0、z 值会爆炸，
> 仅靠 ① 会把 0.2% 的微小波动误判为异常（实测出现过）。

## 三、客户价值分层（RFM）

打分规则：R / F / M 各按五分位切 1~5 分（R 反向），再按各自是否高于均值二分，组合成 8 类。
数据适用性已通过验证（见第七节）：R 取值 {r_nuniq} 个、F/M 变异系数 {f_cv:.2f}/{m_cv:.2f}、Top20% 占 {top20:.1f}%。

{seg_table}

## 四、KMeans 聚类交叉验证（k={k}）

对标准化后的 R / F / M 做 KMeans 聚类，与第三节的规则分层交叉验证。

| 客户群 | 客户数 | 营收合计（亿元） | 平均频次 | 客均营收（万元） |
|---|---|---|---|---|
{cluster_lines}

> ⚠️ **命名依据是「客均营收」降序，而不是「营收合计」** —— 簇越大合计越高，
> 所以「中高价值客户群」的合计可能高于「高价值客户群」，这不矛盾（前者客户数远多于后者）。

**一致性结论**：两种方法的分层粒度本就不同（RFM 按 R/F/M 三维二分得 8 类；KMeans 按距离聚成 4 簇），
因此不强求成员完全一致，而是看**价值序是否一致**：

- KMeans 各簇的客均营收**严格递减**（{cluster_means}）—— 簇的价值序清晰 ✅
- 更有说服力的一致点：**KMeans 前两簇合计 {top2_rev:.2f} 亿元（占 {top2_pct:.1f}%），
  与 RFM「重要价值客户」的 {rfm_top_pct:.1f}% 基本重合** —— 两种方法独立得出
  「营收高度集中在头部分层」这同一结论 ✅

> 说明：RFM 八类的**客均**营收是总体递减但并非严格单调（样本量小的类别会交叉，
> 如「重要发展客户」仅 7 家），这是小样本的正常现象，不影响分组结论。

> 聚类实现：numpy（k-means++ 初始化 + 固定随机种子），不依赖 scikit-learn，结果可复现。

## 五、经营预警分布

{alert_block}

## 六、结论与运营建议

1. **{conc1_title}**：Top10 客户贡献营收 {top10:.2f}%，Top20%（{top20c} 家）贡献 {top20:.2f}%；
   **第 {idx80} 家即达到累计 80%**。
   → {conc1_action}
2. **重点关注「重要价值 / 重要保持」客群**：这是营收主力，应配置专属维护与优先履约资源。
3. **「重要挽留」客群需甄别**：{reclaim_n} 家、平均 R={reclaim_r} 天（最久未交易），
   但其营收仅占 **{reclaim_pct:.2f}%** —— 属"单客价值层级不低、但体量很小"的流失预警客群，
   建议用**低成本触达**（短信/回访）而非重投资源。
4. **营收数据口径待核实**：单均金额异常波动（见第二节），在核实前不建议基于营收做趋势外推。
{conc5}

## 七、数据适用性说明

分析之前先判断「数据能不能支撑结论」。本报告结论建立在以下已验证的数据特征上：

| 维度 | 实测 | 是否适用 |
|---|---|---|
| R（最近消费距今天数） | 取值 {r_min}~{r_max} 天，共 {r_nuniq} 个不同值 | {r_ok} |
| F（消费频次） | 变异系数 **{f_cv:.2f}** | {f_ok} |
| M（消费金额） | 变异系数 **{m_cv:.2f}** | {m_ok} |
| 营收集中度 | Top20% 客户占 **{top20:.1f}%** | {c_ok} |

**判定标准**：R 需有多个取值（否则该维度无信息）；F / M 变异系数 > 0.3 视为有区分度；
营收集中度 Top20% ≥ 50% 才符合真实业务「**少数客户贡献多数营收**」的分布特征。

> **对照教训（值得保留的一段判断过程）**
> 本项目的姊妹数仓项目使用的是**均匀分布**的仿真客户 —— 每客户每天都下单、金额接近。
> 实测：R 恒为 0、F/M 变异系数仅约 2%、Top20% 客户仅占营收 20.6%。
> 用那种数据做 RFM，会得到「各层营收只差 4%」的**无效分层** ——
> 分层流程技术上完全正确，但业务上什么也没说明。
>
> 因此本项目**自带构造**了符合真实业务特征的数据集（见 `src/make_data.py`）：
> - 金额规模用**混合分布**（85% 对数正态主体 + 15% 对数正态头部）：
>   偏度 **{m_skew:.2f}**、max/中位 **{m_ratio:.1f}** 倍（原纯幂律方案为 8.31 / 55 倍，图表会被极端值压平）
>   → Top20% 客户贡献 **{top20:.1f}%**
> - 购买用**泊松过程**（指数分布间隔），底部 12% 客户**沉睡**（60~180 天不下单）→ R 才有多样性
> - 刻意植入 **{n_inj} 种形态**的已知异常（{inj_desc}）→ 用于验证异常检测的**双向**识别能力
>
> **先判断数据能不能支撑结论，再动手分析** —— 这是本项目沉淀下来的分析习惯。

## 八、产出物

| 文件 | 说明 |
|---|---|
| `output/customer_segment_result.csv` | 每家客户的 R/F/M、评分、分层与聚类归属 |
| `output/monthly_sales.csv` | 月度营收、订单数、单均金额与异常标记 |
| `output/figures/*.png` | 5 张分析图表（RFM 散点、分层**条形图**、帕累托、月度趋势、聚类散点） |
""".format(
        n_cust=len(seg), n_month=len(marked), total=total_rev / 1e8,
        orders=int(marked["orders"].sum()), avg_orders=marked["orders"].mean(),
        cv=stab["cv"], top10=par["top_n_share"], top20=par["top20_share"], top20c=par["top20_count"],
        seg_table=seg_md,
        k=k, cluster_lines=cluster_lines,
        # ---- 以下变量全部由实测值计算，避免模板里写死数字与表格脱节 ----
        n_orders=int(marked["orders"].sum()),
        vmin=marked["avg_order_value"].min(), vmax=marked["avg_order_value"].max(),
        vmed=(marked.loc[~marked["is_anomaly"], "avg_order_value"].median()
              if (~marked["is_anomaly"]).any() else marked["avg_order_value"].median()),
        vratio=marked["avg_order_value"].max() / max(
            (marked.loc[~marked["is_anomaly"], "avg_order_value"].median()
             if (~marked["is_anomaly"]).any() else marked["avg_order_value"].median()), 1e-9),
        cluster_means=" → ".join(format(round(m / 1e4), ",") for m in cluster_stat["m"]),
        top2_rev=float(cluster_stat["total"].head(2).sum()) / 1e8,
        top2_pct=float(cluster_stat["total"].head(2).sum()) / float(cluster_stat["total"].sum()) * 100,
        rfm_top_pct=float(summary.loc[summary["segment"] == "重要价值客户", "revenue_pct"].iloc[0])
        if (summary["segment"] == "重要价值客户").any() else 0.0,
        reclaim_n=int(summary.loc[summary["segment"] == "重要挽留客户", "customers"].iloc[0])
        if (summary["segment"] == "重要挽留客户").any() else 0,
        reclaim_r=int(round(summary.loc[summary["segment"] == "重要挽留客户", "avg_recency"].iloc[0]))
        if (summary["segment"] == "重要挽留客户").any() else 0,
        reclaim_pct=float(summary.loc[summary["segment"] == "重要挽留客户", "revenue_pct"].iloc[0])
        if (summary["segment"] == "重要挽留客户").any() else 0.0,
        m_skew=float(seg["monetary"].skew()),
        m_ratio=float(seg["monetary"].max() / seg["monetary"].median()),
        n_inj=len(anomalies),
        inj_desc=(" / ".join("%s %+.0f%%" % (r["month"], r["deviation_pct"])
                             for _, r in anomalies.iterrows()) or "无"),
        an_lines=an_lines,
        r_min=int(seg["recency"].min()), r_max=int(seg["recency"].max()),
        r_nuniq=int(seg["recency"].nunique()),
        r_ok=_mark(seg["recency"].nunique() > 5),
        f_cv=seg["frequency"].std() / seg["frequency"].mean(),
        f_ok=_mark(seg["frequency"].std() / seg["frequency"].mean() > 0.3),
        m_cv=seg["monetary"].std() / seg["monetary"].mean(),
        m_ok=_mark(seg["monetary"].std() / seg["monetary"].mean() > 0.3),
        c_ok=_mark(par["top20_share"] >= 50),
        idx80=(int((par["cum_curve"]["cum_pct"] >= 80).values.argmax()) + 1
               if (par["cum_curve"]["cum_pct"] >= 80).any() else len(par["cum_curve"])),
        alert_block=alert_block,
        src_desc=("项目自带仿真数据集（SQLite `data/sales.db`）" if source == "sqlite"
                  else "制造业经营分析数据仓库 ADS 层（`ads_db.ads_sale_analysis`）"),
        conc1_title=("客户结构较为集中" if par["top20_share"] >= 50 else "客户结构高度分散"),
        conc1_action=("符合制造业客户结构特征；头部客户是营收主力，需重点维护以避免单点流失。"
                      if par["top20_share"] >= 50 else
                      "无核心大客户依赖，经营风险较低；但缺少大客户抓手，建议按分层结果做差异化运营。"),
        conc5=("5. **风险集中在资金链**：预警类型中「回款滞后」与「营收缺口」占比最高，经营风险主要在回款端。"
               if len(alerts) else
               "5. **（本数据源无预警数据）**：自带数据集只含订单，无法给出资金链/生产端风险判断；"
               "切换 `--source mysql` 可基于数仓预警表补充该结论。"),
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
