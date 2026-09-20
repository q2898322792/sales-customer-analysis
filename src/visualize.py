# -*- coding: utf-8 -*-
"""可视化：产出分析图表（全部写到 output/figures/）"""

import os
import numpy as np

from config import FIG_DIR, setup_matplotlib

plt = setup_matplotlib()


def _save(fig, name):
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print("  ✅ 出图 %s" % name)
    return path


def plot_rfm_scatter(df, name="01_rfm_散点_最近消费vs消费金额.png"):
    """R（横轴，越左越近） vs M（纵轴），按八分层着色。"""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    segs = df["segment"].unique()
    cmap = plt.get_cmap("tab10")
    for i, s in enumerate(segs):
        d = df[df["segment"] == s]
        ax.scatter(d["recency"], d["monetary"] / 1e8, s=28, alpha=.75,
                   color=cmap(i % 10), label="%s (%d家)" % (s, len(d)))

    ax.set_xlabel("最近一次消费距今天数 R（天）")
    ax.set_ylabel("累计消费金额 M（亿元）")
    ax.set_title("客户 RFM 分布：最近消费 vs 消费金额")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=.3)
    return _save(fig, name)


def plot_segment_bar(summary, name="02_八分层_客户数与营收占比.png"):
    """八分层：客户数（柱） + 营收占比（折线，次轴）。"""
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    x = np.arange(len(summary))
    ax.bar(x, summary["customers"], color="#4C78A8", alpha=.85, label="客户数")
    ax.set_xticks(x)
    ax.set_xticklabels(summary["segment"], rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("客户数")
    ax.set_title("RFM 八分层：客户规模与营收贡献")

    ax2 = ax.twinx()
    ax2.plot(x, summary["revenue_pct"], "o-", color="#E45756", lw=2, label="营收占比")
    ax2.set_ylabel("营收占比 %")
    for i, v in enumerate(summary["revenue_pct"]):
        ax2.annotate("%.1f%%" % v, (i, v), textcoords="offset points", xytext=(0, 6),
                     fontsize=8, color="#E45756", ha="center")

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=9)
    ax.grid(alpha=.3, axis="y")
    return _save(fig, name)


def plot_pareto(cum_curve, name="03_帕累托_客户营收累计占比.png"):
    """客户营收累计占比曲线。"""
    fig, ax = plt.subplots(figsize=(9, 5))
    n = len(cum_curve)
    ax.bar(range(n), cum_curve["monetary"] / 1e8, color="#72B7B2", alpha=.8, label="单客户营收")
    ax.set_xlabel("客户（按营收降序）")
    ax.set_ylabel("营收（亿元）")

    ax2 = ax.twinx()
    ax2.plot(range(n), cum_curve["cum_pct"], color="#F58518", lw=2, label="累计占比")
    ax2.axhline(80, ls="--", c="grey", lw=1)
    ax2.set_ylabel("累计占比 %")

    ax.set_title("客户营收帕累托分布（共 %d 家）" % n)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=9)
    ax.grid(alpha=.3, axis="y")
    return _save(fig, name)


def plot_monthly_trend(marked, name="04_月度趋势_营收与单均金额异常.png"):
    """月度营收（柱） + 单均金额（折线），异常点红色标注。"""
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(marked))
    ax.bar(x, marked["revenue"] / 1e8, color="#54A24B", alpha=.8, label="月度营收")
    ax.set_xticks(x)
    ax.set_xticklabels(marked["month"], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("营收（亿元）")
    ax.set_title("月度营收与单均金额（红点为异常月份）")

    ax2 = ax.twinx()
    ax2.plot(x, marked["avg_order_value"], "o-", color="#B279A2", lw=1.8, label="单均金额")
    ax2.set_ylabel("单均金额（元）")
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, p: "{:,.0f}".format(v)))

    ab = marked[marked["is_anomaly"]]
    for i, r in ab.iterrows():
        idx = marked.index.get_loc(i)
        ax2.scatter([idx], [r["avg_order_value"]], s=110, facecolors="none",
                    edgecolors="#E45756", lw=2.2, zorder=5)
        ax2.annotate("%s\n%s" % (r["month"], "偏离%+.0f%%" % r["deviation_pct"]),
                     (idx, r["avg_order_value"]), textcoords="offset points",
                     xytext=(0, 14), fontsize=8, color="#E45756", ha="center")

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=9)
    ax.grid(alpha=.3, axis="y")
    return _save(fig, name)


def plot_clusters(df, name="05_KMeans聚类_频次vs金额.png"):
    """KMeans 聚类结果：F vs M 散点，按簇命名着色。"""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    cmap = plt.get_cmap("Set2")
    for i, s in enumerate(sorted(df["cluster_name"].unique())):
        d = df[df["cluster_name"] == s]
        ax.scatter(d["frequency"], d["monetary"] / 1e8, s=28, alpha=.8,
                   color=cmap(i % 8), label="%s (%d家)" % (s, len(d)))

    ax.set_xlabel("消费频次 F（累计订单数）")
    ax.set_ylabel("累计消费金额 M（亿元）")
    ax.set_title("KMeans 聚类结果（基于标准化 R/F/M）")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=.3)
    return _save(fig, name)
