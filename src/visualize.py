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


def plot_rfm_scatter(df, name="01-rfm-scatter.png"):
    """R（横轴，越左越近） vs M（纵轴），按八分层着色。"""
    fig, ax = plt.subplots(figsize=(10.2, 5.8))
    cmap = plt.get_cmap("tab10")
    # 图例按客户数降序，重要类别排在前面
    order = df["segment"].value_counts().index.tolist()

    # 两个维度都是**右偏分布**（R 偏度 2.42、M 偏度 1.86），77% 的客户落在低值端。
    # 线性刻度下点必然全挤在左下角 —— 这不是数据问题，是刻度没配得上分布形态。
    # 处理：M 用 log10；R 用 log10(R+1) 手动压缩（R 含 0，不能直接取对数），
    #       再把刻度标签换回**真实天数**，保证读图语义不变。
    def cmp_r(v):
        return np.log10(np.asarray(v, dtype=float) + 1)

    for i, s in enumerate(order):
        d = df[df["segment"] == s]
        ax.scatter(cmp_r(d["recency"]), d["monetary"] / 1e4, s=32, alpha=.78,
                   color=cmap(i % 10), label="%s（%d 家）" % (s, len(d)))

    ax.set_yscale("log")
    ticks = [0, 1, 3, 10, 30, 60, 100, 200]
    ax.set_xticks(cmp_r(ticks))
    ax.set_xticklabels([str(t) for t in ticks])

    ax.set_xlabel("最近一次消费距今天数 R（天；刻度按对数压缩，标签为真实天数）")
    # 单位用「万元」而非「亿元」：本数据客户营收在百万级，用亿元会出现 0.02/0.08 这类难读的小数刻度
    ax.set_ylabel("累计消费金额 M（万元，对数刻度）")
    ax.set_title("客户 RFM 分布：最近消费 vs 消费金额")
    # 图例移到图外右侧：数据点密集在左下与右侧，图内任何位置都会遮挡
    ax.legend(fontsize=8.5, loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0)
    ax.grid(alpha=.3, which="both")
    return _save(fig, name)


def plot_segment_bar(summary, name="02-segment-bar.png"):
    """八分层：**横向条形图** —— 客户数（条长） + 营收占比（标注在条上）。

    改横向的原因：中文类别名较长，纵向柱状图必须旋转标签且仍然拥挤；
    横向条形图的类别名可以水平书写，一眼能读完。
    按营收占比排序，阅读顺序与重要性一致。
    """
    d = summary.sort_values("revenue_pct", ascending=True).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(9.5, 6))
    y = np.arange(len(d))

    ax.barh(y, d["customers"], color="#4C78A8", alpha=.85)
    ax.set_yticks(y)
    ax.set_yticklabels(d["segment"], fontsize=10)
    ax.set_xlabel("客户数")
    ax.set_xlim(0, d["customers"].max() * 1.38)
    ax.set_title("RFM 八分层：客户规模与营收贡献")

    # 在条上标注：客户数 + 营收占比（把原来那条断崖式折线改成直标，避免 86%→0.1% 的刻度压缩）
    for i, r in d.iterrows():
        ax.text(r["customers"] + d["customers"].max() * 0.02, i,
                "%d 家 · 营收 %.1f%%" % (r["customers"], r["revenue_pct"]),
                va="center", fontsize=9, color="#333333")

    ax.grid(alpha=.3, axis="x")
    return _save(fig, name)


def plot_pareto(par, name="03-pareto.png"):
    """客户营收帕累托：以**累计占比曲线**为主，标注关键读数。

    原版用「柱状 + 累计曲线」双轴：柱子被个别头部客户撑开、其余 298 根全被压成一条基线，
    视觉上等于噪音；而帕累托的核心信息本来就在累计曲线上 —— 所以去掉柱子，把关键读数直接标出来。
    """
    cum = par["cum_curve"]
    n = len(cum)
    x = np.arange(n)
    cum_pct = cum["cum_pct"].to_numpy()

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(x, cum_pct, color="#F58518", lw=2.4, label="营收累计占比")

    # 参考线：50% / 80%
    for lvl, col in [(50, "#54A24B"), (80, "#E45756")]:
        ax.axhline(lvl, ls="--", c=col, lw=1.2, alpha=.8)
        ax.text(n * 0.995, lvl + 1.5, "%d%%" % lvl, color=col, fontsize=9, ha="right")

    # 关键读数标注：Top10 / Top20% / 达到 80% 所需客户数
    def mark(cnt, label, dy):
        if cnt <= 0 or cnt > n:
            return
        yv = cum_pct[cnt - 1]
        ax.scatter([cnt - 1], [yv], s=60, color="#B279A2", zorder=5)
        ax.annotate("%s\n占 %.1f%%" % (label, yv), (cnt - 1, yv),
                    textcoords="offset points", xytext=(10, dy),
                    fontsize=9, color="#B279A2")

    mark(10, "Top10 客户", -34)
    mark(par["top20_count"], "Top20%%（%d 家）" % par["top20_count"], 12)

    idx80 = int(np.argmax(cum_pct >= 80)) + 1 if (cum_pct >= 80).any() else None
    if idx80:
        ax.annotate("第 %d 家达 80%%" % idx80, (idx80 - 1, cum_pct[idx80 - 1]),
                    textcoords="offset points", xytext=(14, -6), fontsize=9, color="#E45756")

    ax.set_xlabel("客户（按营收降序，共 %d 家）" % n)
    ax.set_ylabel("营收累计占比 %")
    ax.set_ylim(0, 105)
    ax.set_xlim(0, n - 1)
    ax.set_title("客户营收帕累托分布")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=.3)
    return _save(fig, name)


def plot_monthly_trend(marked, name="04-monthly-trend.png"):
    """月度营收（柱） + 单均金额（折线），异常点红色标注。"""
    fig, ax = plt.subplots(figsize=(10, 5.8))
    x = np.arange(len(marked))
    rev = marked["revenue"] / 1e4                     # 统一用万元，避免 0.1~0.8 的小数刻度
    ax.bar(x, rev, color="#54A24B", alpha=.8, label="月度营收")
    ax.set_xticks(x)
    ax.set_xticklabels(marked["month"], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("营收（万元）")
    # 顶部留 28% 空白：给异常标注腾位置，避免像上一版那样与标题重叠
    ax.set_ylim(0, rev.max() * 1.28)
    ax.set_title("月度营收与单均金额（红圈为异常月份）", pad=12)

    ax2 = ax.twinx()
    v = marked["avg_order_value"]
    ax2.plot(x, v, "o-", color="#B279A2", lw=1.8, label="单均金额")
    ax2.set_ylabel("单均金额（万元）")
    # 统一用「万元」并去掉千分位，与左轴的纯数字风格保持一致
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda val, p: "%.1f" % (val / 1e4)))
    ax2.set_ylim(v.min() * 0.75, v.max() * 1.10)

    for _, r in marked[marked["is_anomaly"]].iterrows():
        idx = marked.index.get_loc(r.name)
        yv = r["avg_order_value"]
        # 标注避让：点在高位 -> 放右下方；点在低位 -> 放右上方（避免压住柱子）
        if yv > v.max() * 0.8:
            xy, ha = (-6, -34), "right"
        elif yv < v.min() * 1.25:
            # 最低点：折线在该点两侧都向上，标注必须往右下拉，否则会压在线上
            xy, ha = (30, -22), "left"
        else:
            xy, ha = (8, 12), "left"
        ax2.scatter([idx], [yv], s=120, facecolors="none",
                    edgecolors="#E45756", lw=2.2, zorder=5)
        ax2.annotate("%s\n偏离 %+.0f%%" % (r["month"], r["deviation_pct"]),
                     (idx, yv), textcoords="offset points", xytext=xy,
                     fontsize=8.5, color="#E45756", ha=ha,
                     arrowprops=dict(arrowstyle="-", color="#E45756", lw=.9, alpha=.7))

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=9)
    ax.grid(alpha=.3, axis="y")
    return _save(fig, name)


def plot_clusters(df, name="05-kmeans-clusters.png"):
    """KMeans 聚类结果：F vs M 散点，按簇命名着色。"""
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    cmap = plt.get_cmap("Set2")

    # 图例按簇的客均营收降序（而不是字典序），与"高/中高/中/中低"的命名顺序一致
    order = (df.groupby("cluster_name")["monetary"].mean()
               .sort_values(ascending=False).index.tolist())
    for i, s in enumerate(order):
        d = df[df["cluster_name"] == s]
        ax.scatter(d["frequency"], d["monetary"] / 1e4, s=30, alpha=.8,
                   color=cmap(i % 8), label="%s（%d 家）" % (s, len(d)))
        # 画质心十字，缓解散点重叠时看不出簇边界的问题
        ax.scatter([d["frequency"].mean()], [d["monetary"].mean() / 1e4],
                   marker="X", s=170, edgecolors="white", lw=1.4,
                   color=cmap(i % 8), zorder=6)

    ax.set_xlabel("消费频次 F（累计订单数）")
    ax.set_ylabel("累计消费金额 M（万元）")
    ax.set_title("KMeans 聚类结果（基于标准化 R/F/M；✕ 为簇质心）")
    ax.legend(fontsize=8.5, loc="upper left")
    ax.grid(alpha=.3)
    return _save(fig, name)
