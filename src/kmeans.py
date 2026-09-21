# -*- coding: utf-8 -*-
"""KMeans 聚类（numpy 实现）

为什么不用 scikit-learn：本机未安装，且本项目希望保持依赖最小（pandas / numpy / matplotlib / pymysql 即可运行）。
实现要点：
  · k-means++ 初始化，避免随机初始化导致结果不稳
  · 固定随机种子 -> 结果可复现
  · 空簇处理：将最远样本重新赋给空簇
"""

import numpy as np


def _kmeans_plus_plus_init(X, k, rng):
    """k-means++ 初始化中心点。"""
    n = X.shape[0]
    centers = [X[rng.integers(n)]]
    for _ in range(1, k):
        dist = np.min(((X[:, None, :] - np.array(centers)[None, :, :]) ** 2).sum(axis=2), axis=1)
        probs = dist / dist.sum()
        centers.append(X[rng.choice(n, p=probs)])
    return np.array(centers)


def kmeans(X, k=4, max_iter=100, tol=1e-6, seed=42):
    """返回 (labels, centers, inertia)。"""
    X = np.asarray(X, dtype=float)
    rng = np.random.default_rng(seed)
    centers = _kmeans_plus_plus_init(X, k, rng)

    labels = np.zeros(X.shape[0], dtype=int)
    for _ in range(max_iter):
        # E 步：分配
        dist = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        new_labels = dist.argmin(axis=1)

        # 处理空簇：把距离最远的样本塞进去
        for j in range(k):
            if (new_labels == j).sum() == 0:
                far = dist.min(axis=1).argmax()
                new_labels[far] = j

        # M 步：更新中心
        new_centers = np.array([X[new_labels == j].mean(axis=0) if (new_labels == j).any() else centers[j]
                                for j in range(k)])

        shift = np.linalg.norm(new_centers - centers)
        labels, centers = new_labels, new_centers
        if shift < tol:
            break

    inertia = float((((X - centers[labels]) ** 2).sum(axis=1)).sum())
    return labels, centers, inertia


def auto_k(X, k_range=range(2, 9), seed=42):
    """用手肘法算各 k 的 inertia，供选 k 参考。"""
    return {k: kmeans(X, k=k, seed=seed)[2] for k in k_range}


def name_clusters(df, labels, value_col="monetary", extra_cols=("frequency",)):
    """按簇的均值给簇命名（高价值 / 中价值 / 低价值 ...）。

    依据：按 value_col 的簇均值降序，依次命名。
    """
    tmp = df.copy()
    tmp["_cluster"] = labels
    # 三个统计量都要：m=客均（用于排序命名）、total=簇合计、n=客户数。
    # ⚠️ 曾经只返回 m 却在报告里当「营收合计」用，导致合计/客均两列都错、命名顺序也乱。
    stat = tmp.groupby("_cluster").agg(
        m=(value_col, "mean"), total=(value_col, "sum"),
        f=(extra_cols[0], "mean"), n=(value_col, "count")
    ).sort_values("m", ascending=False)

    names = ["高价值客户群", "中高价值客户群", "中价值客户群", "中低价值客户群",
             "低价值客户群", "潜力培育客户群", "长尾客户群", "待激活客户群"]

    mapping = {cid: names[i] if i < len(names) else "客户群%d" % (i + 1)
               for i, cid in enumerate(stat.index)}
    tmp["cluster_name"] = tmp["_cluster"].map(mapping)
    return tmp.drop(columns=["_cluster"]), stat.reset_index(), mapping


if __name__ == "__main__":
    X = np.random.RandomState(0).rand(200, 3) * 10
    lab, cen, inertia = kmeans(X, k=4)
    print("labels:", np.bincount(lab), "inertia:", round(inertia, 3))
    print("手肘法:", {k: round(v, 1) for k, v in auto_k(X).items()})
