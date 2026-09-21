# -*- coding: utf-8 -*-
"""生成自带仿真销售数据 -> data/sales.db（SQLite）

为什么要自带造数：
  1. 与数仓项目彻底解耦 —— 不碰 MySQL，数仓项目零改动
  2. 项目可独立复现 —— 任何人 clone 下来跑 make_data.py + run_all.py 即可出报告

数据设计（v2 · 修复四个数据源缺陷）
------------------------------------------------------------------
设计目标：让客户在 R / F / M 三个维度上都有**真实且相互独立**的区分度。

  · 金额规模 scale：**混合分布**（85% 对数正态主体 + 15% 对数正态头部）
    -> 修复缺陷 B：原纯幂律使偏度达 8.3、max/中位差 55 倍，图表被极端值压平。
       改为混合分布后偏度约 2~3、max/中位约 8~12 倍，视觉上能看清分布。

  · 购买间隔 interval：**泊松过程**（指数分布间隔），而不是"每天按概率决定"
    -> 修复缺陷 C：原二值概率使 85% 客户 R=0、其余跳到 40+ 天，中间完全空档。
       改为泊松过程后 R 自然形成连续分布。

  · 客单价水平 price_level：**客户级对数正态，与购买频次弱相关**
    -> 修复缺陷 A：原设计 M ≈ F × 常数，实测 r(F,M)=0.999984，散点退化成一条直线、
       RFM 的 M 维度沦为 F 的重复。现让"低频客户也能高客单"（项目型采购）、
       "高频客户客单偏低"（耗材型采购），目标把 r(F,M) 降到 0.75~0.85。

  · 异常：植入**三种形态**（+60% / -55% / +120%）
    -> 修复缺陷 D：原来只有一处异常，折线除一个尖峰外全程平直。

规模控制：约 300 客户 × 353 天 × 4 万行订单，总营收 3 亿元以内。

用法：
    python make_data.py
"""

import os
import sys
import argparse
import sqlite3
import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATA_DIR, SQLITE_PATH                      # noqa: E402

REGIONS = ["华东", "华南", "华北", "华中", "西南", "西北", "东北"]
ORDER_STATUS = ["已完成"] * 88 + ["已发货"] * 6 + ["已取消"] * 4 + ["已作废"] * 2
CATEGORY = ["控制器", "传感器", "电机", "气动", "液压", "注塑", "冲压", "视觉", "通信", "电源"]

SURNAME = ["中兴", "海尔", "上汽", "鞍钢", "亿纬锂能", "三一", "徐工", "美的", "格力", "比亚迪",
           "宁德", "京东方", "海康", "大华", "汇川", "中车", "宝钢", "沙钢", "盛虹", "恒力"]
SUFFIX = ["科技集团", "股份公司", "智能科技有限公司", "重工集团", "电子有限公司", "制造有限公司",
          "新材料公司", "装备集团"]

# ---- 可调参数 ----------------------------------------------------------
SEED = 42
HEAD_RATIO = 0.15                   # 头部客户占比
HEAD_MU, HEAD_SIGMA = 2.0, 0.5      # 头部客户规模（对数正态）
BODY_MU, BODY_SIGMA = 0.0, 0.85     # 主体客户规模（对数正态）
INTERVAL_MEDIAN = 8.0               # 平均购买间隔中位数（天）—— 决定 R 的分布范围
INTERVAL_SIGMA = 1.2                # 间隔的离散度（越大 -> R 分布越连续）
PRICE_BASE = 6000.0                 # 单笔金额基准（元）
# 客户级客单价离散度 —— 决定 M 与 F 的解耦程度。
# 实测扫描：0.35 -> r(F,M)=0.86；0.45 -> 0.80；0.55 -> 0.70；0.95 -> 0.44（过度解耦）。
# 取 0.45，使 r(F,M) ≈ 0.80 —— 既保留"大客户贡献更多"的正相关，又让 M 拥有独立于 F 的信息。
PRICE_CUST_SIGMA = 0.45
PRICE_ORDER_SIGMA = 0.45            # 单笔金额离散度
SCALE_PRICE_EXP = 0.15              # 客单价对客户规模的弹性（弱正相关，避免共线）
SLEEP_RATIO = 0.12                  # 沉睡客户占比
ORDERS_PER_DAY_MAX = 2              # 每个购买日额外订单上限

# 植入的异常月份：(年, 月) -> 金额放大倍数。多形态，用于展示双向识别能力
INJECTED_ANOMALY = {(2026, 1): 1.60, (2026, 2): 0.45, (2026, 6): 2.20}


def build_customers(n, rng):
    """客户列表：规模（混合分布）+ 购买间隔（泊松过程参数）+ 是否沉睡。"""
    n_head = max(1, int(n * HEAD_RATIO))
    head = rng.lognormal(HEAD_MU, HEAD_SIGMA, n_head)
    body = rng.lognormal(BODY_MU, BODY_SIGMA, n - n_head)
    scale = np.concatenate([head, body])
    scale = np.sort(scale)[::-1]                 # 降序，便于按规模分配活跃度

    customers = []
    for i in range(n):
        # 购买间隔：规模大的客户买得更勤（间隔更短），但保留随机性
        base_interval = INTERVAL_MEDIAN * (1.0 / (1.0 + scale[i])) ** 0.35
        interval = float(np.clip(rng.lognormal(np.log(base_interval), INTERVAL_SIGMA), 1.0, 90.0))

        # 客单价水平：客户级对数正态，与规模**弱**正相关（弹性 0.15）
        price_level = PRICE_BASE * np.exp(rng.normal(0, PRICE_CUST_SIGMA)) * (scale[i] ** SCALE_PRICE_EXP)

        gap = 0
        if i >= int(n * (1 - SLEEP_RATIO)):      # 沉睡客户：近期不再下单
            gap = int(rng.integers(60, 180))

        name = "%s%s" % (SURNAME[i % len(SURNAME)], SUFFIX[(i // len(SURNAME)) % len(SUFFIX)])
        if i >= len(SURNAME) * len(SUFFIX):
            name = "%s(%d)" % (name, i // (len(SURNAME) * len(SUFFIX)) + 1)

        customers.append({
            "customer_id": "C%04d" % (i + 1),
            "customer_name": name,
            "region": REGIONS[i % len(REGIONS)],
            "scale": float(scale[i]),
            "interval": interval,
            "price_level": float(price_level),
            "gap": gap,
        })
    return customers


def purchase_days(rng, interval, total_days, stop_day):
    """泊松过程：按指数分布间隔生成购买日（0 基）。"""
    if stop_day <= 0:
        return []
    days, d = [], int(rng.integers(0, max(2, int(interval))))
    while d < stop_day:
        days.append(d)
        d += max(1, int(round(rng.exponential(interval))))
    return days


def generate(args):
    rng = np.random.default_rng(SEED)
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(SQLITE_PATH):
        os.remove(SQLITE_PATH)

    start = datetime.date.fromisoformat(args.start_date)
    days = [start + datetime.timedelta(days=i) for i in range(args.days)]

    customers = build_customers(args.customers, rng)
    products = [{"product_id": "P%04d" % (i + 1),
                 "product_name": "%s-%d" % (CATEGORY[i % len(CATEGORY)], i + 1)}
                for i in range(args.products)]
    print("  客户 %d 家 | 产品 %d 款 | 天数 %d（%s ~ %s）"
          % (len(customers), len(products), len(days), days[0], days[-1]))

    conn = sqlite3.connect(SQLITE_PATH)
    cur = conn.cursor()
    cur.execute("""CREATE TABLE sales_order (
        order_id TEXT PRIMARY KEY, customer_id TEXT, customer_name TEXT,
        product_id TEXT, product_name TEXT, region TEXT, order_date TEXT,
        order_amt REAL, order_status TEXT, return_amt REAL)""")
    cur.execute("CREATE INDEX idx_cust ON sales_order(customer_id)")
    cur.execute("CREATE INDEX idx_date ON sales_order(order_date)")

    rows, oid = [], 0
    for c in customers:
        stop = len(days) - c["gap"]
        for di in purchase_days(rng, c["interval"], len(days), stop):
            d = days[di]
            for _ in range(1 + int(rng.integers(0, ORDERS_PER_DAY_MAX + 1))):
                oid += 1
                amt = c["price_level"] * float(np.exp(rng.normal(0, PRICE_ORDER_SIGMA)))
                amt *= INJECTED_ANOMALY.get((d.year, d.month), 1.0)
                p = products[int(rng.integers(0, len(products)))]
                st = ORDER_STATUS[int(rng.integers(0, len(ORDER_STATUS)))]
                ret = round(amt * float(rng.choice([0, 0, 0, 0, 0.05, 0.15])), 2) if st == "已取消" else 0.0
                rows.append(("O%07d" % oid, c["customer_id"], c["customer_name"],
                             p["product_id"], p["product_name"], c["region"],
                             d.isoformat(), round(amt, 2), st, ret))
        if len(rows) >= 50000:
            cur.executemany("INSERT INTO sales_order VALUES (?,?,?,?,?,?,?,?,?,?)", rows)
            conn.commit()
            rows = []

    if rows:
        cur.executemany("INSERT INTO sales_order VALUES (?,?,?,?,?,?,?,?,?,?)", rows)
        conn.commit()

    cur.execute("""SELECT COUNT(*), SUM(order_amt) FROM sales_order
                   WHERE order_status NOT IN ('已取消','已作废')""")
    n, total = cur.fetchone()
    conn.close()
    print()
    print("  ✅ 生成完成：有效订单 %s 行 | 总营收 %.2f 亿元 | 单均 %.0f 元"
          % (format(n, ","), total / 1e8, total / n))
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--customers", type=int, default=300)
    ap.add_argument("--products", type=int, default=201)
    ap.add_argument("--days", type=int, default=353)
    ap.add_argument("--start-date", default="2025-10-01")
    args = ap.parse_args()

    print("=" * 64)
    print("生成自带仿真销售数据 -> %s" % SQLITE_PATH)
    print("=" * 64)
    generate(args)


if __name__ == "__main__":
    main()
