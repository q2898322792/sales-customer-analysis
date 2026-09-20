# -*- coding: utf-8 -*-
"""生成自带仿真销售数据 -> data/sales.db（SQLite）

为什么要自带造数：
  1. 与数仓项目彻底解耦 —— 不碰 MySQL，数仓项目零改动
  2. 项目可独立复现 —— 任何人 clone 下来跑 make_data.py + run_all.py 即可出报告，
     不依赖本地 MySQL 和数仓

数据设计（关键：让客户具备真实业务应有的**异质性**）：
  · 金额规模：幂律分布 w_i ∝ i^(-0.8)，使 Top20% 客户贡献约 60% 营收
    （真实业务通常如此；数仓项目的仿真数据是均匀分布，导致 RFM 分层无意义）
  · 购买活跃度：与规模正相关（大客户买得勤），并让底部客户逐步"沉睡"
    -> 这样 R（最近一次消费）才有区分度，RFM 三个维度才都有效

用法：
    python make_data.py                 # 默认生成到 data/sales.db
    python make_data.py --customers 300 --days 353 --target-orders 800000
"""

import os
import sys
import argparse
import sqlite3
import random
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

POWER = 0.8          # 幂律指数：Top20% 客户约占 60% 营收
SEED = 42


def build_customers(n, rng):
    """返回客户列表（按规模降序），每个含 scale / activity / last_active_gap。"""
    # 规模：幂律 w_i ∝ i^(-POWER)，再归一化
    ranks = np.arange(1, n + 1, dtype=float)
    scale = ranks ** (-POWER)
    scale = scale / scale.sum()

    customers = []
    for i in range(n):
        # 活跃度与规模正相关：前 20% 高频、中间 50% 中频、后 30% 低频（其中一部分沉睡）
        if i < int(n * 0.2):
            activity = 0.90
        elif i < int(n * 0.7):
            activity = 0.45
        else:
            activity = 0.10

        # 沉睡客户：最后 12% 的客户近 60~180 天不再下单 -> R 维度有区分度
        gap = 0
        if i >= int(n * 0.88):
            gap = int(rng.integers(60, 180))
            activity = 0.02

        name = "%s%s" % (SURNAME[i % len(SURNAME)], SUFFIX[(i // len(SURNAME)) % len(SUFFIX)])
        if i >= len(SURNAME) * len(SUFFIX):
            name = "%s(%d)" % (name, i // (len(SURNAME) * len(SUFFIX)) + 1)

        customers.append({
            "customer_id": "C%04d" % (i + 1),
            "customer_name": name,
            "region": REGIONS[i % len(REGIONS)],
            "scale": float(scale[i]),
            "activity": activity,
            "gap": gap,
        })
    return customers


def build_products(n, rng):
    return [{
        "product_id": "P%04d" % (i + 1),
        "product_name": "%s-%d" % (CATEGORY[i % len(CATEGORY)], i + 1),
    } for i in range(n)]


def generate(args):
    rng = np.random.default_rng(SEED)
    random.seed(SEED)

    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(SQLITE_PATH):
        os.remove(SQLITE_PATH)

    start = datetime.date.fromisoformat(args.start_date)
    days = [start + datetime.timedelta(days=i) for i in range(args.days)]
    end = days[-1]

    customers = build_customers(args.customers, rng)
    products = build_products(args.products, rng)
    print("  客户 %d 家 | 产品 %d 款 | 天数 %d（%s ~ %s）" % (len(customers), len(products), len(days), start, end))

    # 每家客户的订单数 ∝ 规模（幂律）→ 营收集中度由 scale 单独决定。
    # ⚠️ 不要再乘 activity：实测会二次放大集中度（Top20% 冲到 80%，目标是 60%）。
    #    activity / gap 只用于制造 R 维度的差异（谁还在买、谁已沉睡）。
    raw = np.array([c["scale"] for c in customers])
    raw = raw / raw.sum()
    per_customer_total = raw * args.target_orders          # 该客户全周期订单数

    conn = sqlite3.connect(SQLITE_PATH)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE sales_order (
            order_id       TEXT PRIMARY KEY,
            customer_id    TEXT,
            customer_name  TEXT,
            product_id     TEXT,
            product_name   TEXT,
            region         TEXT,
            order_date     TEXT,
            order_amt      REAL,
            order_status   TEXT,
            return_amt     REAL
        )
    """)
    cur.execute("CREATE INDEX idx_cust ON sales_order(customer_id)")
    cur.execute("CREATE INDEX idx_date ON sales_order(order_date)")

    rows = []
    oid = 0
    for ci, c in enumerate(customers):
        n_order = int(per_customer_total[ci])
        if n_order <= 0:
            continue

        # 该客户可下单的日期窗口（沉睡客户提前结束）
        last_day = len(days) - 1 - c["gap"]
        if last_day < 1:
            continue

        # 下单日期：在 [0, last_day] 上按活跃度加权随机（偏向近期）
        weights = np.linspace(0.6, 1.0, last_day + 1)
        weights = weights / weights.sum()
        pick_days = rng.choice(last_day + 1, size=n_order, p=weights)

        # 单均金额：与规模**弱相关**。
        # ⚠️ 不要让它正比于 scale —— 订单数已经 ∝ scale，若金额也 ∝ scale，
        #    营收就变成 ∝ scale²，集中度会失控（实测 Top20% 占到 97%）。
        base = rng.uniform(8000, 30000, size=n_order)
        amt = base * (0.9 + 0.2 * rng.random(n_order))

        for k in range(n_order):
            oid += 1
            d = days[int(pick_days[k])]
            p = products[int(rng.integers(0, len(products)))]
            st = ORDER_STATUS[int(rng.integers(0, len(ORDER_STATUS)))]
            ret = float(amt[k]) * float(rng.choice([0, 0, 0, 0, 0.05, 0.15])) if st == "已取消" else 0.0
            # 刻意植入一处已知异常：2026-06 单均金额抬升 ~120%（模拟促销/集中大单）。
            # 目的是给「异常检测」一个可验证的样本 —— 相当于检测方法的 fixture，
            # 在 README 中明确说明，不算数据造假。
            if d.year == 2026 and d.month == 6:
                amt[k] *= 2.2
            rows.append(("O%08d" % oid, c["customer_id"], c["customer_name"], p["product_id"],
                         p["product_name"], c["region"], d.isoformat(), round(float(amt[k]), 2), st, round(ret, 2)))

        if len(rows) >= 200000:
            cur.executemany("INSERT INTO sales_order VALUES (?,?,?,?,?,?,?,?,?,?)", rows)
            conn.commit()
            print("    已写入 %s 行 …" % format(oid, ","))
            rows = []

    if rows:
        cur.executemany("INSERT INTO sales_order VALUES (?,?,?,?,?,?,?,?,?,?)", rows)
        conn.commit()

    cur.execute("SELECT COUNT(*), SUM(order_amt) FROM sales_order")
    n, total = cur.fetchone()
    conn.close()

    print()
    print("  ✅ 生成完成：%s 行 | 总金额 %.2f 亿元" % (format(n, ","), total / 1e8))
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--customers", type=int, default=300)
    ap.add_argument("--products", type=int, default=201)
    ap.add_argument("--days", type=int, default=353)
    ap.add_argument("--start-date", default="2025-10-01")
    ap.add_argument("--target-orders", type=int, default=800000)
    args = ap.parse_args()

    print("=" * 60)
    print("生成自带仿真销售数据 -> %s" % SQLITE_PATH)
    print("=" * 60)
    generate(args)


if __name__ == "__main__":
    main()
