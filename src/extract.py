# -*- coding: utf-8 -*-
"""取数：支持两种数据源

  · sqlite（默认）：项目自带仿真数据 `data/sales.db` —— **clone 即可复现**，不依赖 MySQL
  · mysql        ：直连数仓 ADS 层 `ads_db.ads_sale_analysis`，体现与数仓工程的衔接

两种源都**只读**，不写回任何库。两条路径产出的 DataFrame 结构一致，
因此下游（RFM / KMeans / 异常检测 / 出图）完全不用关心数据来自哪里。

用法：
    from extract import fetch_all
    cust, monthly, alerts = fetch_all()                 # 默认 sqlite
    cust, monthly, alerts = fetch_all(source="mysql")   # 直连数仓
"""

import sqlite3
import pandas as pd
import pymysql

from config import DB_CONFIG, SQLITE_PATH


# ---------------- 内部：两种源各自的原始查询 ----------------

def _sqlite_daily():
    """SQLite 逐客户逐日聚合。"""
    with sqlite3.connect(SQLITE_PATH) as conn:
        return pd.read_sql("""
            SELECT customer_id,
                   MAX(customer_name)  AS customer_name,
                   MAX(region)         AS region,
                   order_date          AS stat_date,
                   COUNT(*)            AS order_cnt,
                   SUM(order_amt)      AS order_amt,
                   SUM(order_amt)      AS paid_amt,
                   SUM(return_amt)     AS return_amt
            FROM sales_order
            WHERE order_status NOT IN ('已取消', '已作废')
            GROUP BY customer_id, order_date
        """, conn)


def _mysql_daily():
    """数仓 ADS 层销售宽表。"""
    with pymysql.connect(**DB_CONFIG) as conn:
        return pd.read_sql("""
            SELECT customer_id,
                   MAX(customer_name)  AS customer_name,
                   MAX(region)         AS region,
                   stat_date,
                   SUM(order_cnt)      AS order_cnt,
                   SUM(order_amt)      AS order_amt,
                   SUM(paid_amt)       AS paid_amt,
                   SUM(return_amt)     AS return_amt
            FROM ads_db.ads_sale_analysis
            WHERE customer_id <> 'UNKNOWN'
            GROUP BY customer_id, stat_date
        """, conn)


# ---------------- 对外统一接口 ----------------

def fetch_customer_rfm(source="sqlite"):
    """客户级 R / F / M。

    R = 数据截止日 - 该客户最后一次下单日（天）
    F = 累计订单数
    M = 累计订单金额
    """
    daily = _sqlite_daily() if source == "sqlite" else _mysql_daily()
    daily["order_amt"] = daily["order_amt"].astype(float)
    daily["order_cnt"] = daily["order_cnt"].astype(int)
    max_date = daily["stat_date"].max()

    g = daily.groupby(["customer_id", "customer_name", "region"], as_index=False).agg(
        frequency=("order_cnt", "sum"),
        monetary=("order_amt", "sum"),
        last_date=("stat_date", "max"),
    )
    g["recency"] = (pd.to_datetime(max_date) - pd.to_datetime(g["last_date"])).dt.days
    return g.drop(columns=["last_date"])


def fetch_monthly(source="sqlite"):
    """月度销售汇总：营收、订单数、单均金额。"""
    daily = _sqlite_daily() if source == "sqlite" else _mysql_daily()
    daily["order_amt"] = daily["order_amt"].astype(float)
    daily["order_cnt"] = daily["order_cnt"].astype(int)
    daily["month"] = daily["stat_date"].astype(str).str.slice(0, 7)

    m = daily.groupby("month", as_index=False).agg(
        revenue=("order_amt", "sum"), orders=("order_cnt", "sum"))
    m["avg_order_value"] = m["revenue"] / m["orders"]
    return m.sort_values("month").reset_index(drop=True)


def fetch_alert_summary(source="sqlite"):
    """经营预警分布。

    sqlite 源没有预警表（自带数据集只造订单），返回空 DataFrame，下游会跳过相关章节。
    """
    if source == "sqlite":
        return pd.DataFrame(columns=["alert_type", "cnt"])
    with pymysql.connect(**DB_CONFIG) as conn:
        return pd.read_sql("""
            SELECT alert_type, COUNT(*) AS cnt
            FROM ads_db.ads_alert_warning
            GROUP BY alert_type ORDER BY cnt DESC
        """, conn)


def fetch_all(source="sqlite"):
    return fetch_customer_rfm(source), fetch_monthly(source), fetch_alert_summary(source)


if __name__ == "__main__":
    import sys
    src = sys.argv[1] if len(sys.argv) > 1 else "sqlite"
    c, m, a = fetch_all(src)
    print("数据源：%s" % src)
    print("客户 %d 家 | 月度 %d 个月" % (len(c), len(m)))
    print(c.head())
    print(m)
