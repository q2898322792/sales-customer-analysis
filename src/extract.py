# -*- coding: utf-8 -*-
"""从数仓 ADS 层抽取分析所需数据

数据来源：ads_db.ads_sale_analysis（销售分析宽表，由数仓 step05 产出）。
本项目**只读**，不写回任何表 —— 与数仓项目解耦。

产出两份数据：
  1. 客户级 RFM 原始值（R 最近一次消费距今天数 / F 消费频次 / M 消费金额）
  2. 月度销售汇总（营收、订单数、单均金额）—— 用于趋势与异常分析
"""

import pandas as pd
import pymysql

from config import DB_CONFIG


def _connect():
    return pymysql.connect(**DB_CONFIG)


def fetch_customer_rfm():
    """客户级 R / F / M 原始值。

    R = 数据截止日 - 该客户最后一次下单日（天）
    F = 累计订单数
    M = 累计订单金额（元）
    """
    sql = """
        SELECT
            s.customer_id,
            MAX(s.customer_name)                        AS customer_name,
            MAX(s.region)                                AS region,
            SUM(s.order_cnt)                             AS frequency,
            SUM(s.order_amt)                             AS monetary,
            DATEDIFF(@v_max_date, MAX(s.stat_date))      AS recency
        FROM ads_db.ads_sale_analysis s
        CROSS JOIN (SELECT @v_max_date := MAX(stat_date) FROM ads_db.ads_sale_analysis) x
        WHERE s.customer_id <> 'UNKNOWN'
        GROUP BY s.customer_id
    """
    with _connect() as conn:
        df = pd.read_sql(sql, conn)
        # 用 pandas 重算一次更直观（避免 SQL 变量在部分客户端下的兼容性问题）
        df["monetary"] = df["monetary"].astype(float)
        df["frequency"] = df["frequency"].astype(int)
        df["recency"] = df["recency"].astype(int)
    return df


def fetch_monthly():
    """月度销售汇总：营收、订单数、单均金额。"""
    sql = """
        SELECT
            DATE_FORMAT(stat_date, '%Y-%m')     AS month,
            SUM(order_amt)                      AS revenue,
            SUM(order_cnt)                      AS orders,
            SUM(paid_amt)                       AS paid_amt,
            SUM(return_amt)                     AS return_amt
        FROM ads_db.ads_sale_analysis
        WHERE customer_id <> 'UNKNOWN'
        GROUP BY month
        ORDER BY month
    """
    with _connect() as conn:
        df = pd.read_sql(sql, conn)
    df["revenue"] = df["revenue"].astype(float)
    df["orders"] = df["orders"].astype(int)
    df["avg_order_value"] = df["revenue"] / df["orders"]
    return df


def fetch_alert_summary():
    """经营预警分布（用于结论部分的资金链判断）。"""
    sql = "SELECT alert_type, COUNT(*) AS cnt FROM ads_db.ads_alert_warning GROUP BY alert_type ORDER BY cnt DESC"
    with _connect() as conn:
        return pd.read_sql(sql, conn)


if __name__ == "__main__":
    rfm = fetch_customer_rfm()
    mon = fetch_monthly()
    print("客户数：%d" % len(rfm))
    print(rfm.head())
    print()
    print("月度：%d 个月" % len(mon))
    print(mon)
