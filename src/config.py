# -*- coding: utf-8 -*-
"""配置：数据库连接、路径、matplotlib 中文字体

数据库连接一律读环境变量，不写死密码（与数仓项目的约定保持一致）。
"""

import os
import matplotlib

matplotlib.use("Agg")            # 无 GUI 环境直接出图

DB_CONFIG = {
    "host": os.environ.get("DB_HOST", "127.0.0.1"),
    "port": int(os.environ.get("DB_PORT", "3306")),
    "user": os.environ.get("DB_USER", "root"),
    "password": os.environ.get("DB_PASSWORD", "root"),
    "charset": "utf8mb4",
}

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
FIG_DIR = os.path.join(OUTPUT_DIR, "figures")

for d in (OUTPUT_DIR, FIG_DIR):
    os.makedirs(d, exist_ok=True)

RESULT_CSV = os.path.join(OUTPUT_DIR, "customer_segment_result.csv")
MONTH_CSV = os.path.join(OUTPUT_DIR, "monthly_sales.csv")
REPORT_MD = os.path.join(OUTPUT_DIR, "销售与客户专题分析报告.md")


def setup_matplotlib():
    """中文字体配置。

    注意两点（数仓项目踩过的坑，这里沿用）：
      · 字体用 SimHei（GB2312 系），不要用字形更全的字体，否则离线预览暴露不出缺字问题
      · 必须关掉 unicode_minus：SimHei 没有 U+2212（真减号）字形，负号会显示成方框
    """
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    return plt
