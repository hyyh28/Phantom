#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import pandas as pd
import numpy as np


def detect_all_period_anomalies(df, date_col, family_col, sales_col, zscore_threshold=2.0):
    """
    基于“同品类的全局平均值与标准差”来检测整个周期内的销售异常。

    Parameters
    ----------
    df : pd.DataFrame
        输入的销售数据表。
    date_col : str
        数据中日期列的列名（如 'date'）。
    family_col : str
        商品品类列的列名（如 'family'）。
    sales_col : str
        销售量或销售额列的列名（如 'sales'）。
    zscore_threshold : float, default 2.0
        超过该 Z 分数阈值则视为异常。

    Returns
    -------
    pd.DataFrame
        包含 [date, family, total_sales, zscore, is_anomaly] 的结果表。
    """

    # 1. 确保日期列是 datetime 类型
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df[date_col] = pd.to_datetime(df[date_col])

    # 2. 以 "日期 + 品类" 为颗粒度做聚合，计算每天每个品类的销售总和
    #    如果数据里还需要细分到不同门店 (store_nbr)，也可在 groupby() 中添加 'store_nbr'
    grouped = (
        df.groupby([date_col, family_col], as_index=False)[sales_col]
        .sum()
        .rename(columns={sales_col: 'total_sales'})
    )
    # 此时 grouped 的列为 ['date', 'family', 'total_sales']

    # 3. 计算每个品类在全时段中的均值和标准差，用于 z-score
    #    这里的“全时段”是指 CSV 文件所覆盖的整个时间范围
    family_stats = (
        grouped.groupby(family_col)['total_sales']
        .agg(mean_val='mean', std_val='std')
        .reset_index()
        .set_index(family_col)
    )

    # 4. 合并 grouped 和 family_stats，计算 z-score
    merged = pd.merge(
        grouped,
        family_stats,
        how='left',
        left_on=family_col,
        right_index=True
    )
    # merged 现在包含: [date, family, total_sales, mean_val, std_val]

    # 5. 计算 z-score
    #    z = (当前销售 - 均值) / 标准差
    #    注意可能会有 std_val=0 的情况，这里加 1e-9 防止除 0
    merged['zscore'] = (merged['total_sales'] - merged['mean_val']) / (merged['std_val'] + 1e-9)

    # 6. 返回 is_anomaly 标记
    merged['is_anomaly'] = merged['zscore'].abs() > zscore_threshold

    # 7. 只保留必要的列，并按照日期排序
    result_cols = [date_col, family_col, 'total_sales', 'zscore', 'is_anomaly']
    result = merged[result_cols].sort_values(by=[date_col, family_col]).reset_index(drop=True)

    return result


if __name__ == "__main__":
    # ========== 读入示例 CSV 文件，下面的文件名和列名需与实际数据相匹配 ==========
    csv_file = "store_11_data.csv"
    df = pd.read_csv(csv_file)

    # ========== 调用函数检测整个周期内的异常 ==========
    anomalies_df = detect_all_period_anomalies(
        df=df,
        date_col='date',
        family_col='family',
        sales_col='sales',
        zscore_threshold=2.0  # 大于2个标准差则视为异常
    )

    # ========== 查看或导出结果 ==========
    # 筛选出确认为异常的记录
    anomalies_only = anomalies_df[anomalies_df['is_anomaly'] == True]

    print("=== 整个周期内的异常记录: ===")
    print(anomalies_only.head(30))  # 仅展示前30条

    # 如果需要导出
    anomalies_only.to_csv("all_period_anomalies.csv", index=False)