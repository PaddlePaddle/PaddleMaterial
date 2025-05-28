import json
from collections import defaultdict

import pandas as pd


def analyze_c13_peaks(csv_path):
    # 读取CSV并自动转换JSON字段
    df = pd.read_csv(csv_path, converters={"tokenized_input": lambda x: json.loads(x)})

    # 初始化统计容器
    c13_stats = defaultdict(int)
    length_records = []

    # 遍历所有数据行
    ccc = 0
    for index, nmr_dict in enumerate(df["tokenized_input"]):
        # 获取13C NMR数据，默认为空列表
        c13_values = nmr_dict.get("13CNMR", [])
        if len(c13_values) > 40:
            ccc += 1
            print(df["smiles"][index])

        length_records.append(len(c13_values))

        # 统计每个位移值（保留2位小数）
        for val in c13_values:
            try:
                # 规范化为保留2位小数（化学位移常规精度）
                rounded_val = round(float(val), 2)
                c13_stats[rounded_val] += 1
            except (TypeError, ValueError):
                continue

    # 转换为排序后的DataFrame
    result_df = pd.DataFrame(
        sorted(c13_stats.items(), key=lambda x: x[0]),
        columns=["Chemical_Shift", "Count"],
    )
    # 保存结果
    result_df.to_csv("c13_shift_distribution.csv", index=False)

    length_result = [
        ["Max_length", max(length_records)],
        ["Min_length", min(length_records)],
    ]

    pd.DataFrame(length_result).to_csv(
        "c13_length_statistics.csv", index=False, header=None
    )

    print(ccc)

    return result_df, length_result


if __name__ == "__main__":
    # df,_ = analyze_c13_peaks("tokenized_dataset_N_new.csv")

    df, _ = analyze_c13_peaks("../tokenized_dataset_N_new20.csv")
    print()
