import json
from collections import defaultdict

import pandas as pd


def j_statistics(nmr_dicts):
    # 初始化统计容器
    j_length_counter = defaultdict(int)
    j_value_counter = defaultdict(int)

    # 遍历所有NMR数据
    for nmr_dict in nmr_dicts:
        for peak in nmr_dict.get("1HNMR", []):
            j_values = peak[-1]

            # 统计耦合常数个数分布
            length = len(j_values)
            j_length_counter[length] += 1

            # 统计具体值出现次数
            for val in j_values:
                # 保留3位小数解决浮点精度问题
                # rounded_val = round(val, 3)
                rounded_val = round(val, 1)
                j_value_counter[rounded_val] += 1

    # 转换长度统计为排序后的DataFrame
    df_length = pd.DataFrame(
        sorted(j_length_counter.items(), key=lambda x: x[0]),
        columns=["Coupling_Constants_Count", "Peak_Count"],
    )

    # 转换值统计为排序后的DataFrame
    df_value = pd.DataFrame(
        sorted(j_value_counter.items(), key=lambda x: x[0]),
        columns=["J_Value", "Count"],
    )

    return df_length, df_value


# 使用示例
if __name__ == "__main__":
    df = pd.read_csv("../tokenized_dataset_N_new25.csv")
    nmrdict = df["tokenized_input"].apply(json.loads)
    df_length, df_value = j_statistics(nmrdict)

    # 保存结果
    df_length.to_csv("j_length_distribution.csv", index=False)
    df_value.to_csv("j_value_distribution.csv", index=False)

    # # 展示结果样例
    # print("耦合常数个数分布：")
    # print(df_length)
    # print("\nJ值频率分布：")
    # print(df_value)
