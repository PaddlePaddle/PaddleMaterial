import pandas as pd
from collections import defaultdict
import json


def analyze_peaks(csv_path):
    # 读取CSV文件并自动转换JSON字符串为字典
    df = pd.read_csv(
        csv_path,
        converters={"tokenized_input": lambda x: json.loads(x)}
    )

    # 初始化四个统计容器
    mean_stats = defaultdict(int)
    delta_stats = defaultdict(int)
    split_stats = defaultdict(int)
    integral_stats = defaultdict(int)

    length_records = []

    # 遍历所有数据行
    for nmr_dict in df["tokenized_input"]:
        # 提取1HNMR数据，若不存在则跳过
        hnmr_data = nmr_dict.get("1HNMR", [])

        length_records.append(len(hnmr_data))

        for peak in hnmr_data:
            # 提取并统计各字段（保留3位小数处理浮点精度）
            if len(peak) >= 4:
                mean = round(peak[0], 2)  # 化学位移均值
                delta = round(peak[1], 2)  # 位移差值
                split_type = peak[2]  # 裂分类型
                integral = peak[3]  # 积分值

                # 更新统计
                mean_stats[mean] += 1
                delta_stats[delta] += 1
                split_stats[split_type] += 1
                integral_stats[integral] += 1

    # 生成排序后的DataFrame
    def create_sorted_df(counter):
        return pd.DataFrame(
            sorted(counter.items()),
            columns=["Value", "Count"]
        )

    # 创建并保存所有统计结果
    create_sorted_df(mean_stats).to_csv("mean_distribution.csv", index=False)
    create_sorted_df(delta_stats).to_csv("delta_distribution.csv", index=False)

    # 分类数据直接转换（保持出现顺序）
    pd.DataFrame(split_stats.items(), columns=["Type", "Count"]) \
        .to_csv("split_type_distribution.csv", index=False)

    pd.DataFrame(integral_stats.items(), columns=["Integral", "Count"]) \
        .to_csv("integral_distribution.csv", index=False)

    length_result = [['Max_length', max(length_records)],
                     ['Min_length', min(length_records)]]

    pd.DataFrame(length_result).to_csv('h1_length_statistics.csv', index=False, header=None)

if __name__ == "__main__":
    analyze_peaks("../tokenized_dataset_N_new25.csv")
