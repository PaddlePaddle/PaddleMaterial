rewards迁移情况

默认情况下，使用的是pymatgen

# 原版的alignn跳过迁移

根据原版 matinvent/rewards/calculators/alignn/README.md 的说明，需要使用特定的预训练ALIGGN模型，而且精度有限，只适用于低成本方案下的场景，故不再处理。

# 原版的syn_score

原版 syn_score 直接在代码仓库中内嵌了 model_pt的权重数据，体积很大。如有必要，自行下载放到对应位置。

