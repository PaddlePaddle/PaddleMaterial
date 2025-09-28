import paddle

def rename_weights(input_path, output_path):
    # 加载权重文件
    weights = paddle.load(input_path)
    
    # 创建新字典并修改键名
    new_weights = {}
    for key in weights.keys():
        if key.startswith('text_encoder'):
            new_key = key.replace('text_encoder', 'spectrum_encoder', 1)
            new_weights[new_key] = weights[key]
        else:
            new_weights[key] = weights[key]
        
        # 保存修改后的权重
        paddle.save(new_weights, output_path)
    print(f"Weights renamed and saved to {output_path}")

if __name__ == "__main__":
    input_path = "./pretrained/step2_onlyH_best.pdparams"
    output_path = "./pretrained/DiffNMR_NMRNet_nless15_onlyH_best.pdparams"
    rename_weights(input_path, output_path)
