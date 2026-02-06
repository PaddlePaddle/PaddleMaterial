import paddle.nn as nn


class LoRALayer(nn.Layer):
    def __init__(
        self,
        in_features,
        out_features,
        rank=8,
        alpha=16,
        dropout=0.0,
    ):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        self.lora_A = self.create_parameter(
            shape=[in_features, rank],
            default_initializer=nn.initializer.KaimingUniform()
        )
        self.lora_B = self.create_parameter(
            shape=[rank, out_features],
            default_initializer=nn.initializer.Constant(0.0)
        )
        
        if dropout > 0.0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None
    
    def forward(self, x):
        if self.dropout is not None:
            x = self.dropout(x)
        return (x @ self.lora_A @ self.lora_B) * self.scaling


class LoRALinear(nn.Layer):
    def __init__(
        self,
        original_layer,
        rank=8,
        alpha=16,
        dropout=0.0,
    ):
        super().__init__()
        self.original_layer = original_layer
        self.original_layer.stop_gradient = True
        
        in_features = original_layer.weight.shape[0]
        out_features = original_layer.weight.shape[1]
        
        self.lora = LoRALayer(in_features, out_features, rank, alpha, dropout)
    
    def forward(self, x):
        result = self.original_layer(x)
        result = result + self.lora(x)
        return result


def apply_lora_to_linear(module, rank=8, alpha=16, dropout=0.0, target_modules=None):
    if target_modules is None:
        target_modules = ["q_proj", "v_proj", "k_proj", "o_proj", "fc1", "fc2"]
    
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            if any(target in name for target in target_modules):
                lora_layer = LoRALinear(child, rank, alpha, dropout)
                setattr(module, name, lora_layer)
        else:
            apply_lora_to_linear(child, rank, alpha, dropout, target_modules)
    
    return module


def get_lora_parameters(module):
    lora_params = []
    for name, param in module.named_parameters():
        if 'lora' in name.lower():
            lora_params.append(param)
    return lora_params


def merge_lora_weights(module):
    for name, child in module.named_children():
        if isinstance(child, LoRALinear):
            original_weight = child.original_layer.weight
            lora_weight = child.lora.lora_A @ child.lora.lora_B * child.lora.scaling
            merged_weight = original_weight + lora_weight
            
            merged_layer = nn.Linear(
                original_weight.shape[0],
                original_weight.shape[1],
                bias_attr=child.original_layer.bias is not None
            )
            merged_layer.weight.set_value(merged_weight)
            if child.original_layer.bias is not None:
                merged_layer.bias.set_value(child.original_layer.bias)
            
            setattr(module, name, merged_layer)
        else:
            merge_lora_weights(child)
    
    return module


def print_trainable_parameters(module):
    trainable_params = 0
    all_params = 0
    
    for param in module.parameters():
        all_params += param.numel()
        if not param.stop_gradient:
            trainable_params += param.numel()
    
    print(
        f"trainable params: {trainable_params:,} || "
        f"all params: {all_params:,} || "
        f"trainable%: {100 * trainable_params / all_params:.2f}%"
    )
