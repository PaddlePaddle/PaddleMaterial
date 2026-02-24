import paddle
import functools
from contextlib import contextmanager

@contextmanager
def place_env(place):
    """
    上下文管理器，用于临时设置PaddlePaddle的运行设备
    
    Args:
        place: paddle.CPUPlace() 或 paddle.CUDAPlace(0) 等设备对象
    
    用法:
        with place_env(paddle.CPUPlace()):
            # 这里的代码在CPU上运行
            x = paddle.rand([2, 3])
            print(x)
    
    @place_env(paddle.CUDAPlace(0))
    def train():
        # 这个函数在GPU上运行
        pass
    """
    # 保存当前的设备设置
    current_device = paddle.get_device()
    
    # 根据place类型设置设备
    if isinstance(place, paddle.CPUPlace):
        paddle.set_device('cpu')
    elif isinstance(place, paddle.CUDAPlace):
        # 获取GPU设备ID
        device_id = place.get_device_id()
        paddle.set_device(f'gpu:{device_id}')
    else:
        raise ValueError(f"不支持的place类型: {type(place)}")
    
    try:
        yield
    finally:
        # 恢复原来的设备设置
        paddle.set_device(current_device)


class PlaceEnv:
    """
    类版本的上下文管理器，也支持装饰器功能
    """
    
    def __init__(self, place):
        """
        初始化PlaceEnv
        
        Args:
            place: paddle.CPUPlace() 或 paddle.CUDAPlace(0) 等设备对象
        """
        self.place = place
        self.original_device = None
    
    def __enter__(self):
        """进入上下文时调用"""
        # 保存当前的设备设置
        self.original_device = paddle.get_device()
        
        # 根据place类型设置设备
        if isinstance(self.place, paddle.CPUPlace):
            paddle.set_device('cpu')
        elif isinstance(self.place, paddle.CUDAPlace):
            device_id = self.place.get_device_id()
            paddle.set_device(f'gpu:{device_id}')
        else:
            raise ValueError(f"不支持的place类型: {type(self.place)}")
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出上下文时调用"""
        # 恢复原来的设备设置
        if self.original_device is not None:
            paddle.set_device(self.original_device)
    
    def __call__(self, func):
        """
        使实例可以作为装饰器使用
        
        Args:
            func: 要装饰的函数
        
        Returns:
            装饰后的函数
        """
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 使用with语句来临时改变设备设置
            with self:
                return func(*args, **kwargs)
        return wrapper


# 为了兼容性，也可以保留函数版本的上下文管理器
@contextmanager
def with_place_env(place):
    """
    with_place_env的别名，与place_env功能相同
    """
    with place_env(place):
        yield


# 使用示例
if __name__ == "__main__":
    # 测试with语句
    print("=== 测试with语句 ===")
    print(f"当前设备: {paddle.get_device()}")
    
    with place_env(paddle.CPUPlace()):
        print(f"with块内设备: {paddle.get_device()}")
        x = paddle.rand([2, 3])
        print(f"创建的张量: {x}")
    
    print(f"with块外设备: {paddle.get_device()}")
    
    print("\n=== 测试类版本with语句 ===")
    with PlaceEnv(paddle.CPUPlace()):
        print(f"with块内设备: {paddle.get_device()}")
        y = paddle.ones([2, 3])
        print(f"创建的张量: {y}")
    
    print(f"with块外设备: {paddle.get_device()}")
    
    # 测试装饰器功能
    print("\n=== 测试装饰器功能 ===")
    
    @PlaceEnv(paddle.CPUPlace())
    def cpu_function():
        """这个函数会在CPU上运行"""
        print(f"函数内设备: {paddle.get_device()}")
        return paddle.rand([2, 2])
    
    # 检查是否有GPU可用
    if paddle.device.cuda.device_count() > 0:
        @PlaceEnv(paddle.CUDAPlace(0))
        def gpu_function():
            """这个函数会在GPU上运行"""
            print(f"函数内设备: {paddle.get_device()}")
            return paddle.rand([2, 2])
    
    # 调用装饰后的函数
    print("调用cpu_function:")
    result_cpu = cpu_function()
    print(f"函数执行后设备: {paddle.get_device()}")
    print(f"结果: {result_cpu}")
    
    if paddle.device.cuda.device_count() > 0:
        print("\n调用gpu_function:")
        result_gpu = gpu_function()
        print(f"函数执行后设备: {paddle.get_device()}")
        print(f"结果: {result_gpu}")
    
    print("\n=== 测试多层嵌套 ===")
    print(f"初始设备: {paddle.get_device()}")
    
    with PlaceEnv(paddle.CPUPlace()):
        print(f"第一层with内设备: {paddle.get_device()}")
        
        if paddle.device.cuda.device_count() > 0:
            with PlaceEnv(paddle.CUDAPlace(0)):
                print(f"第二层with内设备: {paddle.get_device()}")
                z = paddle.rand([2, 2])
                print(f"创建的张量: {z}")
        
        print(f"回到第一层with设备: {paddle.get_device()}")
    
    print(f"最终设备: {paddle.get_device()}")