import paddle
import functools
from paddle._typing.device_like import PlaceLike

class PlaceEnv:
    """
    Class version of context manager, also supports decorator functionality
    """
    
    def __init__(self, place: PlaceLike):
        """
        Initialize PlaceEnv
        
        Args:
            place: device objects like paddle.CPUPlace() or paddle.CUDAPlace(0)
        """
        self.place = place
        self.original_device = None
    
    def __enter__(self):
        """Called when entering the context"""
        # Save current device setting
        self.original_device = paddle.get_device()
        paddle.set_device(self.place)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Called when exiting the context"""
        # Restore original device setting
        if self.original_device is not None:
            paddle.set_device(self.original_device)
    
    def __call__(self, func):
        """
        Allows instance to be used as a decorator
        
        Args:
            func: function to be decorated
        
        Returns:
            Decorated function
        """
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Use with statement to temporarily change device setting
            with self:
                return func(*args, **kwargs)
        return wrapper



# Usage example
if __name__ == "__main__":
    # Test with statement
    print("=== Testing with statement ===")
    print(f"Current device: {paddle.get_device()}")
    
    print("\n=== Testing class version with statement ===")
    with PlaceEnv(paddle.CPUPlace()):
        print(f"Device inside with block: {paddle.get_device()}")
        y = paddle.ones([2, 3])
        print(f"Created tensor: {y}")
    
    print(f"Device outside with block: {paddle.get_device()}")
    
    # Test decorator functionality
    print("\n=== Testing decorator functionality ===")
    
    @PlaceEnv(paddle.CPUPlace())
    def cpu_function():
        """This function will run on CPU"""
        print(f"Device inside function: {paddle.get_device()}")
        return paddle.rand([2, 2])
    
    # Check if GPU is available
    if paddle.device.cuda.device_count() > 0:
        @PlaceEnv(paddle.CUDAPlace(0))
        def gpu_function():
            """This function will run on GPU"""
            print(f"Device inside function: {paddle.get_device()}")
            return paddle.rand([2, 2])
    
    # Call decorated functions
    print("Calling cpu_function:")
    result_cpu = cpu_function()
    print(f"Device after function execution: {paddle.get_device()}")
    print(f"Result: {result_cpu}")
    
    if paddle.device.cuda.device_count() > 0:
        print("\nCalling gpu_function:")
        result_gpu = gpu_function()
        print(f"Device after function execution: {paddle.get_device()}")
        print(f"Result: {result_gpu}")
    
    print("\n=== Testing multiple nesting ===")
    print(f"Initial device: {paddle.get_device()}")
    
    with PlaceEnv(paddle.CPUPlace()):
        print(f"Device inside first with block: {paddle.get_device()}")
        
        if paddle.device.cuda.device_count() > 0:
            with PlaceEnv(paddle.CUDAPlace(0)):
                print(f"Device inside second with block: {paddle.get_device()}")
                z = paddle.rand([2, 2])
                print(f"Created tensor: {z}")
        
        print(f"Back to first with block device: {paddle.get_device()}")
    
    print(f"Final device: {paddle.get_device()}")