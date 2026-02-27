from tqdm import tqdm
import time
import os;os.system("") #兼容windows

def hex_to_ansi(hex_color: str, background: bool = False) -> str:
    """
    将十六进制颜色转换为ANSI转义序列
    
    Args:
        hex_color: 十六进制颜色，如 '#dda0a0' 或 'dda0a0'
        background: True表示背景色，False表示前景色
    
    Returns:
        ANSI转义序列字符串，如 '\033[38;2;221;160;160m'
    
    Example:
        >>> print(f"{hex_to_ansi('#dda0a0')}Hello{hex_to_ansi('#000000')} World")
        >>> print(f"{hex_to_ansi('dda0a0', background=True)}背景色{hex_to_ansi.reset()}")
    """
    # 移除#号并转换为小写
    hex_color = hex_color.lower().lstrip('#')
    
    # 处理简写形式 (#fff -> ffffff)
    if len(hex_color) == 3:
        hex_color = ''.join([c * 2 for c in hex_color])
    
    # 转换为RGB值
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    
    # ANSI真彩色序列
    # 38;2;R;G;B 为前景色，48;2;R;G;B 为背景色
    code = 48 if background else 38
    return f'\033[{code};2;{r};{g};{b}m'

def rgb_to_ansi(r: int, g: int, b: int, background: bool = False) -> str:
    """RGB值直接转ANSI"""
    code = 48 if background else 38
    return f'\033[{code};2;{r};{g};{b}m'

# 重置颜色的ANSI码
hex_to_ansi.reset = '\033[0m'

class ColoredTqdm(tqdm):
    def __init__(self, *args, 
                 start_color=(221, 160, 160),  # RGB: #DDA0A0
                 end_color=(160, 221, 160),    # RGB: #A0DDA0
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.start_color = start_color
        self.end_color = end_color
    
    def get_current_color(self):
        progress = self.n / self.total if self.total > 0 else 0
        current_rgb = tuple(
            int(start + (end - start) * progress)
            for start, end in zip(self.start_color, self.end_color)
        )
        result = current_rgb[0] * 16 ** 4 \
               + current_rgb[1] * 16 ** 2 \
               + current_rgb[2] * 16 ** 0
        return "%06x" % result
    
    def update(self, n=1):
        super().update(n)
        # 使用Rich的真彩色支持
        style = hex_to_ansi(self.get_current_color())
        self.bar_format = f'{{l_bar}}{style}{{bar}}{hex_to_ansi.reset}{{r_bar}}'
        self.refresh()


if __name__ == "__main__":
    # 使用示例
    for i in ColoredTqdm(range(100), desc="🌈 彩虹渐变"):
        time.sleep(0.1)