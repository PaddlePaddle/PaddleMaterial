from tqdm import tqdm
import time
import os;os.system("") # Compatible with Windows

def hex_to_ansi(hex_color: str, background: bool = False) -> str:
    """
    Convert hexadecimal color to ANSI escape sequence
    
    Args:
        hex_color: Hexadecimal color, e.g., '#dda0a0' or 'dda0a0'
        background: True for background color, False for foreground color
    
    Returns:
        ANSI escape sequence string, e.g., '\033[38;2;221;160;160m'
    
    Example:
        >>> print(f"{hex_to_ansi('#dda0a0')}Hello{hex_to_ansi('#000000')} World")
        >>> print(f"{hex_to_ansi('dda0a0', background=True)}Background color{hex_to_ansi.reset()}")
    """
    # Remove # symbol and convert to lowercase
    hex_color = hex_color.lower().lstrip('#')
    
    # Handle shorthand form (#fff -> ffffff)
    if len(hex_color) == 3:
        hex_color = ''.join([c * 2 for c in hex_color])
    
    # Convert to RGB values
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    
    # ANSI true color sequence
    # 38;2;R;G;B for foreground, 48;2;R;G;B for background
    code = 48 if background else 38
    return f'\033[{code};2;{r};{g};{b}m'

def rgb_to_ansi(r: int, g: int, b: int, background: bool = False) -> str:
    """Convert RGB values directly to ANSI"""
    code = 48 if background else 38
    return f'\033[{code};2;{r};{g};{b}m'

# ANSI code to reset color
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

        if self.total is None:
            return "#FFFFFF" 

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
        style = hex_to_ansi(self.get_current_color())
        self.bar_format = f'{{l_bar}}{style}{{bar}}{hex_to_ansi.reset}{{r_bar}}'
        self.refresh()


if __name__ == "__main__":
    # Usage example
    for i in ColoredTqdm(range(10), desc="🌈 Rainbow gradient", leave = False):
        for j in ColoredTqdm(range(100), desc="🌈 Rainbow gradient", leave = False):
            time.sleep(0.01)