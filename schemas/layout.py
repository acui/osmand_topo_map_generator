from dataclasses import dataclass
import numpy as np

@dataclass
class Layout:
    paper_size: np.ndarray
    map_size: np.ndarray

    @property
    def left_margin(self) -> float:
        return (self.paper_size[0] - self.map_size[0]) / 2
    
    @property
    def top_margin(self) -> float:
        return (self.paper_size[1] - self.map_size[1]) / 3
    
    @property
    def right_margin(self) -> float:
        return self.paper_size[0] - self.left_margin - self.map_size[0]
    
    @property
    def bottom_margin(self) -> float:
        return self.paper_size[1] - self.top_margin - self.map_size[1]