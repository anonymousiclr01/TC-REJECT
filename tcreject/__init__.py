"""TC-REJECT: open-set novelty detection for windowed multichannel time series.

One method, one configuration, every dataset:

    from tcreject import run, DEFAULT
    from tcreject.datasets import synthetic
    res = run(synthetic.make_dataset(seed=11), DEFAULT)
"""
from .config import Config, DEFAULT, CHANNELS
from .pipeline import run

__all__ = ["Config", "DEFAULT", "CHANNELS", "run"]
__version__ = "1.0.0"
