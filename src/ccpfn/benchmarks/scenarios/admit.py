# A synthetic dataset based on the simulated dataset generated in 
# https://github.com/CausalTeam/ADMIT/blob/main/data/dataset.py and 
# https://github.com/lushleaf/varying-coefficient-net-with-functional-tr/blob/main/data/simu1.py

import numpy as np
import pandas as pd

from .base import Scenario

class ADMIT(Scenario):
    """Fully-synthetic scenario adapted from 
    https://github.com/CausalTeam/ADMIT/blob/main/data/dataset.py and 
    https://github.com/lushleaf/varying-coefficient-net-with-functional-tr/blob/main/data/simu1.py. 
    """
    name = "ADMIT"