# Import project before running any scripts

import os
import sys

project_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, project_path)
