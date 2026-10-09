import numpy as np
import pandas as pd
import sklearn
import matplotlib
import seaborn as sns
import torch

print("NumPy:", np.__version__)
print("Pandas:", pd.__version__)
print("Scikit-learn:", sklearn.__version__)
print("Matplotlib:", matplotlib.__version__)
print("Seaborn:", sns.__version__)
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())