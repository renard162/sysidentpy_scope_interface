import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('./Temp/freq_correction.csv', sep=';')
data = data[data['base_freq']>3000]
model_params = np.polyfit(x=data['base_freq'], y=data['correction'], deg=1)
correction_model = np.poly1d(model_params)

print('fim')
