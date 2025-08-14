import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_hdf("events_a1a1.h5", key="df")
print(df.columns)
df.to_csv('a1a1.csv')
