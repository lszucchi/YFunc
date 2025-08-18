import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from YFunc import Yfunc
from tkinter import filedialog

path1=filedialog.askopenfilename(title='Choose file')
path2=filedialog.askopenfilename(title='Choose file')


df1=pd.read_csv(path1)
df2=pd.read_csv(path2)

df=pd.concat((df1, df2), ignore_index=True)
df.sort_values(by=['temp'])

df.to_csv('2412 TP7 Parameters.csv')

fig, ax = plt.subplots()

ax.plot(df['temp'], df['Rext1'], 'b.', label="Slope RdYf x Yf")
ax.plot(df['temp'], df['Rext2'], 'r.', label="Intercept Rd x 1/Yf")

ax.set_ylabel("Rext ($\Omega$)")
ax.set_xlabel("Temp (K)")

ax.set_title("Rext x Temp")

ax.legend()
fig.savefig("2412 TP7 RextxT.png")


