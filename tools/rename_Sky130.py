import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from os import walk, rename, listdir
from os.path import splitext
from re import search

import tkinter as tk
from tkinter import filedialog

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size': 14})
plt.rcParams['figure.constrained_layout.use'] = True

def GetDUTfromPath(path):
    # return path.replace('p', '.').replace('l', '')[path.find('_w')+2:].split('u_')[:-1]

    df=pd.read_csv("sky130.csv", header=[0, 1])
    m=search("/([0-9]*)_", path).group()[1:-1]
    n=search("_([0-9]*)_", path).group()[1:-1]
    return df[m][n]

path="D:/Dropbox/Cryochip/Medidas/Sky130/transistormodule" 

now=datetime.now().strftime('%Y%m%d %H%M')

for root, dirs, files in walk(path, topdown=False):
    for file in files:
            if file.endswith("csv") and "IDVD" in file:
                    W, L, DUT=GetDUTfromPath(f"{root}/{file}")
                    WL=[float(W), float(L)]
                    print(DUT, W, L)
                    if len(W)==1: W=f"{W}.0"
                    if len(L)==1: L=f"{L}.0"
                    rename(f"{root}/{file}", f"{root}/{DUT}_w{W.replace('.', 'p')}u_l{L.replace('.', 'p')}u_IDVD.csv")
