import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from re import search
from datetime import datetime
from os import walk, rename, listdir
from os.path import splitext, isfile
from YFunc import YFuncExtraction

import tkinter as tk
from tkinter import filedialog

def GetDUTfromPath(path):
    # return path.replace('p', '.').replace('l', '')[path.find('_w')+2:].split('u_')[:-1]

    df=pd.read_csv("sky130.csv", header=[0, 1])
    m=search("/([0-9]*)_", path).group()[1:-1]
    n=search("_([0-9]*)_", path).group()[1:-1]
    return df[m][n]

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size': 14})
plt.rcParams['figure.constrained_layout.use'] = True

path="D:/Dropbox/Cryochip/Medidas/Sky130/transistormodule" 

now=datetime.now().strftime('%Y%m%d %H%M')

for root, dirs, files in walk(path, topdown=False):
    for file in files:
            if file.endswith("csv") and "IDVG" in file:
                    DUT, W, L, _ =file.rsplit('_', 3)
                    WL=[float(W[1:-1].replace('p', '.')), float(L[1:-1].replace('p', '.'))]
                    print(DUT, WL)

                    if not isfile(f"{path}/Parameters {DUT} {now}.log"):
                        with open(f"{path}/Parameters {DUT} {now}.log", 'w') as myfile:
                            myfile.write('W,L,LIN,Vth,SS1,SS2,migm,miyf,theta1,theta2,Rext1,Rext2,beta,errmax%\n')
                    
                    LIN, Vth, SS1, SS2, migm, miyf, theta1, theta2, Rext1, Rext2, beta, errmax=YFuncExtraction(f"{root}/{file}",4,WL, 4.148 if "3v3" in file else 11.6, 6, 0.1, 1, 'VG', 'ID')
                    print(f"{W},{L},{format(LIN, '1.3f')},{format(Vth, '1.3f')},{format(SS1,'6.2f')},{format(SS2,'6.2f')},{format(migm, '.4e')},{format(miyf, '.4e')},{format(theta1, '+.3e')},{format(theta2, '+.3e')},{format(Rext1, '.3e')},{format(Rext2, '.3e')},{format(beta, '.3e')},{format(errmax, '.3e')}")
                    with open(f"{path}/Parameters {DUT} {now}.log", 'a') as myfile:
                        myfile.write(f"{W},{L},{format(LIN, '1.3f')},{format(Vth, '1.3f')},{format(SS1,'6.2f')},{format(SS2,'6.2f')},{format(migm, '.4e')},{format(miyf, '.4e')},{format(theta1, '+.3e')},{format(theta2, '+.3e')},{format(Rext1, '.3e')},{format(Rext2, '.3e')},{format(beta, '.3e')},{format(errmax, '.3e')}\n")
