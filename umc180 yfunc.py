from os import listdir
from YFunc import YFuncExtraction, GUI
from HPIB.HPT import Plot2P
from HPIB.DevParams import UMC
from datetime import datetime
from tkinter import filedialog
import pandas as pd
import matplotlib.pyplot as plt

path=filedialog.askdirectory(title='Choose folder')

now=datetime.now().strftime('%Y%m%d %H%M')

root, prefix = path.rsplit('/', 1)

if 't' in prefix.lower():
    with open(f"{path}/Parameters {now}.log", 'w') as myfile:
        myfile.write('temp,LIN,Vth,SS,migm,miyf,theta1,theta2,Rext1,Rext2,errmax%\n')

if 'c' in prefix.lower():
    with open(f"{path}/Parameters2P {now}.log", 'w') as myfile:
        myfile.write('temp,res\n')

if 'cb' in prefix.lower():
    with open(f"{path}/Parameters4P {now}.log", 'w') as myfile:
        myfile.write('temp,res\n')

for file in listdir(path):
    if file.endswith('csv'):
        try:
            meas, temp, _ =file.rsplit(' - ')
        except:
            temp=float(file.rsplit(' K')[0])
        temp=float(temp)

        if 't' in prefix.lower():
##            try:
                LIN, Vth, SS, migm, miyf, theta1, theta2, Rext1, Rext2, errmax=YFuncExtraction(f"{path}/{file}",UMC[int(prefix[2:])], 4.2, 3.9, 5, 1, 'Vg', 'Id')
                with open(f"{path}/Parameters {now}.log", 'a') as myfile:
                    myfile.write(f"{format(temp, '7.3f')},{format(LIN, '1.3f')},{format(Vth, '1.3f')},{format(SS,'6.2f')},{format(migm, '.4e')},{format(miyf, '.4e')},{format(theta1, '+.3e')},{format(theta2, '+.3e')},{format(Rext1, '.3e')},{format(Rext2, '.3e')},{format(errmax, '.3e')}\n")
##            except: continue

        if 'd' in prefix.lower():
            try:
                TreatDiode()
            except: continue
    
        if '4P' in file or '2P' in file:
            Res=Plot2P(f"{root}/{prefix}/{file}")
            plt.close('all')
            with open(f"{root}/{prefix}/Parameters{meas} {now}.log", 'a') as myfile:
                myfile.write(f"{format(temp, '7.3f')},{format(Res, '7.3f') if Res > 1 else format(Res, '.4e')}\n")
