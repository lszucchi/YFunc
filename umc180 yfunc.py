from os import listdir, walk, rename
from os.path import isfile
from YFunc import YFuncExtraction, DiodeExtraction, GUI
from HPIB.HPT import Plot2P
from HPIB.DevParams import UMC
from datetime import datetime
from tkinter import filedialog
import pandas as pd
import matplotlib.pyplot as plt

path=filedialog.askdirectory(title='Choose folder')
print(path)

now=datetime.now().strftime('%Y%m%d %H%M')

def getpd(df, trace):
    return df[trace][df[trace].columns[0]].to_numpy()

for root, dirs, files in walk(path, topdown=False):
    
    for file in files:
        prefix=root.rsplit("/", 2)[-1]
        if file.endswith('csv'):
            meas, temp, _ =file.rsplit(' - ')
            
            if 't' in prefix.lower():            
                temp=float(temp.rsplit(' K')[0])
                
                if not isfile(f"{root}/Parameters {now}.log"):
                    with open(f"{root}/Parameters {now}.log", 'w') as myfile:
                        myfile.write('temp,LIN,Vth,SS1,SS2,migm,miyf,theta1,theta2,Rext1,Rext2,beta,errmax%\n')

                try:
                    LIN, Vth, SS1, SS2, migm, miyf, theta1, theta2, Rext1, Rext2, beta, errmax=YFuncExtraction(f"{root}/{file}",temp,UMC[int(prefix.split('T')[1].strip('NP'))], 4.2, 3.9, 5, 1, 'Vg', 'Id')
                    print(f"{format(temp, '7.3f')},{format(LIN, '1.3f')},{format(Vth, '1.3f')},{format(SS1,'6.2f')},{format(SS2,'6.2f')},{format(migm, '.4e')},{format(miyf, '.4e')},{format(theta1, '+.3e')},{format(theta2, '+.3e')},{format(Rext1, '.3e')},{format(Rext2, '.3e')},{format(beta, '.3e')},{format(errmax, '.3e')}")
                    with open(f"{root}/Parameters {now}.log", 'a') as myfile:
                        myfile.write(f"{format(temp, '7.3f')},{format(LIN, '1.3f')},{format(Vth, '1.3f')},{format(SS1,'6.2f')},{format(SS2,'6.2f')},{format(migm, '.4e')},{format(miyf, '.4e')},{format(theta1, '+.3e')},{format(theta2, '+.3e')},{format(Rext1, '.3e')},{format(Rext2, '.3e')},{format(beta, '.3e')},{format(errmax, '.3e')}\n")
                except:
                    LIN, Vth, SS1, SS2, migm, miyf, theta1, theta2, Rext1, Rext2, beta, errmax=YFuncExtraction(f"{root}/{file}",temp,UMC[int(prefix.split('T')[1].strip('NP'))], 4.2, 3.9, 5, 1, 'VG', 'ID')
                    print(f"{format(temp, '7.3f')},{format(LIN, '1.3f')},{format(Vth, '1.3f')},{format(SS1,'6.2f')},{format(SS2,'6.2f')},{format(migm, '.4e')},{format(miyf, '.4e')},{format(theta1, '+.3e')},{format(theta2, '+.3e')},{format(Rext1, '.3e')},{format(Rext2, '.3e')},{format(beta, '.3e')},{format(errmax, '.3e')}")
                    with open(f"{root}/Parameters {now}.log", 'a') as myfile:
                        myfile.write(f"{format(temp, '7.3f')},{format(LIN, '1.3f')},{format(Vth, '1.3f')},{format(SS1,'6.2f')},{format(SS2,'6.2f')},{format(migm, '.4e')},{format(miyf, '.4e')},{format(theta1, '+.3e')},{format(theta2, '+.3e')},{format(Rext1, '.3e')},{format(Rext2, '.3e')},{format(beta, '.3e')},{format(errmax, '.3e')}\n")
            if 'd' in prefix.lower():
                temp=float(temp.rsplit(' K')[0])
    ##                    rename(f"{root}/{file}", f"{root}/Diode - {format(temp, '07.3f')} K - 2509.csv")
                DiodeExtraction(f"{root}/{file}", temp)
                plt.close('all')
        
            if '4P' in file or '2P' in file:
                Res=Plot2P(f"{root}/{prefix}/{file}")
                plt.close('all')
                with open(f"{root}/{prefix}/Parameters{meas} {now}.log", 'a') as myfile:
                    myfile.write(f"{format(temp, '7.3f')},{format(Res, '7.3f') if Res > 1 else format(Res, '.4e')}\n")


print(path, "Done")
