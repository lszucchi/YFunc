import pandas as pd
import numpy as np
from os.path import splitext
import os
root='./transistormodule/'
from re import search

def MDMtoCSV(path):
        outpath, ext=splitext(path)
        if ext != '.mdm':
                raise RuntimeError('Filetype not .mdm')

        params=[]
        header=[]
        header2=[]

        with open(path) as file:
                try:
                        for measurement in range(20):

                            ##### Wait BEGIN_DB
                            for line in file:
                                if line == 'BEGIN_DB\n':
                                    break
                                
                            ##### Treat params                
                            for line in file:
                                if line == '\n':
                                    break
                                header2+=[y for y in [[x for x in line.split(' ') if x][1]] if y not in header2]
                                params+=[[x for x in line.split(' ') if x][-2]]
                            header+=[x.strip('#').replace(',', '') for x in file.readline().split(' ') if x][:3]

                            ##### Read until END_DB
                            for line in file:
                                if line == 'END_DB\n':
                                    break
                                if 'trace' not in locals():
                                    trace=[float(x) for x in line.split(' ')[:-1] if x][:3]
                                else:
                                    trace=np.vstack((trace, [float(x) for x in line.split(' ')[:-1] if x][:3]))
                                    
                            ##### INIT dataframe if needed
                            if 'df' not in locals():
                                df=trace
                                del trace
                                continue
                                
                            ##### Stack each trace into dataframe
                            df=np.hstack((df, trace))
                            del trace
                except:
                        ##### Fail to read = EOF
                        header=pd.MultiIndex.from_tuples(tuple(zip(header, params)), names=('Trace', '|'.join(header2)))

                        df=pd.DataFrame(data=df, index=range(len(df)), columns=header)
                        df.to_csv(outpath+'.csv')

for root, dirs, files in os.walk(root, topdown=False):
    
    for file in files:
            if file.endswith("mdm"):
                    MDMtoCSV(f"{root}/{file}")
                    os.remove(f"{root}/{file}")

