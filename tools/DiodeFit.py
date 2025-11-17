import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lmfit.models import Model
from lmfit import Parameters
from scipy.constants import k, e
from os import listdir

def Diode_Model(Vf, Is, Rs, n, T):
##    print(Is, Rs, n)
    Vt=k*T/e
    I=Is*(np.exp(Vf/(n*Vt))-1)
    for i in range(100):
        Iinit=I
        I=Is*(np.exp((Vf-Rs*I)/(n*Vt)))
        if np.max(np.abs((I-Iinit)/I)) < 1e-3:
##            print(np.max(np.abs((I-Iinit)/I)))
            return I
    return I

path='C:\\Users\\Zucchi-Note\\Dropbox\\Cryochip\\Medidas\\PreFix\\240621 TN3 TN12 TP1 DN1 DN2 CB1\\Cooldown\\DN1\\'
path=path.replace('\\', '/')


for file in listdir(path)[::-1]:
    T=float(file.split(' K')[0])
    try:
        df=pd.read_csv(path+file, header=[0, 1])
        
        If=df['If'][df['If'].columns[0]].to_numpy()
        Vf=df['Vf'][df['Vf'].columns[0]].to_numpy()
    except:
        df=pd.read_csv(path+file)
        
        If=df['If'].to_numpy()
        Vf=df['Vf'].to_numpy()

    If=If[np.where(Vf>0)]
    Vf=Vf[np.where(Vf>0)]


    IFit=If[If>1e-7]
    VFit=Vf[-len(IFit):]

    ##IFit=IFit[IFit<1e-4]
    ##VFit=VFit[:len(IFit)]

    ##VFit=VFit-IFit*30

    ItIf=np.array([np.trapz(IFit[:i+1], VFit[:i+1]) for i in range(len(VFit))])

    p=np.polyfit(IFit, ItIf, 2)

    Rsa=p[0]*2
    na=p[1]*e/(k*T)

    p=Parameters()

    p.add_many(('Is', 1e-12, True),
                ('Rs', Rsa, True, 0, None),
                ('n', na, True),
                ('T', T, False))

    gmodel=Model(Diode_Model)
    res=gmodel.fit(IFit, Vf=VFit, params=p, weights=np.sqrt(IFit))

    Is, Rs, n = (res.params['Is'].value, res.params['Rs'].value, res.params['n'].value)

    print(Is, Rs, n)
    plt.plot(Vf, If, 'x')

    x=np.linspace(VFit[0], VFit[-1])
    plt.plot(x, Diode_Model(x, Is, Rs, n, T))
    plt.text(plt.xlim()[1]*0.1, 0.9*plt.ylim()[1], f"Is={format(Is, '.2e')} A, n={format(n, '.2f')}, Rs={format(Rs, '.2f')} ohm")
    plt.savefig(file.replace('.csv', '.png'))
    plt.close()



