import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lmfit.models import Model
from datetime import datetime
from supersmoother import SuperSmoother
from scipy.constants import epsilon_0

##### Definições de Modelos

def Yfunc(Id, gm):
    return Id/np.sqrt(gm)

def Xi_Model(VGt, a, b, c):
    out=c*np.power(VGt,-3)+b*np.power(VGt, -2)+a
    return out

def Id_Model(Vg, Vth, Vd, beta, theta1, theta2):
    VGt=Vg-Vth-Vd/2
    return (beta*Vd*VGt)/(1+theta1*VGt+theta2*(VGt**2))

def YFuncExtraction(path, WL, t_ox, e_ox, Vd=25e-3, nfins=1, VgName='Vg', IdName='Id', ind=0, PlotIdxVgs=True, SaveIntermediary=False, SaveFinal=True, Interpolate=False):
    path=path.replace('\\', '/')
    logpath, filename=path.rsplit('/', 1)
    COX=(e_ox*epsilon_0)/(t_ox*1e-9)

    W=WL[0]
    L=WL[1]

    try:
        df=pd.read_csv(path, header=[0, 1])
        
        Id=df[IdName][df[IdName].columns[0]].to_numpy()
        Vg=df[VgName][df[VgName].columns[0]].to_numpy()
        Vd=float(df.columns[2][1])
    except:
        df=pd.read_csv(path)
        
        Id=df[IdName].to_numpy()
        Vg=df[VgName].to_numpy()
        
    gm=np.diff(Id)/np.diff(Vg)

    ##### Tratando p-type e nfins
    if np.average(Id)<0:
        Id=-Id
        Vg=-Vg
        Vd=-Vd

    Id=Id/nfins

    ##### Id x Vgs tradicional

    try:
        VGfit=Vg[np.argmax(gm)-2:np.argmax(gm)+2]
        IDfit=Id[np.argmax(gm)-2:np.argmax(gm)+2]
        
        m, b= np.polyfit(VGfit, IDfit, 1)
        LIN=-b/m+Vd/2
        fitID=m*Vg[:np.argmax(gm)]+b
    except:
        LIN=0

    if PlotIdxVgs:
        i=5
        j=5

        Y=Id
        while np.max(np.abs(Y)) < 0.3:
            i-=1
            Y=Y*1e3
    
        Y2=gm
        while np.max(np.abs(Y2)) < 0.3:
            j-=1
            Y2=Y2*1e3

        prefix=['f','p','n','u','m','']

        fig, ax=plt.subplots()
        ax.plot(Vg, Y, 'b', label=f'Vd={int(np.around(Vd*1e3))} mV')
        ax2=ax.twinx()
        ax2.plot(Vg[1:], Y2, 'r--', label="$g_m")
        ax.set_title("$I_D$ x $V_{GS}$" +f" - $V_D$={Vd} mV")
        ax.set_xlabel("$V_{GS}$ (V)")
        ax.set_ylabel(f"$I_D$ ({prefix[i]}A)")
        ax2.set_ylabel(f"$g_m$ ({prefix[j]}S)")
        ax.legend()
        prefix=['f','p','n','u','m','']

        fig.savefig(f"{logpath}/{filename.replace('csv', 'png')}")
    
    ##### Smooth gm e tomando índice de máximo gm
    model = SuperSmoother()
    model.fit(Vg[1:], gm)

    maxgm=np.argmax(model.predict(Vg))
    gmmax=np.max(model.predict(Vg))

    ##### Cálculo Subthreshold Slope, menor SS discreto
    VglogId=np.diff(Vg)/np.diff(np.log10(Id))
    SS=1e3*np.min(VglogId[VglogId > 0])
        
    ##### Pegando parametros somente para região de inversão (Vg > Vg[maxgm])
    Id_inv=Id[maxgm:]
    Vg_inv=Vg[maxgm:]
    gm_inv=gm[maxgm-1:]

    ##### Regressão polinomial para redução de ruído
    FitOrder=4
    
    if Interpolate:
        gm_inv_prefit=gm_inv
        gm_inv=np.polyval(np.polyfit(Vg_inv, gm_inv, FitOrder), Vg_inv)
        Id_inv_prefit=Id_inv
        Id_inv=np.polyval(np.polyfit(Vg_inv, Id_inv, FitOrder), Vg_inv)

    ##### Calulando Yfunc e Xi
    Y=Yfunc(Id_inv, gm_inv)
    Xi=(1/Y)**2

    ##### Vth inicial (Fleury, 2001)
    Vt=[Vg_inv[0]-Id_inv[0]/gm_inv[0]-Vd/2]

    ##### Parâmetros iniciais para a regressão
    p0=[-1e3, 1e3, 0]
    beta=0
    e=0

    ##### Critérios de parada para cálculo de beta e Vth
    eps=1e-14
    Max_It=100

    ##### Iterando erro = Vth-Vth* tendendo a zero
    for i in range(Max_It): 
        VGt=Vg_inv-Vt[-1]-Vd/2

        gmodel=Model(Xi_Model)
        res=gmodel.fit(Xi, VGt=VGt, a=p0[0], b=p0[1], c=p0[2])
        p=[res.params['a'].value, res.params['b'].value, res.params['c'].value]

        beta=1/(p[1]*Vd)
        e=(beta*Vd*p[2])/2
        
        Vt+=[Vt[-1]+e]
        if np.abs(e) < eps:
            break
            
    if SaveIntermediary:
        fig, ax=plt.subplots()
        ax.plot(VGt, Xi, '.', label="Experimental data")
        ax.plot(VGt, Xi_Model(VGt, p[0], p[1], p[2]), label="Model fit")
        ax.set_title("$\\xi$ x $V_{Gt}$")
        ax.set_xlabel("$V_{Gt}$ (V)")
        ax.set_ylabel(f"$\\xi$")
        ax.legend()
        fig.savefig(f"{logpath}/{filename.replace('.csv', '.xiFit.png')}")

    ##### Vth
    Vth=Vt[-1]
    Theta=beta*Vd*p[0]

    ##### Calculando VGt
    VGt=Vg_inv-Vth-Vd/2

    ##### Calculando theta_eff
    t_eff=((beta*Vd)/Id_inv)-(1/VGt)

    ##### Regressão Linear thetaeff x VGt
    Theta2, Theta1 = np.polyfit(VGt, t_eff, 1)

    if SaveIntermediary:
        fig, ax=plt.subplots()
        ax.plot(VGt, t_eff, '.', label="Experimental data")
        ax.plot(VGt, Theta2*VGt+Theta1, label="Model fit")
        ax.set_title("$\Theta_{Eff}$ x $V_{Gt}$")
        ax.set_xlabel("$V_{Gt}$ (V)")
        ax.set_ylabel("$\Theta_{Eff}$")
        ax.legend()
        fig.savefig(f"{logpath}/{filename.replace('.csv', 'ThetaFit.png')}")
    
    ##### Cálculo delta Vth
    deltaVt=(np.sqrt(beta*Vd/gmmax)-1-Theta1*VGt[0])/(2*Theta2*VGt[0])

    ##### Cálculo theta1 e theta2
    theta2=Theta2/(1-Theta2*(deltaVt**2))
    theta1=Theta1*(1+theta2*(deltaVt**2))+2*theta2*deltaVt

    ##### Cálculo mobilidade por Y-Funcion e maxgm
    miyf=beta*(L/W)/COX*1e4
    migm=gmmax*(L/W)/(Vd*COX)*1e4

    ##### Sobreposição Fitting Final
    Id_Final=(beta*Vd*(Vg[Vg > Vth]-Vth-Vd/2))/(1+Theta1*(Vg[Vg > Vth]-Vth-Vd/2)+Theta2*(Vg[Vg > Vth]-Vth-Vd/2)**2)
    maxarg=len(Id)-maxgm
    errmax=np.max(np.abs((Id_Final[-maxarg:]-Id[-maxarg:])/Id[-maxarg:]))*100
    plt.plot(range(0, maxarg), Id_Final[-maxarg:])
    plt.plot(range(0, maxarg), Id[-maxarg:])
    plt.show()

    if SaveFinal:
        fig, ax=plt.subplots()
        ax.plot(Vg, Id, 'xb', label="Experimental data")
        ax.plot(Vg[Vg > Vth], Id_Final, 'k', label="Model fit")
        ax.set_title("$I_D$ x $V_{GS}$" +f"")
        ax.set_xlabel("$V_{GS}$ (V)")
        ax.set_ylabel("$I_D$ (A)")
        ax.legend()
        fig.savefig(f"{logpath}/{filename.replace('.csv', 'IdFit.png')}")

    plt.close('all')

    return LIN, Vth, SS, migm, miyf, theta1, theta2, errmax

def WriteLog(root, name, now, LIN, Vth, SS, migm, miyf, theta1, theta2, errmax):
##### String de formatação da saída. Padrão = '+5.4e' (com sinal, 5 digitos totais, 4 digitos após o ponto, notação científica)
    FStr='+.5e'
    with open(f"{root}/parameters {now}.csv", 'a') as myfile:
        print(f"{format(LIN, FStr)},{format(Vth, FStr)},{format(SS, '3.1f')},{format(migm, FStr)},{format(miyf, FStr)},{format(theta1, FStr)},{format(theta2, FStr)},{format(errmax, FStr)}")
        myfile.write(f"{name},{format(LIN, FStr)},{format(Vth, FStr)},{format(SS, '3.1f')},{format(migm, FStr)},{format(miyf, FStr)},{format(theta1, FStr)},{format(theta2, FStr)},{format(errmax, FStr)}\n")