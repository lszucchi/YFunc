import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lmfit.models import Model
from datetime import datetime
from supersmoother import SuperSmoother
from scipy.constants import epsilon_0
from sekve.extractor.ss_extractor import extract_ss_cryo

##### Definições de Modelos

def CalcGM(Vg, Id):
    return np.gradient(Id, Vg, edge_order=2)

def Yfunc(Id, gm):
    return np.divide(Id, np.sqrt(gm))

def Xi_Model(VGt, a, b, c):
    out=c*np.power(VGt,-3)+b*np.power(VGt, -2)+a
    return out

def Id_Model(Vg, Vth, Vd, beta, theta1, theta2):
    VGt=Vg-Vth-Vd/2
    return (beta*Vd*VGt)/(1+theta1*VGt+theta2*(VGt**2))

def YFuncExtraction(path, temp, WL, t_ox, e_ox, Vd=None, nfins=1, VgName='Vg', IdName='Id', PlotIdxVgs=True, SaveIntermediary=False, SaveRs=False, SaveFinal=True, Interpolate=True):
    LIN, Vth, SS, migm, miyf, theta1, theta2, Rext1, Rext2, errmax = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0] 

    path=path.replace('\\', '/')
    logpath, filename=path.rsplit('/', 1)

    W=WL[0]
    L=WL[1]

    try:
        df=pd.read_csv(path, header=[0, 1])
        
        Id=df[IdName][df[IdName].columns[0]].to_numpy()
        Vg=df[VgName][df[VgName].columns[0]].to_numpy()
        if Vd==None:
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
    Id=np.clip(Id, np.min(Id[Id>0]), 1)

    ##### Smooth gm e tomando índice de máximo gm
    model = SuperSmoother()
    model.fit(Vg[1:], gm)

    maxgm=np.argmax(model.predict(Vg))
    gmmax=np.max(model.predict(Vg))

    ##### Id x Vgs tradicional

    try:
        Vgfit=Vg[maxgm-2:maxgm+2]
        Idfit=Id[maxgm-2:maxgm+2]
        
        m, b= np.polyfit(Vgfit, Idfit, 1)
        LIN=-b/m+Vd/2
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
        ax.set_title("$I_D$ x $V_{GS}$" +f" - $V_D$={int(np.around(Vd*1e3))} mV")
        ax.set_xlabel("$V_{GS}$ (V)")
        ax.set_ylabel(f"$I_D$ ({prefix[i]}A)")
        ax2.set_ylabel(f"$g_m$ ({prefix[j]}S)")
        ax.legend()
        prefix=['f','p','n','u','m','']

        fig.savefig(f"{logpath}/{filename.replace('csv', 'png')}")

    ##### Cálculo Subthreshold Slope, menor SS discreto. Tratamento pra impedir warning quando Id <= 0
    logId=np.log10(Id[Id>1e-9])
    VglogId=np.diff(Vg[Id>1e-9])/np.diff(logId)
    SS1=1e3*np.min(VglogId)

    SS2=extract_ss_cryo(v=Vg, i=Id, t=temp, w=W, l=L)

    try:
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
        COX=(e_ox*epsilon_0)/(t_ox*1e-9)
        miyf=beta*(L/W)/COX*1e4
        migm=gmmax*(L/W)/(Vd*COX)*1e4

        ##### Sobreposição Fitting Final
        Id_Final=(beta*Vd*(Vg[Vg > Vth]-Vth-Vd/2))/(1+Theta1*(Vg[Vg > Vth]-Vth-Vd/2)+Theta2*(Vg[Vg > Vth]-Vth-Vd/2)**2)
        maxarg=len(Id)-maxgm
        errmax=np.max(np.abs((Id_Final[-maxarg:]-Id[-maxarg:])/Id[-maxarg:]))*100

        if SaveFinal:
            fig, ax=plt.subplots()
            ax.plot(Vg, Id, 'xb', label="Experimental data")
            ax.plot(Vg[Vg > Vth], Id_Final, 'k', label="Model fit")
            ax.set_title("$I_D$ x $V_{GS}$" +f"")
            ax.set_xlabel("$V_{GS}$ (V)")
            ax.set_ylabel("$I_D$ (A)")
            ax.legend()
            fig.savefig(f"{logpath}/{filename.replace('.csv', 'IdFit.png')}")

        ##### Cálculo Rs
        Rt=Vd/Id_inv
        
        max_y=np.max(Y[Y<np.inf])
##        n=np.where(Y/max_y>0.05)

        Xfit=Y[Y/max_y>0.6]

        Yfit=Rt[Y/max_y>0.6]*Xfit

        p1=np.polyfit(Xfit, Yfit, 1)
        Rext1=p1[0]


        if SaveRs:
            fig, ax=plt.subplots()
            
            ax.plot(Y, Rt*Y, '.')
            ax.plot(Xfit, np.polyval(p1, Xfit), '--')
            ax.set_xlabel("Yf")
            ax.set_ylabel("Rd.Yf")
            
            fig.savefig(f"{logpath}/{filename.replace('.csv', 'RtYf x Yf.png')}")

        n=np.argmax(gm)
        Xfit=1/Y[Y/max_y>0.6]
        Yfit=Rt[Y/max_y>0.6]

        p2=np.polyfit(Xfit, Yfit, 1)
        Rext2=p2[1]

        if SaveRs:
            fig, ax=plt.subplots()
            
            ax.plot(1/Y, Rt, '.')
            ax.plot(Xfit, np.polyval(p2, Xfit), '--')
            ax.set_xlabel("Inv Yf")
            ax.set_ylabel("Rd")
            
            fig.savefig(f"{logpath}/{filename.replace('.csv', 'Rt x InvYf.png')}")

    except Exception as e: print(">>> ", e)

    plt.close('all')

    return LIN, Vth, SS1, SS2, migm, miyf, theta1, theta2, Rext1, Rext2, beta, errmax

def DiodeExtraction(path, temp, VfName="Vf", IfName="If"):
    try:
        df=pd.read_csv(path, header=[0, 1])
        
        If=df[IfName][df[IfName].columns[0]].to_numpy()
        Vf=df[VfName][df[VfName].columns[0]].to_numpy()
    except:
        df=pd.read_csv(path)
        
        If=df[IfName].to_numpy()
        Vf=df[VfName].to_numpy()

    log_if=np.log10(np.clip(If, 1e-10, 1))
    np.diff(log_if)/np.diff(Vf)
    
    fig, ax=plt.subplots()

    ax.plot(Vf, If)

    ax.set_title("If x Vf")
    
    ax.set_ylabel("$I_f$ (A)")
    ax.set_xlabel("$V_f$ (V)")

    plt.savefig(path.replace(".csv", ".png"))

    ax.set_yscale('log')

    fig.savefig(path.replace(".csv", " log.png"))

    fig2, ax2=plt.subplots()

    ax2.plot(Vf[1:], gm)

    ax2.set_title("gm x Vf")
    
    ax2.set_ylabel("$g_m$ (S)")
    ax2.set_xlabel("$V_f$ (V)")

    fig2.savefig(path.replace(".csv", " gm.png"))

def PlotVgs(path, sizex=640, draw=False):

    try:
        df=pd.read_csv(path, header=[0, 1])
        
        Id=df['Id'][df['Id'].columns[0]].to_numpy()
        Vg=df['Vg'][df['Vg'].columns[0]].to_numpy()
        Vd=float(df.columns[2][1])
    except:
        df=pd.read_csv(path)
        
        Id=df['Id'].to_numpy()
        Vg=df['Vg'].to_numpy()

    if np.average(Id) < 0:
        Vg=-Vg
        Vd=-Vd
        Id=-Id
        ptype=True

    gm=np.diff(Id)/np.diff(Vg)
    maxgm=np.argmax(gm)

    i=0
    j=0

    while np.max(np.abs(Id)) < 0.3:
        i+=1
        Id=Id*1e3

    while np.max(np.abs(gm)) < 0.3:
        j+=1
        gm=gm*1e3

    try:
        Vgfit=Vg[maxgm-2:maxgm+2]
        Idfit=Id[maxgm-2:maxgm+2]
        
        m, b= np.polyfit(Vgfit, Idfit, 1)
        LIN=-b/m+Vd/2
    except Exception as err:
        print(f">>> Error: {err}")
        LIN=0

    fig, ax=plt.subplots()

    prefix=['f','p','n','u','m','']

    ax.plot(Vg, Id, 'b', label=f'Vd={int(np.around(Vd*1e3))} mV')
    ax.set_ylim(bottom=0)

    if LIN != 0:
        Vgfit=np.linspace(-b/m, Vg[np.argmax(gm)])
        ax.plot(Vgfit, m*Vgfit+b, 'k', alpha=0.5, label="Vth LinFit")

    ax.set_title("$I_D$ x $V_{GS}$" +f" - $V_D$={Vd} mV")
    ax.set_xlabel("-$V_{GS}$ (V)" if 'ptype' in locals() else "-$V_{GS}$")
    ax.set_ylabel(f"-$I_D$ ({prefix[-i]}A)" if 'ptype' in locals() else f"$I_D$ ({prefix[-i]}A)")

    ax2=ax.twinx()
    ax2.plot(Vg[1:], gm, 'r--', label="$g_m$")
    ax2.set_ylim(bottom=0)

    ax2.set_ylabel(f"$g_m$ ({prefix[-j]}S)")
    fig.legend(loc='upper left', bbox_to_anchor=(0.12, 0.89))

    fig.savefig(f"{path.replace('csv', 'png')}")

    return np.around(LIN, 3)

def CalcIsSat(path, T):
    global e, k
    n, Ispec= 0, 0

    try:
        df=pd.read_csv(path, header=[0, 1])
        
        Id=df['Id'][df['Id'].columns[0]].to_numpy()
        Vg=df['Vg'][df['Vg'].columns[0]].to_numpy()
        Vd=float(df.columns[2][1])
    except:
        df=pd.read_csv(path)
        
        Id=df['Id'].to_numpy()
        Vg=df['Vg'].to_numpy()
    
    if np.average(Id) < 0:
        Vd=-Vd
        Vg=-Vg
        Id=-Id

    model = SuperSmoother()
    model.fit(Vg[1:], gm)

    maxgm=np.argmax(model.predict(Vg))
    gmmax=np.max(model.predict(Vg))

    PlotVgs(path)

    y=Id/(gm*k*T/e)
    x=Id
    
    # print(format(Ispec, '.3e'))
    n=np.nanmin(y[y>0])
    # print(format(n, '.3f'))

    fig = plt.figure()
    ax=plt.gca() 
    ax.set_yscale('log')
    ax.set_xscale('log')
    
    ax.plot(x, np.array([n for c in x]), '--k')
    try:
        m, b = np.polyfit(np.log10(x[-7:]), np.log10(y[-7:]), 1, w=np.sqrt(y[-7:]))
        
        Ispec=10**((-b)/m)
        ax.plot(x, y[-1]/(b * np.power(x[-1], m))*(b * np.power(x, m)), '--r')
    except Exception as err:
        print(f">>> Error: {err}")
    ax.plot(x, y, '.b')
    
    ax.set_ylabel('$I_D/(g_m.U_T)$ (log)')
    ax.set_xlabel('$I_D$ (log)')
    
    ax.set_ylim((1, 1.2*y[-1]))

    save_path=f"{path.rsplit('.',1)[0]}.IdGmUt.png"
    plt.savefig(save_path) 
    
    return n, Ispec
    
