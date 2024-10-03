import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.optimize import least_squares
from supersmoother import SuperSmoother
from scipy.constants import epsilon_0
import wx

##### Parametros da janela
margin=10
MarginX=80
MarginY=25

##### Parâmetros do Transistor (Carregados na UI)
WStart=3000
LStart=1000

##### Critérios de parada para cálculo de beta e Vth
eps=1e-14
Max_It=100

##### Mostrar plot das regressões (debug, trava o calculo a cada plot. Alternativamente usar plt.draw(), mas pode crashar o PC por excesso de figuras)
##draw=False

##### Extrair Vth por Extrapolação Linear
VthLinex=True

##### Parâmetros regressão polinomial para redução de ruído
FitOrder=4

##### String de formatação da saída. Padrão = '+5.4e' (com sinal, 5 digitos totais, 4 digitos após o ponto, notação científica)
FStr='5.4e'

##### Definições de Modelos

def Yfunc(Id, gm):
    return Id/np.sqrt(gm)

def Xi_Model(p, VGt):
    return p[2]*np.power(VGt,-3)+p[1]*np.power(VGt, -2)+p[0]

def Xi_Model2(p, VGt, Vd):
    beta, eps, t2 = p
    return (1/(beta*Vd))*((1/(VGt+eps)**2)-t2) 

def Xi_ObjFun(p, VGt, Xi):
    return Xi_Model(p, VGt)-Xi

def Xi_ObjFun2(p, VGt, Xi, Vd):
    return Xi_Model2(p, VGt, Vd)-Xi

def Id_Model(Vg, Vth, Vd, beta, theta1, theta2):
    VGt=Vg-Vth-Vd/2
    return (beta*Vd*VGt)/(1+theta1*VGt+theta2*(VGt**2))

def Id_ObjFun(p, VGt, Id, Vd, beta):
    return Id_Model(p, VGt, Vd, beta)-Id

def YFuncExtraction(pathlist, W, L, t_ox, e_ox, Vd, VgName='Vg', IdName='Id', SaveIntermediary=False, SaveFinal=True, Interpolate=False):
    pathlist=[path.replace('\\', '/') for path in pathlist]
    logpath=f"{pathlist[0].rsplit('/', 1)[0]}/"
    COX=(e_ox*epsilon_0)/(t_ox*1e-9)

    logtime=datetime.now().strftime('%y%m%d %H%M%S')
    with open(logpath+f"Parameters {logtime}.csv", 'w') as myfile:
        print("Temp,VthLinEx,Vth,SS,mi_yf,theta1,theta2,mi_maxgm")
        print("K,V,V,cm2/(V.s),,,cm2/(V.s)")
        myfile.write("Temp,VthLinEx,Vth,SS,mi_yf,theta1,theta2,mi_maxgm\n")
        myfile.write("K,V,V,mV/dec,cm2/(V.s),,,cm2/(V.s)\n")
        
    for path in pathlist:
        ##### Carregando Dados

        try:
            df=pd.read_csv(path, header=[0, 1])
            
            Id=df[IdName][df[IdName].columns[0]].to_numpy()
            Vg=df[VgName][df[VgName].columns[0]].to_numpy()
        except:
            df=pd.read_csv(path)
            
            Id=df[IdName].to_numpy()
            Vg=df[VgName].to_numpy()
            
        gm=np.diff(Id)/np.diff(Vg)

        try: temp=path.rsplit(' - ', 1)[1].split(' K')[0]
        except: temp=path.rsplit(' - ', 1)[1]

        ##### Tratando p-type 
        if np.average(Id)<0:
            Id=-Id
            Vg=-Vg
            Vd=-Vd

        #Smooth gm e tomando índice de máximo gm
        model = SuperSmoother()
        model.fit(Vg[1:], gm)

        maxgm=np.argmax(model.predict(Vg))
        gmmax=np.max(model.predict(Vg))

        ##### Vth por extrapolação linear
        if VthLinex:
            m, b= np.polyfit(Vg[np.argmax(gm)-2:np.argmax(gm)+2], Id[np.argmax(gm)-2:np.argmax(gm)+2], 1)
            VthLinEx=-b/m+Vd/2

        ##### Cálculo Subthreshold Slope, menor SS discreto
        SS=np.min(np.diff(Vg)/np.clip(np.diff(np.log10(np.clip(Id, 1e-11, 1))), 1e-11, 1))
            
        ##### Pegando parametros somente para região de inversão (Vg > Vg[maxgm])
        Id_inv=Id[maxgm+1:]
        Vg_inv=Vg[maxgm+1:]
        gm_inv=gm[maxgm:]

        ##### Regressão polinomial para redução de ruído
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
        p=[-1e3, 1e3, 0]
        beta=0
        e=0

        ##### Iterando erro = Vth-Vth* tendendo a zero
        for i in range(Max_It): 
            VGt=Vg_inv-Vt[-1]-Vd/2
            
            res=least_squares(Xi_ObjFun, p, args=(VGt, Xi), ftol=1e-15, xtol=1e-10)

            p=res.x

            beta=1/(p[1]*Vd)
            e=(beta*Vd*p[2])/2
            
            Vt+=[Vt[-1]+e]
            if e < eps:
                break
        ##    print(Vt[-1])
        if SaveIntermediary:
            fig, ax=plt.subplots()
            ax.plot(VGt, Xi, '.')
            ax.plot(VGt, Xi_Model(p, VGt))
            fig.savefig(f"./XixVGt {temp}.png")

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
            ax.plot(VGt, t_eff, '.')
            ax.plot(VGt, Theta2*VGt+Theta1)
            fig.savefig(f"./FitTheta_Eff {temp}.png")

        ##### Cálculo delta Vth
        deltaVt=(np.sqrt(beta*Vd/gm_inv[0])-1-Theta1*VGt[0])/(2*Theta2*VGt[0])

        ##### Cálculo theta1 e theta2
        theta2=Theta2/(1-Theta2*(deltaVt**2))
        theta1=Theta1*(1+theta2*(deltaVt**2))+2*theta2*deltaVt

        ##### Cálculo mobilidade por Y-Funcion e maxgm
        miyf=beta*(L/W)/COX*1e4
        migm=gmmax*(L/W)/(Vd*COX)*1e4

        ##### Sobreposição Fitting Final
        Id_Final=(beta*Vd*(Vg[Vg > Vth]-Vth-Vd/2))/(1+Theta1*(Vg[Vg > Vth]-Vth-Vd/2)+Theta2*(Vg[Vg > Vth]-Vth-Vd/2)**2)

        if SaveFinal:
            fig, ax=plt.subplots()
            ax.plot(Vg, Id, 'xb')
            ax.plot(Vg[Vg > Vth], Id_Final, 'k')
            fig.savefig(f"./FitIdxVgs {temp}.png")

        with open(logpath+f"Parameters {logtime}.csv", 'a') as myfile:
            print(f"{temp},{format(VthLinEx, FStr)},{format(Vth, FStr)},{format(SS*1e3, FStr)},{format(miyf, FStr)},{format(theta1, FStr)},{format(theta2, FStr)},{format(migm, FStr)}")
            myfile.write(f"{temp},{format(VthLinEx, FStr)},{format(Vth, FStr)},{format(SS*1e3, FStr)},{format(miyf, FStr)},{format(theta1, FStr)},{format(theta2, FStr)},{format(migm, FStr)}\n")

class MyPanel(wx.Panel):
    def __init__(self, *args, **kwargs):
        wx.Panel.__init__(self, *args, **kwargs)
        X, Y = (10, 10)

        self.WBoxTx = wx.StaticText(self, label='W (nm):', pos=(X, Y))
        self.WBox  = wx.TextCtrl(self, value=str(WStart), pos=(X, Y+MarginY), size=(60,20))

        self.TBoxTx  = wx.StaticText(self, label='t_ox (nm):', pos=(X+MarginX, Y))
        self.TBox = wx.TextCtrl(self, value='4.2', pos=(X+MarginX, Y+MarginY), size=(60,20))

        self.LBoxTx = wx.StaticText(self, label='L (nm)', pos=(X, Y+2*MarginY))
        self.LBox = wx.TextCtrl(self, value=str(LStart), pos=(X, Y+3*MarginY), size=(60,20))

        self.eBoxTx = wx.StaticText(self, label='e_ox:', pos=(X+MarginX,  Y+2*MarginY))
        self.eBox = wx.TextCtrl(self, value='3.9', pos=(X+MarginX, Y+3*MarginY), size=(60,20))

        self.VdBoxTx = wx.StaticText(self, label='Vd (mV):', pos=(X+2*MarginX, Y))
        self.VdBox = wx.TextCtrl(self, value='25', pos=(X+2*MarginX, Y+MarginY), size=(60,20))

        self.nFinsBoxTx = wx.StaticText(self, label='n Fins', pos=(X+2*MarginX,  Y+2*MarginY))
        self.nFinsBox = wx.TextCtrl(self, value='1', pos=(X+2*MarginX, Y+3*MarginY), size=(60,20))

        self.VgBoxTx = wx.StaticText(self, label='Vg Column:', pos=(X+2*margin+3*MarginX,  Y))
        self.VgBox = wx.TextCtrl(self, value='Vg', pos=(X+2*margin+3*MarginX, Y+MarginY), size=(60,20))

        self.IdBoxTx = wx.StaticText(self, label='Id Column:', pos=(X+2*margin+3*MarginX,  Y+2*MarginY))
        self.IdBox = wx.TextCtrl(self, value='Id', pos=(X+2*margin+3*MarginX, Y+3*MarginY), size=(60,20))

        self.SaveIntermediary=wx.CheckBox(self, label='Save Intermediaries', pos=(X, Y+4*MarginY+margin))
        self.SaveIntermediary.SetValue(1)
        self.SaveFinalFits=wx.CheckBox(self, label='Save Id x Vgs Fit', pos=(X, Y+5*MarginY+margin))
        self.SaveFinalFits.SetValue(1)
        self.Interpolate=wx.CheckBox(self, label='Interpolate', pos=(X, Y+6*MarginY+margin))
        self.Interpolate.SetValue(1)

        self.Select=wx.Button(self, label='Select File(s):', pos=(X+1*MarginX+6*margin, Y+4*MarginY+margin), size=(150, 70))
        self.Select.Bind(wx.EVT_BUTTON, self.OnButton)
        

    def OnButton(self,e):
        dialog = wx.FileDialog(self, "Choose file(s)", "", "", "*.*", wx.FD_MULTIPLE)
        if dialog.ShowModal() == wx.ID_OK:
            YFuncExtraction(dialog.GetPaths(), float(self.WBox.GetValue()), float(self.LBox.GetValue()), float(self.TBox.GetValue()),
                            float(self.eBox.GetValue()), float(self.VdBox.GetValue())*1e-3, float(self.nFinsBox.GetValue()), self.VgBox.GetValue(),
                            self.IdBox.GetValue(), self.SaveIntermediary.GetValue(), self.SaveFinalFits.GetValue(), self.Interpolate.GetValue())

class MyFrame(wx.Frame):
    def __init__(self, *args, **kwargs):
        wx.Frame.__init__(self, None, wx.ID_ANY,
                          "Y-Function"
                          )
        MyPanel(self)
        self.SetSize(360,240)
        self.Show()

if __name__ == "__main__":
    app = wx.App()
    frame = MyFrame()
    app.MainLoop()
