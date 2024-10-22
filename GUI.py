import wx
from YFunc import YFuncExtraction

##### Parametros da janela
margin=10
MarginX=80
MarginY=25

##### Parâmetros do Transistor (Carregados na UI)
WStart=3000
LStart=1000

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
            for path in dialog.GetPaths():
                YFuncExtraction(path, [float(self.WBox.GetValue()), float(self.LBox.GetValue())], float(self.TBox.GetValue()), float(self.eBox.GetValue()),
                                float(self.VdBox.GetValue())*1e-3, float(self.nFinsBox.GetValue()), self.VgBox.GetValue(), self.IdBox.GetValue(),
                                self.SaveIntermediary.GetValue(), self.SaveFinalFits.GetValue(), self.Interpolate.GetValue())

class MyFrame(wx.Frame):
    def __init__(self, *args, **kwargs):
        wx.Frame.__init__(self, None, wx.ID_ANY,
                          "Y-Function"
                          )
        MyPanel(self)
        self.SetSize(360,240)
        self.Show()

def StartApp():
    # if __name__ == "__main__":
       app = wx.App()
       frame = MyFrame()
       app.MainLoop()
