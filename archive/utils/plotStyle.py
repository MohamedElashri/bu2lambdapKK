import ROOT as rt

def frameStyle(frame):
    
    frame.SetTitle("")
    # frame.GetYaxis().SetMaxDigits(3)
    # frame.GetXaxis().SetTitleOffset(0.0)
    frame.SetMaximum(frame.GetMaximum()*1.2)
    frame.GetYaxis().SetNoExponent()
    frame.GetYaxis().SetLabelSize(0.063)
    frame.GetYaxis().SetTitleOffset(0.8)

    frame.GetXaxis().SetTitleSize(0)
    frame.GetXaxis().SetLabelSize(0)

    return None

def pullStype(framePull):
    
    framePull.SetFillStyle(0)
    framePull.GetYaxis().SetLabelSize(0.15)
    framePull.GetYaxis().SetTitleOffset(0.4)

    framePull.GetYaxis().CenterTitle(1)
    framePull.GetYaxis().SetNdivisions(507)
    framePull.GetYaxis().SetNdivisions(3, rt.kTRUE)
    framePull.GetYaxis().SetTitle("Pull")
    framePull.GetYaxis().SetTitleSize(0.15)

    framePull.GetXaxis().SetTitleSize(0.2)
    framePull.GetXaxis().CenterTitle(1)
    framePull.GetXaxis().SetLabelSize(0.15)
    framePull.GetXaxis().SetTitleOffset(0.8)
    framePull.GetXaxis().SetNdivisions(507)
    framePull.SetMaximum(6)
    framePull.SetMinimum(-6)
    framePull.SetTitle("")

    return None
