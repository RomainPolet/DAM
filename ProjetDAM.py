from numpy import *
from timeit import default_timer as timer
import time
from numpy import *
from numpy.linalg import solve
from scipy.sparse import dok_matrix
from scipy.sparse.linalg import spsolve
from matplotlib import cm
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.animation as animation
import matplotlib as mpl
from matplotlib.gridspec import GridSpec




def myfunc(rpm, s, theta, thetaC, deltathetaC) :
    gamma = 1.3
    m = 0.001161 #[kg] ?
    Qtot = 2800e3 * m #[J]
    mp = 0.8 #[kg] ?
    mb = 1.5  #[kg] ?
    w = rpm*2*pi/60
    #Source : 1.4L Ford Focus

    D = 0.075970 #[m]
    R = 0.03825 #[m]
    t = 11
    beta = 3.2  #?
    L = beta*R
    Td = -thetaC*pi/180 #[rad]
    dT = deltathetaC*pi/180 #[rad]

    Vc = pi*pi*D*2*R/(4)

    def V(T):
        Tr = T #- (2 * pi)
        return Vc * (1 - cos(Tr) + beta - sqrt(beta * beta - sin(Tr) ** 2)) / 2 + Vc / (t - 1)

    def dV(T):
        Tr = T #- (2 * pi)
        return Vc * sin(Tr) * (1 + cos(Tr) / sqrt(beta * beta - sin(Tr) ** 2)) / 2

    def Q(T):
        Tr = T #- (2 * pi)
        try:
            res = zeros(len(Tr))
            for i in range(len(Tr)):
                if Td <= Tr[i] < Td + dT:
                    res[i] = Qtot * (1 - cos(pi * (Tr[i] - Td) / dT)) / 2
            return res

        except TypeError:
            if Td <= Tr < Td + dT:
                return Qtot * (1 - cos(pi * (Tr - Td) / dT)) / 2
            else:
                return 0

    def dQ(T):
        Tr = T #- (2 * pi)
        try:
            res = zeros(len(Tr))
            for i in range(len(Tr)):
                if Td <= Tr[i] < Td + dT:
                    res[i] = Qtot * pi * sin(pi * (Tr[i] - Td) / dT) / (2 * dT)
            return res
        except TypeError:
            if Td <= Tr < Td + dT:
                return Qtot * pi * sin(pi * (Tr - Td) / dT) / (2 * dT)
            else:
                return 0

    def dp(p, T):
        Tr = T #- (2 * pi)
        Vol = V(Tr)
        return (gamma - 1) * dQ(Tr) / Vol - gamma * p * dV(Tr) / Vol

    def RungeKutta(theta, n, s):
        h = (theta[-1]-theta[0]) / n
        p = zeros(n + 1)
        p[0] = s * 1e5  # [Pa] ? Pression à theta = -2pi
        for i in range(0, n):
            if (-2 * pi < theta[i + 1] < -pi) : #% (4 * pi) - 2 * pi
                p[i + 1] = s * 1e5
            elif (pi < theta[i + 1] < 2 * pi):  #% (4 * pi) - 2 * pi
                p[i + 1] = s * 1e5
            else:
                K1 = dp(p[i], theta[i])
                K2 = dp(p[i] + h * K1 / 2, theta[i] + h / 2)
                K3 = dp(p[i] + h * K2 / 2, theta[i] + h / 2)
                K4 = dp(p[i] + h * K3, theta[i]+ h)
                p[i + 1] = p[i] + h * (K1 + 2 * K2 + 2 * K3 + K4) / 6
        return p


    n = 100000
    theta = linspace(-2*pi, 2 * pi, n+1) #-(2*pi)
    pression = RungeKutta(theta, n,s)
    plt.plot(theta, V(theta), "-r", label="V")
    print("Vol min : ", min(V(theta)), "\nVol max : ", max(V(theta)))
    plt.show()
    plt.plot(theta, Q(theta)*1e-3, "-b")
    #plt.plot(theta, dQ(theta)*1e-3, "-r")
    plt.show()
    plt.plot(theta, pression*1e-5, "-b", label="Echapement")
    plt.plot(theta[where((theta >= -2*pi) & (theta < -pi))], pression[where((theta >= -2*pi) & (theta < -pi))]*1e-5, "y", label="Admission")
    plt.plot(theta[where((theta >= -pi) & (theta < Td))], pression[where((theta >= -pi) & (theta < Td))]*1e-5, "g", label="Compression")
    plt.plot(theta[where((theta >= Td) & (theta < Td+dT))], pression[where((theta > Td)&(theta < Td+dT))]*1e-5, "r", label="Combustion")
    plt.plot(theta[where((theta >= Td+dT) & (theta < pi))], pression[where((theta >= Td+dT) & (theta < pi))]*1e-5, "black",label="Détente")

    print("Pression max [bar] : ",max(pression)*1e-5, "\nPression min [bar] : ", min(pression)*1e-5,
          "\nPression admission : ", pression[0]*1e-5, "\nPression echappement : ", pression[-1]*1e-5)
    plt.legend()
    plt.xlabel("Angle [rad]")
    plt.ylabel("Pression [bar]")
    plt.show()
    plt.plot(V(theta), pression*1e-5, label="Echapement")  #%(4*pi)-2*pi
    plt.plot(V(theta[where((theta >= -2 * pi) & (theta < -pi))]),
             pression[where((theta >= -2 * pi) & (theta < -pi))] * 1e-5, "y", label="Admission")
    plt.plot(V(theta[where((theta >= -pi) & (theta < Td))]), pression[where((theta >= -pi) & (theta < Td))] * 1e-5, "g", label="Compression")
    plt.plot(V(theta[where((theta >= Td) & (theta < Td + dT))]), pression[where((theta > Td) & (theta < Td + dT))] * 1e-5,
             "r", label="Combustion")
    plt.plot(V(theta[where((theta >= Td + dT) & (theta < pi))]), pression[where((theta >= Td + dT) & (theta < pi))] * 1e-5,
             "black", label="Détente")
    plt.ylabel("Pression [bar]")
    plt.xlabel("Volume [m^3]")
    plt.legend()
    plt.show()
    Fpied = pi*D*D*pression/4 - mp*R*w*w*cos(theta)
    plt.plot(theta, Fpied, "b")
    Ftete = -pi*D*D*pression/4 + (mp+mb)*R*w*w*cos(theta)
    plt.plot(theta, Ftete, "-r")
    plt.plot(theta, Ftete+Fpied, "--")
    plt.show()


    sigma = 450e6; #[Pa]
    E = 200e9 #[Pa]
    lb = linspace(0.00001, 0.03, 10000) #[m]
    A = 11*lb*lb
    Ix = 419*(lb**4)/12
    Iy = 131*(lb**4)/12
    Feulx = pi*pi*E*Ix/(L*L)
    Fcritx = Feulx*A*sigma/(Feulx+A*sigma)
    Feuly = pi*pi*E*Iy/(0.5*0.5*L*L)
    Fcrity = Feuly*A*sigma/(Feuly+A*sigma)
    plt.plot(lb, Fcritx)
    plt.plot(lb, Fcrity, "-r")
    print(Fcritx[where((lb>0.00899) & (lb < 0.009))])
    print(Fcrity[where((lb>0.00899) & (lb < 0.009))])
    plt.scatter(lb[where((lb>0.00899) & (lb < 0.009))], Fcritx[where((lb>0.00899) & (lb < 0.009))])
    plt.show()





myfunc(3000,1,0, 20, 50)
