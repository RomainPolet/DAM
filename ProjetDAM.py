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


# Source : 1.4L Ford Focus
D = 0.075970 #[m]
R = 0.03825 #[m]
C = 2*R
tau = 11
t = tau
beta = 3.2  #?
L = beta*R
m = 0.0004044 #[kg] ?
Q = 2800e3
Qtot = Q * m #[J]
mpiston =  6.36*1e-2+1.46*1e-8*D**(3.84)#[kg] ?
mbielle = 0.4+4.17*1e-6*D**(2.632)  #[kg] ?
mp = mpiston
mb = mbielle
gamma = 1.3
cp_R = 1.3 / 0.3
Vc = pi*D*D*2*R/4



def myfunc(rpm, s, theta, thetaC, deltathetaC) :


    def V(T):
        Tr = T #- (2 * pi)
        return Vc * (1 - cos(Tr) + beta - sqrt(beta * beta - sin(Tr) ** 2)) / 2 + Vc / (t - 1)


    def dV(T):
        Tr = T #- (2 * pi)
        return Vc * sin(Tr) * (1 + cos(Tr) / sqrt(beta * beta - sin(Tr) ** 2)) / 2


    def Q(T, p, use_p = True):
        Tr = T #- (2 * pi)
        try:
            res = zeros(len(Tr))
            for i in range(len(Tr)):
                if (Td <= Tr[i] < Td + dT) :
                    res[i] = Qtot * (1 - cos(pi * (Tr[i] - Td) / dT)) / 2
                elif (use_p) and ((-2 * pi < Tr[i] < -pi) or (pi < Tr[i] < 2 * pi)):
                    res[i] = cp_R * p[i] * V(Tr[i])
            return res

        except TypeError:
            if Td <= Tr < Td + dT:
                return Qtot * (1 - cos(pi * (Tr - Td) / dT)) / 2
            elif (use_p) and ((-2 * pi < Tr < -pi) or (pi < Tr < 2 * pi)):
                return cp_R * p * V(Tr)
            else:
                return 0



    def dQ(T, p, use_p= True):
        Tr = T #- (2 * pi)
        try:
            res = zeros(len(Tr))
            for i in range(len(Tr)):
                if Td <= Tr[i] < Td + dT:
                    res[i] = Qtot * pi * sin(pi * (Tr[i] - Td) / dT) / (2 * dT)
                elif (use_p) and ((-2 * pi < Tr[i] < -pi) or (pi < Tr[i] < 2 * pi)):
                    res[i] = cp_R * p[i] * dV(Tr[i])
            return res

        except TypeError:
            if Td <= Tr < Td + dT:
                return Qtot * pi * sin(pi * (Tr - Td) / dT) / (2 * dT)
            elif (use_p) and ((-2*pi < Tr < -pi) or (pi<Tr<2*pi)):
                return cp_R * p * dV(Tr)
            else :
                return 0


    def dp(p, T):
        Tr = T #- (2 * pi)
        Vol = V(Tr)
        return (gamma - 1) * dQ(Tr, p, use_p=True) / Vol - gamma * p * dV(Tr) / Vol



    def RungeKutta(theta, n, s, ci):

        h = (theta[-1]-theta[0]) / n
        p = zeros(n + 1)
        p[0] = ci  # [Pa] Pression à theta en ci
        for i in range(0, n):
            K1 = dp(p[i], theta[i])
            K2 = dp(p[i] + h * K1 / 2, theta[i] + h / 2)
            K3 = dp(p[i] + h * K2 / 2, theta[i] + h / 2)
            K4 = dp(p[i] + h * K3, theta[i]+ h)
            p[i + 1] = p[i] + h * (K1 + 2 * K2 + 2 * K3 + K4) / 6
        return p

    #=================================================================================================
    #-------------------------------------------CODE--------------------------------------------------
    #=================================================================================================



    w = rpm * 2 * pi / 60
    Td = -thetaC * pi / 180  # [rad]
    dT = deltathetaC * pi / 180  # [rad]
    array = False
    n = 10000
    try :
        test = len(theta)
        array = True     #il faut retourner pression
        pression = zeros(test)
        starter = -2*pi
        init_cond = s*1e5
        theta_lin = theta*pi/180   #Pour les plot(s)
        for i in range(test) :
            theta_lin2 = linspace(starter, theta[i] * pi / 180, n + 1)  # -(2*pi)
            pression_computed = RungeKutta(theta_lin2, n,s, init_cond)
            pression[i] = pression_computed[-1]
            starter = theta[i]*pi/180
            init_cond = pression[i]
            if (i==0) :
                n = 100

    except TypeError:
        theta_lin = linspace(-2 * pi, theta*pi/180, n + 1)  # -(2*pi)
        pression = RungeKutta(theta_lin, n,s, s*1e5)
        pression_res = pression[-1]           #Il faudra retourner pression_res (ou pression[-1)

    print("Vol min : ", min(V(theta_lin)), "\nVol max : ", max(V(theta_lin)))
    print("Cylindrée [cm^3] : ", (max(V(theta_lin)) - min(V(theta_lin))) * 1e6, " ou ", Vc * 1e6)
    print("Pression max [bar] : ", max(pression) * 1e-5, "\nPression min [bar] : ", min(pression) * 1e-5,
          "\nPression admission : ", pression[0] * 1e-5, "\nPression echappement : ", pression[-1] * 1e-5)


    plt.plot(theta_lin, V(theta_lin), "-r", label="V")
    plt.show()
    plt.plot(theta_lin, V(theta_lin)*pression/(m*287.1))
    plt.show()
    plt.plot(theta_lin, Q(theta_lin, pression, use_p=True)*1e-3, "-b")
    plt.show()
    plt.plot(theta_lin, pression*1e-5, "-b", label="Echapement")

    plt.plot(theta_lin[where((theta_lin >= -2*pi) & (theta_lin <= -pi))], pression[where((theta_lin >= -2*pi) & (theta_lin <= -pi))]*1e-5, "y", label="Admission")
    plt.plot(theta_lin[where((theta_lin >= -pi) & (theta_lin <= Td+0.1))], pression[where((theta_lin >= -pi) & (theta_lin <= Td+0.1))]*1e-5, "g", label="Compression")
    plt.plot(theta_lin[where((theta_lin >= Td-0.1) & (theta_lin <= Td+dT+0.1))], pression[where((theta_lin >= Td-0.1)&(theta_lin <= Td+dT+0.1))]*1e-5, "r", label="Combustion")
    plt.plot(theta_lin[where((theta_lin >= Td+dT-0.1) & (theta_lin <= pi))], pression[where((theta_lin >= Td+dT-0.1) & (theta_lin <= pi))]*1e-5, "black",label="Détente")

    
    plt.legend()
    plt.xlabel("Angle [rad]")
    plt.ylabel("Pression [bar]")
    plt.show()
    plt.plot(V(theta_lin), pression*1e-5, label="Echapement")  #%(4*pi)-2*pi

    plt.plot(V(theta_lin[where((theta_lin >= -2 * pi) & (theta_lin <= -pi))]),
             pression[where((theta_lin >= -2 * pi) & (theta_lin <= -pi))] * 1e-5, "y", label="Admission")
    plt.plot(V(theta_lin[where((theta_lin >= -pi) & (theta_lin <= Td+0.1))]), pression[where((theta_lin >= -pi) & (theta_lin <= Td+0.1))] * 1e-5, "g", label="Compression")
    plt.plot(V(theta_lin[where((theta_lin >= Td-0.1) & (theta_lin <= Td + dT+0.1))]), pression[where((theta_lin >= Td-0.1) & (theta_lin <= Td + dT+0.1))] * 1e-5,
             "r", label="Combustion")
    plt.plot(V(theta_lin[where((theta_lin >= Td + dT-0.1) & (theta_lin <= pi))]), pression[where((theta_lin >= Td + dT-0.1) & (theta_lin <= pi))] * 1e-5,
             "black", label="Détente")


    plt.ylabel("Pression [bar]")
    plt.xlabel("Volume [m^3]")
    plt.legend()
    plt.show()


    Fpied = pi*D*D*pression/4 - mp*R*w*w*cos(theta_lin)
    if not array :
        Fpied_solo = pi*D*D*pression_res/4 - mp*R*w*w*cos(theta*pi/180)
    plt.plot(theta_lin, Fpied, "b")
    Ftete = -pi*D*D*pression/4 + (mp+mb)*R*w*w*cos(theta_lin)
    if not array :
        Ftete_solo = pi*D*D*pression_res/4 - mp*R*w*w*cos(theta*pi/180)


    plt.plot(theta_lin, Ftete, "-r")
    plt.plot(theta_lin, Ftete+Fpied, "--")
    plt.show()
    plt.plot(theta_lin, Ftete*R*sin(theta_lin))
    plt.show()

    Kx = 1
    Ky = 0.5
    Ix = 419/12
    Iy = 131/12
    Fcritx = max(max(abs(Ftete)),max(abs(Fpied)))
    Fcrity = max(max(abs(Ftete)),max(abs(Fpied)))
    bx = (L*L*Kx*Kx)/(pi*pi*2*1e11*Ix)
    ax = 1/(4.5*1e8*11)
    by = (L*L*Ky*Ky)/(pi*pi*2*1e11*Iy)
    ay = ax
    t1 = sqrt(Fcritx*(ax+sqrt(ax*ax+4*bx/Fcritx))/2)
    t2 = sqrt(Fcrity*(ay+sqrt(ay*ay+4*by/Fcrity))/2)
    print("Force max : [N] ", Fcritx, " tx : [m] ", t1, "|ty = ", t2, "| t3 = ", sqrt(0.025*D*D/11))


    if array :
        return V(theta), Q(theta, 0, use_p=False)/m, Fpied, Ftete, pression, max(t1,t2)

    else :
        return V(theta), Q(theta, 0, use_p=False)/m, Fpied_solo, Ftete_solo, pression_res, max(t1, t2)





myfunc(3000,1,linspace(-360, 360,100), 20, 50)
