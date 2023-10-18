import matplotlib.pyplot as plt
from scipy.integrate import odeint
import numpy as np
import math

#  System Parameters
k = 20 #  N/m
m = 1 #  kg
L0 = 0.5 #  meters

#====================================
#  Case Parameter Setting
#====================================

def case_variables(case: int):
    if case == 1:
        h = 1.05*L0
        b = 0.5 # Ns/m
        mu = 0 #  Coulomb friction
        x0 = 0 #  m
        dx0 = 0.1 #  m.s
        
    elif case == 2:
        h = 1.05*L0
        mu = 0.1 #  Coulomb friction
        b = 0
        x0 = 0 #  m
        dx0 = 0.1 #  m.s
        
    elif case == 3:
        h = 1.05*L0
        b = 0.5 # Ns/m
        mu = 0
        x0 = 0 #  m
        dx0 = 7.5 #  m.s   
        
    elif case == 4:
        h = 1.05*L0
        b = 0
        mu = 0.1 #  Coulomb friction
        x0 = 0 #  m
        dx0 = 7.5 #  m.s
        
    elif case == 5:
        h = 0.95*L0
        b = 0.5 # Ns/m
        mu = 0
        x0 = 0 #  m
        dx0 = 0.1 #  m.s 
        
    elif case == 6: 
        h = 0.95*L0
        b = 0
        mu = 0.1 #  Coulomb friction
        x0 = 0 #  m
        dx0 = 0.1 #  m.s
        
    elif case == 7:
        h = 0.95*L0
        b = 0.5 # Ns/m
        mu = 0
        x0 = 0 #  m
        dx0 = 7.5 #  m.s
        
    elif case == 8:
        h = 0.95*L0
        b = 0
        mu = 0.1 #  Coulomb friction
        x0 = 0 #  m
        dx0 = 7.5 #  m.s
    else:
        IndexError()   
        print("Enter case number 1 through 8 only")

    vars = [h, b, mu, x0, dx0, case]
    return vars

#====================================
#  Initialization
#====================================
tstart = 0
tstop = 30 #  seconds
delta_t = 0.01 #  seconds
I = m #  kg
C = 1/k # m/N
t = np.arange(tstart, tstop, delta_t)

#  Function to return dx/dt
def linear_system_ode(x, t):
    
    L = math.sqrt(x[0]*x[0] + h*h)


    dx1 = x[1]
    dx2 =  (-b*x[1])  + ((-k)*(L0 - L)*(-x[0]/L))
    dxdt = [dx1, dx2]
    return dxdt


def nonlinear_system_ode(x, t):
    
    L = math.sqrt(x[0]*x[0] + h*h)
    g = 9.81 #  m/s^2   #  m = 1 kg
    theta = math.atan(x[0]/h) #  [rads]
    Fk = k * (L0 - L)
    Fn = m*g + Fk*math.cos(theta)

    dx1 = x[1]
    dx2 =  (-mu*Fn*np.sign(x[1]))  + ((-k)*(L0 - L)*(-x[0]/L))
    dxdt = [dx1, dx2]
    return dxdt

def plotter(x1, x2, t, case: int, save: bool = False):
    plt.title("Case " + str(case))
    plt.plot(t, x1, label= "x(t) [m]")
    plt.plot(t, x2, label= "v(t) [m/s]")
    plt.xlabel("Time [s]")
    plt.legend()
    plt.grid()
    plt.show()

    if save:
        plt.savefig()
            
def plot_pairs(x1_1, x1_2, x2_1, x2_2, t, case1, case2, save: bool = False):
    
    return
        
if __name__ == "__main__":
    
    #vars = [h, b, mu, x0, dx0, case]
    vars = case_variables(2)
    h = vars[0]
    b = vars[1]
    mu = vars[2]
    x0 = vars[3]
    dx0 = vars[4]
    case_index = vars[5]
    
    #  Initial Condition
    x_init = [x0, dx0] 
    
    print(b, mu)
    #  Solve the ODE
    out = odeint(nonlinear_system_ode, x_init, t)
    print(out)
    x1 = out[:,0]
    x2 = out[:,1]
    
    #  Plot & Show outputs
    plotter(x1, x2, t, case_index)