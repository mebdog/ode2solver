import math
import matplotlib.pyplot as plt
import numpy as np

def main():
    ans1=euler(fcn,0,2,10,0.5)
    ans2=midpoint(fcn,0,2,10,0.5)
    ans3=modifiedeuler(fcn,0,2,10,0.5)
    ans4=rungekutta(fcn,0,2,10,0.5)
    er1=errorcalc(ans1,sol)
    er2=errorcalc(ans2,sol)
    er3=errorcalc(ans3,sol)
    er4=errorcalc(ans4,sol)
    # Solution Plotting
    plt.figure("1st Order Method Approximations")
    plt.title("1st Order Method Approximations")
    plt.xlabel("t")
    plt.ylabel("y")
    plt.plot(ans1[0],ans1[1],label="Euler")
    plt.plot(ans2[0],ans2[1],label="Midpoint")
    plt.plot(ans3[0],ans3[1],label="Modified Euler")
    plt.plot(ans4[0],ans4[1],label="Runge-Kutta")
    plt.legend()
    # Error Plotting
    plt.figure("1st Order Method Errors")
    plt.title("1st Order Method Errors")
    plt.xlabel("t")
    plt.ylabel("Error")
    plt.semilogy(er1[0],er1[1],label="Euler")
    plt.semilogy(er2[0],er2[1],label="Midpoint")
    plt.semilogy(er3[0],er3[1],label="Modified Euler")
    plt.semilogy(er4[0],er4[1],label="Runge-Kutta")
    plt.legend()
    # Stability Plotting
    plt.figure("Stability for different methods")
    plt.title("Stability for different methods")
    plt.xlabel("Re(hk)")
    plt.ylabel("Im(hk)")
    x=np.linspace(-2,2,1000)
    y=np.linspace(-2,2,1000)
    X,Y=np.meshgrid(x,y)
    eulerF=(X+1)**2+Y**2-1
    midpointF=X+(X**2-Y**2)/2
    plt.contour(X,Y,eulerF,[0])
    plt.contour(X,Y,midpointF,[0])
    plt.contour
    plt.legend()

    plt.show()

# Solves for y'=f(t,y)
# a < t < b ; t(0)=alpha
def euler(f,a,b,N,alpha):
    h=(b-a)/N
    t=[a+h*i for i in range(N+1)]
    w=[alpha for i in range(N+1)]
    for i in range(1,N+1):
        w[i] = w[i-1]+h*f(t[i-1],w[i-1])
    return(t,w)
def midpoint(f,a,b,N,alpha):
    h=(b-a)/N
    t=[a+h*i for i in range(N+1)]
    w=[alpha for i in range(N+1)]
    for i in range(1,N+1):
        w[i] = w[i-1]+h*f(t[i-1]+h/2,w[i-1]+(h/2)*f(t[i-1],w[i-1]))
    return(t,w)
def modifiedeuler(f,a,b,N,alpha):
    h=(b-a)/N
    t=[a+h*i for i in range(N+1)]
    w=[alpha for i in range(N+1)]
    for i in range(1,N+1):
        w[i] = w[i-1]+(h/2)*(f(t[i-1],w[i-1])+f(t[i],w[i-1]+h*f(t[i-1],w[i-1])))
    return(t,w)
def rungekutta(f,a,b,N,alpha):
    h=(b-a)/N
    t=[a+h*i for i in range(N+1)]
    w=[alpha for i in range(N+1)]
    for i in range(1,N+1):
        k1=h*f(t[i-1],w[i-1])
        k2=h*f(t[i-1]+h/2,w[i-1]+k1/2)
        k3=h*f(t[i-1]+h/2,w[i-1]+k2/2)
        k4=h*f(t[i],w[i-1]+k3)
        w[i]=w[i-1]+(1/6)*(k1+2*k2+2*k3+k4)
    return(t,w)

# Solves for x"=f(x)
# with Initial conditions x(a) = alpha and x'(a) = beta, a < t < b
def leapfrog(f,a,b,N,alpha,beta):
    h=(b-a)/N
    t=[a+h*i for i in range(N+1)]
    w=[alpha for i in range(N+1)]
    v=[beta for i in range(N+1)]
    for i in range(1,N+1):
        w[i] = w[i-1]+h*v[i-1]+(1/2)*f(w[i-1])*h**2
        v[i] = v[i-1]+(1/2)*(f(w[i-1])+f(w[i]))*h
    return(t,w)

# Solving for Mx" + Cx' + Kx = f(t)
# with x(a) = x0, y(a) = y0
def euler_2nd(F,G,a,b,N,x0,y0):
    # substituting x' = y = F(t,x,y) and y' = (f(t) - Cy - Kx)/M = G(t,x,y)
    h=(b-a)/N
    x=[x0 for i in range(N+1)]
    y=[y0 for i in range(N+1)]
    for i in range(1,N+1):
        x[i] = x[i-1]+h*F(t[i-1],x[i-1],y[i-1])
        y[i] = y[i-1]+h*G(t[i-1],x[i-1],y[i-1])
    return(t,x)

#solves for Mx" + Cx' +Kx = f(t)
def newmark(f,M,C,K,t_i,t_f,N,x_i, v_i,gamma, beta):
    # M is the mass matrix, C is the damping matrix, K is the stiffness matrix
    #x_i and v_i are initial condition vectors
    # f is a column vector
    h = (t_f - t_i)/N
    
    t = np.linspace(t_i, t_f,N + 1)
    x = np.empty(len(x_i),N+1)
    v = np.empty(len(v_i,N+1))
    a = np.empty(len(v_i,N+1))
    
    x[:,0] = x_i
    v[:,0] = v_i
    a[:,0] = np.linalg.inv(M)*(f[:,0] - C*v[:,0] - K*x[:,0])
    
    A = M/(beta*h**2) + gamma*C/(beta*h) + K
    invA = np.linalg.inv(A)
    
    for i in range(0,N):
        B = f[:,i+1] + M * (x[:,i]/(beta*h**2) + v[:,i]/(beta*h) + (1/(2*beta) - 1)*a[:,i]) + C*(gamma*x[:,i]/(beta*h)-v[:,i]*(1-gamma/beta)-h*a[:,i]*(1-gamma/(2*beta)))
        x[:,i+1] = invA * B
        a[:,i+1] = (x[:,i+1]-x[:,i])/(beta*h**2) - v[:,i]/(beta*h)-(1/(2*beta) - 1)*a[:,i]
        v[:,i+1] = v[:,i]+(1-gamma)*h*a[:,i]+gamma*h*a[:,i+1]
        
    return (t,x[:,N+1])
    
def errorcalc(tw,y):
    return([tw[0],[abs(tw[1][i]-y(tw[0][i])) for i in range(len(tw[0]))]])

def funceval(fcn,a,b,N):
    h=(b-a)/N
    t=[a+h*i for i in range(N+1)]
    w=[fcn(t[i]) for i in range(N+1)]
    return(t,w)

def fcn(t,y):return(y-t**2+1)
def sol(t):return(t**2+2*t+1-0.5*math.exp(t))
def fcn2(y):return(-y)
def sol2(t):return(math.sin(t))

if __name__ == '__main__':
    main()
