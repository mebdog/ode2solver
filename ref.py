import math
import matplotlib.pyplot as plt

def main():
    print("Euler")
    print(euler(fcn,0,2,10,0.5))
    print("Midpoint")
    print(midpoint(fcn,0,2,10,0.5))
    print("Modified Euler")
    print(modifiedeuler(fcn,0,2,10,0.5))
    print("Runge Kutta")
    print(rungekutta(fcn,0,2,10,0.5))


# Solves for y'=f(t,y)
# a < t < b ; t(0)=alpha
def euler(f,a,b,N,alpha):
    h=(b-a)/N
    t=[round(a+h*i,3) for i in range(N+1)]
    w=[alpha for i in range(N+1)]
    for i in range(1,N+1):
        w[i] = w[i-1]+h*f(t[i-1],w[i-1])
    return(t,w)
def midpoint(f,a,b,N,alpha):
    h=(b-a)/N
    t=[round(a+h*i,3) for i in range(N+1)]
    w=[alpha for i in range(N+1)]
    for i in range(1,N+1):
        w[i] = w[i-1]+h*f(t[i-1]+h/2,w[i-1]+(h/2)*f(t[i-1],w[i-1]))
    return(t,w)
def modifiedeuler(f,a,b,N,alpha):
    h=(b-a)/N
    t=[round(a+h*i,3) for i in range(N+1)]
    w=[alpha for i in range(N+1)]
    for i in range(1,N+1):
        w[i] = w[i-1]+(h/2)*(f(t[i-1],w[i-1])+f(t[i],w[i-1]+h*f(t[i-1],w[i-1])))
    return(t,w)
def rungekutta(f,a,b,N,alpha):
    h=(b-a)/N
    t=[round(a+h*i,3) for i in range(N+1)]
    w=[alpha for i in range(N+1)]
    for i in range(1,N+1):
        k1=h*f(t[i-1],w[i-1])
        k2=h*f(t[i-1]+h/2,w[i-1]+k1/2)
        k3=h*f(t[i-1]+h/2,w[i-1]+k2/2)
        k4=h*f(t[i],w[i-1]+k3)
        w[i]=w[i-1]+(1/6)*(k1+2*k2+2*k3+k4)
    return(t,w)



#Solves for x"=f(x)
# with Initial conditions x(a) = alpha and x'(a) = beta, a < t < b
def leapfrog(f,a,b,N,alpha,beta,k):
    h = (b-a)/N
    
    t = np.linspace(a, k*b,k*N + 1)
    x = np.empty(k*N+1)
    v = np.empty(k*N+1)
    
    x[0] = alpha
    v[0] = beta
    
    for i in range(1, k*N+1):
        x[i] = x[i-1] + v[i-1]*h + 0.5*f(x[i-1])*h**2
        v[i] = v[i-1] + 0.5*(f(x[i-1]) + f(x[i]))*h
        
    return (t,x)

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

# def prettyplot()
# def errorplot()
def fcn(t,y):return(y-t**2+1)
if __name__ == '__main__':
    main()
