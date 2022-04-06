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
#     def newmmark
#     def leapfrog
# def prettyplot()
# def errorplot()
def fcn(t,y):return(y-t**2+1)
if __name__ == '__main__':
    main()