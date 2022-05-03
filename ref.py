import math
import matplotlib.pyplot as plt
import numpy as np

def main():
    # Stability :)
    d=.01
    i=1j
    x=np.arange(-3,3,d)
    y=np.arange(-3,3,d)
    X,Y=np.meshgrid(x,y)
    Z=X+Y*i
    Z1=abs(1+Z)
    Z2=abs(1+Z+Z**2/2)
    Z3=abs(1+Z+Z**2/2+Z**3/3+Z**4/24)
    Z4=abs(1+2*Z)
    Z5=abs(Z*math.sqrt(1/2))
    plt.figure("Euler")
    plt.contour(X,Y,Z1,[1])
    plt.figure("Modified Euler")
    plt.contour(X,Y,Z2,[1])
    plt.figure("RK4")
    plt.contour(X,Y,Z3,[1])
    plt.figure("Leapfrog")
    plt.contour(X,Y,Z4,[1])
    plt.figure("Newmark")
    plt.contour(X,Y,Z5,[1])
    # Resonance Example
    appres = newmark(.5, .1, 5, yeehaw, 0 , 30, 1000, 1, 0)
    plt.figure("Resonance")
    plt.title("Resonance- Newmark")
    plt.xlabel('t')
    plt.ylabel('x')
    plt.plot(appres[0],appres[1],color='green',label="Newmark")
    # Simple Harmonic Motion
    # All stable simple harmonic motion
    exa1 = funceval(u1sol, 0, 5, 1000)
    app0 = leapfrog(xpp,0,5,1000,1,0)
    app1 = rk4sys(0,5,1000,[1,0],[u1,u2])
    app2 = euler_2nd(u1,u2,0,5,1000,1,0)
    app3 = modifiedeuler_2nd(u1,u2,0,5,1000,1,0)
    app4 = newmark(.5, 0, 5, dummyzero, 0 , 5, 1000, 1, 0)
    plt.figure("All Stable - SHM")
    plt.title("Simple Harmonic Motion; N=1000")
    plt.xlabel('t')
    plt.ylabel('x')
    plt.plot(app0[0],app0[1],color='green',label="Leapfrog")
    plt.plot(app1[0],np.transpose(app1[1])[0],color='blue',label="RK4")
    plt.plot(app2[0],app2[1],color="orange",label="Euler")
    plt.plot(app3[0],app3[1],color='red',label="Modified Euler")
    plt.plot(app4[0],app4[1],color='black',label="Newmark")
    plt.plot(exa1[0],exa1[1],label="Exact Solution")
    plt.legend(loc='lower left')
    plt.figure("All Stable Error - SHM")
    plt.title("Simple Harmonic Motion Error; N=1000")
    plt.xlabel('t')
    plt.ylabel('Error')
    plt.semilogy(app0[0],errorcalc(app0[1],exa1[1]),'g',label="Leapfrog")
    plt.semilogy(app1[0],errorcalc(np.transpose(app1[1])[0],exa1[1]),'b',label="RK4 ")
    plt.semilogy(app2[0],errorcalc(app2[1],exa1[1]),color="orange",label="Euler")
    plt.semilogy(app3[0],errorcalc(app3[1],exa1[1]),'r',label="Modified Euler")
    plt.semilogy(app4[0],errorcalc(app4[1],exa1[1]),color='black',label="Newmark")
    plt.legend(loc='lower left')
    # Some stable simple harmonic motion
    exa1 = funceval(u1sol, 0, 5, 100)
    app0 = leapfrog(xpp,0,5,100,1,0)
    app1 = rk4sys(0,5,100,[1,0],[u1,u2])
    app2 = euler_2nd(u1,u2,0,5,100,1,0)
    app3 = modifiedeuler_2nd(u1,u2,0,5,100,1,0)
    app4 = newmark(.5, 0, 5, dummyzero, 0 , 5, 100, 1, 0)
    plt.figure("Some Stable - SHM")
    plt.title("Simple Harmonic Motion; N=100")
    plt.xlabel('t')
    plt.ylabel('x')
    plt.plot(app0[0],app0[1],'g',label="Leapfrog")
    plt.plot(app1[0],np.transpose(app1[1])[0],'b',label="RK4")
    plt.plot(app2[0],app2[1],color="orange",label="Euler")
    plt.plot(app3[0],app3[1],'r',label="Modified Euler")
    plt.plot(app4[0],app4[1],color='black',label="Newmark")
    plt.plot(exa1[0],exa1[1],label="Exact Solution")
    plt.legend(loc='lower left')
    plt.figure("Some Stable Error - SHM")
    plt.title("Simple Harmonic Motion Error; N=100")
    plt.xlabel('t')
    plt.ylabel('Error')
    plt.semilogy(app0[0],errorcalc(app0[1],exa1[1]),'g',label="Leapfrog")
    plt.semilogy(app1[0],errorcalc(np.transpose(app1[1])[0],exa1[1]),'b',label="RK4 ")
    plt.semilogy(app2[0],errorcalc(app2[1],exa1[1]),color="orange",label="Euler")
    plt.semilogy(app3[0],errorcalc(app3[1],exa1[1]),'r',label="Modified Euler")
    plt.semilogy(app4[0],errorcalc(app4[1],exa1[1]),color='black',label="Newmark")
    plt.legend(loc='lower left')
    # All unstable simple harmonic motion
    exa1 = funceval(u1sol, 0, 5, 10)
    app0 = leapfrog(xpp,0,5,10,1,0)
    app1 = rk4sys(0,5,10,[1,0],[u1,u2])
    app2 = euler_2nd(u1,u2,0,5,10,1,0)
    app3 = modifiedeuler_2nd(u1,u2,0,5,10,1,0)
    app4 = newmark(.5, 0, 5, dummyzero, 0 , 5, 10, 1, 0)
    plt.figure("All Unstable - SHM")
    plt.title("Simple Harmonic Motion; N=10")
    plt.xlabel('t')
    plt.ylabel('x')
    plt.plot(app0[0],app0[1],'g',label="Leapfrog")
    plt.plot(app1[0],np.transpose(app1[1])[0],'b',label="RK4")
    plt.plot(app2[0],app2[1],color="orange",label="Euler")
    plt.plot(app3[0],app3[1],'r',label="Modified Euler")
    plt.plot(app4[0],app4[1],color='black',label="Newmark")
    plt.plot(exa1[0],exa1[1],label="Exact Solution")
    plt.legend(loc='lower left')
    plt.figure("All Unstable Error - SHM")
    plt.title("Simple Harmonic Motion Error; N=10")
    plt.xlabel('t')
    plt.ylabel('Error')
    plt.semilogy(app0[0],errorcalc(app0[1],exa1[1]),'g',label="Leapfrog")
    plt.semilogy(app1[0],errorcalc(np.transpose(app1[1])[0],exa1[1]),'b',label="RK4 ")
    plt.semilogy(app2[0],errorcalc(app2[1],exa1[1]),color="orange",label="Euler")
    plt.semilogy(app3[0],errorcalc(app3[1],exa1[1]),'r',label="Modified Euler")
    plt.semilogy(app4[0],errorcalc(app4[1],exa1[1]),color='black',label="Newmark")
    plt.legend(loc='lower left')
    # Damped Harmonic Motion
    # All stable damped harmonic motion
    exa1 = funceval(u3sol, 0, 5, 1000)
    app1 = rk4sys(0,5,1000,[1,0],[u3,u4])
    app2 = euler_2nd(u3,u4,0,5,1000,1,0)
    app3 = modifiedeuler_2nd(u3,u4,0,5,1000,1,0)
    app4 = newmark(.5, .1, 5, dummyzero, 0 , 5, 1000, 1, 0)
    app5 = newmark(.5, .1, 5, dummyzero, 0 , 5, 1000, 1, 0, 1/6)
    plt.figure("All Stable - DHM")
    plt.title("Damped Harmonic Motion; N=1000")
    plt.xlabel('t')
    plt.ylabel('x')
    plt.plot(app1[0],np.transpose(app1[1])[0],'b',label="RK4")
    plt.plot(app2[0],app2[1],color="orange",label="Euler")
    plt.plot(app3[0],app3[1],'r',label="Modified Euler")
    plt.plot(app4[0],app4[1],color='black',label="Newmark")
    plt.plot(app5[0],app5[1],color='green',label="Newmark - Linear")
    plt.plot(exa1[0],exa1[1],label="Exact Solution")
    plt.legend(loc='lower left')
    plt.figure("All Stable Error - DHM")
    plt.title("Damped Harmonic Motion Error; N=1000")
    plt.xlabel('t')
    plt.ylabel('Error')
    plt.semilogy(app1[0],errorcalc(np.transpose(app1[1])[0],exa1[1]),'b',label="RK4 ")
    plt.semilogy(app2[0],errorcalc(app2[1],exa1[1]),color="orange",label="Euler")
    plt.semilogy(app3[0],errorcalc(app3[1],exa1[1]),'r',label="Modified Euler")
    plt.semilogy(app4[0],errorcalc(app4[1],exa1[1]),color='black',label="Newmark")
    plt.semilogy(app5[0],errorcalc(app5[1],exa1[1]),color='green',label="Newmark - Linear")
    plt.legend(loc='lower left')
    # Some stable damped harmonic motion
    exa1 = funceval(u3sol, 0, 5, 100)
    app1 = rk4sys(0,5,100,[1,0],[u3,u4])
    app2 = euler_2nd(u3,u4,0,5,100,1,0)
    app3 = modifiedeuler_2nd(u3,u4,0,5,100,1,0)
    app4 = newmark(.5, .1, 5, dummyzero, 0 , 5, 100, 1, 0)
    app5 = newmark(.5, .1, 5, dummyzero, 0 , 5, 100, 1, 0, 1/6)
    plt.figure("Some Stable - DHM")
    plt.title("Damped Harmonic Motion; N=100")
    plt.xlabel('t')
    plt.ylabel('x')
    plt.plot(app1[0],np.transpose(app1[1])[0],'b',label="RK4")
    plt.plot(app2[0],app2[1],color="orange",label="Euler")
    plt.plot(app3[0],app3[1],'r',label="Modified Euler")
    plt.plot(app4[0],app4[1],color='black',label="Newmark")
    plt.plot(app5[0],app5[1],color='green',label="Newmark - Linear")
    plt.plot(exa1[0],exa1[1],label="Exact Solution")
    plt.legend(loc='lower left')
    plt.figure("Some Stable Error - DHM")
    plt.title("Damped Harmonic Motion Error; N=100")
    plt.xlabel('t')
    plt.ylabel('Error')
    plt.semilogy(app1[0],errorcalc(np.transpose(app1[1])[0],exa1[1]),'b',label="RK4 ")
    plt.semilogy(app2[0],errorcalc(app2[1],exa1[1]),color="orange",label="Euler")
    plt.semilogy(app3[0],errorcalc(app3[1],exa1[1]),'r',label="Modified Euler")
    plt.semilogy(app4[0],errorcalc(app4[1],exa1[1]),color='black',label="Newmark")
    plt.semilogy(app5[0],errorcalc(app5[1],exa1[1]),color='green',label="Newmark - Linear")
    plt.legend(loc='lower left')
    # All unstable damped harmonic motion
    exa1 = funceval(u3sol, 0, 5, 10)
    app1 = rk4sys(0,5,10,[1,0],[u3,u4])
    app2 = euler_2nd(u3,u4,0,5,10,1,0)
    app3 = modifiedeuler_2nd(u3,u4,0,5,10,1,0)
    app4 = newmark(.5, .1, 5, dummyzero, 0 , 5, 10, 1, 0)
    app5 = newmark(.5, .1, 5, dummyzero, 0 , 5, 10, 1, 0, 1/6)
    plt.figure("All Unstable - DHM")
    plt.title("Damped Harmonic Motion; N=10")
    plt.xlabel('t')
    plt.ylabel('x')
    plt.plot(app1[0],np.transpose(app1[1])[0],'b',label="RK4")
    plt.plot(app2[0],app2[1],color="orange",label="Euler")
    plt.plot(app3[0],app3[1],'r',label="Modified Euler")
    plt.plot(app4[0],app4[1],color='black',label="Newmark")
    plt.plot(app5[0],app5[1],color='green',label="Newmark - Linear")
    plt.plot(exa1[0],exa1[1],label="Exact Solution")
    plt.legend(loc='lower left')
    plt.figure("All Unstable Error - DHM")
    plt.title("Damped Harmonic Motion Error; N=10")
    plt.xlabel('t')
    plt.ylabel('Error')
    plt.semilogy(app1[0],errorcalc(np.transpose(app1[1])[0],exa1[1]),'b',label="RK4")
    plt.semilogy(app2[0],errorcalc(app2[1],exa1[1]),color="orange",label="Euler")
    plt.semilogy(app3[0],errorcalc(app3[1],exa1[1]),'r',label="Modified Euler")
    plt.semilogy(app4[0],errorcalc(app4[1],exa1[1]),color='black',label="Newmark")
    plt.semilogy(app5[0],errorcalc(app5[1],exa1[1]),color='green',label="Newmark - Linear")
    plt.legend(loc='lower left')
    plt.show()
    return()

def newmark(M,C,K,f,a,b,N,x0,v0,beta=(1/4),gamma=(1/2)):
    h=(b-a)/N
    a0=(f(a)-C*v0-K*x0)/M
    t=[a+h*i for i in range(N+1)]
    x=[x0 for i in range(N+1)]
    v=[v0 for i in range(N+1)]
    a=[a0 for i in range(N+1)]
    A=M/(beta*h**2)+(gamma*C)/(beta*h)+K
    for i in range(1,N+1):
        x[i] = f(t[i]) \
            + M*(x[i-1]/(beta*h**2)+v[i-1]/(beta*h)+(1/(2*beta)-1)*a[i-1]) \
            + C*((gamma*x[i-1])/(beta*h)-v[i-1]*(1-(gamma/beta))-h*a[i-1]*(1-gamma/(2*beta)))
        x[i] = x[i]/A
        a[i] = (x[i]-x[i-1])/(beta*h**2)-v[i-1]/(beta*h)-(1/(2*beta)-1)*a[i-1]
        v[i] = gamma*(x[i]-x[i-1])/(beta*h)+v[i-1]*(1-gamma/beta)+h*a[i-1]*(1-gamma/(2*beta))
    return(t,x)

def leapfrog(f,a,b,N,alpha,beta):
    h=(b-a)/N
    t=[a+h*i for i in range(N+1)]
    w=[alpha for i in range(N+1)]
    v=[beta for i in range(N+1)]
    for i in range(1,N+1):
        w[i] = w[i-1]+h*v[i-1]+(1/2)*f(w[i-1])*h**2
        v[i] = v[i-1]+(1/2)*(f(w[i-1])+f(w[i]))*h
    return(t,w)

def euler_2nd(F,G,a,b,N,x0,y0):
    h=(b-a)/N
    t=[a+h*i for i in range(N+1)]
    x=[x0 for i in range(N+1)]
    y=[y0 for i in range(N+1)]
    for i in range(1,N+1):
        x[i] = x[i-1]+h*F(t[i-1],x[i-1],y[i-1])
        y[i] = y[i-1]+h*G(t[i-1],x[i-1],y[i-1])
    return(t,x)

def modifiedeuler_2nd(F,G,a,b,N,x0,y0):
    h=(b-a)/N
    t=[a+h*i for i in range(N+1)]
    x=[x0 for i in range(N+1)]
    y=[y0 for i in range(N+1)]
    for i in range(1,N+1):
        x[i] = x[i-1]+(h/2)*(F(t[i-1],x[i-1],y[i-1])+F(t[i],x[i-1]+h*F(t[i-1],x[i-1],y[i-1]),y[i-1]))
        y[i] = y[i-1]+(h/2)*(G(t[i-1],x[i-1],y[i-1])+G(t[i],x[i-1],y[i-1])+h*G(t[i-1],x[i-1],y[i-1]))
    return(t,x)

def midpoint_2nd(F,G,a,b,N,x0,y0):
    h=(b-a)/N
    t=[a+h*i for i in range(N+1)]
    x=[x0 for i in range(N+1)]
    y=[y0 for i in range(N+1)]
    for i in range(1,N+1):
        x[i] = x[i-1]+h*F(t[i-1]+h/2,x[i-1]+(h/2)*F(t[i-1],x[i-1],y[i-1]),y[i-1])
        y[i] = y[i-1]+h*G(t[i-1]+h/2,x[i-1],y[i-1]+(h/2)*G(t[i-1],x[i-1],y[i-1]))
    return(t,x)

def rk4sys(a,b,N,alpham,fjtu):
    m = len(alpham)
    h = (b-a)/N
    t = a
    k1,k2,k3,k4,wd,w,ti,xi = [],[],[],[],[],[],[],[]
    for i in range(m):
        k1.append(0);k2.append(0);k3.append(0);k4.append(0);wd.append(0);w.append(alpham[i])
    ti.append(t)
    xi.append(w[:])
    for i in range(1,N+1):
        for j in range(m): wd[j] = w[j]
        for j in range(m): k1[j] = h*fjtu[j](t,*wd)
        for j in range(m): wd[j] = w[j] + k1[j]/2
        for j in range(m): k2[j] = h*fjtu[j](t+h/2,*wd)
        for j in range(m): wd[j] = w[j] + k2[j]/2
        for j in range(m): k3[j] = h*fjtu[j](t+h/2,*wd)
        for j in range(m): wd[j] = w[j] + k3[j]
        for j in range(m): k4[j] = h*fjtu[j](t+h,*wd)
        for j in range(m): w[j] = w[j]+(k1[j]+2*k2[j]+2*k3[j]+k4[j])/6
        t = a + i*h
        ti.append(t)
        xi.append(w[:])
    return(ti,xi)
    
def errorcalc(w,y):
    return([abs(w[i]-y[i]) for i in range(len(w))])

def funceval(fcn,a,b,N):
    h=(b-a)/N
    t=[a+h*i for i in range(N+1)]
    w=[fcn(t[i]) for i in range(N+1)]
    return(t,w)

# This is the system for Simple Harmonic Motion
def u1(t,y1,y2):return(y2)
def u2(t,y1,y2):return(-10*y1)
def u1sol(t):return(math.cos(math.sqrt(10)*t))
def u2sol(t):return(-math.sqrt(10)*math.sin(math.sqrt(10)*t))

# This is to input a zero forcing function for newmark
def dummyzero(t):return(0)

# This is the forcing equation to demonstrate resonance
def yeehaw(t):return(math.sin(math.sqrt(10)*t))

# This is the equation governing Simple Harmonic Motion
def xpp(x):return(-10*x)

# This is the system for Damped Harmonic Motion
def u3(t,y1,y2):return(y2)
def u4(t,y1,y2):return(-.2*y2-10*y1)
def u3sol(t):return(math.exp(-0.1*t)*math.cos(math.sqrt(9.99)*t))
def u4sol(t):return(-math.exp(-0.1*t)*(math.sqrt(9.99)*math.sin(math.sqrt(9.99)*t)+0.1*cos(math.sqrt(9.99)*t)))


if __name__ == '__main__':
    main()
