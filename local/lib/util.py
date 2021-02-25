import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
import pandas as pd

def sigmoide(u):
        g = np.exp(u)/(1 + np.exp(u))
        return g

#Aprendizaje
def Gradiente(X2,y2,MaxIter = 100000):
    w = np.ones(3).reshape(3, 1)
    eta = 0.001
    N = len(y2)
    Error =np.zeros(MaxIter)
    Xent = np.concatenate((X2,np.ones((100,1))),axis=1)

    for i in range(MaxIter):
        tem = np.dot(Xent,w)
        tem2 = sigmoide(tem.T)-np.array(y2)
        Error[i] = np.sum(abs(tem2))/N
        tem = np.dot(Xent.T,tem2.T)
        wsig = w - eta*tem/N
        w = wsig
    return w, Error

def plot_frontera():
    iris = datasets.load_iris()
    X, y = iris.data, iris.target
    X2 = X[:100][:,:2]
    y2 = y[:100]
    #fig, (ax0, ax1) = plt.subplots(1,2)
    #x0.scatter(X2[:,0], X2[:,1], c=y2, cmap="Accent")
    
    w,Error = Gradiente(X2,y2)
    #print(w)
    #print('Error=',Error[-1])
    #Grafica de la frontera encontrada
    iris = datasets.load_iris()
    X, y = iris.data, iris.target
    X2 = X[:100][:,:2]
    y2 = y[:100]
    plt.scatter(X2[:,0], X2[:,1], c=y2,s=40, alpha=.5, cmap="Accent",label='data set')
    p = np.random.permutation(50)
    p = p[0]
    f = np.dot(np.r_[X2[p,:],1],w)
    plt.scatter(X2[p,0], X2[p,1], s=40, alpha=.5, color="red", label='f='+str(f))
    p = p+50
    f = np.dot(np.r_[X2[p,:],1],w)
    plt.scatter(X2[p,0], X2[p,1], s=40, alpha=.5, color="blue", label='f='+str(f))
    x1 = np.linspace(4,8,20)
    x2 = -(w[0]/w[1])*x1 - (w[2]/w[1])
    plt.plot(x1,x2,'k',label='frontera')
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.legend()

def plot_model_reg(t, prediction):
    d = pd.read_csv("local/data/trilotropicos.csv")
    xr = np.linspace(np.min(d.longitud), np.max(d.longitud), 100)
    plt.scatter(d.longitud, d.densidad_escamas, s=40, alpha=.2, color="blue", label="")
    plt.plot(xr,prediction(t,xr), lw=2, color="black")
    #plt.title("   ".join([r"$\theta_%d$=%.2f"%(i, t[i]) for i in range(len(t))]));

    p = d.iloc[np.random.randint(len(d))]
    pred = prediction(t, p.longitud)
    plt.plot([p.longitud, p.longitud], [p.densidad_escamas, pred], ls="--", color="gray", label=u"error de predicción")
    plt.scatter(p.longitud, p.densidad_escamas, s=70, alpha=.5, color="red", label="random sample")
    plt.scatter(p.longitud, pred, s=70, alpha=1., color="black", label=u"predicción")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.xlabel('$x$')
    plt.ylabel('$t$')