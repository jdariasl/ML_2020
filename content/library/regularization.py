from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from matplotlib.patches import Ellipse
from sklearn.neighbors import KernelDensity

# coding=utf-8
def plot_ellipse(ax, mu ,sigma, color='k'):

    vals, vecs = np.linalg.eigh(sigma)
    
    x , y = vecs[:, 0]
    theta = np.degrees(np.arctan2(y,x))
    
    w,h = 4* np.sqrt(vals)
    
    ax.tick_params(axis='both',which='major',labelsize=20)
    ellipse = Ellipse(mu,w,h,theta,color=color)
    ellipse.set_alpha(0.2)
    ax.add_artist(ellipse)

def mapFeature(X1, X2, degree = 6):

    Nm = X1.shape[0]       
    out = np.ones(Nm).reshape(Nm, 1)
    for i in range(1,degree+1):
        for j in range(i+1):
            tem = (X1**(i-j))*(X2**j)
            out = np.hstack([out,tem.reshape(Nm,1)])
    return out
def sigmoide(u):
    g = np.exp(u)/(1 + np.exp(u))
    return g

def plotDecisionBoundary(w, X, Y):
    plt.figure()
    plt.plot(X[Y.flat==1,1],X[Y.flat==1,2],'r+',label='Clase 1')
    plt.plot(X[Y.flat==0,1],X[Y.flat==0,2],'bo', label='Clase 0')

    if X.shape[1] <= 3:
        # Only need 2 points to define a line, so choose two endpoints
        plot_x = [np.min(X[:,1])-2,  np.max(X[:,1])+2]

        # Calculate the decision boundary line
        plot_y = (-1/w[3])*(w[2]*plot_x + w[1])

        # Plot, and adjust axes for better viewing
        plt.plot(plot_x, plot_y,color = 'black',label = 'Frontera de decision')
    
        # Legend, specific for the exercise
        plt.legend()
        #axis([30, 100, 30, 100])
    else:
        #Here is the grid range
        u = np.linspace(-1, 1.5, num=50)
        v = np.linspace(-1, 1.5, num =50)

        z = np.zeros([len(u), len(v)])
        #Evaluate z = theta*x over the grid
        for i in range(len(u)):
            for j in range(len(v)):
                z[i,j] = np.dot(mapFeature(u[i].reshape(1,1), v[j].reshape(1,1)),w)
        
        xv, yv = np.meshgrid(u, v)
        plt.contour(xv, yv, z.T, 0)

def StandardLogisticRegression(X,Y,lam=0):
    #Aprendizaje
    MaxIter = 100000
    eta = 10
    N = len(Y)
    Error =np.zeros(MaxIter)
    Xent = mapFeature(X[:,0], X[:,1])
    w = np.ones(Xent.shape[1]).reshape(Xent.shape[1], 1)

    for i in range(MaxIter):
        tem = np.dot(Xent,w)
        tem2 = sigmoide(tem.T)-Y
        Error[i] = np.sum(np.abs(tem2))/N
        tem = np.dot(Xent.T,tem2.T) + lam*np.concatenate((np.array([0]).reshape(1,1),w[1:]),axis=0)
        w = w - eta*(1/N)*tem
    print('Error=',Error[-1])
    plt.plot(Error)
    plt.title("Error de entrenamiento")
    plt.xlabel("Iteraciones")
    plotDecisionBoundary(w, Xent, Y)

def PrintOverfittingReg():
    print(__doc__)

    def true_fun(X):
        return np.cos(1.5 * np.pi * X)

    np.random.seed(0)

    n_samples = 30
    degrees = [1, 4, 15]

    X = np.sort(np.random.rand(n_samples))
    y = true_fun(X) + np.random.randn(n_samples) * 0.1

    plt.figure(figsize=(14, 5))
    for i in range(len(degrees)):
        ax = plt.subplot(1, len(degrees), i + 1)
        plt.setp(ax, xticks=(), yticks=())

        polynomial_features = PolynomialFeatures(degree=degrees[i],
                                             include_bias=False)
        linear_regression = LinearRegression()
        pipeline = Pipeline([("polynomial_features", polynomial_features),
                         ("linear_regression", linear_regression)])
        pipeline.fit(X[:, np.newaxis], y)

        # Evaluate the models using crossvalidation
        scores = cross_val_score(pipeline, X[:, np.newaxis], y,
                             scoring="neg_mean_squared_error", cv=10)

        X_test = np.linspace(0, 1, 100)
        plt.plot(X_test, pipeline.predict(X_test[:, np.newaxis]), label="Model")
        plt.plot(X_test, true_fun(X_test), label="True function")
        plt.scatter(X, y, edgecolor='b', s=20, label="Samples")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.xlim((0, 1))
        plt.ylim((-2, 2))
        plt.legend(loc="best")
        plt.title("Degree {}\nMSE = {:.2e}(+/- {:.2e})".format(
            degrees[i], -scores.mean(), scores.std()))
    plt.show()

def PrintOverfittingClassify():
    Gamma=[10,1,0.01]
    n_samples = 50
    mean = np.array((1, 2))
    cov = [[1, 0], [0, 1]]
    X1 = np.random.multivariate_normal(mean, cov, n_samples)
    X2 = np.random.multivariate_normal(mean+1, cov, n_samples)
    X = np.r_[X1,X2]
    Y = np.r_[np.ones(n_samples),np.zeros(n_samples)]
    plt.figure(figsize=(20,5))

    for i,gamma in enumerate(Gamma):
        #gamma = Gamma[i]
        clf = SVC(gamma=gamma)
        clf.fit(X,Y)
        plt.subplot(1,3,i+1)
        plt.scatter(X1[:,0],X1[:,1], marker='+')
        plt.scatter(X2[:,0],X2[:,1], c= 'green', marker='o')
        h = .02  # step size in the mesh
        # create a mesh to plot in
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))


        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, m_max]x[y_min, y_max].
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        plt.contour(xx, yy, Z, cmap=plt.cm.Blues)
        plt.title('SVC gamma={:.2f}'.format(gamma))
    plt.show()

def Fronteras(clf_ini,N,Rep):
    Cov = np.identity(2) * 1.1
    Cov2 = np.array([[1.1,0.5],[0.5,1.1]])
    Mean = [1.1,2.1]
    Mean2 = [4.1,4.1]
    plt.figure(figsize=(15,5))
    plt.subplot(121)
    for i in range(Rep):
        clf = clf_ini
        x, y  = np.random.multivariate_normal(Mean, Cov, N).T
        x2, y2  = np.random.multivariate_normal(Mean2, Cov2, N).T
        X = np.r_[np.c_[x,y],np.c_[x2,y2]]
        Y = np.r_[np.ones((N,1)),np.zeros((N,1))]
        clf.fit(X,Y.flatten())
        if i == 0:
            plt.scatter(X[:,0],X[:,1],c=Y.flatten(), cmap='Set2',alpha=0.5)

        h = .02  # step size in the mesh
        # create a mesh to plot in
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))

        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, m_max]x[y_min, y_max].
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        plt.contour(xx, yy, Z, colors='k',linewidths=0.5)
    plt.title('Fronteras para N='+str(N))
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.grid()
    plt.subplot(122)
    N = N*10
    for i in range(Rep):
        clf = clf_ini
        x, y  = np.random.multivariate_normal(Mean, Cov, N).T
        x2, y2  = np.random.multivariate_normal(Mean2, Cov2, N).T
        X = np.r_[np.c_[x,y],np.c_[x2,y2]]
        Y = np.r_[np.ones((N,1)),np.zeros((N,1))]
        clf.fit(X,Y.flatten())
        if i == 0:
            plt.scatter(X[:,0],X[:,1],c=Y.flatten(), cmap='Set2',alpha=0.5)

        h = .02  # step size in the mesh
        # create a mesh to plot in
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))

        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, m_max]x[y_min, y_max].
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        plt.contour(xx, yy, Z, colors='k',linewidths=0.5)
    plt.title('Fronteras para N='+str(N))
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.grid()

class Kernel_classifier:
    def __init__(self, bandwidth=1.0):
        self.h = bandwidth
        self.n_class = 2
        self.clfs = []
        
    def fit(self,X,Y):
        classes=np.unique(Y)
        self.n_class = classes.shape[0]
        for k in range(self.n_class):
            self.clfs.append(KernelDensity(bandwidth=self.h))
            X_tem = X[Y==k,:]
            self.clfs[k].fit(X_tem)
            
    def predict(self,X):
        pred = np.zeros((X.shape[0],self.n_class))
        for k in range(self.n_class):
            pred[:,k] = self.clfs[k].score_samples(X)
        predict_class = np.argmax(pred,axis=1)
        return predict_class 

class PolynomialLinearRegression:
    def __init__(self, degree):
        self.degree = degree
        self.model = LinearRegression()
        
    def fit(self, X,y):
        self.model.fit(np.vander(X, self.degree + 1), y)
        
    def predict(self, X):
        return self.model.predict(np.vander(X, self.degree + 1))

    def score(self, X, y):
        # rmse
        return np.sqrt(np.mean((y-self.predict(X))**2))
        
def main():
    print("Module Loaded")
if __name__ == '__main__':
    main()