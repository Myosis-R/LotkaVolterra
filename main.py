import numpy as np 
import matplotlib.pyplot as plt
from functools import partial
from mpl_toolkits.mplot3d import Axes3D
plt.style.use('ggplot')
from scipy import stats,spatial
from multiprocessing import Pool
import warnings
import sys
#import manim as mn

class Model_glv:
    """docstring for glv"""
    def __init__(self, nb_species):
        self.nb_species = nb_species
        self.A = np.ones((nb_species,nb_species))
        self.S = np.ones((nb_species,nb_species)) # copy for bifurcation
        self.R = np.ones(nb_species)
        self.nb_dots = 1000
        
        self.dt = 1e-2

    def glv(self,X): #X(nb_species,1or+) !!!
        return X*(self.R.reshape((self.nb_species,1))+self.S@X)
    
    def Jglv(self,X): #one dot
        return X*self.S+np.diag((self.R.reshape((self.nb_species,1))+self.S@X).reshape(self.nb_species))
 
    # Finds value of y for a given x using step size h
    # and initial value y0 at x0.
    def rungeKutta(self,X):
        X = X.reshape((self.nb_species,-1))
        k1 = self.glv(X)
        k2 = self.glv(X + 0.5 * k1*self.dt)
        k3 = self.glv(X + 0.5 * k2*self.dt)
        k4 = self.glv(X + k3*self.dt)
        # Update next value of y
        X = X + (1.0 / 6.0)*self.dt*(k1 + 2 * k2 + 2 * k3 + k4)
        return X
    
    def eulerForward(self,X):
        X = self.glv(X)*self.dt + X 
        return X

    def step(self,X): # résolution plus complexe ?
        return self.rungeKutta(X)

    def gen_params(self):
        if self.nb_species == 2 :
            # Classic Lotka-Volterra
            #self.R = np.array([1.1,-0.4])
            #self.A = np.array([[0,-0.4],[0.1,0]])
            # Competitive Lotka-Volterra
            #self.R = np.array([-1,1])
            #self.A = np.array([[1,0],[-2,1]])
            #self.R = np.array([3,2])
            #self.A = np.array([[-1,-2],[-1,-1]])
            # det(A)=0
            #self.R = np.array([3,2])
            #self.A = np.array([[-0.5,-0.5],[-0.5,-0.5]])
            self.R = np.array([1,1])
            self.A = np.array([[-1,-1],[-1,-1]])
        # 4d Chaotic Lotka-Voltera
        elif self.nb_species == 4 :
            self.R = np.array([1,0.72,1.53,1.27])
            self.A = np.array([[-1.    , -1.09  , -1.52  , -0.    ],
                                   [-0.    , -0.72  , -0.3168, -0.9792],
                                   [-3.5649, -0.    , -1.53  , -0.7191],
                                   [-1.5367, -0.6477, -0.4445, -1.27  ]])
            #limit cycle
#            self.A = np.array([[-1.      , -1.0355  , -1.444   , -0.      ],
#                                   [-0.      , -0.72    , -0.30096 , -0.93024 ],
#                                   [-3.386655, -0.      , -1.53    , -0.683145],
#                                   [-1.459865, -0.615315, -0.422275, -1.27    ]])



        else :
            sys.exit('no gen params')
        self.S = self.A # copy for bifurcation

    def equilibrium_points(self):
        #self.jacob_cp0 = np.diag(self.R)
        if np.abs(np.linalg.det(self.A))>10**(-4):
            self.equilibrium_point_finder()
            #self.jacob_cp1 = self.A*self.equilibrium_points[0,:]
        else:
            warnings.warn("Warning Det(A)~0")


    def equilibrium_point_finder(self): #!!!!aucune verif des determinants et calcul Jacob
        self.equilibrium_points = np.zeros((2**self.nb_species,self.nb_species))
        for i in range(2**self.nb_species-1):
            species = np.flatnonzero(np.logical_not(np.unpackbits(np.array(i, dtype=np.uint8),bitorder='little'))[:self.nb_species]) #resoud le systeme pour un sous ensemble d'especes non nul limite le nb d'espece a 2^8-1
            species_0 = np.flatnonzero(np.unpackbits(np.array(i, dtype=np.uint8),bitorder='little')[:self.nb_species])
            A_ = np.delete(np.delete(self.A, species_0, 0), species_0, 1)
            R_ = np.delete(self.R, species_0)
            equilibrium_point = np.delete(self.equilibrium_points[i,:],species)
            self.equilibrium_points[i,species] = -np.linalg.inv(A_)@R_
        print(self.equilibrium_points)
    
    def plot4D(self,n,start):
        steps = int(3e5)
        X = np.zeros((self.nb_species,steps,n))
        X[:,0,:] = np.random.random((self.nb_species,n))

        for i in range(steps-1):
            X[:,i+1,:] = self.step(X[:,i,:])

        fig, axes = plt.subplots(3,2)
        axes[0,0].plot(X[0,start:],X[1,start:])
        axes[1,0].plot(X[0,start:],X[2,start:])
        axes[2,0].plot(X[0,start:],X[3,start:])
        axes[0,1].plot(X[1,start:],X[2,start:])
        axes[1,1].plot(X[1,start:],X[3,start:])
        axes[2,1].plot(X[2,start:],X[3,start:])
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        fig.tight_layout()
        img = ax.scatter(X[0,start:],X[1,start:],X[2,start:],c=X[3,start:],cmap=plt.hot(),s=0.4)
        clb = fig.colorbar(img)
        ax.set_xlabel(r'$x_1$')
        ax.set_ylabel(r'$x_2$')
        ax.set_zlabel(r'$x_3$')
        clb.ax.set_ylabel(r'$x_4$')

        plt.show()

    def bassinOfAttraction(self):
        steps = int(3e5)
        record = int(3e4)
        x = np.random.random((self.nb_species,self.nb_dots))
        X = np.zeros((self.nb_species,self.nb_dots,record))
        for i in range(steps):
            x = self.step(x)
        for i in range(record):
            x = self.step(x)
            X[:,:,i] = x
        print('ok')
        kde = stats.gaussian_kde(X.reshape((self.nb_species,-1)))
        kdt = spatial.KDTree(X.reshape((self.nb_species,-1)).T)
        return kde,kdt
    
    def cvAttractor(self,n,stop):#no optimization
        dots = np.mgrid[0:1:n*1j,0:1:n*1j].reshape((2,-1)) #ndim!!!!
        kde,kdt = self.bassinOfAttraction()
        fixVar = np.array([0.3,0.46])
        dots = np.vstack((fixVar.reshape((2,1))*np.ones((2,n**2)),dots))
        cv = np.array(list(map(partial(self.convergence,stop,kdt),dots.T)))
        plt.matshow(cv.reshape((n,n)))
        plt.show()

    def convergence(self,stop,kdt,dot):
        for i in range(stop):
            if i%1000==0 and kdt.query(dot.flatten(),k=1)[0]<0.005 :
                return np.log(i+1)
            dot = self.step(dot)
        return np.log(stop)

    def evolution(self):
        steps = int(3e5)
        record = int(4e3)
        x = np.random.random((self.nb_species,self.nb_dots))
        X = np.zeros((record,self.nb_dots))
        for i in range(steps):
            x = self.step(x)
        for i in range(record):
            x = self.step(x)
            X[i,:] = x[0,:]
        return x,np.amax(X,axis=0)

    def lyapunov_exponent(self):
        x = np.random.random((self.nb_species,self.nb_dots))
        #x = np.array([0.301303,0.4586546,0.13076546,0.35574162]).reshape((self.nb_species,1))+(0.5-np.random.random((self.nb_species,self.nb_dots)))*0.1
        d_0=1e-5
        epsilon = d_0*np.ones(self.nb_species)*(1/self.nb_species**(1/2.))
        steps = int(1e6)
        lya_exp = np.zeros(self.nb_dots)
        lya_exp_plot = np.zeros((steps,self.nb_dots)) #plot
        y = x+epsilon.reshape((self.nb_species,1))
        
        start = int(2e6)
        test = int(1e5)
        ite = int(1e2)    
        
        for i in range(start):
            x = self.step(x)
            y = self.step(y)
            d_1 = np.sum((x-y)**2, axis=0)**(1/2.)
        
        y = x + d_0*((y-x)/d_1)
        self.dt = 1e-4
        for i in range(test):
            for j in range(ite):
                x = self.step(x)
                y = self.step(y)
                
            d_1 = np.sum((x-y)**2, axis=0)**(1/2.)
            lya_exp = lya_exp + ((1/(self.dt*ite))*np.log(d_1/d_0)-lya_exp)/(i+1)
            lya_exp_plot[i,:] = (1/(self.dt*ite))*np.log(d_1/d_0)
            y = x + d_0*((y-x)/d_1)
        self.dt = 1e-2
        print(np.mean(lya_exp),np.amax(lya_exp),np.amin(lya_exp))
        print()
        
        fig,ax = plt.subplots()
        ax.plot(lya_exp_plot[:test,0])
        plt.show()

    def bifurcations(self,min,max,n):
        bif = np.zeros((n,self.nb_species,self.nb_dots))
        s = np.linspace(min,max,n)#np.linspace(0.8,1.3,n)
        max = np.zeros((n,self.nb_dots))
        for i in range(n): 
            self.S = s[i]*(self.A-np.diag(np.diag(self.A)))+np.diag(np.diag(self.A))
            bif[i,:,:],max[i,:] = self.evolution()
            print(i)
        
        fig,ax =plt.subplots()
        plt.hist2d((s.reshape((n,1))*np.ones((n,self.nb_dots))).flatten(),max[:,:].flatten(), (n,100),density=True, facecolor='g', alpha=0.75,cmap=plt.cm.jet)
        plt.show()
        if self.nb_species == 2 :
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            fig.tight_layout()
            for i in range(n):
                ax.scatter(s[i]*np.ones(self.nb_dots),bif[i,0,:],bif[i,1,:])
            ax.set_xlabel(r'$s$')
            ax.set_ylabel(r'$x$')
            ax.set_zlabel(r'$y$')
            plt.show()
        else :
            fig,ax = plt.subplots()
            fig.tight_layout()
            for i in range(n):
                ax.scatter(s[i]*np.ones(self.nb_dots),max[i,:])#bif[i,0,:])
            ax.set_xlabel(r'$s$')
            ax.set_ylabel(r'$x_{max}$')
            plt.show()
        self.S = self.A
    
    def vector_field_2D(self,x_min,y_min,x_max,y_max,dx): #grid unclear
        x, y = np.meshgrid(np.arange(x_min,x_max,dx),np.arange(y_min,y_max,dx))
        vf = np.zeros((2,x.shape[0],x.shape[1]))
        for i in range(x.shape[0]):
            (vf[0,i,:],vf[1,i,:])=self.glv(np.array([x[i,:],y[i,:]]))
        self.print_vector_field(x,y,vf)

    def print_vector_field(self,x,y,vf):
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$y$')
        plt.axis('equal')
        M = np.hypot(vf[0,:,:], vf[1,:,:])
        Q = plt.quiver(x, y, vf[0,:,:]/M, vf[1,:,:]/M, M,minlength=2,headaxislength=5,headwidth=5)
        plt.scatter(x, y, color='k', s=5)
        cb = plt.colorbar(Q)
        if False and np.abs(np.linalg.det(self.S))>10**(-4):
            plt.plot(self.equilibrium_points[:,0],self.equilibrium_points[:,1],'ro') 
        fig.tight_layout()
        #plt.savefig('case2D.eps')
        plt.show()


        
def main():
    model_1 = Model_glv(2)
    model_1.gen_params()
    #model_1.equilibrium_points()
    #model_1.lyapunov_exponent()
    #model_1.bifurcations(0.90,0.98,20)
    model_1.S = 2*(model_1.A-np.diag(np.diag(model_1.A)))+np.diag(np.diag(model_1.A))
    #model_1.plot4D(1,15000)
    model_1.vector_field_2D(0.99,0,1.01,0.02,0.002)
    #model_1.cvAttractor(100,100000)
            




if __name__ == "__main__":
    main()
