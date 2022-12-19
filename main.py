import numpy as np 
import matplotlib.pyplot as plt 
#import scipy as sc
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
		self.nb_dots = 10
		self.dt = 1e-5

	def glv(self,X):
		return X*(self.R.reshape((self.nb_species,1))+self.S@X)
	
	def Jglv(self,X): #one dot
		return X*self.S+np.diag((self.R.reshape((self.nb_species,1))+self.S@X).reshape(self.nb_species))
 
	# Finds value of y for a given x using step size h
	# and initial value y0 at x0.
	def rungeKutta(self,X):
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
		X = self.rungeKutta(X)
		return X

	def gen_params(self):
		if self.nb_species == 2 :
			# Classic Lotka-Volterra
			self.R = np.array([1.1,-0.4])
			self.A = np.array([[0,-0.4],[0.1,0]])
			# Competitive Lotka-Volterra
			#self.R = np.array([3,2])
			#self.A = np.array([[-1,-2],[-1,-1]])
			# det(A)=0 
			#self.R = np.array([3,2])
			#self.A = np.array([[-0.5,-0.5],[-0.5,-0.5]])
			#self.R = np.array([1,1])
			#self.A = np.array([[-0.5,-0.5],[-0.5,-0.5]])
		# 4d Chaotic Lotka-Voltera
		elif self.nb_species == 4 :
			self.R = np.array([1,0.72,1.53,1.27])
			self.A = np.array([[-1.    , -1.09  , -1.52  , -0.    ],
	       						[-0.    , -0.72  , -0.3168, -0.9792],
	       						[-3.5649, -0.    , -1.53  , -0.7191],
	       						[-1.5367, -0.6477, -0.4445, -1.27  ]])
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

	def lyapunov_exponent(self):
		x = np.random.random((self.nb_species,self.nb_dots))
		#x = np.array([0.301303,0.4586546,0.13076546,0.35574162]).reshape((self.nb_species,1))+(0.5-np.random.random((self.nb_species,self.nb_dots)))*0.1
		d_0=1e-5
		epsilon = d_0*np.ones(self.nb_species)*(1/self.nb_species**(1/2.))
		steps = int(1e7)
		lya_exp = np.zeros(self.nb_dots)
		lya_exp_plot = np.zeros((steps,self.nb_dots)) #plot
		X=np.zeros((steps,self.nb_species,self.nb_dots)) #plot
		Y=np.zeros((steps,self.nb_species,self.nb_dots))
		#z=y
		y = x+epsilon.reshape((self.nb_species,1))
		
		start = int(1e6)
		test = int(1e5)
		ite = int(1e2)	
		
		for i in range(start):
			x = self.step(x)
			y = self.step(y)
			d_1 = np.sum((x-y)**2, axis=0)**(1/2.)
		
		y = x + d_0*((y-x)/d_1)
			
		for i in range(test):
			for j in range(ite):
				x = self.step(x)
				y = self.step(y)
				X[i*ite+j,:,:]=x
				#z = self.step(z)
				
			d_1 = np.sum((x-y)**2, axis=0)**(1/2.)
			lya_exp = lya_exp + ((1/(self.dt*ite))*np.log(d_1/d_0)-lya_exp)/(i+1)
			lya_exp_plot[i,:] = (1/(self.dt*ite))*np.log(d_1/d_0)
			y = x + d_0*((y-x)/d_1)
			#Y[i,:]=z[:,0]

		print(lya_exp)
		print()
		print(np.mean(lya_exp_plot,axis=0),np.std(lya_exp_plot,axis=0))
		fig, axes = plt.subplots(3,2)
		axes[0,0].plot(X[:,0,0],X[:,1,0],'r')
		axes[1,0].plot(X[:,0,0],X[:,2,0],'r')
		axes[2,0].plot(X[:,0,0],X[:,3,0],'r')
		axes[0,1].plot(X[:,1,0],X[:,2,0],'r')
		axes[1,1].plot(X[:,1,0],X[:,3,0],'r')
		axes[2,1].plot(X[:,2,0],X[:,3,0],'r')
		#axes[0,0].plot(Y[:,0],Y[:,1],'b')
		#axes[1,0].plot(Y[:,0],Y[:,2],'b')
		#axes[2,0].plot(Y[:,0],Y[:,3],'b')
		#axes[0,1].plot(Y[:,1],Y[:,2],'b')
		#axes[1,1].plot(Y[:,1],Y[:,3],'b')
		#axes[2,1].plot(Y[:,2],Y[:,3],'b')
		fig,ax = plt.subplots()
		ax.plot(X[:,0,0])
		ax.plot(X[:,1,0])
		ax.plot(X[:,2,0])
		ax.plot(X[:,3,0])
		fig,ax = plt.subplots()
		ax.plot(lya_exp_plot[:test,0])
		plt.show()

	def bifurcations(self):
		s = np.linspace(0.9,1.1,10)
		for i in s : 
			self.S = i*(self.A-np.diag(np.diag(self.A)))+np.diag(np.diag(self.A))
			self.lyapunov_exponent()

		self.S = self.A
	
	def vector_field_2D(self,x_min,y_min,x_max,y_max,dx): #grid unclear
		self.x, self.y = np.meshgrid(np.arange(x_min,x_max,dx),np.arange(y_min,y_max,dx))
		self.vf = np.zeros((self.x.shape[0],self.x.shape[1],2))
		for idx in np.ndindex(self.x.shape):
			self.vf[idx]=self.glv(np.array([self.x[idx],self.y[idx]]))

	def print_vector_field(self):
		plt.figure()
		M = np.hypot(self.vf[:,:,0], self.vf[:,:,1])
		Q = plt.quiver(self.x, self.y, self.vf[:,:,0]/M, self.vf[:,:,1]/M, M, units='width')
		plt.scatter(self.x, self.y, color='k', s=5)
		if np.abs(np.linalg.det(self.A))>10**(-4):
			plt.plot(self.equilibrium_points[:,0],self.equilibrium_points[:,1],'ro') 
		plt.show()


		
def main():
	model_1 = Model_glv(4)
	model_1.gen_params()
	#model_1.equilibrium_points()
	model_1.lyapunov_exponent()
	#model_1.bifurcations()
	#model_1.vector_field_2D(0,0,2,2,0.1)
	#model_1.print_vector_field()




if __name__ == "__main__":
    main()
