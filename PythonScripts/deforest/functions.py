import numpy as np
from scipy import special
import deforest.data
def ale(x,y):
	return (np.maximum(x,y) + np.log(1 + np.exp(-np.abs(x-y))))



def dualBinomial(k,nu,gamma,sigma,sigma2):
	# return -0.5 * ((k-nu)*1.0/25)**2

	
	
	p = nu/(nu + sigma*sigma)
	n = (nu/sigma)**2


	if nu > 0:
		v = special.gammaln(n+k) - special.gammaln(k+1) - special.gammaln(n)
		w = k * np.log(1 - p) + n * np.log(p)
		p1 = np.log(gamma) +v + w

	else:
		p1 = -9999999999
		if type[k] == type(0):
			p1[k==0] = np.log(gamma)
		# if k == 0:
		# 	p1 =np.log(gamma)
		# else:
		# 	p1 = -99999999
	
	p2 = np.log(1-gamma)-0.5* ((k-nu)*1.0/sigma2)**2
	return ale(p1,p2)

def loadUMAPData(umapfile):

	load_x = []
	load_y = []
	names = []
	with open(umapfile) as f:
		for line in f:	
			vals = line.rstrip().split(' ')
			x = []
			res = int(vals[0].split('_')[1])
			name = vals[0]
			
			for i in range(2,len(vals)):
				x.append(float(vals[i]))
			# if x[-1] > 0:
			# 	x = np.array(x)/x[-1]
			load_x.append(x)
			tag = int(vals[1])
			load_y.append(tag)
			names.append(name)
	N = len(load_x)
	# p = np.random.permutation(N)
	load_x = np.array(load_x)
	load_y = np.array(load_y)

	return load_x,load_y,names

def GetEncoding(file,gap,distribution,resolution,mode):
	s = deforest.DataStruct(file,0,5000)
	N = deforest.Network(10,gap)
	optimalPath = N.Navigate(s,distribution)
	return optimalPath.Encode(s,N.JumpSize,mode,resolution)
