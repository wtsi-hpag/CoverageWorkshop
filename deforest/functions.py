import numpy as np
from scipy import special
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
		p1[k==0] = np.log(gamma)
		# if k == 0:
		# 	p1 =np.log(gamma)
		# else:
		# 	p1 = -99999999
	
	
	p2 = np.log(1-gamma)-0.5* ((k-nu)*1.0/sigma2)**2
	return ale(p1,p2)