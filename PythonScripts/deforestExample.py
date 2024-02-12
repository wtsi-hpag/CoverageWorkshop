import matplotlib.pyplot as plt
import numpy as np
import deforest



###### CELL ONE

file = "../Data/Human.dat"
dataThinning = 0.999 #This value controls the level of smoothing applied to the data. 0 is the raw data, 0.99(etc) is very smooth.
baseSkipping = 1000 #This value controls the base-sampling rate. Higher values makes the later code run quicker, but uses less information
s = deforest.DataStruct(file,dataThinning,baseSkipping) #We have provided an inbuilt data-parser for you, because we're nice.

plt.figure(1)
plt.clf() #comment this out if you do want to layer multiple plots
plt.plot(s.Index,s.Coverage)
plt.ylabel("Coverage")
plt.xlabel("Chromosome Index")
plt.pause(0.05)
plt.draw()


#### CELL TWO

def jackLogProbability(k,nu):
	#This is a (somewhat) biologically motivated distribution function, which assumes that the distribution is Poisson 
	#distributed around the norm, but with some statistical noise distributed according to the Gamma distribution; which 
	#in turn means the distribution is a Negative Binomial distribution. 
	#Then on *top* of that, I add a secondary noise term which is just a big gaussian 
	#background noise centred on nu - representing a general background uncertainty in everything
	
	fractionOfDataWhichIsReal = 0.99
	dataNoise = 5
	backgroundNoise = 10
	return deforest.dualBinomial(k,nu,fractionOfDataWhichIsReal,dataNoise,backgroundNoise)

def myLogProbability(k,nu):
	sigma = 5 
	return -0.5 * ((k-nu)/sigma)**2  #this is just a boring Gaussian that I put down. You make up your own function. Get weird.

nu = 45
k = np.arange(0,100)

plt.figure(2)
plt.clf()
y1 = jackLogProbability(k,nu)
plt.plot(k,y1 - np.max(y1),label="Biological Function")
y2 = myLogProbability(k,nu)
plt.plot(k,y2-np.max(y2),label="My Function")
plt.xlabel("Observed Coverage")
plt.legend()
plt.ylabel(f"Log-Probability with Mean {nu}")
plt.pause(0.02)
plt.draw()


### CELL THREE

preSmoothing = 0
dataSkip = 5000 #if you are running on the web hosted version (it's very slow!)
# dataSkip = 500 #if you are running locally
maxHarmonic = 16
minimumJump = 1e6


s = deforest.DataStruct(file,preSmoothing,dataSkip)
sPlot = deforest.DataStruct(file,0.999,dataSkip) #load a smoothed version for plotting!

N = deforest.Network(maxHarmonic,minimumJump)
N.LogJumpPrior = -50 #Set this to zero to permit more jumps, set to a large negative value (< -1000) to remove jumps that the code is less certain about
N.LogDiploidPrior = -0.1 #Set this to zero to disable the Diploid Prior (for maximum impact, also increase maxHarmonic to see what happens!)
N.SearchResolution = 50 # Turn this value down to make it faster, but too low and your results become nonsense as nu cannot accurately be determined
optimalPath = N.Navigate(s,jackLogProbability)



## Now we plot the model
fig,axs = plt.subplots(1,3)
axs[0].plot(sPlot.Index,sPlot.Coverage,'k',label="Data (thin= " + str(preSmoothing) + ")")
[xx,yy] = optimalPath.GetPlottingPath(s) 
axs[0].plot(xx,yy,label="Harmonic Fit (thin= " + str(preSmoothing) + ")")
axs[0].legend()
axs[0].set_xlabel("Chromosome Index")
axs[0].set_ylabel("Coverage")
axs[1].plot(xx,yy/optimalPath.Nu)
axs[1].set_xlabel("Chromosome Index")
axs[1].set_ylabel("Harmonic")
axs[2].plot(N.nus,N.scores)
axs[2].set_xlabel("Fundamental Frequency (nu)")
axs[2].set_ylabel("Inferred Score")
plt.pause(0.02)
plt.draw()

#### CELL FOUR

fig,axs=plt.subplots(1,3)
modes = ['diff','sum','sqdiff']
for i in range(len(modes)):
	enc = optimalPath.Encode(s,N.JumpSize,modes[i],10)
	axs[i].plot(np.linspace(0,1,len(enc)),enc)
	axs[i].set_title(modes[i])
plt.show()


input("Enter to exit")