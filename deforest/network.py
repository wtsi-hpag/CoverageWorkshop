import deforest.node
from tqdm import tqdm
import time
class Network:
	
	
	def __init__(self,qmax,jump=50000):
		self.LogDiploidPrior = -0.1
		self.PermittedPloidies = [2,3,4]
		self.LogJumpPrior = -50
		self.Accelerate = 50
		self.SearchResolution = 25
		self.Q = qmax
		self.JumpSize = jump

	def Resize(self,size):
		self.Nodes = []
		for i in range(size):
			r = []
			for q in range(self.Q+1):
				r.append(deforest.node.Node(i,q))
			self.Nodes.append(r)

	def CheatPrior(self,nu,meanNu):
		sigma = 15
		return -0.5 * ((nu - meanNu/2)/sigma)**2

	def Navigate(self,data,probabilityFunction):
		startTime = time.time()
		self.StartNode = deforest.node.Node(-1,0)

		nuMin = 0.2*data.Mean
		nuMax = 0.6*data.Mean
		accelerator = self.Accelerate
		fullSize = len(data.Index)
		reducedSize = int(fullSize/accelerator)

		self.Resize(reducedSize)
		res = self.SearchResolution
		self.nus = []
		self.scores = []
		self.UncorrectedScores = []
		for i in tqdm(range(res),leave=False):
			testNu = nuMin + i * (nuMax-nuMin)/(res -1)
			v = self.NetworkPass(data,probabilityFunction,testNu,accelerator)
			self.nus.append(testNu)
			self.scores.append(v.Score + self.CheatPrior(testNu,data.Mean))
			self.UncorrectedScores.append(v.Score)
			if i == 0 or v.Score > bestScore:
				bestScore = v.Score
				nu = testNu

		self.Resize(fullSize)
		r = self.NetworkPass(data,probabilityFunction,nu,1,True)
		r.SetEnd(nu)
		elapsedTime = time.time() - startTime
		print(f"Navigation complete. {fullSize} nodes navigated in {elapsedTime} seconds.\nMaximum Score was {r.Score}")
		return r

	def NetworkPass(self,data,probabilityFunction,nu,scanSpeed,mode=False):
		dataGap = data.Index[1] - data.Index[0]
		jumpSteps = int(self.JumpSize / dataGap)
		for i in tqdm(range(len(self.Nodes)),disable=not mode,leave=False):
			for q in range(0,self.Q+1):

				if i == 0:
					prevNode = self.StartNode
				else:
					prevNode = self.Nodes[i-1][q]

				nodeCost = probabilityFunction(data.Coverage[i*scanSpeed],nu*q) 
				# print(nodeCost)
				if q not in self.PermittedPloidies:
					nodeCost += self.LogDiploidPrior
				self.Nodes[i][q].CumulativeLinearScore = prevNode.CumulativeLinearScore + nodeCost
				bestCost = prevNode.Score + nodeCost
				if i >= jumpSteps:
					for jumpQ in range(0,self.Q+1):
						if jumpQ != q:

							cumDiff = self.Nodes[i][q].CumulativeLinearScore - self.Nodes[i-jumpSteps][q].CumulativeLinearScore

							jumpScore = self.Nodes[i-jumpSteps][jumpQ].Score + cumDiff + self.LogJumpPrior

							if (jumpScore > bestCost):
								bestCost = jumpScore
								prevNode = self.Nodes[i-jumpSteps][jumpQ]

				self.Nodes[i][q].Connected = prevNode
				self.Nodes[i][q].Score = bestCost
				# print(prevNode.Score, bestCost, self.Nodes[i][q].Score)
				# print('\ti=%d q=%d has score %f and is connecting to %d-%d' % (i,q,bestCost,prevNode.Id,prevNode.Q))
				
		bestRoute = None
		bestScore = -9e99
		for q in range(0,self.Q+1):
			if self.Nodes[-1][q].Score > bestScore:
				bestRoute = self.Nodes[-1][q]
				bestScore = self.Nodes[-1][q].Score
		return bestRoute