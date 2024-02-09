import deforest.node
from tqdm import tqdm

class Network:
	LogDiploidPrior = -0.1
	PermittedPloidies = [2,3,4]
	LogJumpPrior = -50
	JumpSize = 100000
	
	def __init__(self,qmax,jump=50000):
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
		self.StartNode = deforest.node.Node(-1,0)

		nuMin = 0.2*data.Mean
		nuMax = 0.6*data.Mean
		accelerator = 50
		fullSize = len(data.Index)
		reducedSize = int(fullSize/accelerator)

		self.Resize(reducedSize)
		res = 25
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