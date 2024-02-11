import numpy as np

class Node:
	Id = 0
	Q = -1
	Score = 0
	IsEnd = False
	def __init__(self,id,q):
		self.Id = id
		self.Q = q
		self.Connected = None
		self.CumulativeLinearScore = 0
	def SetEnd(self,nu):
		self.IsEnd = True
		self.Nu = nu

	def GetPlottingPath(self,data):

		if not self.IsEnd:
			raise Exception("A Plotting-Path can only be generated by a terminal network node")
		x = [data.Index[self.Id]]
		y = [self.Q * self.Nu]

		prevx = data.Index[self.Id]
		prevy = self.Q
		c = self.Connected
		while c.Id > -1:
			if c.Q != prevy:
				x.append(data.Index[c.Id])
				x.append(data.Index[c.Id])
				y.append(prevy*self.Nu)
				# x.append(prevx)
				y.append(c.Q * self.Nu)
			prevx = data.Index[c.Id]
			prevy = c.Q
			c = c.Connected
		x.append(data.Index[0])
		y.append(prevy * self.Nu)
		x.reverse()
		y.reverse()
		x = np.array(x).reshape((len(x),))
		y = np.array(y).reshape((len(y),))		
		return [x,y]
	def Encode(self,data,jump,mode='basic',resolution=10000):
		if not self.IsEnd:
			raise Exception("A Hash can only be generated by a terminal network node")

		x = []
		if mode == "basic":
			x = [data.Index[self.Id],self.Q]
			# y = [self.Q]
			prevx = data.Index[self.Id]
			prevy = self.Q
			c = self.Connected
			while c.Id > -1:
				if c.Q != prevy:
					x.append(c.Q)
					x.append(data.Index[c.Id]-jump)
				prevy = c.Q
				c = c.Connected
			x.append(prevy)
			x.append(data.Index[0])
			x.reverse()
		elif mode in ['sum','diff','sqdiff']:
			#gather
			tx = []
			ty = []
			prevx = data.Index[self.Id]
			prevy = self.Q
			c = self.Connected
			while c.Id > -1:
				if c.Q != prevy:
					# x.append(c.Q)
					ty.append(prevy)
					tx.append(data.Index[c.Id]-jump)
				prevy = c.Q
				c = c.Connected
			
			tx.append(0)
			ty.append(prevy)
			tx.reverse()
			ty.reverse()
			tx = np.array(tx)/data.Index[-1]
			
			space= np.linspace(0,1,resolution)
			x = []
			index = 0
			run = 0
			prev = -1
			for pos in space:
				caughtNpthing = True
				while index < len(tx) and tx[index] <= pos:
					caughtNothing = False
					catch = ty[index]
					if mode=='sum':
						run += catch
						val = run
					elif mode == 'diff':
						if prev == -1:
							diff = 0
						else:
							diff += catch - prev
						val = diff
					elif mode == 'sqdiff':
						if prev == -1:
							diff = 0
						else:
							diff += (catch - prev)**2
						val = diff
					prev = catch
					index += 1
				if caughtNothing:
					if mode == 'sum':
						val = run
					else:
						val = 0
				x.append(val)

			# x.append()
		# print(x)
		c = x.copy()
		if mode not in ['diff']:
			
			c[1:] = np.diff(c)
		return c