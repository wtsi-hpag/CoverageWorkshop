import numpy as np
import deforest
import tensorflow as tf

trainingType = "AllData" #choices are AllData, BigGap, SmallGap, HighConfidence or one of the SingleModes (requires rewriting the trainingFile)
resolution = 100 #choices are 10, 20 50, 100 or 1000
encodingMethod = 'diff' # choices are diff, sum or sqdiff
trainingFraction = 0.8
epoch_count = 5
testFile = "../Data/Diane.dat"
testGap = 1e6
trainingFile = f"../ProcessedData/AllData/Res_{resolution}_{encodingMethod}.dat"
def loadTrainingData(file,trainingFraction):

	load_x = []
	load_y = []
	names = []
	with open(file) as f:
		for line in f:	
			vals = line.rstrip().split(' ')
			x = []

			
			params = vals[0].split('_')
			name = params[0]
			## if you want to get fancy, you can additionally provide the network with the deforester parameters which were used to generate it
			## uncomment these lines to add three dimensions to your feature vector -- but remember that you'll need to do that for your encoding later on!
			# gap = int(params[1])
			# noise = float(params[2])/10000
			# sigma = float(params[3])
			# x = [gap,noise,sigma]
			for i in range(2,len(vals)):
				x.append(float(vals[i]))
			tag = int(vals[1])			
			load_x.append(x)
			load_y.append(tag)
			names.append(name)
	N = len(load_x)
	if trainingFraction < 1.0:
		p = np.random.permutation(N)
		load_x = np.array(load_x)[p]
		load_y = np.array(load_y)[p]

	trainN  = int(trainingFraction*N)

	return (load_x[:trainN],load_y[:trainN]), (load_x[trainN:],load_y[trainN:]), names


(x_train, y_train), (x_test, y_test),names = loadTrainingData(trainingFile,trainingFraction)
outputDimension = 3 ## We have 3 categories and are using 1-hot encoding
model = tf.keras.models.Sequential([
	tf.keras.layers.Dense(len(x_train[0]), activation='relu'),
	tf.keras.layers.Dense(2*len(x_train[0]), activation='relu'),
	tf.keras.layers.Dense(1000, activation='relu'),
	tf.keras.layers.Dense(500, activation='relu'),
	tf.keras.layers.Dense(100, activation='relu'),
	tf.keras.layers.Dense(10, activation='relu'),
	tf.keras.layers.Dense(outputDimension), 
	tf.keras.layers.Softmax() #this final layer converts values -inf -> + inf into probabilities
])


def Prediction(model,encoding):
	pred = model.predict([encoding],verbose=0)
	indexSort = np.argsort(pred[0])[::-1]
	types = ['Normal','DFTD1','DFTD2']
	print("Model Predictions:")
	for i in range(len(indexSort)):
		print(f"{types[indexSort[i]]}:\t{pred[0][indexSort[i]]*100:.4f}%")


def jackLogProbability(k,nu):
	fractionOfDataWhichIsReal = 0.99
	dataNoise = 5
	backgroundNoise = 10
	return deforest.dualBinomial(k,nu,fractionOfDataWhichIsReal,dataNoise,backgroundNoise)


loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])


M = model.fit(x_train, y_train, epochs=epoch_count, validation_data=(x_test,y_test))



myEnc = deforest.GetEncoding(testFile,testGap,jackLogProbability,resolution,encodingMethod) 

print("Input file:",testFile)
Prediction(model,myEnc)