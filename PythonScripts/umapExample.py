import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import umap.umap_ as umap
import deforest
umapfile =  "../ProcessedData/SingleMode/Mode_10_Res_100_diff.dat"



(data_x,data_y,names) = deforest.loadUMAPData(umapfile)
scaled_data = StandardScaler().fit_transform(data_x) #this converts the array into standard-deviations from the mean, which means the model doesn't have to spend effort learning the mean!

reducer = umap.UMAP().fit(scaled_data)
embedding = reducer.embedding_
plt.figure(5)
plt.clf()
plt.scatter(
    embedding[:, 0],
    embedding[:, 1],
	s=10,
    c=data_y)

# This function formatter will replace integers with target names
formatter = plt.FuncFormatter(lambda val, loc: ["Normal","DFTD1","DFTD2"][int(val)])

# We must be sure to specify the ticks matching our target names
plt.colorbar(ticks=[0, 1, 2], format=formatter)
plt.xlabel("UMAP1")
plt.ylabel("UMAP2")

## Enable this to add annotations to the plot
# for i in range(len(names)):
# 	plt.annotate(names[i],(embedding[i,0],embedding[i,1]),size=5)
plt.show()