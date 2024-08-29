import numpy as np
import seaborn as sns
def embedding(pos,i):
    return np.sin(pos/10000**(2*i/1028)),np.cos(pos/10000**(2*i/1028))

pos_array = np.arange(0,500)
dim_array = np.arange(0,250)


embedding_sin = np.array([[embedding(pos,dim)[0] for pos in pos_array] for dim in dim_array[0::2]])
print(embedding_sin.shape)
embedding_cos = [[embedding(pos,dim)[1] for pos in pos_array ] for dim in dim_array[1::2]]
embedding_result = np.zeros(shape=(250,500))
embedding_result[0::2,:] = embedding_sin
embedding_result[1::2,:] = embedding_cos

plt.figure()
sns.heatmap(embedding_result.transpose(1,0),cmap='hsv')
plt.xlabel('Variable dimension')
plt.ylabel('Sequence Length')
plt.show()
