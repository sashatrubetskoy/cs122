import final_model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

#N_VOCA = 10000

# conf_errors = 4*[0]
# for conf, nullval in zip(range(4), [3.5, 3.5, 3.5, 3.5]):
# 	model = final_model.CollaborativeTopicModel(10, N_VOCA, nullval, params = conf)
# 	model.fit()
# 	conf_errors[conf] = model.train_error
# 	model = None
# print('The Best confidence params are represented by {}'.format(conf_errors.index(min(conf_errors))))
# print(conf_errors)


#TRY DIFFERENT NUM_TOPICS
ks = range(50, 110, 10)

# train_errors_k = np.zeros(len(ks))
# test_errors_k = np.zeros(len(ks))
# for i in range(len(ks)):
# 	model = final_model.CollaborativeTopicModel(ks[i], N_VOCA, 0)
# 	model.fit()
# 	train_errors_k[i] = model.train_error
# 	test_errors_k[i] = model.test_error
# 	model = None

# plot.figure()
# plot.plot(ks, train_errors_k)
# plot.title('Training error vs. number of topics')
# plot.xlabel('k (num topics)')
# plot.ylabel('RMSE')
# plot.savefig('recctrainkerrors.png')

# plot.figure()
# plot.plot(ks, test_errors_k)
# plot.title('Testing errors vs. number of topics')
# plot.xlabel('k (num topics)')
# plot.ylabel('RMSE')
# plot.ylim(0, 20)
# plot.savefig('recctestkerrors.png')

BEST_NUM_TOPICS = 50

# #TRY CONSIDERING DIFFERENT VOCABULARY SIZES

# sizes = [1000, 5000, 10000, 20000]

# train_errors_k = np.zeros(len(sizes))
# test_errors_k = np.zeros(len(sizes))

# for i, s in enumerate(sizes):
# 	model = final_model.CollaborativeTopicModel(n_topic=BEST_NUM_TOPICS, n_voca=s, nullval=0, params=1)
# 	model.fit()
# 	train_errors_k[i] = model.train_error
# 	test_errors_k[i] = model.test_error

# plot.figure()
# plot.plot(sizes, train_errors_k)
# plot.title('TRAIN ERRORS VS VOCAB SIZE')
# plot.xlabel('v')
# plot.ylabel('RMSE')
# plot.savefig('recctrainverrors.png')


# plot.figure()
# plot.plot(sizes, test_errors_k)
# plot.title('TEST ERRORS VS VOCAB SIZE')
# plot.xlabel('v')
# plot.ylabel('RMSE')
# plot.savefig('recctestverrors.png')

BEST_VOCAB = 10000

MODEL = final_model.CollaborativeTopicModel(BEST_NUM_TOPICS, BEST_VOCAB, 3)
MODEL.fit()

# V = item(doc)_topic matrix, n_topic x n_item
pca = KernelPCA(n_components = 2)
projected = pd.DataFrame(pca.fit_transform(MODEL.V.T))
print('pca.explained_variance_ratio',pca.explained_variance_ratio_) 

# plot.figure()

projected.columns = ['x', 'y']
print(projected)
projected.plot(kind='scatter', x='x', y='y')
plt.show()

# plt.show()
# plot.scatter(projected[:0], projected[:1])
# plot.title("Projecting Latent Item Matrix into 2-dimensions using PCA")
# plot.xlabel("Principal Component 1")
# plot.ylabel("Principal Component 2")

# AFTER VINAY WRITES HIS CODE, YOU CAN LABEL THE POINTS BY MOVIE NAME HERE BASED ON HOW HE INDEXES!!!!!
plot.savefig('movielatentspace.png')


MODEL.lda.print_all_topics(20)






