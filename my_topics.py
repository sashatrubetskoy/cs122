import re

from time import time

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer,\
	ENGLISH_STOP_WORDS
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.datasets import fetch_20newsgroups

COMMON_NAMES = ['nick', 'michael', 'mike', 'sam', 'sarah', 'frank', 'tom',
	'alan', 'kevin', 'jamie', 'tyler', 'josh', 'david']
n_samples = 20000
n_features = 3000
n_topics = 10
n_top_words = 20


def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
    print()


def go( n_topics=10, n_top_words=20, n_samples=20000, n_features=3000):
	stop_words = ENGLISH_STOP_WORDS.union(COMMON_NAMES)


	t0 = time()
	print('Filtering data...')
	data_samples = [l for l in raw[-n_samples:] if len(l)>100]
	print('Data filtered in {0:.3f}s.'.format(time()-t0))

	print('Extracting tf features for LDA...')
	tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2,
	                                max_features=n_features,
	                                stop_words=stop_words)
	t0 = time()
	tf = tf_vectorizer.fit_transform(data_samples)
	print('done in {0:.3f}s.'.format(time() - t0))

	print('Fitting LDA models with tf features,', \
	      'n_samples={} and n_features={}...'.format(n_samples, n_features))
	lda = LatentDirichletAllocation(n_topics=n_topics, max_iter=5,
	                                learning_method='online',
	                                learning_offset=50.,
	                                random_state=0)
	t0 = time()
	lda.fit(tf)
	print('done in {0:.3f}s.'.format(time() - t0))

	print('\nTopics in LDA model:')
	tf_feature_names = tf_vectorizer.get_feature_names()
	print_top_words(lda, tf_feature_names, n_top_words)


t0 = time()
print('Reading scripts...')
f = open("data/scripts.csv")
raw = f.readlines()
print('Scripts read in {}s.'.format(time()-t0))

go()
#for n_topics in range(10, 21):
#	go(n_topics=n_topics)
