from time import time
import pandas as pd
import numpy as np
import numpy.linalg
import my_topics
import random
import pickle

class CollaborativeTopicModel:
    """
    Building hybrid recommender system based on the following paper:

    Wang, Chong, and David M. Blei. "Collaborative topic modeling for recommending scientific articles."
    Proceedings of the 17th ACM SIGKDD international conference on Knowledge discovery and data mining. ACM, 2011.
    PAPER HERE: 
    Arguments
    ----------
    n_topic: number of topics (hyperparameter)
    n_vica: size of vocabulary
    n_movie: int
        number of movie
    n_user: int
        number of movie
    nullval: number to use for 
    """

    def __init__(self, printout = True, full = False, n_topic = 75, lamu = 0.05, lamv = 3000, n_voca = 5000, nullval = 3.5, e = 1e-100, error_diff = 0.001, params = 2, ratingsfile = 'ratings_100K.csv', scriptsfile = "matched1.csv"):
        
        #User x Item ratings matrix, Splitting into Train and Test, making dict of names to movieIDs
        print('Cleaning Ratings')
        self.R, self.R_test, self.movienames = self.get_ratings_matrix(scriptsfile, ratingsfile, 0.9, full)

        #Build Topic Model

        #self.lda = my_topics.Lda_model(n_topic, n_voca, scriptsfile)
        self.lda = pickle.load(open('final_LDA.p', 'rb'))
        self.theta = self.lda.topic_dist

        #lambda_u = sigma_u^2 / sigma^2
        self.lambda_u = lamu
        self.lambda_v = lamv

        self.n_topic = n_topic
        self.n_voca = n_voca
        self.n_user = len(self.R)
        self.n_item = len(self.R.iloc[0])
        self.threshold = error_diff
        self.nullval = nullval


        #Set confidence matrix params [A,B], values are either A if an item has been rated, or B, if it hasn't
        if params == 0:
            self.params = [1/np.nanstd(self.R) ** 2, 0]
        elif params == 1:
            self.params = [1, 0]
        elif params == 2:
            self.params = [1/np.nanstd(self.R) ** 2, 0.01]
        elif params == 3:
            self.params = [1, 0.01]
        elif params == 4:
            self.params = 2 * [1/np.nanstd(self.R) ** 2]
        elif params == 5:
            self.params = [5, 0]
        
        # INIT U = user_topic matrix, n_topic x n_user
        self.U = pd.DataFrame(np.random.multivariate_normal(np.zeros(self.n_user), np.identity(self.n_user) * (1. / self.lambda_u),size=self.n_topic))

        # INIT V = item(doc)_topic matrix, n_topic x n_item
        self.V = pd.DataFrame(np.random.multivariate_normal(np.zeros(self.n_item), np.identity(self.n_item) * (1. / self.lambda_v), size = self.n_topic))


        self.V.columns = self.R.columns

        #INIT confidence matrix
        self.C = self.R.applymap(self.get_c)
        
        self.errors = []
        self.train_error = None
        self.test_error = None

        self.fit(printout)

    def binary(self, val):
        if np.isnan(val):
            return 0
        elif val <= 2:
            return 0
        else:
            return 1

    def get_c(self, val):
        if np.isnan(val):
            return self.params[1]
        else:
            return self.params[0]

    @staticmethod
    def get_ratings_matrix(scriptsfile, ratingsfile, hyperparameter, full):
        print('Combining movie and item indexing...')
        # Reads in movies from the matched file
        df_movies = pd.read_csv(scriptsfile, usecols = ['movieId', 'title'])
        #print('df movies', df_movies)
        # Drops duplicates
        df_movies = df_movies.drop_duplicates(subset='movieId')
        # Reads in ratings from the movie lens rating file
        df_ratings = pd.read_csv(ratingsfile, usecols=['userId', 'movieId', 'rating'])
        
        if not full:
            df_ratings = df_ratings.groupby('movieId').filter(lambda x: len(x) > 50)

        s = set(df_ratings['movieId'])
        # Joins ratings and movies from matched final using movieId
        df = df_movies.merge(df_ratings, on='movieId')

        print('Saving Names...')
        # Creates a dictionary mapping movieIds to movie titles
        movieName_dict = df_movies.set_index('movieId')['title'].to_dict()

        train = df.copy()
        test = df.copy()

        # Creates a set numbered from 1:len(rows of df)
        df_rows = set((range(df.shape[0])))

        print('Splitting Train and Test')
        # Creates a random sample that has a hyperparameter 
        test_rows = set(random.sample(df_rows, int(hyperparameter*len(df_rows))))


        # Splits data into training and test
        for i in test_rows:
            test.iat[i, 3] = np.nan

        train_rows = df_rows - test_rows
        for j in train_rows:
            train.iat[j, 3] = np.nan
        
        print('Constructing ratings matrix')
        # Creates the training matrix
        train = train.pivot(index='userId', columns='movieId', values='rating')

        # Takes a percentage of the movies to include in training
        # to test out-of-matrix prediction
        train=train.sample(frac=hyperparameter, axis=1)
        test = test.pivot(index='userId', columns='movieId', values='rating')

        return train, test, movieName_dict



    def fit(self, print_out, n_iter = 20):
        print("Learning U and V...\n")
        t0 = time()
        old_err = 0
        num_iter = 0
        for iteration in range(n_iter):
            print('Finished {} iterations\n'.format(num_iter))

            self.do_e_step(print_out)

            err = self.error()
            self.errors.append(err)

            num_iter += 1
            
            if abs(old_err - err) < self.threshold:
                print('Error threshold reached!')
                self.errors = self.errors[:num_iter]
                break
            else:
                old_err = err

        self.errors = self.errors[:num_iter]
            
        print('Finished training with {} iterations in {} seconds'.format(num_iter, time() - t0))

        self.train_error = err
        self.test_error = self.error(True)
        print('Achieved mean training error of {}'.format(self.train_error))
        print('Achieved mean test error of {}'.format(self.test_error))

    # reconstructing matrix for prediction
    def predict_item(self, test=False):
        if test:
            missing_ids = self.R_test.columns.difference(self.R.columns)
            add_V = self.theta.ix[missing_ids].T
            old_V = self.V
            new_V = pd.concat([old_V, add_V], axis=1).as_matrix()
            return np.dot(self.U.T.as_matrix(), new_V)

        return np.dot(self.U.T.as_matrix(), self.V.as_matrix())

    # reconstruction using U and V error
    def error(self, test=False):
        ratings_matrix = self.R
        if test:
            ratings_matrix = self.R_test
        err = np.sqrt(np.nanmean((ratings_matrix - self.predict_item(test)) ** 2))
        return err

    #Perform one iteration of updates
    def do_e_step(self, printout):
        self.update_u(printout)
        self.update_v(printout)

    #Update all the elements of U
    def update_u(self, printout, hybrid = True):
        v = np.matrix(self.V)
        t0 = time()
        print('Updating Users...')

        #Vectorized operation of applying update step
        new_u = np.array(list(map(self.get_new_uservec, range(self.n_user), self.n_user * [v], self.n_user * [printout]))).T

        #Changing elements of U
        self.U.update(new_u)
        print('Finished Users in {}\n'.format(time() - t0))

    def get_new_uservec(self, ui, v, printout):
        if printout:
            print('Updating User {}'.format(ui))
        #retrieving all components of equation
        c_i = np.matrix(np.diag(self.C.iloc[ui]))
        r_i = np.copy(self.R.iloc[ui])
        r_i[np.isnan(r_i)] = self.nullval
        r_i = np.matrix(r_i).T
        
        #performing computation, equation comes from page 4 of paper linked at the top of this file
        left = v * c_i * v.T + (self.lambda_u * np.identity(self.n_topic))
        return np.array(numpy.linalg.solve(left, v * c_i * r_i)).flatten()

    def update_v(self, printout, hybrid = True):
        print('Updating Movies...')
        t0 = time()
        u = np.matrix(self.U)

        #Changing each item vector
        for vj in self.V.columns:
        	self.V[vj] = self.get_new_movievec(vj, u, printout)
        print('Finished Movies in {}\n'.format(time() - t0))


    def get_new_movievec(self, vj, u, printout):
        if printout:
            print('Updating Movie: {}'.format(self.movienames[vj]))

        #retrieving components of equation
        c_j = np.matrix(np.diag(np.copy(self.C[vj])))
        r_j = np.copy(self.R[vj])
        r_j[np.isnan(r_j)] = self.nullval
        r_j = np.matrix(r_j).T
        theta_j = np.matrix(self.theta.loc[vj]).T

        #performing computation, equation comes from page 4 of paper linked at the top of this file
        left = u * c_j * u.T + (self.lambda_v * np.identity(self.n_topic))
        out =np.array(np.linalg.solve(left, u * c_j * r_j + (self.lambda_v * theta_j))).flatten()
        return out

    #movie_ratings is a dictionarry with movieIDs for keys and ratings for values
    def add_user(self, movie_ratings, topics_to_see = 10, words_to_see = 20):


        new_user_id = self.R.index.values.max() + 1
        self.R.loc[new_user_id] = np.nan
        self.R_test.loc[new_user_id] = np.nan

        #set ratings of new user
        for movieid in movie_ratings.keys():
            if movieid in self.R.columns and movie_ratings[movieid] != '':
                    self.R.set_value(new_user_id, movieid, movie_ratings[movieid])
                    self.R_test.set_value(new_user_id, movieid, movie_ratings[movieid])

        #updates to model attributes U and C
        self.C.ix[new_user_id] = (list(map(self.get_c, self.R.ix[new_user_id])))

        new_latent_vector = np.random.multivariate_normal(np.zeros(self.n_topic), np.identity(self.n_topic) * (1. / self.lambda_u))
        self.U[new_user_id] = new_latent_vector
        self.n_user += 1

        #relearning
        self.fit()

        #retrieving recommendations based on predicted ratings
        predictions_matrix = self.predict_item()
        predictions = pd.Series(predictions_matrix[predictions_matrix.shape[0]-1], index=self.V.columns)
        recs = predictions.sort_values(ascending=False)[:10]

        result = []
        for r in recs.index.values:
            result.append((r,self.movienames[r]))

        #topics = self.U[new_user_id].sort_values(ascending = False).index.values[:topics_to_see]
        #self.topics = self.lda.get_topics(topics, words_to_see)

        return result


my_model = CollaborativeTopicModel()
pickle.dump(model, open( "model.p", "wb" )