# CS 122 Final Project: Hybrid Movie Recommender
*Our website uses a hybrid content and collaborative model to recommend the best movies for users.*

### How to run the website
We have already trained and fitted a model for the web interface to use. In order to run the site with this model, please do the following:
1. Download this file from this Google Drive link. It should be close to 500 MB when unzipped.
2. Move the file, `fittedmodelfile.p`, into `cs122/py/site`.
3. To run the web interface demo, simply run this command from the `cs122` directory:
    ```sh
    $ python3 py/site/cherry.py
    ```
4. Now you can go to http://localhost:8080/ and rate the movies, as instructed on the page.
5. Click "Submit" when you are finished rating the movies.

The model will update based on the ratings you provided, and you will be take to a page that shows the top 10 movies for you to watch, plus your 10 favorite latent topics as determined by the model. 

### Explanation of files
- `data/all_movies.csv` - A table of all the MovieLens data movies, which is used by linkage.py to match records with the Springfield data.
- `data/names.txt` - A list of personal names that is used to expand the stop words used in the content model.
- `proposal/RecommenderProposal.pdf` - The original project proposal.
- `py/linkage.py` - The program used to link MovieLens ratings with Springfield scripts.
- `py/scrape_ss.py` - The program used to scrape all scripts from the Springfield website.
- `py/site/cherry.py`- This file uses the CherryPy framework to run the website.
- `py/site/final_model.py` - The final hybrid model, used by the site.
- `py/site/public` - This folder contains CSS and images for the site.
- `py/testing.py` - We used this file to find the best hyperparameters.

### Data that is too big for GitHub
- The zipped movie script data can be found at [this URL](https://drive.google.com/open?id=0B-Zg2Odn-W_wR0RCbkZQN2IzR2s). It contains matched.csv, which has 9755 rows, and 3 columns: 'title', 'script', 'movieId'. This was used in the LDA portion of the model.
- The saved model can be downloaded here.