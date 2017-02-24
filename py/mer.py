#Record Linkage
#Link movie titles
import pandas as pd
import matplotlib.pyplot as plt
import csv
import re
import time


def read_in(lens_file, scripts_file):
    '''
    Read in lens and scripts data
    '''
    lens_df = pd.read_csv(lens_file)
    lens = lens_df['title'].values.tolist()
    scripts_df = pd.read_csv(scripts_file)
    scripts = scripts_df['title'].values.tolist()
    #lens = ['Good & Evil, a (2002)']
    #scripts = ['a Good and Evil (2002)']

    return lens, scripts


def basic_linkage(lens, scripts):
    exact = 0
    mismatched_fe = 0
    a_matches = 0
    an_matches = 0
    the_matches = 0

    for scripts_title in scripts:
        #print(scripts_title)
        t0 = time.time()
        for lens_title in lens:
            #print(lens_title)
            l = lens_title.lower().replace('&', 'and').replace(' ', '')
            s = scripts_title.lower().replace('&', 'and').replace(' ', '')
            l_np = re.sub(r'\W', '', l)
            s_np = re.sub(r'\W', '', s)
            #print(l, s)
            # CASE 1 and 2
            if l_np == s_np:
                exact += 1
                break
            else:
                l_year = l[-4:]
                s_year = s[-4:]
                # Case: Mismatched Foreign and English
                if l_year == s_year:
                    l_foreng = l.split('(')
                    s_foreng = s.split('(')
                    if len(l_foreng) == 3: 
                        if len(s_foreng) == 3:
                            l_combo = ''.join(l_foreng)
                            l_stripped = l_combo.replace(' ', '')
                            s_combo = ''.join(s_foreng)
                            s_stripped = s_combo.replace(' ', '')

                            if l_stripped == s_stripped:
                                mismatched_fe += 1
                                break
                # Case: misplaced a, an, or the     
                    if ',a' in l:
                        if ''.join(l_np.split('a')) == ''.join(s_np.split('a')):
                            a_matches += 1
                    elif ',an' in l:
                        if ''.join(l_np.split('an')) == ''.join(s_np.split('an')):
                            an_matches += 1
                    elif ',the' in l:
                        if ''.join(l_np.split('the')) == ''.join(s_np.split('the')):
                            the_matches += 1
                            break
        print(time.time()-t0)
    print('Exact matches: {}\nForeign-English: {}\n"a": {}\n"an": {}\n"the": {}'\
        .format(exact, mismatched_fe, a_matches, an_matches, the_matches))

    # Drop rows that didn't match with the movieLens dataset
    # scripts = scripts[pd.notnull(scripts['movieId'])] 
    # print(scripts)
    return scripts

if __name__ == '__main__':
    lens_file = "data/ml-20m/movies.csv"
    scripts_file = "data/small.csv"

    lens, scripts = read_in(lens_file, scripts_file)
    linked = basic_linkage(lens, scripts)
    # scripts.to_csv("ml-20m/smallMatched.csv")
