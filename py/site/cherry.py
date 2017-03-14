import os, os.path
import random
import string
import pickle
import cherrypy

from final_model import CollaborativeTopicModel

model = pickle.load(open('py/site/model.p', 'rb'))

CHOSEN_MOVIES = {1704: "Good Will Hunting (1997)", 4896: "Harry Potter and the Sorcerer's Stone (2001)", 780: "Independence Day (1996)",
    8376: "Napoleon Dynamite (2004)", 2028: "Saving Private Ryan (1998)", 1968: "The Breakfast Club (1985)",
    48516: "The Departed (2006)", 48780: "The Prestige (2007)", 48385: "Borat (2006)",
    33493: "Star Wars: Episode III - Revenge of the Sith", 589: "Terminator 2: Judgment Day (1991)",
    7153: "The Lord of the Rings: The Return of the King", 1: "Toy Story (1995)", 2858: "American Beauty (1999)",
    72998: "Avatar (2009)", 2918: "Ferris Bueller's Day Off (1986)", 56174: "I Am Legend (2007)",
    45722: "Pirates of the Caribbean: Dead Man's Chest (2006)", 5618: "Spirited Away (2001)", 69122: "The Hangover (2009)",
    71379: "Paranormal Activity (2007)", 2571: "The Matrix (1999)", 111: "Taxi Driver (1976)", 79132: "Inception (2010)",
    8464: "Super Size Me (2004)", 1732: "The Big Lebowski (1998)", 364: "The Lion King (1994)",
    6377: "Finding Nemo (2003)", 72641: "The Blind Side (2009)", 593: "The Silence of the Lambs (1991)",
    1721: "Titanic (1997)", 1265: "Groundhog Day (1993)", 318: "The Shawshank Redemption (1994)",
    6942: "Love Actually (2003)", 2502: "Office Space (1999)", 5299: "My Big Fat Greek Wedding (2002)",
    919: "The Wizard of Oz (1939)", 82459: "True Grit (2010)", 356: "Forrest Gump (1994)",
    296: "Pulp Fiction (1994)", 1270: "Back to the Future (1985)", 2710: "Blair Witch Project (1999)",
    85774: "Senna (2010)", 7318: "The Passion of the Christ (2004)", 527: "Schindler's List (1993)",
    58559: "The Dark Knight (2008)", 72407: "The Twilight Saga: New Moon (2009)", 54503: "Superbad (2007)"}

class StringGenerator(object):
    def build_rating_screen(self):
        contents = ''
        row = '<div class="row">'
        i = 0
        j = 1

        for movie_id in CHOSEN_MOVIES:
            title = '<br>' + str(CHOSEN_MOVIES[movie_id])

            if len(title) > 32:
                title = title[4:]

            item = '<div class="three columns" align="center" id="form_id" style="margin-top: 2%">\n<br>'\
              + title + '<br><br><img src="static/images/'\
              + str(movie_id) + '.jpg" alt="Inception poster" align="bottom" style="width:150px;height:225px"><br>\
              <br><input type="text" placeholder="rate out of 5" name="rating" style="width: 120px;"/>\n</div>'
            row += item

            if j % 4 == 0:
                if i % 2 == 1:
                    row = '<div class="row"  style="background-color: #f2f2f2;"' + row[15:]
                contents += (row + '\n</div>')
                row = '<div class="row">'
                j = 0
                i += 1

            j += 1

        return contents

    @cherrypy.expose
    def index(self):
        contents = self.build_rating_screen()
        html = '''
        <html>
        <head>
          <meta charset="utf-8">
          <title>Hybrid movie recommender</title>
          <!-- FONT–––––––––––––––––––––––––––––––––––––––––––––– -->
          <link href="//fonts.googleapis.com/css?family=Raleway:400,300,600" rel="stylesheet" type="text/css">

          <!-- CSS-–––––––––––––––––––––––––––––––––––––––––––––– -->
          <link rel="stylesheet" href="static/css/normalize.css">
          <link rel="stylesheet" href="static/css/skeleton.css">

          <!-- Favicon––––––––––––––––––––––––––––––––––––––––––– -->
          <link rel="icon" type="image/png" href="images/favicon.png">

        <script>
        function Validate() {
            var ratings_list = [];
            var ratings_nodes = document.getElementsByName("rating");

            for (i = 0; i < ratings_nodes.length; i++) { 
                var r = parseInt(ratings_nodes[i].value)
                ratings_list.push(r)

                if ( (r > 5 || r < 1) && (r != "") ) {
                    alert("Ratings must be an integer from 1 to 5.");
                    return false;
                }

            }

            if (ratings_list.join("").length < 10) {
                alert("Please rate at least 10 movies.");
                return false;
            }

            return ratings_list
        }
        </script>
        </head>
        <body>
          <div class="container">
            <h1 style="margin-top: 10%">Hybrid Movie Recommender</h1>
            <p><strong>Please rate (from 1-5) at least 10 movies.</strong> If you have not seen a movie, simply skip it. Decimals will be rounded down.<p>
            <form method="post" id = "myid" onsubmit="return Validate()" action="results">'''\
            + contents +\
        '''
            <input type="hidden" name="params" > <br>
            <input align="center" class="button-primary" type="submit" value="Submit">
            </form>
          </div>
        </body>
        </html>
        '''
        return html

    @cherrypy.expose
    def results(self, params, **kwargs):
        ratings = {}
        for r, movie_id in zip(params.split(','), CHOSEN_MOVIES):
            if r == '':
                ratings[movie_id] = ''
            else:
                ratings[movie_id] = float(r)
        prediction = model.add_user(ratings)
        contents = ''

        for rec in prediction:
          index, title = rec
          contents+='''\n
            <div class="two columns" align="center" style="margin-top: 2%">
            <br><img src="static/images/film.jpg" alt="placeholders poster" align="bottom" style="width:150px;height:225px"><br><br>'''+title+'''
            </div>
            '''
        html = \
        '''
        <html>
        <head>
          <meta charset="utf-8">
          <title>Results</title>
          <!-- FONT–––––––––––––––––––––––––––––––––––––––––––––– -->
          <link href="//fonts.googleapis.com/css?family=Raleway:400,300,600" rel="stylesheet" type="text/css">

          <!-- CSS-–––––––––––––––––––––––––––––––––––––––––––––– -->
          <link rel="stylesheet" href="static/css/normalize.css">
          <link rel="stylesheet" href="static/css/skeleton.css">

          <!-- Favicon––––––––––––––––––––––––––––––––––––––––––– -->
          <link rel="icon" type="image/png" href="images/favicon.png">
        </head>
        <body>
          <div class="container">
            <h2>You should watch...</h2>
            <div class="row">''' +\
            contents +\
            '''</div>
          </div>
        </body>
        </html>

        '''
        return html

if __name__ == '__main__':
    conf = {
        '/': {
            'tools.sessions.on': True,
            'tools.staticdir.root': os.path.abspath(os.getcwd())
        },
        '/static': {
            'tools.staticdir.on': True,
            'tools.staticdir.dir': '/Users/sasha/Documents/cs122/py/site/public'
        }
    }
    cherrypy.quickstart(StringGenerator(), '/', conf)