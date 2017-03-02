import random
import string

import cherrypy

movies = {1:'Inception', 2:'21 Jump Street', 3:'Office Space',
    4:'La La Land', 5:'Interstellar', 6:'Mad Max: Fury Road', 7: 'Lmao'}

form_contents = ''.join([movies[key] + \
    '<br><input type="text" value="rate out of 5" name="r' + str(key) + '" />\n<br><br>' \
    for key in movies])

adds = '+'.join(['r'+str(key)+'.value' for key in movies])

class StringGenerator(object):
    @cherrypy.expose
    def index(self):
        html = '''<html>
            <head>
            </head>
            <body>
                <h2>Rate these movies:</h2>
                <form method="post" oninput="params.value =''' + adds + '''" action="results">
                ''' +\
                form_contents +\
                '''
                <input type="hidden" name="params" >
                <input type="submit" value="Submit">
                </form>
                <script>
                    function combine_fields() {
                        var spacer = "";
                        var newValue=""; 
                        var elements =document.forms["frm"].elements;
                        for(i=0; i<elements.length;i++) {
                            newValue+=elements[i].value+spacer;
                            };
                        return newValue.trim();
                        }
                </script>
            </body>
        </html>'''
        return html

    @cherrypy.expose
    def results(self, params, **kwargs):
        ratings = [r for r in params]
        max_i = ratings.index(max(ratings))
        return 'Your favorite movie is ' + movies[max_i + 1]


if __name__ == '__main__':
    cherrypy.quickstart(StringGenerator())