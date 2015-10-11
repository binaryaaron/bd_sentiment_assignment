#predefined stopwords 
from nltk.corpus import stopwords
import sys

# set of stopwords from only the english language
# TODO: sub these for the proper word_id based on the vocab
sw = stopwords.words('english')

if __name__ == "__main__":
    # every line should be of the form: <rating> word_id:word_count
    for l in sys.stdin:
        #all words separated by spaces
        words = l.strip().split()

        #the rating is the first indexed word
        rating = words[0]
        #the word pairs are anything after that
        words = words[1:]
        
        for w in words:
            [word,c] = w.split(':')
            #if word not in sw:
            print '%s\t(%s,%s)' % (rating, word, c)
