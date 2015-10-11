#!/usr/bin/env python

#predefined stopwords 
from nltk.corpus import stopwords
import sys
import string


# set of stopwords from only the english language
# TODO: sub these for the proper word_id based on the vocab

sw = stopwords.words('english')
punctuation = list(string.punctuation)
for p in punc:
        sw.append(p)
sw = set(sw)

if __name__ == "__main__":
    # every line should be of the form: <rating> word_id:word_count

    for l in sys.stdin:
        # get the basic fields we'll work with
        docid, rating, words = l.split(',', 2)

        words = [word for word in words.lower().split()
                if word not in sw]

        for w in words:
            #if word not in sw:
            print("(%s, %s)\t%s:1" % (docid, rating, w))
