#!/usr/bin/env python

#predefined stopwords 
# from nltk.corpus import stopwords
import sys
from stopwords import stopwords
from stopwords import punc
# import string


def punct_killer(review):
        return "".join(list(map(lambda x: x if x not in punc else ' ', review)))


if __name__ == "__main__":
    # every line should be of the form: <rating> word_id:word_count

    for l in sys.stdin:
        # get the basic fields we'll work with
        docid, rating, words = l.split(',', 2)

        words = punct_killer(words)
        words = [word for word in words.lower().split()
                if word not in stopwords]

        for w in words:
            #if word not in sw:
            print("(%s, %s)\t%s:1" % (docid, rating, w))
