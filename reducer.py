#!/usr/bin/env python

from itertools import groupby
from operator import itemgetter
import sys

# generator for the input so we don't read it all in at once
def gen(f):
    for line in f:
        yield line.rstrip().split('\t', 1)

if __name__ == '__main__':
    data = gen(sys.stdin)

    # group all the same ratings together and construct the (rating,
    # iterator) pair where iterator will return all values that were
    # previously of the form: <rating>\t<word_id>:<count>
    for (rating, iterator) in groupby(data, itemgetter(0)):
        # keep track of all current counts for a given rating
        count = {}

        #iterator returns ["<rating>", "<word_id>:<count>"]
        for [rating, pair] in iterator:
            #extract the <word_id>
            w_id,c = pair.split(':')

            # update the total count for the <word_id>
            if w_id in count.keys():
                count[w_id] += c
            else:
                count[w_id] = c

        # for this rating print out the total count for every word in
        # this rating
        for w_id in count.keys():
            print '%s\t%s:%s' % (rating, w_id, count[w_id])
