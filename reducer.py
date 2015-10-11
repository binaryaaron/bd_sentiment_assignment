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

    # group all the same docids together and construct the (docid,
    # iterator) pair where iterator will return all values that were
    # previously of the form: (<docid>, rating)\t<word>:<count>
    for (docid, iterator) in groupby(data, itemgetter(0)):
        # keep track of all current counts for a given docid
        count = {}

        #iterator returns ["<docid>", "<word_id>:<count>"]
        for [docid, pair] in iterator:
            #extract the <word_id>
            w_id, c = pair.split(':')

            # update the total count for the <word_id>
            if w_id in count.keys():
                count[w_id] += c
            else:
                count[w_id] = c

        # for this docid print out the total count for every word in
        # this docid
        for w_id in count.keys():
            print('%s\t%s:%s' % (docid, w_id, count[w_id]))
