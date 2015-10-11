#!/usr/bin/env python

from itertools import groupby
from operator import itemgetter
import sys


def gen(f):
    for line in f:
        yield line.rstrip().split('\t', 1)

def main():
    data = gen(sys.stdin)

    for (rating, iterator) in groupby(data, itemgetter(0)):
        count = {}
        for [rating, pair] in iterator:
            w_id,c = pair.split(':')
            if w_id in count.keys():
                count[w_id] += c
            else:
                count[w_id] = c

        for w_id in count.keys():
            print '%s\t%s:%s' % (rating, w_id, count[w_id])
                

if __name__ == '__main__':
    main()
