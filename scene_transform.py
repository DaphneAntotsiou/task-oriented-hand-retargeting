__author__ = 'DafniAntotsiou'

from numpy import array


def skeleton2mjcs(skelcoord):
    return array([-skelcoord[0], -skelcoord[2], - skelcoord[1]])


def skeleton2hpe(skelcoord):
    return array([-skelcoord[0], skelcoord[1], -skelcoord[2]])


def hpe2mjcs(modelcoord):
    return array([modelcoord[0], modelcoord[2], -modelcoord[1]])


def hpe2mjcsrot(modelcoord):
    return array([modelcoord[0], modelcoord[1], modelcoord[2]])
