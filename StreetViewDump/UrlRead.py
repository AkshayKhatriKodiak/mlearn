#!/usr/bin/env python

import urllib2
from urllib2 import URLError, HTTPError

from StreetViewXMLParse import *

def urlRead(urlStr, fileName = None):
    data = None
    file = None

    try:
        file = urllib2.urlopen(urlStr)
        data = file.read()

    except HTTPError as e:
        print("urlRead: URL %s caused error %s code %d\n" %
            (urlStr, repr(e), e.code))
        return None

    except (URLError, IOError) as e:
        print("urlRead: URL %s caused error %s\n" % (urlStr, repr(e)))
        return None
        
    if file:
        file.close()
    if fileName and data:
        try:
            with open(fileName, "wb") as f:
                f.write(data)
        except IOError as e:
            print("urlRead: Could not write to file %s: %s\n" % e)
            return None
    return data


def test():

    print urlRead("http://cbk0.google.com/cbk?output=xml&panoid=Q6Z2Rhy5_lHSoFAWz-Qmww")
    urlRead("http://cbk0.google.com/cbk?output=thumbnail&w=300&h=128&panoid=Q6Z2Rhy5_lHSoFAWz-Qmww", "image2.jpeg")

if __name__ == "__main__":
    test()
