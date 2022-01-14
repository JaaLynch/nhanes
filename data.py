import xport
import numpy as np
import os

#import urllib.request
#ftp_path = 'ftp://ftp.cdc.gov/pub/Health_Statistics/nchs/nhanes/2009-2010/DEMO_F.XPT'
#urllib.request.urlretrieve(ftp_path, 'data/DEMO_F.XPT')
#wget.download(ftp_path)


"""
Read five NHANES data files and merge them into a 2d array.

These values can be exported:

Z : an array of the data, with rows corresponding to subjects, and
    columns corresponding to variables

VN : an array of variable names, in 1-1 correspondence with the
     columns of Z; the variable names consist of the file name,
     followed by a ":", followed by the variable name

KY : the sequence numbers (subject identifiers), in 1-1 correspondence 
     with the rows of Z
"""

## Data file names (the files are in /data)
FN = ["DEMO_F.XPT", "BMX_F.XPT", "BPX_F.XPT", "DR1TOT_F.XPT", "DR2TOT_F.XPT"]

def get_data(fname):
    """ 
    Place all the data in the file `fname` into a dictionary indexed 
    by sequence number.

    Arguments:
    ----------
    fname : The file name of the data

    Returns:
    --------
    Z : A dictionary mapping the sequence numbers to lists of data values
    H : The names of the variables, in the order they appear in Z
    """

    ## The data, indexed by sequence number
    Z = {}

    ## The variable names, in the order they appear in values of Z.
    H = None

    with xport.XportReader(fname) as reader:
        for row in reader:

            ## Get the variable names from the first record
            if H is None:
                H = row.keys()
                H.remove("SEQN")
                H.sort()

            Z[row["SEQN"]] = [row[k] for k in H]

    return Z,H

with open('data/DEMO_F.XPT', 'rb') as f:
    library = xport.load(f)

with xport.XportReader('data/DEMO_F.XPT') as reader:
    for row in reader:
        print (row)


## Read all the data files
D,VN = [],[]
for fn in FN:
    fn_full = os.path.join("data/", fn)
    X,H = get_data(fn_full)
    s = fn.replace(".XPT", "")
    H = [s + ":" + x for x in H]
    D.append(X)
    VN += H

## The sequence numbers that are in all data sets
KY = set(D[0].keys())
for d in D[1:]:
    KY &= set(d.keys())
KY = list(KY)
KY.sort()

def to_float(x):
    try:
        return float(x)
    except ValueError:
        return float("nan")

## Merge the data
Z = []
for ky in KY:

    z = []

    map(z.extend, (d[ky] for d in D))
    ## equivalent to
    ## for d in D:
    ##     z.extend(d[ky])

    z = [to_float(a) for a in z]
    ## equivalent to
    ## map(to_float, z)

    Z.append(z)

Z = np.array(Z)