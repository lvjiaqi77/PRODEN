import numpy as np
import torch
from sklearn.preprocessing import OneHotEncoder
import os
import hashlib 
import errno

def binarize_class(y):  
    label = y.reshape(len(y), -1)
    enc = OneHotEncoder(categories='auto') 
    enc.fit(label)
    label = enc.transform(label).toarray().astype(np.float32)     
    label = torch.from_numpy(label)
    return label


def partialize(y, y0, t, p):
    new_y = y.clone()
    n, c = y.shape[0], y.shape[1]
    avgC = 0

    if t=='binomial':
        for i in range(n):
            row = new_y[i, :] 
            row[np.where(np.random.binomial(1, p, c)==1)] = 1
            while torch.sum(row) == 1:
                row[np.random.randint(0, c)] = 1
            avgC += torch.sum(row)
            new_y[i] = row / torch.sum(row) 

    if t=='pair':
        P = np.eye(c)
        for idx in range(0, c-1):
            P[idx, idx], P[idx, idx+1] = 1, p
        P[c-1, c-1], P[c-1, 0] = 1, p
        for i in range(n):
            row = new_y[i, :]
            idx = y0[i] 
            row[np.where(np.random.binomial(1, P[idx, :], c)==1)] = 1
            avgC += torch.sum(row)
            new_y[i] = row / torch.sum(row)

    avgC = avgC / n    
    return new_y, avgC


def check_integrity(fpath, md5): 
    if not os.path.isfile(fpath):
        return False
    md5o = hashlib.md5()
    with open(fpath, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''): 
            md5o.update(chunk)
    md5c = md5o.hexdigest()
    if md5c != md5:
        return False
    return True


def download_url(url, root, filename, md5):
    import urllib.request

    root = os.path.expanduser(root)
    fpath = os.path.join(root, filename)

    try:
        os.makedirs(root)
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise

    if os.path.isfile(fpath) and check_integrity(fpath, md5):
        print('Using downloaded and verified file: ' + fpath)
    else:
        print('Downloading ' + url + ' to ' + fpath)
        urllib.request.urlretrieve(url, fpath) 
