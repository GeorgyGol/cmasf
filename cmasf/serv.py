import numpy as np

def neighbours(ar, cur_index, cnt_of_neiboors=3, exclude_from_neibors_index=[]):
    """return cnt_of_neiboors from left-right for cur_index in array ar, with exclude some indexes"""
    rmax = np.max([0, cur_index + cnt_of_neiboors - len(ar)])
    lmin = np.max([cur_index - (cnt_of_neiboors + rmax), 0])

    excl = set(exclude_from_neibors_index) | {cur_index}
    nbs = [i for i in range(lmin, len(ar)) if i not in excl]
    return ar[nbs[:cnt_of_neiboors * 2]]


def as_matrix(row, period, fill_val=np.nan):
    offs=period - len(row) % period
    # kper = len(row) // period + int(offs>0)
    offs = 0 if offs==period else offs
    wr=np.append(np.asarray(row, dtype='float'), [fill_val]*offs)

    wr=np.reshape(wr, (-1, period))
    return wr

def iterate_group(iterator, count):
    itr = iter(iterator)
    for i in range(0, len(iterator), count):
        yield iterator[i:i + count]