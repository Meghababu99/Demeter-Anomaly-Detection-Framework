import os,sys
sys.path.insert(0, os.path.abspath('../../'))
import argparse
import ciso8601
import json
import numpy as np
import idlwrap

from app import CFG as cfg
from Lib.database.base import DBUtils
from Lib.stations import Stations

import Lib.database.models as md

from app import SWARMLogger

def schgt_fcn_np(swarm_ne, swarm_height, fof2, hmf2):
    ref_height = 6371.20
    h_swarm = swarm_height # swarm_height - ref_height #; in Km
    n_t = swarm_ne
    nmf2 = (1.240 * 1E+10 * (fof2)**2.0) * 1E-6 # NmF2 in electrons / m3 and foF2 in MHz

    #----- Ratio of N_T to NmF2 to solve the α-Chapman equation:

    x = np.arange(80000).astype(np.float64)*0.001
    ratio1 = n_t/nmf2
    ln_ratio1 = np.log(ratio1)
    eq_1 = ln_ratio1 - 0.5*(1.0- x - np.exp(-x))

    rs = np.where((eq_1>-0.0001) & (eq_1<0.0001))[0]
    count = np.count_nonzero(rs)
    if count==0:
        h_t = 0.0
    else:
        rs2 = np.where(np.min(np.abs(eq_1[rs]))==np.abs(eq_1[rs]))[0]
        count2 = np.count_nonzero(rs2)
        a = rs[rs2]
        z = float(x[a])
        h1 = hmf2
        h2 = h_swarm
        h_t = (h2 - h1) / z

    return h_t

def schgt_fcn_idl(swarm_ne, swarm_height, fof2, hmf2):
    ref_height = 6371.20
    h_swarm = swarm_height # swarm_height - ref_height #; in Km
    n_t = swarm_ne
    nmf2 = (1.240 * 1E+10 * (fof2)**2.0) * 1E-6 # NmF2 in electrons / m3 and foF2 in MHz

    #----- Ratio of N_T to NmF2 to solve the α-Chapman equation:
    #Print, 'Ratio: ', STRING((N_T/NmF2), FORMAT='(F10.7)')

    x = idlwrap.findgen(80000)*0.001
    ratio1 = n_t/nmf2
    ln_ratio1 = idlwrap.alog(ratio1)
    eq_1 = ln_ratio1 - 0.5*(1.0- x - np.exp(-x))

    rs = idlwrap.where(idlwrap.operator_(eq_1, 'GT', -0.0001) & idlwrap.operator_(eq_1, 'LT', 0.0001))
    count = np.count_nonzero(rs)
    if count==0:
        h_t = 0.0
    else:
        rs2 = idlwrap.where(idlwrap.operator_(
            np.min(idlwrap.abs(eq_1[rs])), 'EQ', idlwrap.abs(eq_1[rs])
        ))
        count2 = np.count_nonzero(rs2)
        a = rs[rs2]
        z = float(x[a])
        h1 = hmf2
        h2 = h_swarm
        h_t = (h2 - h1) / z

    return h_t


def main(argv):
    SWARMLogger.logger.info('SWARM Scale Height Function')
    swarm_height = 454.54
    swarm_ne = 103917.80
    fof2 = 4.050
    hmf2 = 231.90

    h_t = schgt_fcn_idl(swarm_ne, swarm_height, fof2, hmf2)
    h_t2 = schgt_fcn_np(swarm_ne, swarm_height, fof2, hmf2)
    print(f'ht: {h_t}, h_t2: {h_t2}')


if __name__ == '__main__':
    sys.exit(main(sys.argv))