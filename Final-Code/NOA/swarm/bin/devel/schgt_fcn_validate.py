import os,sys
sys.path.insert(0, os.path.abspath('../../'))
from datetime import datetime
import argparse
import ciso8601
from pathlib import Path
import json
import numpy as np
import pandas as pd
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
    validate_path = Path(cfg['DATA_PATH']).joinpath('validate')
    vfiles = validate_path.glob('*PrintOut*.txt')

    pddate_ = lambda *_: datetime.strptime(
        f'{int(_[0])}-{int(_[1]):02d}-{int(_[2]):02d} {int(_[3]):02d}:{int(_[4]):02d}:{int(_[5]):02d}.{int(_[6]):03d}',
        '%Y-%m-%d %H:%M:%S.%f') if all([not isinstance(_el, str) or len(_el.strip()) > 0 for _el in _]) else None
    pdcolnames = ['yyyy', 'MM', 'dd', 'HH', 'mm', 'ss', 'ms', 'swne', 'swhgt', 'fof2', 'hmf2', 'scalef2', 'scalehgt']
    pdatecol = [0, 1, 2, 3, 4, 5, 6]
    pdtype = {0: np.int16, 1: np.int8, 2: np.int8, 3: np.int8, 4: np.int8, 5: np.int8, 6: np.int16, 7: np.float32, 8: np.float32,
              9: np.float32, 10: np.float32, 11: np.float32, 12: np.float32}

    for file in vfiles:
        poutpath = Path(file)
        df = pd.read_fwf(poutpath, names=pdcolnames, header=None, parse_dates={'timestamp': pdatecol}, date_parser=pddate_, dtype=pdtype)
        df['h_t'] = df.apply(lambda row: schgt_fcn_np(row.swne, row.swhgt, row.fof2, row.hmf2), axis=1)
        df_ = df[df["h_t"] > 0]
        mean = (df_['scalehgt'] - df_['h_t']).mean()
        min = (df_['scalehgt'] - df_['h_t']).min()
        max = (df_['scalehgt'] - df_['h_t']).max()
        std = (df_['scalehgt'] - df_['h_t']).std()

        print(f'File: {poutpath.stem}, Convergence: {len(df[df["h_t"]>0])}/{len(df)}, Mean: {mean}, Min: {min}, Max: {max}, SD: {std}')


if __name__ == '__main__':
    sys.exit(main(sys.argv))