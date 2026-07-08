import os,sys
sys.path.insert(0, os.path.abspath('../'))
import typer
import numpy as np
from datetime import datetime
import ciso8601
from enum import Enum
from typing import Optional, List, Union
from app import CFG as cfg, DBUtils

import Lib.database.models as md
import Lib.database.crud as cr

from app import SWARMLogger

app = typer.Typer()

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

    rs = np.where((eq_1>-0.001) & (eq_1<0.001))[0]
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
        h_t = (h2 - h1) / z if z > 0 else 0.0

    return h_t

def setHt(rec):
    args = [rec.swarm.plasmaDensity, rec.swarm.altitude, rec.sao.foF2, rec.sao.phF2lyr] # args = [rec.swarm.plasmaDensity, rec.swarm.altitude, rec.sao.foF2, rec.sao.hF2]
    invalid = any([_==9999 or np.isnan(_) or _ is None for _ in args])
    if invalid:
        ht = 9999.0
    else:
        ht = schgt_fcn_np(*args)
    rec.scHgtF2pk = ht


def compute_ht(coinc, dbo, overwrite=False):
    cnt = cr.count_coincidences(dbo=dbo, log=not overwrite, onlyvalid=True)
    if not overwrite and cnt > 0:
        SWARMLogger.logger.warning(f'Found {cnt} Valid Coincidences for table: {md.Coincidence.__table__.fullname}')
        SWARMLogger.logger.warning(f'Set <Overwrite> to insert, skipping data insert for table: {md.Coincidence.__table__.fullname}')
        return

    for i, c_ in enumerate(coinc):
        SWARMLogger.logger.info(f'Record SAO {c_.sao.date} | SWARM {c_.swarm.date} {i + 1}/{len(coinc)}')
        if c_.scHgtF2pk is None or overwrite is True:
            setHt(c_)
    dbo.ormdb.session.commit()

    cnt = cr.count_coincidences(dbo=dbo, log=not overwrite, onlyvalid=True)
    SWARMLogger.logger.warning(f'Found {cnt} Valid Coincidences for table: {md.Coincidence.__table__.fullname}')


def aggregate_ht(dbo, overwrite=False):
    cnt = cr.count_coincidence_groups(dbo=dbo, log=not overwrite)
    if not overwrite and cnt > 0:
        SWARMLogger.logger.warning(f'Found {cnt} Grouped Coincidences for table: {md.CoincidenceGroup.__table__.fullname}')
        SWARMLogger.logger.warning(f'Set <Overwrite> to insert, skipping data insert for table: {md.CoincidenceGroup.__table__.fullname}')
        return

    cr.insert_coincidence_groups(dbo=dbo, overwrite=overwrite)

    cnt = cr.count_coincidence_groups(dbo=dbo, log=not overwrite)
    SWARMLogger.logger.warning(f'Found {cnt} Valid Coincidence Groups for table: {md.CoincidenceGroup.__table__.fullname}')

    cr.create_coincidences_view(dbo=dbo)
    SWARMLogger.logger.info(f'Created concidences View')


@app.command()
def main(
    overwrite: bool = typer.Argument(False, help='Overwrite and reset existing Coincidence Tables State'),
):
    """
    Application IonoStation-SWARM Coincidences calculator:
    Estimated coincidences between Ionosonde SAO acquisitions and SWARM Satellite Measurements,
    calculates the effective scale height Ht at the SWARM altitude for those coincidences and then
    calculates Ht mean and standard deviation values for each coincidence group (coinciding Ionosonde SAO acquisitions).
    All the above datasets are finally stored into the DB.
    """
    overwrite = True

    with DBUtils() as dbo:
        SWARMLogger.logger.info(f"Clear stored Coincidences and Coincidence Groups")
        cr.clear_coincidences(dbo=dbo, overwrite=overwrite)
        SWARMLogger.logger.info(f"Detect and store Coincidences")
        cr.insert_coincidences(dbo=dbo, overwrite=overwrite)
        coinc = cr.get_coincidences(dbo=dbo)
        compute_ht(coinc, dbo, overwrite=overwrite)
        aggregate_ht(dbo, overwrite=overwrite)


if __name__ == '__main__':
    sys.exit(app())