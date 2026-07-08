import os,sys
sys.path.insert(0, os.path.abspath('../'))
import typer
from datetime import datetime
from pathlib import Path
import ciso8601
from enum import Enum
import re
from typing import Optional, List
from app import CFG as cfg, DBUtils

import Lib.database.models as md
import Lib.database.crud as cr
from Lib.swarm import Parser as SWARMParser

from app import SWARMLogger

cliapp = typer.Typer()


_SWARM_PATTERN = re.compile(r'^(\w{5})_(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})_(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2}).txt$')


def swarmiter(base=None):
    base = base or cfg['REPO']['SWARM_PATH']
    swarmfiles = list()
    satellite = None
    for root, dirs, files in os.walk(base, topdown=True):
        root_ = Path(root)
        if re.match(r'SWARM_([A,B])', root_.name):
            satellite = root_.name
        for f in files:
            dmatch = re.findall(_SWARM_PATTERN, f)
            if not (dmatch and len(dmatch[0]) == 13):
                continue
            dmatch = dmatch[0]

            station = dmatch[0]
            start = datetime(year=int(dmatch[1]), month=int(dmatch[2]), day=int(dmatch[3]),
                             hour=int(dmatch[4]), minute=int(dmatch[5]), second=int(dmatch[6]))
            end = datetime(year=int(dmatch[7]), month=int(dmatch[8]), day=int(dmatch[9]),
                             hour=int(dmatch[10]), minute=int(dmatch[11]), second=int(dmatch[12]))

            rpath = root_.joinpath(f)

            swarmfiles.append({'satellite':satellite, 'start': start, 'end': end, 'path': rpath})
            if len(swarmfiles)%100==0:
                yield swarmfiles
                swarmfiles = list()

    if swarmfiles:
        yield swarmfiles

@cliapp.command()
def main(
        start: Optional[datetime] = typer.Argument(None, help='Start query date'),
        end: Optional[datetime] = typer.Argument(None, help='End query date'),
    ):
    """
    Application Swarm Exporter:
    Parses SWARM input datasets requested by time interval,
    and then ingests parsed SWARM datasets into the DB.
    """
    recnt = 0
    fopen = 0
    records = list()
    stations = cr.get_stations(asdict=True, type='iono')

    def flash(dbo):
        nonlocal recnt, records
        SWARMLogger.logger.info('Saving')
        for p in records:
            p.store(stations, dbo=dbo)

        records = list()


    SWARMLogger.logger.info(f"Ingesting SWARM data for [{start.isoformat() if start else 'MINDATE'} {end.isoformat() if end else 'MAXDATE'}]")

    filescounter = 0

    with DBUtils() as dbo:
        for swarmfiles in swarmiter():
            filescounter += len(swarmfiles)
            for sf in swarmfiles:
                sparser = SWARMParser(sf['path'], sf['satellite'], sf['start'],sf['end'], stations = stations)
                sparser.parse()
                recnt += sparser.count
                fopen+=1
                records.append(sparser)

                if fopen%10==0:
                    SWARMLogger.logger.info(f'Processed: {fopen} files|{recnt} records')

                if (recnt > 0) & (fopen % 500 == 0):
                    flash(dbo)
        flash(dbo)

    SWARMLogger.logger.info(f"Parser {filescounter} swarm files")


if __name__ == '__main__':
    sys.exit(cliapp())