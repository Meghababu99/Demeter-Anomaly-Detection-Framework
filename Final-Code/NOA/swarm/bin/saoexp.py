import os,sys
sys.path.insert(0, os.path.abspath('../'))
import typer
from datetime import datetime
import ciso8601
from enum import Enum
from typing import Optional, List, Union
from app import CFG as cfg, DBUtils

from Lib.techtide import API, FSScanner
from Lib.utils import ufunc #getExternalScaled, ingestScaled


import Lib.database.models as md
import Lib.database.crud as cr

from app import SWARMLogger

app = typer.Typer()

_STATIONS_AIS = ['MZ152', 'OL246', 'RM041','TNJ20','GM037']
_STATIONS_DONE = []
_STATIONS_LST = ['AL945','AS00Q','AT138','AU930','BC840', 'BLJ03','BP440','BVJ03','CAJ2M','CGK21','DB049',
                 'EA036','EB040','EG931','FF051','FZA0M','GR13L','GU513','HE13N','IC437','IF843', 'JI91J','JJ433',
                 'JR055','KS759','LM42B','LV12P', 'LL721', 'ME929', 'MH453',
                 'MHJ45','MO155', 'MU12K', 'NI135', 'PA836', 'PQ052', 'PRJ18', 'PSJ5J', 'RL052', 'RO041', 'SA418',
                 'SAA0K','SMK29', 'VT139', 'WP937', 'WU430']
_list2enum = lambda x: Enum('Type', {el.upper():el for el in set(x)})
_STATIONS = _list2enum(_STATIONS_LST + _STATIONS_AIS)

@app.command()
def main(
        start: datetime = typer.Argument(None, help='Start query date'),
        end: datetime = typer.Argument(None, help='End query date'),
        station: List[_STATIONS] = typer.Option(_STATIONS_LST, help='Ionospheric station code', case_sensitive=False),
    ):
    """
    Application SAO Param Exporter:
    Fetches remote SAO data for acquisitions requested by time intervals and station(s) codes,
    and then ingests parsed SAO datasets into the DB.
    """

    stations_arg = [s_.name for s_ in station] if station else Noneppyhto

    recnt = 0
    records = list()
    stations = cr.get_stations(asdict=True, type='iono')

    def flush(dbo):
        nonlocal recnt, records
        SWARMLogger.logger.info('Saving Digisonde records')
        dbrec = [s.toDB(stations=stations) for s in records]
        cr.upcert_sao(dbrec, dbo=dbo)
        records = list()

    SWARMLogger.logger.info(f"Ingesting Digisonde data for [{start.isoformat() if start else 'MINDATE'} {end.isoformat() if end else 'MAXDATE'}]")

    scanner = FSScanner(input=cfg['REPO']['SAO_PATH'], stations=stations)
    scanner.scan(start=start, end=end, stations=stations_arg)

    with DBUtils() as dbo:
        for station, sao in scanner.iter():
            SWARMLogger.logger.info(f'Station {station.code}, Record: {sao.sounding.timestamp}')
            records.append(sao)
            recnt += 1
            if (recnt > 0) & (recnt % 5000 == 0):
                flush(dbo)
        flush(dbo)


if __name__ == '__main__':
    sys.exit(app())

