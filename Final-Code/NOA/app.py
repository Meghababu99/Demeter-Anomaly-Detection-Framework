import os
import uuid
import yaml
import psutil
import psycopg2
from psycopg2.extras import register_uuid
register_uuid()
from sqlalchemy.ext.declarative import declarative_base

from Lib.utils.logger import Logger as _Logger

def join(loader, node):
    seq = loader.construct_sequence(node)
    return ''.join([str(i) for i in seq])


yaml.add_constructor('!join', join)

_F = os.path.abspath(__file__)


def normpath(extpath, basepath = _F):
    if basepath is None:
        return None
    elif os.path.isabs(extpath):
        return extpath
    elif os.path.isfile(basepath):
        return str(os.path.normpath(os.path.join(os.path.dirname(basepath),extpath)))
    else:
        return str(os.path.normpath(os.path.join(basepath,extpath)))


def _fixGPaths():
    CFG['DATA_PATH'] = normpath(CFG['DATA_PATH'], _F)
    CFG['LOG_PATH'] = normpath(CFG['LOG_PATH'], _F)
    for rk in ('REPO','STATIONS'):
        for k,v in CFG[rk].items():
            CFG[rk][k] = normpath(v, _F)


_CFG = normpath('./var/SWARMconf.yml')
CFG = dict()
with open(_CFG) as f:
    CFG = yaml.load(f, Loader=yaml.Loader)

_fixGPaths()

NCPU = psutil.cpu_count()
UUID = uuid.UUID(CFG['UUID'])

from pydantic import Extra

class PydanticConfig:
    arbitrary_types_allowed = True
    extra = Extra.ignore

SWARMLogger = _Logger(CFG['LOG_PATH'])

from Lib.database.base import DBBase, DBUtils
Base = declarative_base(cls = DBBase)
from Lib.database.models import *
from Lib.database.schemas import *
from Lib.database import crud

