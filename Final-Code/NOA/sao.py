import uuid as uuid_mod
from datetime import datetime
from typing import Optional, Literal, List, Union
from pydantic import BaseModel, validator, Extra, Field, root_validator
from _fields import Hex, Bool

# from app import CFG as cfg, DBUtils, UUID as APPUUID

# import Lib.database.schemas as schemas
# import Lib.database.models as md

class GeophysicalConst(BaseModel):
    gyrofrequency: float
    dipAngle: float
    lat: float
    lon: float
    ssn: float

    def __init__(self,data):
        super().__init__(**{k:v for k,v in zip(self.__fields__.keys(),data)})
        for field, value in self.__dict__.items():
            print(f"{field}: {value}")

    class Config:
        arbitrary_types_allowed = True
        extra = Extra.ignore


class System(BaseModel):
    sounder: str
    stationid: str
    ursicode: Optional[str] = None
    name: Optional[str] = Field(None, alias='NAME')
    artist: Optional[str] = Field(None, alias='ARTIST')
    nhVer: Optional[str] = Field(None, alias='NH')
    adepVer: Optional[str] = Field(None, alias='ADEP')
    operMsg: Optional[str] = Field(None, alias='opermsg')

    class Config:
        arbitrary_types_allowed = True
        extra = Extra.ignore


class SoundingDPS(BaseModel):
    version: str
    timestamp: datetime
    rcvstation: str
    transtation: str
    dpsSched: Hex
    dpsProg: Hex
    startFreq: int
    coarseFreq: int
    stopFreq: int
    dpsFineFreqStep: int
    multiplexingDSBL: Bool
    ndpsSmallSteps: Hex
    dpsPhaseCode: Hex
    altANT1Setup: int
    dpsANT1Opts: Hex
    totalFFTSamplesPOW: int
    dpsRadioSilentMode: int
    pulseRepRate: int
    rangeStart: int
    dpsRangeIncr: str
    numRages: int
    scanDelay: int
    dpsBaseGain: Hex
    dpsFreqSearchEnabled: Bool
    dpsOpMode: int
    artistEnabled: Bool
    dpsDataFmt: int
    onlinePrinterSel: int
    ionoThreshFTP: int
    highInterference: int

    def __init__(self,data):
        print('input',data)
        data = [data[0],f'{data[1]}-{data[3]}-{data[4]}T{data[5]}:{data[6]}:{data[7]}'] + data[8:]
        super().__init__(**{k:v for k,v in zip(self.__fields__.keys(),data)})

    class Config:
        arbitrary_types_allowed = True
        extra = Extra.ignore


class SoundingDIGI256(BaseModel):
    version: str
    timestamp: datetime
    programSet: int
    programType: str
    journal: str
    nominalFreq: int
    outputCtrl: str
    startFreq: int
    incrFreq: Hex
    stopFreq: int
    testOutput: str
    stationid: str
    phaseCode: Hex
    ant1Azimuth: Hex
    ant1Scan: Hex
    ant1OptionDoppler: Hex
    numSamples: int
    repRate: Hex
    pulseWidthCode: Hex
    timeCtrl: Hex
    freqCorrection: Hex
    gainCorrection: Hex
    rangeIncr: Hex
    rangeStart: Hex
    freqSearch: Hex
    nominalGain: Hex
    spare: int

    def __init__(self,data):
        data = [data[0],f'{data[1]}-{data[3]}-{data[4]}T{data[5]}:{data[6]}:{data[7]}'] + data[9:]
        super().__init__(**{k:v for k,v in zip(self.__fields__.keys(),data)})
        print('soundinf data')
        # \print("Assigned values:")
        for field, value in self.__dict__.items():
            print(f"{field}: {value}")

    class Config:
        arbitrary_types_allowed = True
        extra = Extra.ignore


class SoundingAISINGV(BaseModel):
    version: str
    timestamp: datetime

    def __init__(self,data):
        data = [data[0],f'{data[1]}-{data[3]}-{data[4]}T{data[5]}:{data[6]}:{data[7]}']
        super().__init__(**{k:v for k,v in zip(self.__fields__.keys(),data)})

    class Config:
        arbitrary_types_allowed = True
        extra = Extra.ignore


def Sounding(data):
    parser = SoundingDPS if data[0]=='FF' else SoundingDIGI256 if data[0]=='FE' else SoundingAISINGV if data[0]=='AA' else None
    return parser(data)


class ScaledIono(BaseModel):
    foF2: Optional[float] = None
    foF1: Optional[float] = None
    mD: Optional[float] = None
    mufD: Optional[float] = None
    fmin: Optional[float] = None
    foEs: Optional[float] = None
    fminF: Optional[float] = None
    fminE: Optional[float] = None
    foE: Optional[float] = None
    fxI: Optional[float] = None
    hF: Optional[float] = None
    hF2: Optional[float] = None
    hE: Optional[float] = None
    hEs: Optional[float] = None
    zmE: Optional[float] = None
    yE: Optional[float] = None
    qf: Optional[float] = None
    qe: Optional[float] = None
    downF: Optional[float] = None
    downE: Optional[float] = None
    downEs: Optional[float] = None
    ff: Optional[float] = None
    fe: Optional[float] = None
    d: Optional[float] = None
    fMUF: Optional[float] = None
    hfMUF: Optional[float] = None
    delta_foF2: Optional[float] = None
    foEp: Optional[float] = None
    fhF: Optional[float] = None
    fhF2: Optional[float] = None
    foF1p: Optional[float] = None
    phF2lyr: Optional[float] = None
    phF1lyr: Optional[float] = None
    zhalfNm: Optional[float] = None
    foF2p: Optional[float] = None
    fminEs: Optional[float] = None
    yF2: Optional[float] = None
    yF1: Optional[float] = None
    tec: Optional[float] = None
    scHgtF2pk: Optional[float] = None
    b0IRI: Optional[float] = None
    b1IRI: Optional[float] = None
    d1IRI: Optional[float] = None
    foEa: Optional[float] = None
    hEa: Optional[float] = None
    foP: Optional[float] = None
    hP: Optional[float] = None
    fbEs: Optional[float] = None
    typeEs: Optional[float] = None

    def __init__(self,data):
        super().__init__(**{k:v for k,v in zip(self.__fields__.keys(),data)})
        print('ionosonde data')
        print("Assigned values:")
        for field, value in self.__dict__.items():
            print(f"{field}: {value}")

    class Config:
        arbitrary_types_allowed = True
        extra = Extra.ignore


class VTHADFGroup(BaseModel):
    virtualHeight: List[float]= Field(None, alias='VH')
    trueHeight: Optional[List[float]] = Field(None, alias='TH')
    amplitude: Optional[List[int]] = Field(None, alias='AMPL')
    dopplerNumber: Optional[List[int]] = Field(None, alias='DN')
    frequency: List[float] = Field(None, alias='FREQ')

    # class Config:
    #     arbitrary_types_allowed = True
    #     extra = Extra.ignore

    def __str__(self):
        return (
            f"VTHADFGroup(\n"
            f"  virtualHeight (VH): {self.virtualHeight}\n"
            f"  trueHeight (TH): {self.trueHeight}\n"
            f"  amplitude (AMPL): {self.amplitude}\n"
            f"  dopplerNumber (DN): {self.dopplerNumber}\n"
            f"  frequency (FREQ): {self.frequency}\n"
            f")"
        )
    class Config:
        arbitrary_types_allowed = True
        extra = Extra.ignore

class VHADFGroup(BaseModel):
    virtualHeight: List[float]= Field(None, alias='VH')
    amplitude: Optional[List[int]] = Field(None, alias='AMPL')
    dopplerNumber: Optional[List[int]] = Field(None, alias='DN')
    frequency: List[float]= Field(None, alias='FREQ')

    class Config:
        arbitrary_types_allowed = True
        extra = Extra.ignore


class TFEGroup(BaseModel):
    trueHeight: List[float]= Field(None, alias='TH')
    frequency: List[float] = Field(None, alias='FREQ')
    electronDensity: List[float] = Field(None, alias='ELDENS')

    class Config:
        arbitrary_types_allowed = True
        extra = Extra.ignore


class SAO(BaseModel):
    geoConst: GeophysicalConst
    system: System
    sounding: Union[SoundingDPS,SoundingDIGI256, SoundingAISINGV]
    scaled: ScaledIono
    analysisFlags: Optional[List[int]] = None
    dopplerTrans: Optional[List[float]] = None
    f2layerO: Optional[VTHADFGroup] = None
    f1layerO: Optional[VTHADFGroup] = None
    elayerO: Optional[VTHADFGroup] = None
    f2layerX: Optional[VHADFGroup] = None
    f1layerX: Optional[VHADFGroup] = None
    elayerX: Optional[VHADFGroup] = None
    medAmplF: Optional[List[int]] = None
    medAmplE: Optional[List[int]] = None
    medAmplEs: Optional[List[int]] = None
    trueHeightsCoefF2: Optional[List[float]] = None
    trueHeightsCoefF1: Optional[List[float]] = None
    trueHeightsCoefE: Optional[List[float]] = None
    quasiParabSegm: Optional[List[float]] = None
    editFlagsChar: Optional[List[int]] = None
    valleyDescrWDUM: Optional[List[float]] = None
    eslayerO: Optional[VHADFGroup] = None
    eauroralayerO: Optional[VHADFGroup] = None
    trueheightProf: Optional[TFEGroup] = None
    qualifLTR: Optional[List[str]] = None
    descrLTR: Optional[List[str]] = None
    editFlgTraceProf: Optional[List[int]] = None


    def hasTrueHeightProfile(self):
        if self.trueheightProf is not None and \
                self.trueheightProf.trueHeight is not None and \
                self.trueheightProf.electronDensity is not None:
            return True

        return False

    # def toDB(self,stations):
    #     from Lib.utils import ufunc
    #     hashtuple_ = (self.system.ursicode, str(int(self.sounding.timestamp.timestamp())))
    #     hash_ = ufunc.serialize(hashtuple_)
    #     id = uuid_mod.uuid5(APPUUID, hash_)
    #     _thpdct = self.trueheightProf.dict() if self.trueheightProf else dict()
    #     _saodct = dict(id=id, date=self.sounding.timestamp, stationid=stations[self.system.ursicode].id)
    #     _saodct = _saodct | self.scaled.dict()
    #     _saodct = _saodct | {
    #         'thpheight': _thpdct.get('trueHeight', None),
    #         'thpfreq': _thpdct.get('frequency', None),
    #         'thpeldens': _thpdct.get('electronDensity', None)
    #     }
    #     _saodct['meta'] = self.json()
    #     return md.SAO(_saodct)

    # def todict(self):
    #     print("Assigned values of SAO:")
    #     for field, value in self.__dict__.items():
    #         print(f"{field}: {value}")

    #     return self.dict()
