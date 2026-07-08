# %%
import re
from pathlib import Path
from datetime import datetime
import ciso8601
import fortranformat as ff
from enum import Enum, Flag
import zipfile
from io import StringIO
from sao import SAO, GeophysicalConst, System, Sounding, ScaledIono, VTHADFGroup, VHADFGroup, TFEGroup

class SAOParser(object):
    _GC = Flag('Group', 'VH TH AMPL DN FREQ ELDENS')

    _DFI = ff.FortranRecordReader('2(40I3)')
    # _GEOCONST = ff.FortranRecordReader('16F7.3')
    _SYSDESCR = ff.FortranRecordReader('A240')
    # _sounder =  ff.FortranRecordReader('120A1')
    _GEOCONST = ff.FortranRecordReader('16F7.3')
    _SCALEDCHAR = ff.FortranRecordReader('15F8.3')
    _ANALYSISFLAGS = ff.FortranRecordReader('60I2')
    _DOPPLERTRANS = ff.FortranRecordReader('16F7.3')

    _VIRTUALHEIGHTS= ff.FortranRecordReader('15F8.3')
    _TRUEHEIGHTS= ff.FortranRecordReader('15F8.3')
    _AMPLITUDES= ff.FortranRecordReader('40I3')
    _DOPPLER= ff.FortranRecordReader('120I1')
    _FREQ= ff.FortranRecordReader('15F8.3')
    _ELDENSITY= ff.FortranRecordReader('15E8.3E1')

    _TRUEHEIGHTSCOEFF = ff.FortranRecordReader('10E11.6E1')
    _QUAZIPARABOLIC = ff.FortranRecordReader('6E20.12E2')

    _EDITFLAGSCHAR = ff.FortranRecordReader('120I1')
    _VALLEYDESCR = ff.FortranRecordReader('10E11.6E1')

    _LETTERS = ff.FortranRecordReader('120A1')
    _EDITFLAGSTRPROF = ff.FortranRecordReader('120I1')

    @staticmethod
    # def _SYSDESCR_FN(input):
    #     opermsg = None
    #     input = input.rstrip()
    #     print(input)
    #     input_ = input
    #     if len(input)>120:
    #         opermsg = input[120:]
    #         input_ = input[0:120]
    #     tokens = [t.split() for t in [r.strip() for r in input_.split(',')]]
    #     token_ = tokens.pop(0)
    #     sysd = dict()
    #     for i,t in enumerate(tokens):
    #         k = t[0].strip()
    #         print('k',k)
    #         v = [v.strip() for v in t[1:]]
    #         if len(input)>120 and i==len(tokens)-1 and len(v)>1:
    #             opermsg = v.pop() + opermsg
    #         sysd[k] = ' '.join(v)

    #     if len(token_)==2:
    #         stationid, ursicode = [t.strip() if t else None for t in token_[1].split('/')]
    #         print(stationid,ursicode)
    #         if stationid=='038':
    #             stationid='138'
    #         if ursicode=='AT038':
    #             ursicode='AT138'
    #         sysd = sysd | {'sounder': token_[0].strip(), 'stationid': stationid, 'ursicode': ursicode, 'opermsg': opermsg if opermsg else None}
    #     # else:
    #     #     if sysd['NAME'] not in cfg['AIS-INGV']:
    #     #         raise
    #     #     aisid = cfg['AIS-INGV'][sysd['NAME']]
    #     #     sysd = sysd | {'sounder': 'AIS-INGV', 'stationid': aisid['id'], 'ursicode': aisid['URSI'], 'opermsg': opermsg if opermsg else None}

    #     return sysd
    def _SYSDESCR_FN2(input):
        opermsg = None
        input = input.rstrip()
        print(input)
        input_ = input
        if len(input) > 120:
            opermsg = input[120:]
            input_ = input[:120]
        
        parts = [t.strip() for t in input_.split(',') if t.strip()]
        tokens = [t.split() for t in parts]


        token_ = tokens.pop(0)
        sysd = dict()
        
        for i, t in enumerate(tokens):
            k = t[0].strip()
            v = [v.strip() for v in t[1:]]
            if len(input) > 120 and i == len(tokens) - 1 and len(v) > 1:
                opermsg = v.pop() + opermsg
            sysd[k] = ' '.join(v)

        if len(token_) == 2:
            stationid, ursicode = [t.strip() if t else None for t in token_[1].split('/')]
            if stationid == '038':
                stationid = '138'
            if ursicode == 'AT038':
                ursicode = 'AT138'
            sysd = sysd | {
                'sounder': token_[0].strip(),
                'stationid': stationid,
                'ursicode': ursicode,
                'opermsg': opermsg if opermsg else None
            }

        return sysd
    @staticmethod
    def _TIMEST_FN(input, nelem = 0):
        dpspos = (0,2,6,9,11,13,15,17,19,22,25,26,27,32,36,41,45,46,47,48,49,50,51,52,55,59,60,64,68,69,70,71,72,73,74,76,77)
        digipos = (0,2,6,9,11,13,15,17,19,30,31,32,38,44,51,53,54,56,59,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77)
        pos = dpspos if input[0:2]=='FF' else digipos if input[0:2]=='FE' else dpspos
        timest = [input[pos[i]:pos[i+1]] for i in range(len(pos)-1) if pos[i+1]<=nelem]
        return timest

    def _read(self,parser = None, pos = None, lines = None):
        if (pos is None or (type(pos)==int and self._dfi and self._dfi[pos]==0)) and lines is None:
            return None

        ret = []
        nelem = self._dfi[pos] if type(pos)==int else None
        while len(ret)<(nelem or 1):
            input = ''
            for l in range(lines or 1):
                input += self._reader.readline().rstrip('\n')
            ret = ret + (parser.read(input) if parser else [input, ])

        lret = len(ret)
        ret = ret[0:nelem] if nelem else ret
        #ret = [None if _ in (9999.0,'/','*') else _ for _ in ret]
        return ret[0] if len(ret) == 1 and lret==1 else ret

    def _load(self):
        gc = SAOParser._GC
        def gcol(groups,ipos = None):
            collection = {}
            parser_ = {
                gc.VH: SAOParser._VIRTUALHEIGHTS,
                gc.TH: SAOParser._TRUEHEIGHTS,
                gc.AMPL: SAOParser._AMPLITUDES,
                gc.DN: SAOParser._DOPPLER,
                gc.FREQ: SAOParser._FREQ,
                gc.ELDENS: SAOParser._ELDENSITY,
            }

            pydparser_ = {
                gc.VH|gc.TH|gc.AMPL|gc.DN|gc.FREQ : VTHADFGroup,
                gc.VH|gc.AMPL|gc.DN|gc.FREQ: VHADFGroup,
                gc.TH|gc.FREQ|gc.ELDENS: TFEGroup
            }

            groupsl = [g for g in SAOParser._GC if g in groups]
            for i,g in enumerate(groupsl):
                collection[g.name] = self._read(parser=parser_[g], pos=ipos+i)

            return pydparser_[groups](**collection) if any(collection.values()) else None

        self._dfi = self._read(parser=SAOParser._DFI, lines=2)
        self._gc= self._read(parser=SAOParser._GEOCONST,pos=0)
        self._scr = self._read(parser=SAOParser._SYSDESCR, lines=self._dfi[1])
        # self._s = self._read(parser=SAOParser._sounder,lines=self._dfi[2])
        # self._gc = self._read(parser=SAOParser._GEOCONST,pos=0)
        self._sr=self._read(pos=False)
        # print(self._gc)
        # print('len',len(self._scr ))
        # print('sccr',self._scr)
        # # print('son',self._s)
        # print('sr',self._sr)
        # result = SAOParser._SYSDESCR_FN(self._scr)

        # # Print the result
        # print('result',result)
        # # Parse the input data
        data = SAOParser._SYSDESCR_FN2(self._scr)
        print('sao data',data)

        # Ensure required fields are present
        if 'sounder' not in data or 'stationid' not in data:
            raise ValueError("Required fields 'sounder' and 'stationid' are missing in the input data.")

        # Transform the dictionary to match the expected aliases
        transformed_data = {
            'sounder': data.get('sounder'),
            'stationid': data.get('stationid'),
            'ursicode': data.get('ursicode'),
            'NAME': data.get('NAME'),  # Ensure this key exists in the input
            'ARTIST': data.get('ARTIST'),
            'NH': data.get('NH'),
            'ADEP': data.get('ADEP'),
            'opermsg': data.get('opermsg')
        }

        # Create the System object
        system = System(**transformed_data)

        # Print the system object for debugging
        # print("System Object:", system)
        
        args =  {
            'geoConst': GeophysicalConst(self._gc),
            'system': System(**transformed_data),
            # 'system': System(description=SAOParser._SYSDESCR_FN(self._scr)),
            # 'sounding': Sounding(SAOParser._TIMEST_FN(self._read(parser=SAOParser._sounder, lines=1),nelem=self._dfi[2])),#, nelem=self._dfi[2])
            # 'system': System(**SAOParser._SYSDESCR_FN(self._scr)),
            'sounding': Sounding(SAOParser._TIMEST_FN(self._sr, nelem=self._dfi[2])),
            'scaled': ScaledIono(self._read(parser=SAOParser._SCALEDCHAR, pos=3)),
            # 'scaled': ScaledIono(self._read(parser=SAOParser._SCALEDCHAR, pos=3)),
            'analysisFlags': self._read(parser=SAOParser._ANALYSISFLAGS, pos=4),
            'dopplerTrans': self._read(parser=SAOParser._DOPPLERTRANS, pos=5),
            'f2layerO': gcol(gc.VH|gc.TH|gc.AMPL|gc.DN|gc.FREQ,ipos=6),
            'f1layerO': gcol(gc.VH|gc.TH|gc.AMPL|gc.DN|gc.FREQ,ipos=11),
            'elayerO': gcol(gc.VH|gc.TH|gc.AMPL|gc.DN|gc.FREQ,ipos=16),
            'f2layerX': gcol(gc.VH|gc.AMPL|gc.DN|gc.FREQ,ipos=21),
            'f1layerX': gcol(gc.VH|gc.AMPL|gc.DN|gc.FREQ,ipos=25),
            'elayerX': gcol(gc.VH|gc.AMPL|gc.DN|gc.FREQ,ipos=29),
            'medAmplF': self._read(parser=SAOParser._AMPLITUDES, pos=33),
            'medAmplE': self._read(parser=SAOParser._AMPLITUDES, pos=34),
            'medAmplEs': self._read(parser=SAOParser._AMPLITUDES, pos=35),
            'trueHeightsCoefF2': self._read(parser=SAOParser._TRUEHEIGHTSCOEFF, pos=36),
            'trueHeightsCoefF1': self._read(parser=SAOParser._TRUEHEIGHTSCOEFF, pos=37),
            'trueHeightsCoefE': self._read(parser=SAOParser._TRUEHEIGHTSCOEFF, pos=38),
            'quasiParabSegm': self._read(parser=SAOParser._QUAZIPARABOLIC, pos=39),
            'editFlagsChar': self._read(parser=SAOParser._EDITFLAGSCHAR, pos=40),
            'valleyDescrWDUM': self._read(parser=SAOParser._VALLEYDESCR, pos=41),
            'eslayerO': gcol(gc.VH|gc.AMPL|gc.DN|gc.FREQ,ipos=42),
            'eauroralayerO': gcol(gc.VH|gc.AMPL|gc.DN|gc.FREQ,ipos=46),
            'trueheightProf': gcol(gc.TH|gc.FREQ|gc.ELDENS,ipos=50),
            'qualifLTR': self._read(parser=SAOParser._LETTERS, pos=53),
            'descrLTR': self._read(parser=SAOParser._LETTERS, pos=54),
            'editFlgTraceProf':  self._read(parser=SAOParser._EDITFLAGSTRPROF, pos=55),
        }
        return args

    def _readSAO(self):
        with open(self.path, 'r') as f:
            self._data = f.read()
            print('data read',self._data)

    def parse(self, path = None, data = None):
        if path and path!=self.path:
            self.path = path
            self._readSAO()
        elif data:
            self._data = data

        self._reader = None
        self._dfi = None
        self.sao = None
        self.valid = None

        if not self._data:
            raise AttributeError('Either define <path> or <data>')

        try:
            with StringIO(self._data) as _reader:
                self._reader = _reader
                self.sao = SAO(**self._load())
                # print(vars(self.sao))  # Prints all attributes inside SAO

                for attr, value in self.sao.__dict__.items():
                    print(f"{attr}: {value}")

                # print('SAO ALL',self.sao)
                # print(len(self.sao))
               

                self.valid = True
        except IOError:
            print('IO Error while parsing product')
            self.valid = False
        # except Exception as e:
        #     print('Error {} while parsing SAO product'.format(e))
        #     SWARMLogger.logger.error(f'Error {e} while parsing SAO product')
        #     self.valid=False

        return self.sao

    def __init__(self, path = None, data = None):
        self.path = path
        self._data = None
        self._reader = None
        self._dfi = None
        self.sao = None
        self.valid = None
        if self.path:
            self._readSAO()
        elif data:
            self._data = data


class SAODataset(object):


    def _readZIP(self):

        def _findSAO(loc):
            reqsao = f'{self.station.code.upper()}_{self.start.strftime("%Y%j%H%M%S")}.SAO$'
            for _szc in loc:
                matched = re.match(reqsao, _szc, re.IGNORECASE)
                if bool(matched):
                    return _szc
            return None

        content = None
        with zipfile.ZipFile(self.path) as _zf:
            loc = _zf.namelist()
            _szf = _findSAO(loc)
            if _szf:
                with _zf.open(_szf, mode='r') as _szf:
                    content = '\n'.join(_szf.read().decode('UTF-8').splitlines())
        return content

    def read(self, path=None):
        self.path = path or self.path
        content = None
        try:
            if self.container=='sao':
                content = self.path.read_text()
            elif self.container=='zip':
                content = self._readZIP()
            else:
                raise AssertionError(f'File: {self.path} Unknown Container: {self.container}')
            assert content is not None, (f'File: {self.path} Empty Container')
        except Exception as e:
            print(f"An error occurred: {e}")
            # return

        #matches = re.finditer(r'^\s\s5(?:[\s\d]{3}){39}\n(?:[\s\d]{3}){40}\n(?:\s+\d*\.\d+|\d+){5}\n', content, re.M)
        matches = re.finditer(r'^\s\s5(?:[\s\d]{3}){39}\n(?:[\s\d]{3}){40}\n(?:[\+\-\s\d\.]{7}){5}\n', content, re.M)
        matches = list(matches)
        for i in range(len(matches)):
            start = matches[i].regs[0][0]
            end = matches[i+1].regs[0][0] if i < len(matches) -1 else None
            if end:
                content_ = content[start:end]
            else:
                content_ = content[start:]

            sparser = SAOParser(data=content_)
            sao = sparser.parse()
            try:
                isvalid = sparser.valid
            except Exception as e:
                isvalid = sparser.valid

            if isvalid:
                if not (hasattr(sao, 'sounding') and sao.sounding and hasattr(sao.sounding, 'timestamp') and sao.sounding.timestamp):
                    print(f'Invalid sounding: {sao}')
                yield sao

    def __init__(self, path, start, end=None, station = None):
        self.path = Path(path)
        self.station = station
        self.station_code=station.code
        self.container = self.path.suffix.lower().lstrip('.')

        self.start = start if isinstance(start, datetime) else ciso8601.parse_datetime(start)
        if end is not None:
            self.end = end if isinstance(end, datetime) else ciso8601.parse_datetime(end)



