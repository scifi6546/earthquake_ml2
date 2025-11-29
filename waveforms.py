import requests
import sqlite3
import io
from obspy import read as wf_read, UTCDateTime
from typing import Optional
class WaveformNotFound(Exception):
    _stationName: str
    _channel: str
    _startTime: UTCDateTime
    _endTime: UTCDateTime
    def __init__(self, station_name: str, channel: str, start_time: UTCDateTime, end_time: UTCDateTime):
        
        self._stationName = station_name
        self._channel = channel
        self._startTime = start_time
        self._endTime = end_time
        super().__init__()
    def _getMessage(self) -> str:
        return f"{self._stationName}, ({self._startTime}-{self._endTime}) not found"
    def __str__(self):
        return self._getMessage()
class InvalidStation(Exception):
    _station_name: str
    def __init__(self, station_name: str):
        self._station_name = station_name
        super().__init__()
    def _getMessage(self) -> str:
        return f"station {self._station_name} is invalid"
    def __str__(self):
        return self._getMessage()
WAVEFORMS_DATABASE = "waveforms.db"
def get_stream(station_name, channel: Optional[str], start_time, end_time):
    def is_station_valid(station_name: str) -> str:
        INVALID_STATIONS = ["TT01"]
        if station_name in INVALID_STATIONS:
            return False
        else:
            return True
    def get_network_code(station_name: str) -> str:
        """
        Some stations are not operated by the AEC, therefore we have to manually map them
        """
        STATION_MAPPINGS = {
            "ACH": "AV",
            "ADAG": "AV",
            "ADK":"IU",
            'AHB': "AV",
            "AKBB": "AV",
            "AKGG": "AV",
            "AKHS": "AV",
            "AKLV": "AV",
            "AKMO": "AV",
            "AKRB": "AV",
            "AKS": "AV",
            "BPBC": "AV",
            "BPPC": "AV",
            "BRPK": "AV",

            "AKSA": "AV",
            "AKUT": "AT",
            "AKT": "AV",
            "AKV": "AV",
            "AMKA": "AV",
            "ANCK": "AV",
            "AU22": "AV",
            "AUCH": "AV",
            "AUJA": "AV",
            "AUL":"AV",
            "AUNO": "AV",
            "AUSB": "AV",
            "AUSS": "AV",
            "AUW": "AV",
            "AUWS": "AV",
            "AULG": "AV",
            "CAHL": "AV",
            "DRR3": "AV",
            "DT1": "AV",
            "DTN": "AV",
            "HAG": "AV",
            "ILS": "AV",
            "ILSW": "AV",
            "ILW": "AV",
            "INE":"AV",
            "ISNN":"AV",
            "ISTK": "AV",

            "IVE": "AV",
            "IVS": "AV",
            "KAB2": "AV",
            "KABR": "AV",
            "KABU": "AV",
            "KAHC": "AV",
            "KAHG": "AV",

            "KAG": "AV",
            "KAKN": "AV",
            "KAPH": "AV",
            "KAPI": "AV",
            "KARR": "AV",
            "KAVE":"AV",
            "KAWS": "AV",
            "KAWH": "AV",
            "KBM": "AV",
            "KCE": "AV",
            "KDAK":"II",
            "KJL": "AV",
            "KEL": "AV",
            "KVT": "AV",
            "M22K": "TA",
            "MID": "AT",
            
            "Q23K": "TA", # before 2019 afterwards it is AK, need to figure out error handling somehow
            "RDDF": "AV",
            "RDJH": "AV",
            "RDSO": "AV",
            "RDT": "AV",
            "RDW":"AV",
            "RDWB": "AV",
            "RED": "AV",
            "SDPT": "AT",
            "SPBG": "AV",
            "SPBL": "AV",
            "SPCG": "AV",
            "SPCL": "AV",
            "SPCN": "AV",
            "SPCP": "AV",
            "SPNN": "AV",
            "SPNW": "AV",
            "SPU": "AV",
            "SPWE": "AV",
            "SPWE": "AV",
            "SSBA": "AV",
            "SSLN": "AV",
            "SSLN": "AV",
            "SSLW": "AV",
            "STLK": "AV",
            "NCT": "AV",
            "MAPS": "AV",
            "MGLS": "AV",
            "MGOD": "AV",
            "MNAT": "AV",
            "MSW": "AV",
            "OHAK": "AT",
            "O22K": "TA",
            "OPT": "AV",
            "PLK1": "AV",
            "PLK2": "AV",
            "PLK3": "AV",
            "PLK5": "AV",
            "PLBL": "AV",
            "PMR": "AT",
            "PS1A": "AV",
            "PS4A": "AV",
            "PV6A": "AV",
            "PVV": "AV",

            "SVW2": "AT",
            "SDPI": "AV",
            "SPDG": "AV",
            
            "SPCG": "AV",
            "SPCL": "AV",
            "SPCN": "AV",
            "SPCR": "AV",
            "SPCP": "AV",
            
            "SSBA": "AV",
            "SSLN": "AV",
            "SSLS": "AV",
            "SSLW": "AV",
            "SLTK": "AV",
            "TT01": "IM",
            "TTA": "AT",

            "VNBL": "AV",
            "VNCG": "AV",
            "VNDA": "AV",
            "VNFG": "AV",
            "VBNKR": "AV",
            "VNHG": "AV",
            "VNSG": "AV",
            "VNMSO":"AV",
            "VNSW": "AV",
            "VNWF": "AV",

            "WECS": "AV",
            "WESE": "AV",
            "WESP": "AV",
            "WESS": "AV",
            "WFAR" :"AV",
            "WHTTR": "AV",
            "WPOG": "AV",
            "WTUG": "AV"

        }
        if station_name in STATION_MAPPINGS:
            return STATION_MAPPINGS[station_name]
        else:
            return "AK"
    def get_format_string(station_name: str, channel, start_time, end_time):
        network_str = f"net={get_network_code(station_name)}"
        station_str = "sta={}".format(station_name)
        if channel is None:
            channel_str = None
        else:
            channel_str = "cha={}".format(channel)
        start_str = "start={}".format(start_time)
        end_str = "end={}".format(end_time)
        format_str = "format=mseed"

        format_array = [
            network_str,
            station_str,
            channel_str,
            start_str,
            end_str,
            format_str
        ]
        format_array = list(filter(lambda x: x is not None, format_array))
        format_string = "&".join(format_array)
      
        base = "https://service.iris.edu/fdsnws/dataselect/1/query?"
        return base+format_string
    def get_channel_string(channel: Optional[str]) -> str:
        if channel is None:
            return "*"
        else:
            return channel
    def read_waveform_from_database(station_name: str, channel: Optional[str], start_time: UTCDateTime, end_time: UTCDateTime, database: sqlite3.Connection):
        cursor = database.cursor()
        channel_string = get_channel_string(channel)
        raw_data =  cursor.execute(
            "SELECT waveform_data FROM waveforms WHERE station_name = ? AND start_time = ? AND end_time = ? AND channel = ?", 
            (
                station_name, str(start_time), str(end_time), channel_string
            )
            ).fetchone()
        if raw_data is None:
            return None
        else:
            return wf_read(io.BytesIO(raw_data[0]))
    def fetch_from_website(station_name: str, channel: Optional[str], start_time: UTCDateTime, end_time: UTCDateTime, database: sqlite3.Connection):
        cursor = database.cursor()
        full_url = get_format_string(station_name, channel, start_time, end_time)
        result = requests.get(full_url)
        if result.status_code != 200:
            raise WaveformNotFound(station_name, channel, start_time, end_time)
        cursor.execute(
            "INSERT INTO waveforms(station_name, channel, start_time, end_time, waveform_data) VALUES (?, ?, ?, ?, ?)", 
            (station_name, get_channel_string(channel), str(start_time), str(end_time), result.content)
        )
        database.commit()
        return read_waveform_from_database(station_name, channel, start_time, end_time, database)
    if not is_station_valid(station_name):
        raise InvalidStation(station_name)
    db = sqlite3.connect(WAVEFORMS_DATABASE)
    cursor = db.cursor()
    cursor.execute(
        """CREATE TABLE IF NOT EXISTS waveforms(
                station_name TEXT NOT NULL,
                channel TEXT NOT NULL,
                start_time TEXT NOT NULL, 
                end_time TEXT NOT NULL, 
                waveform_data BLOB NOT NULL,
                PRIMARY KEY(station_name, channel, start_time, end_time),
                UNIQUE(station_name, channel, start_time, end_time)
            ) 
            STRICT;
        """
    )
    db.commit()
    data = read_waveform_from_database(station_name, channel, start_time, end_time, db)
    if data is not None:
        return data
    else:
        return fetch_from_website(station_name, channel, start_time, end_time, db)
    
