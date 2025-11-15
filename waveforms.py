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
WAVEFORMS_DATABASE = "waveforms.db"
def get_stream(station_name, channel: Optional[str], start_time, end_time):
    def get_format_string(station_name, channel, start_time, end_time):
        network_str = "net=AK"
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
    
