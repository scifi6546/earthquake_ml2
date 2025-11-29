import sqlite3
from constants import EVENT_DATABASE_PATH
from obspy import UTCDateTime
from matplotlib import pyplot as plt
import numpy
from obspy.geodetics import gps2dist_azimuth
from waveforms import get_stream
from datetime import timedelta
class InvalidOrigin(Exception):
    _message: str
    def __init__(self, message: str):
        self._message =  message
        super().__init__(self._message)
class InvalidPick(Exception):
    _message: str
    def __init__(self, message: str):
        self._message =  message
        super().__init__(self._message)
class Site:
    _station_name: str
    _latitude: float
    _longitude: float
    def __init__(self, station_name: str, cursor):
        result = cursor.execute("SELECT lat, lon FROM site WHERE sta = ? LIMIT 1", (station_name,))
        self._station_name = station_name
        
        row = result.fetchone()
        if row == None:

            self._latitude = None
            self._longitude = None
        else:
            self._latitude = row["lat"]
            self._longitude = row["lon"]

        
    def latitude(self) -> float:
        return self._latitude
    def longitude(self) -> float:
        return self._longitude
    def stationName(self) -> str:
        return self._station_name
class Pick:
    _time: UTCDateTime
    _phase: str
    _aridId: int
    _site: Site
    def __init__(self, arid: int, cursor):
        result = cursor.execute("SELECT time, iphase, sta FROM arrival WHERE arid = ?;", (arid,))

        row = result.fetchone()
        if row is None:
            raise InvalidPick(f"pics do not exist for arrival: {arid}")
        self._time = UTCDateTime(row["time"])

        self._phase = row["iphase"]
        self._aridId = arid
        self._site = Site(row["sta"], cursor)
    def time(self) -> UTCDateTime:
        return self._time
    def phase(self) -> str:
        return self._phase
    def aridId(self) -> int:
        return self._aridId
    def site(self) -> Site:
        return self._site
class Origin:
    """
    Class Reperesenting a origin
    """
    _longitude: float
    _latitude: float
    _magnitude: float
    _originId: int
    _time: UTCDateTime
    _depth: float
    _pics: list[Pick]
    def __init__(self, orid: int, cursor):
        result = cursor.execute("SELECT origin.lat, origin.lon, origin.depth, origin.time FROM origin "\
               " WHERE orid = ?;", (orid,))
        row = result.fetchone()
        self._latitude = row["lat"]
        self._longitude = row["lon"]
        self._originId = orid
        self._depth = float(row["depth"])
        self._time = UTCDateTime(row["time"])
        magnidude_result = cursor.execute(
            "SELECT netmag.magnitude FROM origin JOIN netmag ON origin.mlid = netmag.magid WHERE origin.orid = ?;", (orid,))
        mag_row = magnidude_result.fetchone()
        if mag_row is None:
            raise InvalidOrigin("Event does not have a magnitude")
        self._magnitude = mag_row["magnitude"]

        result = cursor.execute("SELECT arid FROM assoc WHERE orid = ?;", (orid,))
        self._pics = []
        for row in result.fetchall():
            try:

                self._pics.append(Pick(row["arid"], cursor))
            except InvalidPick as e:
                print(f"skipping pick: {row["arid"]} reason: {e}")

        if len(self._pics) == 0:
            raise InvalidOrigin(f"origin {self.originId()} has zero pics")
    def picks(self) -> list[Pick]:
        return self._pics
    def picksFilter(self, filter_function) -> list[Pick]:
        return list(filter(filter_function, self._pics))
    def time(self) -> UTCDateTime:
        return self._time
    def depth(self) -> float:
        return self._depth
    def originId(self) -> int:
        return self._originId
    def magnitude(self) -> float:
        return self._magnitude
    def latitude(self) -> float:
        return self._latitude
    def longitude(self) -> float:
        return self._longitude
    def distance_from_site(self, site: Site) -> float:
        """
        Returns distance in meters"""
        (distance_meters, a, _a) = gps2dist_azimuth(self.latitude(), self.longitude() ,site.latitude(), site.longitude())
        return distance_meters
    
    def numberPicks(self, cursor) -> int:
        result = cursor.execute("SELECT COUNT(*) FROM assoc WHERE orid = ?;", (self.originId(),))
        row = result.fetchone()
        return row[0]


class Event:
    _origin: Origin
    _event_id: int
    _event_name: str
    def origin(self) -> Origin:
        return self._origin
    def latutude(self) -> float:
        return self.origin().latitude()
    def longitude(self) -> float:
        return self.origin().longitude()
    def time(self) -> UTCDateTime:
        return self.origin().time()
    def magnitude(self) -> float:
        return self.origin().magnitude()
    def eventId(self) -> int:
        return self._event_id
    def eventName(self) -> str:
        return self._event_name
    def depth(self) -> float:
        return self.origin().depth()
    def usgsLink(self) -> str:
        return f"https://earthquake.usgs.gov/earthquakes/eventpage/ak{self.eventName()}/executive"
    def numberPicks(self, cursor) -> int:
        return self.origin().numberPicks(cursor)
    def picks(self) -> list[Pick]:
        return self.origin().picks()
    def picksFilter(self, filter_function) -> list[Pick]:
        return self.origin().picksFilter(filter_function)
    def distance_from_site(self, site: Site) -> float:
        """Returns distance in meters"""
        return self.origin().distance_from_site(site)
    def __init__(self, event_id: int, cursor):

        prefor = cursor.execute("SELECT prefor FROM event WHERE evid = ?;", (event_id,)).fetchone()
        if prefor == None:
            print(f"ERROR: no prefor:, evid: {event_id}")
        self._origin = Origin(prefor[0], cursor)
        self._event_id = int(event_id)
        event_name = cursor.execute(
            "SELECT aecevent.eventname FROM EVENT "\
            "JOIN aecevent ON event.evid = aecevent.evid WHERE event.evid = ?;", (event_id,)
        ).fetchone()
        self._event_name = str(event_name["eventname"])

    @staticmethod
    def loadFromDb(start_time = None, end_time = None):
        if start_time is not None:
            start_time = UTCDateTime(start_time)
        if end_time is not None:
            end_time = UTCDateTime(end_time)
        event_database = sqlite3.connect(EVENT_DATABASE_PATH)
        cursor = event_database.cursor()
        cursor.row_factory = sqlite3.Row

        if start_time is None and end_time is None:
            query = """
                    SELECT 
                    event.evid as evid
                    FROM event JOIN origin ON event.prefor = origin.orid
                """
            cursor.execute(query)
        elif start_time is None and end_time is not None:
            query = """
                    SELECT 
                    event.evid as evid
                       
                    FROM event JOIN origin ON event.prefor = origin.orid
                    WHERE event.time < ?
                """
            cursor.execute(query, (end_time.timestamp,))
        elif start_time is not None and end_time is None:
            query = """
                    SELECT 
                    event.evid as evid
                       
                    FROM event JOIN origin ON event.prefor = origin.orid
                    WHERE origin.time > ?
                """
            cursor.execute(query, (start_time.timestamp,))
        elif start_time is not None and end_time is not None:
            query = """
                    SELECT 
                        event.evid as evid
                    FROM event JOIN origin ON event.prefor = origin.orid
                    WHERE origin.time > ? AND origin.time < ?
            """
            cursor.execute(query, (start_time.timestamp, end_time.timestamp))
        else:
            raise Exception("should never get here!")

        res = cursor.fetchall()

        for row in res:
            try:
                output_event = Event(row["evid"],cursor)

                yield output_event
            except InvalidOrigin as e:
                print(f"skipping evid: {row["evid"]} reason: {e}")

