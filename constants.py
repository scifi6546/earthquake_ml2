from pathlib import Path
EVENT_DATABASE_PATH = Path("./all_events.db")
SECOND = 1.0
MINUTE = 60.0 * SECOND
"""
Number of seconds in a minute
"""
HOUR = 60.0 * MINUTE
"""
Number of seconds in an hour
"""
DAY = 24.0 * HOUR
"""
Number of seconds in a day
"""
MAGNITUDE_BINS_NAME = "magnitude_bins"