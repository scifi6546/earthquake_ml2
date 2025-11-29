from event import Event
from running_histogram import histogram_from_event
for event in Event.loadFromDb():
    if event.magnitude() > 6:
        histogram_from_event(event, max_distance=300 * 1000.)
        print(event.magnitude())