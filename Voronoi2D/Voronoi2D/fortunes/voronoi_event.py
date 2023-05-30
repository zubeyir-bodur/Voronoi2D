from functools import total_ordering

@total_ordering
class VoronoiEvent:
    def __init__(self, point_, place_event_flag_):
        self.point = point_
        self.place_event_flag = place_event_flag_
        self.y = point_[1]
        self.arch = None

    def __eq__(self, other):
        return self.y == other.y
        
    def __lt__(self, other):
        return self.y < other.y


    #def compare_with_event(self, another_event):
    #    return -1 if (self.y > another_event.y) else 1


    def __str__(self):
        return "VEvent : placeEvent : " + str(self.place_event_flag) + ", y : " + str(self.y) + ", arch : " + str(self.arch != None)
