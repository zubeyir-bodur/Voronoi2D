from functools import total_ordering

@total_ordering
class VoronoiEvent:
    def __init__(self, point_, place_event_flag_):
        self.point = point_
        self.place_event_flag = place_event_flag_
        self.y = point_[1]
        self.arch = None

    def __eq__(self, other):
        if other != None:
            return self.y == other.y
        else:
            return False
        
    def __lt__(self, other):
        if other != None:
            return self.y > other.y
        else: 
            return False
    

    def __hash__(self):
        return hash(id(self))


    def __str__(self):
        return "VEvent : placeEvent : " + str(self.place_event_flag) + ", y : " + str(self.y) + ", arch : " + str(self.arch != None)
