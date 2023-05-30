class VoronoiEdge:
    def __init__(self, start_vector, left_vector, right_vector):
        self.start = start_vector
        self.left = left_vector
        self.right = right_vector
        self.neighbour = None
        self.end = None
        self.f = (self.right[0] - self.left[0]) / (self.left[1] - self.right[1])
        self.g = self.start[1] - self.f*self.start[0]
        self.direction = (right_vector[1] - left_vector[1], -(right_vector[0] - left_vector[0])) # normal vector of right to left
        self.intersected = False
        self.iCounted = 0
    

    def __str__(self):
        return "VEdge : " + str(self.start) + ", " + str(self.end ) + ", intersected : " + str(self.intersected) + ", iCounted : " + str(self.iCounted) + ", f : " + str(self.f) + ", g : " + str(self.g) + ", neighbour : " + str(self.neighbour != None)
