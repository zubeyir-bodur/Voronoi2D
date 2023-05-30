from turtle import left


class VoronoiParabola:
    """
    Binary Tree that holds the parabolas, also itself is a parabola
    """
    def __init__(self):
        self.site = None            # Site vector, 2-tuple
        self.leftParabola = None    # Left Child
        self.rightParabola = None   # Right Child
        self.parent = None          # Parent parabola
        self.edge = None            
        self.circleEvent = None
        self.isLeaf = False         # True if there is no left or right child
    

    def __init__(self, site_vector):
        self.site = site_vector
        self.leftParabola = None
        self.rightParabola = None
        self.parent = None
        self.edge = None
        self.circleEvent = None
        self.isLeaf = True


    def left_child(self):
        return self.leftParabola


    def right_child(self):
        return self.rightParabola


    def set_left_child(self, left_par):
        self.leftParabola = left_par
        if left_par != None:
            left_par.parent = self
        if left_par != None or self.rightParabola != None:
            self.isLeaf = False
        else:
            self.isLeaf = True


    def set_right_child(self, right_par):
        self.rightParabola = right_par
        if right_par != None:
            right_par.parent = self
        if right_par != None or self.leftParabola != None:
            self.isLeaf = False
        else:
            self.isLeaf = True


    def left_parabola_on_upper_level(self):
        current_parent = self.parent
        previous_parent = self
        while previous_parent.left_child() != current_parent:
            if current_parent.parent != None:
                previous_parent = current_parent
                current_parent = current_parent.parent
            else:
                return None
        return current_parent


    def right_parabola_on_upper_level(self):
        current_parent = self.parent
        previous_parent = self
        while previous_parent.right_child() != current_parent:
            if current_parent.parent != None:
                previous_parent = current_parent
                current_parent = current_parent.parent
            else:
                return None
        return current_parent


    def left_parabola_on_lower_level(self):
        current_parent = self.left_child()
        while not current_parent.isLeaf:
            current_parent = current_parent.right_child()
        return current_parent


    def right_parabola_on_lower_level(self):
        current_parent = self.right_child()
        while not current_parent.isLeaf:
            current_parent = current_parent.left_child()
        return current_parent


    def left_parabola_on_same_level(self):
        upper = self.left_parabola_on_upper_level()
        if upper != None:
            return upper.left_parabola_on_lower_level()
        else:
            return None


    def right_parabola_on_same_level(self):
        upper = self.right_parabola_on_upper_level()
        if upper != None:
            return upper.right_parabola_on_lower_level()
        else:
            return None


    def __str__(self):
        return "VParabola : circleEvent : " + str(self.circleEvent != None) + ", leftParabola : " + str(self.leftParabola != None) + ", rightParabola : " + str(self.rightParabola != None) + ", parent : " + str(self.parent != None) + ", isLeaf : " + str(self.isLeaf)         + ", edge : " + str(self.edge != None)
