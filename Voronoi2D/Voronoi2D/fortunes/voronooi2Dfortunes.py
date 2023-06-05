import numpy as np
import math
from queue import PriorityQueue
from fortunes.voronoi_parabola import VoronoiParabola
from fortunes.voronoi_event import VoronoiEvent
from fortunes.voronoi_edge import VoronoiEdge
import pygame

class Voronoi2DFortunes:
    def __init__(self, surface_, voronoi_seeds, width_, height_):
        self.surface = surface_         # PyGame surface where everything will be drawn to
        self.done = False
        self.root_parabola = None       # BinaryTree of parabolas
        self.width = width_             # Surface dimensions, width
        self.height = height_           # Surface dimensions, height
        self.sweep_line_pos = 0.0       # Beach line, which is horizontal
        self.places = voronoi_seeds     # an array of two tuples, representing voronoi seeds (or delaunay vertices)
        self.edges = []
        self.deleted_events = set()
        self.points = []
        self.event_queue = PriorityQueue()
        for p in self.places:
            self.event_queue.put(VoronoiEvent((p[0], p[1]), True))


    def get_edges(self):
        """
        Retrieves the voronoi edges without animation
        """
        while not self.event_queue.empty():
            event = self.event_queue.get()
            self.sweep_line_pos = event.point[1]
            if event in self.deleted_events:
                self.deleted_events.remove(event)
            elif event.place_event_flag:
                self.insert_parabola(event.point)
            else:
                self.remove_parabola(event)
        self.finish_edge(self.root)
        for edge in self.edges:
            if edge.neighbour != None:
                edge.start = edge.neighbour.end
                edge.neighbour = None
        return self.edges


    def update(self):
        """
        Will be called to update the beach line, supports animation
        """
        if not self.done:
            if not self.event_queue.empty():
                event = self.event_queue.get()
                self.sweep_line_pos = event.point[1]
                if event in self.deleted_events:
                    self.deleted_events.remove(event)
                elif event.place_event_flag:
                    self.insert_parabola(event.point)
                else:
                    self.remove_parabola(event)
                self.finish_edge(self.root_parabola)
            else:
                self.done = True
                self.finish_edge(self.root_parabola)
                for edge in self.edges:
                    if edge.neighbour != None:
                        edge.start = edge.neighbour.end
                        edge.neighbour = None
        return

    def draw(self):
        """
        """
        # Draw currently completed voronoi edges
        for edge in self.edges:
            p0 = edge.start
            p1 = edge.end
            if edge.neighbour != None:
                p0 = edge.neighbour.end
            # Draw line from p0 to p1
            if p0 != (float('nan'), float('nan')) and p1 != (float('nan'), float('nan')) and p0 != None and p1 != None:
                pygame.draw.line(surface=self.surface, color="#CC00FF", start_pos=p0, end_pos=p1, width=2)
        # if not finished
        if not self.done:
            # Draw sweep line
            pygame.draw.line(surface=self.surface, color="#000000", start_pos=(0, self.sweep_line_pos), end_pos=(self.width, self.sweep_line_pos), width=2)
            # Draw beach line 
            self.draw_beach_line()
        # Draw the voronoi site points as a circle
        for v_site_point in self.places:
            pygame.draw.circle(self.surface, "#FF0000", v_site_point, 2)
        return

    def draw_beach_line(self):
        """
        Draws the green parabolas that are extensions of the incomplete voronoi diagram
        """
        prev_x = 0.0
        prev_y = float('nan')
        par = self.get_parabola_by_x(prev_x)
        num_iterations = 1000
        for i in range(1, num_iterations):
            x = self.width*float(i)/float(num_iterations)
            y = float('nan')
            par = self.get_parabola_by_x(x)
            if par.site != None:
                y = self.get_y(par.site, x)
            else:
                print(par)
            if prev_y != float('nan') and y != float('nan'):
                # draw line from (prev_x, prev_y) to (x, y), color is pure green
                pygame.draw.line(surface=self.surface, color=(0, 255, 0), start_pos=(prev_x, prev_y), end_pos=(x, y), width=2)
            prev_x = x
            prev_y = y
        return

    def insert_parabola(self, parabola_vector):
        """
        """
        # Base case
        if self.root_parabola == None:
            self.root_parabola = VoronoiParabola(parabola_vector)
            return

        # Degenerate event - heights are the same
        if self.root_parabola.isLeaf and (self.root_parabola.site[1] - parabola_vector[1]) < 1:
            fp = self.root_parabola.site
            self.root_parabola.isLeaf = False
            self.root_parabola.set_left_child(VoronoiParabola(fp))
            self.root_parabola.set_right_child(VoronoiParabola(parabola_vector))
            s = ((parabola_vector[0] + fp[0])/2.0, self.height) # starting edge
            self.points.append(s)
            if parabola_vector[0] > fp[0]:
                self.root_parabola.edge = VoronoiEdge(s, fp, parabola_vector)
            else:
                self.root_parabola.edge = VoronoiEdge(s, parabola_vector, fp)
            self.edges.append(self.root_parabola.edge)
            return

        # General case
        par = self.get_parabola_by_x(parabola_vector[0])

        if par.circleEvent != None:
            self.deleted_events.add(par.circleEvent)
            par.circleEvent = None

        start = (parabola_vector[0], self.get_y(par.site, parabola_vector[0]))
        self.points.append(start)

        edgeLeft = VoronoiEdge(start, par.site, parabola_vector)
        edgeRight = VoronoiEdge(start, parabola_vector, par.site)

        edgeLeft.neighbour = edgeRight
        self.edges.append(edgeLeft)

        par.edge = edgeRight

        p0 = VoronoiParabola(par.site)
        p1 = VoronoiParabola(parabola_vector)
        p2 = VoronoiParabola(par.site)
        
        par.isLeaf = False
        par.set_right_child(p2)
        par.set_left_child(VoronoiParabola())
        par.left_child().edge = edgeLeft
        par.left_child().set_left_child(p0)
        par.left_child().set_right_child(p1)

        self.check_circle(p0)
        self.check_circle(p2)
        return

    def remove_parabola(self, parabola_event):
        """
        """
        p1 = parabola_event.arch
        xl = p1.left_parabola_on_upper_level()
        xr = p1.right_parabola_on_upper_level()
        p0 = xl.left_parabola_on_lower_level()
        p2 = xr.right_parabola_on_lower_level()

        if p0 == p2:
            print("Error: parabola left and right have the same focus")

        if p0.circleEvent != None:
            self.deleted_events.add(p0.circleEvent)
            p0.circleEvent = None
        if p2.circleEvent != None:
            self.deleted_events.add(p2.circleEvent)
            p2.circleEvent = None

        p_vector = (parabola_event.point[0], self.get_y(p1.site, parabola_event.point[0]))
        self.points.append(p_vector)

        xl.edge.end = p_vector
        xr.edge.end = p_vector

        higher = VoronoiParabola()
        par = p1
        while par != self.root_parabola:
            par = par.parent
            if par == xl:
                higher = xl;
            elif par == xr:
                higher = xr
        higher.edge = VoronoiEdge(p_vector, p0.site, p2.site)
        self.edges.append(higher.edge)

        grand_parent = p1.parent.parent
        if grand_parent != None:
            if p1.parent.left_child() == p1:
                if grand_parent.left_child() == p1.parent:
                    grand_parent.set_left_child(p1.parent.right_child())
                if grand_parent.right_child() == p1.parent:
                    grand_parent.set_right_child(p1.parent.right_child())
            else:
                if grand_parent.left_child() == p1.parent:
                    grand_parent.set_left_child(p1.parent.left_child())
                if grand_parent.right_child() == p1.parent:
                    grand_parent.set_right_child(p1.parent.left_child())
        self.check_circle(p0)
        self.check_circle(p2)
        return

    def finish_edge(self, voronoi_parabola):
        """
        """
        if voronoi_parabola.isLeaf:
            return
        mx = -1.0
        if voronoi_parabola.edge.direction[0] > 0.0:
            mx = max(self.width, voronoi_parabola.edge.start[0] + 10)
        else:
            mx = min(0.0, voronoi_parabola.edge.start[0] - 10)

        end_vector = (mx, mx*voronoi_parabola.edge.f + voronoi_parabola.edge.g)
        voronoi_parabola.edge.end = end_vector
        self.points.append(end_vector)
        self.finish_edge(voronoi_parabola.left_child())
        self.finish_edge(voronoi_parabola.right_child())


    def get_x_of_edge(self, voronoi_parabola, y):
        """
        Returns the current x position of an intersection point
        """
        left = voronoi_parabola.left_parabola_on_lower_level()
        right = voronoi_parabola.right_parabola_on_lower_level()
        p = left.site
        r = right.site

        delta_p_1 = 2.0*(p[1] - self.sweep_line_pos)
        a1 = 1 / delta_p_1
        b1 = -2* p[0] / delta_p_1
        c1 = self.sweep_line_pos + delta_p_1/4 + (p[0]**2)/delta_p_1

        delta_p_2 = 2.0*(r[1] - self.sweep_line_pos)
        a2 = 1 / delta_p_2
        b2 = -2* r[0] / delta_p_2
        c2 = self.sweep_line_pos + delta_p_2/4 + (r[0]**2)/delta_p_2

        a = a1 - a2
        b = b1 - b2
        c = c1 - c2

        discriminant = b**2 - 4*a*c
        x_1 = (-b + math.sqrt(discriminant)) / (2*a)
        x_2 = (-b - math.sqrt(discriminant)) / (2*a)

        return_y = -1.0
        if p[1] < r[1]:
            return_y = max(x_1, x_2)
        else:
            return_y = min(x_1, x_2)
        return return_y


    def get_parabola_by_x(self, x):
        """
        Returns the parabola object on the beach line given the x coordinate
        """
        new_parbola = self.root_parabola
        x_new = 0.0
        while not new_parbola.isLeaf:
            x_new = self.get_x_of_edge(new_parbola, self.sweep_line_pos)
            if x_new > x:
                new_parbola = new_parbola.left_child()
            else:
                new_parbola = new_parbola.right_child()
        return new_parbola


    def get_y(self, parabola_vector, focus_x):
        """
        Solves y for focus_x in the parabola, given the parabola_vector as
        the second degree polynomial
        """
        delta_p = 2.0*(parabola_vector[1] - self.sweep_line_pos)
        a1 = 1 / delta_p
        b1 = -2* parabola_vector[0] / delta_p
        c1 = self.sweep_line_pos + delta_p/4 + (parabola_vector[0]**2)/delta_p
        return a1*focus_x*focus_x + b1*focus_x + c1


    def check_circle(self, voronoi_parabola):
        """
        """
        left_parent_parabola = voronoi_parabola.left_parabola_on_upper_level()
        right_parent_parabola = voronoi_parabola.right_parabola_on_upper_level()
        parabola_a = None
        parabola_c = None
        if left_parent_parabola != None:
            parabola_a = left_parent_parabola.left_parabola_on_lower_level()
        if right_parent_parabola != None:
            parabola_c = right_parent_parabola.right_parabola_on_lower_level()

        if parabola_a == None or parabola_c == None or parabola_a.site == parabola_c.site:
            return

        vector_s = self.get_edge_intersection(left_parent_parabola.edge, right_parent_parabola.edge)
        if vector_s == None:
            return

        delta_x = parabola_a.site[0] - vector_s[0]
        delta_y = parabola_a.site[1] - vector_s[1]
        delta_distance = math.sqrt(delta_x**2 + delta_y**2)
        if vector_s[1] - delta_distance >= self.sweep_line_pos:
            return

        new_event = VoronoiEvent((vector_s[0], vector_s[1] - delta_distance), False)
        self.points.append(new_event.point)
        voronoi_parabola.circleEvent = new_event
        new_event.arch = voronoi_parabola
        self.event_queue.put(new_event)
        return

    def get_edge_intersection(self, edge_a, edge_b):
        """
        """
        # Find edge intersections through normals of edges
        x_intersect = (edge_b.g - edge_a.g) / (edge_a.f - edge_b.f)
        y_intersect = edge_a.f*x_intersect + edge_a.g

        if (x_intersect - edge_a.start[0]) / edge_a.direction[0] < 0:
           return None
        if (y_intersect - edge_a.start[1]) / edge_a.direction[1] < 0:
           return None

        if (x_intersect - edge_b.start[0]) / edge_b.direction[0] < 0:
            return None
        if (y_intersect - edge_b.start[1]) / edge_b.direction[1] < 0:
            return None

        intersection_point = (x_intersect, y_intersect)
        self.points.append(intersection_point)
        return intersection_point