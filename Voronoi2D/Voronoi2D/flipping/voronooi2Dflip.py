import numpy as np
import math
from math import acos, sqrt

def left_test(p0, p1, p2):
        return (p0[0] - p2[0]) * (p1[1] - p2[1]) - (p0[1] - p2[1]) * (p1[0] - p2[0]) > 0


def intersection_test(p0, p1, q0, q1):
    return (left_test(q0, q1, p0) != left_test(q0, q1, p1) and
        left_test(p0, p1, q0) != left_test(p0, p1, q1)
    )


class Voronoi2DFlipping:
    def __init__(self, center=(0, 0), radius=36500):
        self.reset(center, radius)

    
    def reset(self, center, radius):
        center = np.asarray(center)
        # Corners for super-triangles
        self.coords = [center + radius * np.array((-1, -1)),
                       center + radius * np.array((+1, -1)),
                       center + radius * np.array((+1, +1)),
                       center + radius * np.array((-1, +1))]
        self.triangles = {}
        self.circles = {}
        # The super triangles
        T1 = (0, 1, 3)
        T2 = (2, 3, 1)
        self.triangles[T1] = [T2, None, None]
        self.triangles[T2] = [T1, None, None]
        # Compute circumcircles
        for t in self.triangles:
            self.circles[t] = self.circumcenter(t)
        return


    def circumcenter(self, tri):
        """
        TODO May need to change the code to avoid singular matrix error
        """
        pts = np.asarray([self.coords[v] for v in tri])
        pts2 = np.dot(pts, pts.T)
        A = np.bmat([[2 * pts2, [[1],
                                 [1],
                                 [1]]],
                     [[[1, 1, 1, 0]]]])

        b = np.hstack((np.sum(pts * pts, axis=1), [1]))
        x = np.linalg.solve(A, b)
        bary_coords = x[:-1]
        center = np.dot(bary_coords, pts)
        radius = np.sum(np.square(pts[0] - center))
        return (center, radius)

    def area(self, x1, y1, x2, y2, x3, y3):
        return abs((x1*(y2-y3) + x2*(y3-y1)+ x3*(y1-y2))/2.0)


    def outer_vertices_of_triangle(self, tri_indices):
        """
        Gives a list of vertices of a triangle,
        which points to an outer triangle. This vertex
        will eventually point to an outer edge of the
        final triangulation when virtual triangles are 
        removed.
        
        Outer triangle = virtual triangles
        that were generated with the result of the super 
        triangle
        """
        outsides = []
        for i, neigh in enumerate(self.triangles[tri_indices]):
            if neigh == None or neigh[0] <= 3 or neigh[1] <= 3 or neigh[2] <= 3:
                outsides.append(tri_indices[i])
        return outsides

    def angle_of_vertex(self, tri_indices, resp_tri_index):
        """
        Computes the angle of a vertex, given the triangle
        Raise an error of resp_tri_index is not in the triangle
        """
        assert(resp_tri_index in tri_indices)
        P1 = self.coords[resp_tri_index]
        Ps = [P1]
        for idx in tri_indices:
            if resp_tri_index != idx:
                Ps.append(self.coords[idx])
        a_vec = (Ps[0][0] - Ps[1][0], Ps[0][1] - Ps[1][1]) #P1 - P2
        b_vec = (Ps[0][0] - Ps[2][0], Ps[0][1] - Ps[2][1]) #P1 - P3
        dot_prod = a_vec[0]*b_vec[0] + a_vec[1]*b_vec[1]
        length_a = math.sqrt(a_vec[0]**2 + a_vec[1]**2)
        length_b = math.sqrt(b_vec[0]**2 + b_vec[1]**2)
        theta = math.acos(dot_prod / (length_a*length_b))
        if theta > math.pi:
            theta = 2*math.pi - theta
        return theta


    def in_triangle_test(self, x, y, tri_indices):
        """
        Point location for a single triangle
        """
        A = self.area(self.coords[tri_indices[0]][0], self.coords[tri_indices[0]][1], self.coords[tri_indices[1]][0], self.coords[tri_indices[1]][1], self.coords[tri_indices[2]][0], self.coords[tri_indices[2]][1])
        A1 = self.area(x, y, self.coords[tri_indices[1]][0], self.coords[tri_indices[1]][1], self.coords[tri_indices[2]][0], self.coords[tri_indices[2]][1])
        A2 = self.area(self.coords[tri_indices[0]][0], self.coords[tri_indices[0]][1], x, y, self.coords[tri_indices[2]][0], self.coords[tri_indices[2]][1])
        A3 = self.area(self.coords[tri_indices[0]][0], self.coords[tri_indices[0]][1], self.coords[tri_indices[1]][0], self.coords[tri_indices[1]][1], x, y)
        return (A == A1 + A2 + A3)


    def on_triangle_edge_test(self, point, tri_indices):
        """
        Returns the edge if on edge, otherwise returns None
        """
        a, b, c = tri_indices
        for (e1, e2) in [(a, b), (b, c), (a, c)]:
            line_segment_length = np.sum(np.square(self.coords[e1] - self.coords[e2]))
            dist_e1_p = np.sum(np.square(self.coords[e1] - point))
            dist_e2_p = np.sum(np.square(self.coords[e2] - point))
            if line_segment_length == dist_e1_p + dist_e2_p:
                return (e1, e2)
        return None


    def circumcircle_test(self, tri, p):
        """ 
        Circumcircle test, will be used for locally Delaunay property
        """
        center, radius = self.circles[tri]
        return np.sum(np.square(center - p)) <= radius


    def get_neighbours_of_edge(self, common_edge, t1, t2):
        """ 
        Retrieves the neighbours of a triangles of a common edge.
        Throws an error if t1 and t2 do not contain the common edge
        """
        assert((common_edge[0] in t1) and (common_edge[1] in t1) and(common_edge[0] in t2) and(common_edge[1] in t2))
        # t1's neighbours
        N1 = None     
        N1_index = -1 # index of N1 that points to t1
        N2 = None     
        N2_index = -1 # index of N2 that points to t1
        # t2's neighbours
        N3 = None
        N3_index = -1 # index of N3 that points to t2
        N4 = None
        N4_index = -1 # index of N4 that points to t2
        for i, x in enumerate(t1):
            if x == common_edge[1]:
                N1 = self.triangles[t1][i] # tri in front of common_edge_1 for t1
                for j, (ja, jb, jc) in enumerate(self.triangles[N1]):
                    if (ja, jb, jc) == t1:
                        N1_index = j
        for i, x in enumerate(t1):
            if x == common_edge[0]:
                N2 = self.triangles[t1][i] # tri in front of common_edge_0 for t1
                for j, (ja, jb, jc) in enumerate(self.triangles[N2]):
                    if (ja, jb, jc) == t1:
                        N2_index = j
        for i, x in enumerate(t2):
            if x == common_edge[1]:
                N3 = self.triangles[t2][i] # tri in front of common_edge_1 for t2
                for j, (ja, jb, jc) in enumerate(self.triangles[N3]):
                    if (ja, jb, jc) == t2:
                        N3_index = j
        for i, x in enumerate(t1):
            if x == common_edge[0]:
                N4 = self.triangles[t1][i] # tri in front of common_edge_0 for t2
                for j, (ja, jb, jc) in enumerate(self.triangles[N4]):
                    if (ja, jb, jc) == t2:
                        N4_index = j
        return N1, N1_index, N2, N2_index, N3, N3_index, N4, N4_index


    def flip_edge(self, common_edge, diagonal, t1, t2):
        """ 
        Just flips an edge. Doesn't check for legality.
        """
        N1, N1_index, N2, N2_index, N3, N3_index, N4, N4_index = self.get_neighbours_of_edge(common_edge, t1, t2)
        self.triangles.pop(t1)
        self.triangles.pop(t2)
        self.circles.pop(t1)
        self.circles.pop(t2)
        t1_new = (common_edge[0], diagonal[0], diagonal[1])
        t2_new = (common_edge[1], diagonal[1], diagonal[0])
        self.triangles[t1_new] = [t2, N3, N1]
        self.triangles[t2_new] = [t1, N2, N4]
        self.circles[t1_new] = self.circumcenter(t1_new)
        self.circles[t2_new] = self.circumcenter(t2_new)
        if N1 != None and N1_index != -1:
            self.triangles[N1][N1_index] = t1_new
        if N2 != None and N2_index != -1:
            self.triangles[N2][N2_index] = t2_new
        if N3 != None and N3_index != -1:
            self.triangles[N3][N3_index] = t1_new
        if N4 != None and N4_index != -1:
            self.triangles[N4][N4_index] = t2_new
        return


    def register_point_and_split_triangle_into_three(self, tri_indices, point):
        """
        Removes the indicated triangle, registers the 
        new triangles and then updates the neighbours.
        Returns the index of the generated triangles as list
        Throws an error if the point lies on an edge, or the point is outside
        """
        """
        triangle1 = [pi, pj, pr]
        triangle2 = [pj, pk, pr]
        triangle3 = [pk, pi, pr]
        # Replace the original triangle with the new triangles in T
        for triangle in T:
            if pi in triangle and pj in triangle and pk in triangle:
                T.remove(triangle)
        T.append(triangle1)
        T.append(triangle2)
        T.append(triangle3)
        return T
        """
        assert(self.on_triangle_edge_test(point, tri_indices) == None)
        assert(self.in_triangle_test(point[0], point[1], tri_indices))
        return [None, None, None]


    def register_point_and_split_triangles_into_four_from_edge(self, common_edge, diagonal, t1, t2, point):
        """ 
        Removes the indicated triangle, registers the 
        new triangles and then updates the neighbours.
        Returns the index of the generated triangles as list
        N1 and N2 are neighbours of t1 other than t2
        N3 and N4 are neighbours of t1 other than t1
        Throws an error if the point is not on the common edge
        Throws an error if the common edge is actually not the common edge
        """
        """
        triangle1 = [pr, pi, pl]
        triangle2 = [pr, pl, pj]
        triangle3 = [pr, pj, pk]
        triangle4 = [pr, pk, pi]
        # Replace the original triangle with the new triangles in T
        for triangle in T:
            if pi in triangle and pj in triangle and pk in triangle:
                T.remove(triangle) #T.remove([pi, pj, pk])
        for triangle in T:
            if pi in triangle and pj in triangle and pl in triangle:
                T.remove(triangle) #T.remove([pi, pj, pl])
        T.append(triangle1)
        T.append(triangle2)
        T.append(triangle3)
        T.append(triangle4)
        return T
        """
        assert((common_edge[0] in t1) and (common_edge[1] in t1) and(common_edge[0] in t2) and(common_edge[1] in t2))
        assert(self.on_triangle_edge_test(point, t1) == common_edge)
        assert(self.on_triangle_edge_test(point, t2) == common_edge)
        return [None, None, None, None]


    def add_point(self, point):
        """
        Adds the point to the triangulation, and legalizes the edges
        """
        """ 
        pr = P[r]
        pi, pj, pk = find_triangle_containing_point(pr, T)
        if is_on_edge(pr, (pi, pj)) or is_on_edge(pr, (pk, pj)) or is_on_edge(pr, (pi, pk)):
            pl = find_third_vertex(pr, pi, pj, T)
            T = split_triangles_with_edge(pr, pi, pj, pk, pl, T)
            T = LEGALIZEEDGE(pr, (pi, pl), T)
            T = LEGALIZEEDGE(pr, (pl, pj), T)
            T = LEGALIZEEDGE(pr, (pj, pk), T)
            T = LEGALIZEEDGE(pr, (pk, pi), T)
        else:
            T = split_triangle(pr, pi, pj, pk, T)
            T = LEGALIZEEDGE(pr, (pi, pj), T)
            T = LEGALIZEEDGE(pr, (pj, pk), T)
            T = LEGALIZEEDGE(pr, (pk, pi), T)
        """
        return


    def export_triangles(self):
        """
        Export the current list of Delaunay triangles with 
        neighboring information. Filters the super triangles from
        both lists
        """
        triangles__ = []
        neighbours__ = []
        for (a, b, c) in self.triangles:
            if a > 3 and b > 3 and c > 3:
                triangles__.append((a - 4, b - 4, c - 4))
                my_n = []
                for (a_n, b_n, c_n) in self.triangles[(a, b, c)]:
                    if a_n > 3 and b_n > 3 and c_n > 3:
                        my_n.append((a_n - 4, b_n - 4, c_n -4))
                    else:
                        my_n.append(None)
                neighbours__.append(my_n)
        return (triangles__, neighbours__)


    def generate_voronoi(self):
        """
        Generate the Voronoi diagram from the Delaunay triangulation
        """
        # use a dict of edges to avoid duplicates
        voronoi_edges = {}
        voronoi_vertices = {}
        for (a, b, c) in self.triangles:
            if a > 3 and b > 3 and c > 3:
                neighbors = self.triangles[(a, b, c)]
                circumcenter_a = self.circles[(a, b, c)][0]
                triangle_edges = [(self.coords[a], self.coords[b]), (self.coords[a], self.coords[c]), (self.coords[b], self.coords[c])]
                outer_vertices = self.outer_vertices_of_triangle((a, b, c))
                outer_edges = []
                for ov in outer_vertices:
                    for e in ((a, b), (a ,c), (b, c)):
                        if ov != e[0] and ov != e[1]:
                            outer_edges.append(e)
                for i, outer_e in enumerate(outer_edges):
                    u = (self.coords[outer_e[0]]+self.coords[outer_e[1]]) / 2.0
                    opposite_vec = tuple([circumcenter_a[0] - u[0], circumcenter_a[1] - u[1]])  # from u to C
                    vec =  tuple([u[0] - circumcenter_a[0], u[1] - circumcenter_a[1]])          # from C to u
                    len_vec = math.sqrt(vec[0]**2 + vec[1]**2)
                    if self.in_triangle_test(circumcenter_a[0], circumcenter_a[1], (a, b, c)):
                        unit_vec = tuple([vec[0] / len_vec, vec[1] / len_vec])
                        target_p_at_inf = tuple([circumcenter_a[0] + 2000*unit_vec[0], circumcenter_a[1] + 2000*unit_vec[1]])
                        voronoi_edges[(tuple(circumcenter_a), target_p_at_inf)] = 0
                    else:
                        if self.angle_of_vertex((a, b, c), outer_vertices[i]) > math.pi/2.0:
                            opposite_unit_vec = tuple([(opposite_vec[0]) / len_vec, (opposite_vec[1]) / len_vec])
                            target_p_at_inf = tuple([circumcenter_a[0] + 2000*opposite_unit_vec[0], circumcenter_a[1] + 2000*opposite_unit_vec[1]])
                            voronoi_edges[(tuple(circumcenter_a), target_p_at_inf)] = 0
                        else:
                            unit_vec = tuple([(vec[0]) / len_vec, (vec[1]) / len_vec])
                            target_p_at_inf = tuple([circumcenter_a[0] + 2000*unit_vec[0], circumcenter_a[1] + 2000*unit_vec[1]])
                            voronoi_edges[(tuple(circumcenter_a), target_p_at_inf)] = 0
                for (a_n, b_n, c_n) in neighbors:
                    if a_n > 3 and b_n > 3 and c_n > 3:
                        voronoi_vertices[tuple(circumcenter_a)] = 0
                        circumcenter_b = self.circles[(a_n, b_n, c_n)][0]
                        voronoi_edge = (tuple(circumcenter_a), tuple(circumcenter_b))
                        voronoi_edges[voronoi_edge] = 0
        return voronoi_edges, voronoi_vertices