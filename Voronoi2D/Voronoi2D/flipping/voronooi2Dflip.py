from xml.etree.ElementTree import PI
import numpy as np
import math
from math import acos, sqrt


class Voronoi2DFlipping:
    def __init__(self, center=(0, 0), radius=36500):
        center = np.asarray(center)
        # Corners for super-triangles
        self.coords = [center + radius * np.array((-1, -1)),
                       center + radius * np.array((+1, -1)),
                       center + radius * np.array((+1, +1)),
                       center + radius * np.array((-1, +1))]
        self.triangles = {}
        self.circles = {}
        self.currently_iterated = 0
        self.convex_hull_prev

        # The super triangles
        T1 = (0, 1, 3)
        T2 = (2, 3, 1)
        self.triangles[T1] = [T2, None, None]
        self.triangles[T2] = [T1, None, None]

        # Compute circumcircles
        for t in self.triangles:
            self.circles[t] = self.circumcenter(t)

    def circumcenter(self, tri):
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

    def outerVerticesOfTriangle(self, tri_indices):
        outsides = []
        for i, neigh in enumerate(self.triangles[tri_indices]):
            if neigh == None or neigh[0] <= 3 or neigh[1] <= 3 or neigh[2] <= 3:
                outsides.append(tri_indices[i])
        return outsides

    def angleOfTriangleVertex(self, tri_indices, resp_tri_index):
        P1 = self.coords[resp_tri_index]
        Ps = [P1]
        for idx in tri_indices:
            if resp_tri_index != idx:
                Ps.append(self.coords[idx])
        a_vec = (Ps[0][0] - Ps[1][0], Ps[0][1] - Ps[1][1]) #P1 - P2
        b_vec = (Ps[0][0] - Ps[2][0], Ps[0][1] - Ps[2][1]) #P1 - P3
        dot_prod = a_vec[0]*b_vec[0] + a_vec[1]*b_vec[1]
        length_a = sqrt(a_vec[0]**2 + a_vec[1]**2)
        length_b = sqrt(b_vec[0]**2 + b_vec[1]**2)
        theta = acos(dot_prod / (length_a*length_b))
        if theta > math.pi:
            theta = 2*math.pi - theta
        return theta

    def inTriangleTest(self, x, y, tri_indices):
        A = self.area(self.coords[tri_indices[0]][0], self.coords[tri_indices[0]][1], self.coords[tri_indices[1]][0], self.coords[tri_indices[1]][1], self.coords[tri_indices[2]][0], self.coords[tri_indices[2]][1])
        A1 = self.area(x, y, self.coords[tri_indices[1]][0], self.coords[tri_indices[1]][1], self.coords[tri_indices[2]][0], self.coords[tri_indices[2]][1])
        A2 = self.area(self.coords[tri_indices[0]][0], self.coords[tri_indices[0]][1], x, y, self.coords[tri_indices[2]][0], self.coords[tri_indices[2]][1])
        A3 = self.area(self.coords[tri_indices[0]][0], self.coords[tri_indices[0]][1], self.coords[tri_indices[1]][0], self.coords[tri_indices[1]][1], x, y)
        return (A == A1 + A2 + A3)


    def inCircleTest(self, tri, p):
        center, radius = self.circles[tri]
        return np.sum(np.square(center - p)) <= radius


    def registerPoints(self, points):
        """ Gets the array of unsorted random points, and sorts them and stores them inside
        self.coords, which will be sorted w.r.t x axis first, then y axis if x values are the same, 
        in ascencing order. Sorting will be done with numpy.lexsort()
        """
        return


    def addPointLexicographical(self):
        """ Updates the current index and adds the sorted point at the current index to the triangulation
        At the end, updates the convex hull
        """
        return

    def flipAllEdges(self):
        return

    def getNextIllegalEdge(self):
        return

    def exportTriangles(self):
        """Export the current list of Delaunay triangles with neighboring information
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

    def generateVoronoi(self):
        """Generate the Voronoi diagram from the Delaunay triangulation"""
        # use a dict of edges to avoid duplicates
        voronoi_edges = {}
        voronoi_vertices = {}
        for (a, b, c) in self.triangles:
            if a > 3 and b > 3 and c > 3:
                neighbors = self.triangles[(a, b, c)]
                circumcenter_a = self.circles[(a, b, c)][0]
                triangle_edges = [(self.coords[a], self.coords[b]), (self.coords[a], self.coords[c]), (self.coords[b], self.coords[c])]
                outer_vertices = self.outerVerticesOfTriangle((a, b, c))
                outer_edges = []
                for ov in outer_vertices:
                    for e in ((a, b), (a ,c), (b, c)):
                        if ov != e[0] and ov != e[1]:
                            outer_edges.append(e)
                for i, outer_e in enumerate(outer_edges):
                    u = (self.coords[outer_e[0]]+self.coords[outer_e[1]]) / 2.0
                    opposite_vec = tuple([circumcenter_a[0] - u[0], circumcenter_a[1] - u[1]]) # from u to C
                    vec =  tuple([u[0] - circumcenter_a[0], u[1] - circumcenter_a[1]]) # from C to u
                    len_vec = sqrt(vec[0]**2 + vec[1]**2)
                    if self.inTriangleTest(circumcenter_a[0], circumcenter_a[1], (a, b, c)):
                        unit_vec = tuple([vec[0] / len_vec, vec[1] / len_vec])
                        target_p_at_inf = tuple([circumcenter_a[0] + 2000*unit_vec[0], circumcenter_a[1] + 2000*unit_vec[1]])
                        voronoi_edges[(tuple(circumcenter_a), target_p_at_inf)] = 0
                    else:
                        if self.angleOfTriangleVertex((a, b, c), outer_vertices[i]) > math.pi/2.0:
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