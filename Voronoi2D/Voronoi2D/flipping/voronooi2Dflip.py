import numpy as np
import math
from math import acos, sqrt

def isPointLeftOfLine(p, q, r):
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]) > 0


def doLinesIntersect(p1, p2, p3, p4):
    return (isPointLeftOfLine(p1, p3, p4) != isPointLeftOfLine(p2, p3, p4) and
        isPointLeftOfLine(p3, p1, p2) != isPointLeftOfLine(p4, p1, p2)
    )

def doesEdgeIntersectTriangle(edge, triangle, points):
    p1, p2 = edge
    t1, t2, t3 = [points[i] for i in triangle]
    r1 = doLinesIntersect(p1, p2, t1, t2)
    r2 = doLinesIntersect(p1, p2, t2, t3)
    r3 = doLinesIntersect(p1, p2, t3, t1)
    return (r1 or r2 or r3)

#class Triangle:
#    def __init__(self, vertices):
#        self.vertices = vertices
#        self.neighbors = [None, None, None]

#    def getVertexIndex(self, vertex):
#        for i in range(3):
#            if np.array_equal(self.vertices[i], vertex):
#                return i

#    def getAdjacentTriangle(self, vertex):
#        index = self.getVertexIndex(vertex)
#        return self.neighbors[index]

#    def setAdjacentTriangle(self, vertex, triangle):
#        index = self.getVertexIndex(vertex)
#        self.neighbors[index] = triangle

#    def getThirdVertex(self, edge):
#        for vertex in self.vertices:
#            if not np.array_equal(vertex, edge[0]) and not np.array_equal(vertex, edge[1]):
#                return vertex


#class Triangulation:
#    def __init__(self):
#        self.triangles = []
#        self.edges = []

#    def addTriangle(self, triangle):
#        self.triangles.append(triangle)

#        for i in range(3):
#            edge = (triangle.vertices[i], triangle.vertices[(i + 1) % 3])
#            self.edges.append(edge)

#    def getAdjacentTriangles(self, edge):
#        triangle1 = None
#        triangle2 = None

#        for triangle in self.triangles:
#            if np.array_equal(edge[0], triangle.vertices[0]) and np.array_equal(edge[1], triangle.vertices[1]):
#                triangle1 = triangle
#            elif np.array_equal(edge[0], triangle.vertices[1]) and np.array_equal(edge[1], triangle.vertices[0]):
#                triangle1 = triangle
#            elif np.array_equal(edge[0], triangle.vertices[1]) and np.array_equal(edge[1], triangle.vertices[2]):
#                triangle2 = triangle
#            elif np.array_equal(edge[0], triangle.vertices[2]) and np.array_equal(edge[1], triangle.vertices[1]):
#                triangle2 = triangle
#            elif np.array_equal(edge[0], triangle.vertices[2]) and np.array_equal(edge[1], triangle.vertices[0]):
#                triangle2 = triangle
#            elif np.array_equal(edge[0], triangle.vertices[0]) and np.array_equal(edge[1], triangle.vertices[2]):
#                triangle2 = triangle

#        return triangle1, triangle2

#    def flipEdge(self, edge, new_vertex):
#        triangle1, triangle2 = self.getAdjacentTriangles(edge)

#        if triangle1 is None or triangle2 is None:
#            return

#        vertex1 = edge[0]
#        vertex2 = edge[1]
#        vertex3 = triangle1.getThirdVertex(edge)
#        vertex4 = triangle2.getThirdVertex(edge)

#        triangle1.vertices = [vertex3, vertex4, new_vertex]
#        triangle2.vertices = [vertex4, vertex3, new_vertex]

#        triangle1_neighbor = triangle1.getAdjacentTriangle(vertex1)
#        triangle2_neighbor = triangle2.getAdjacentTriangle(vertex2)

#        triangle1.setAdjacentTriangle(vertex1, triangle2_neighbor)
#        triangle1.setAdjacentTriangle(vertex3, triangle2)
#        triangle2.setAdjacentTriangle(vertex2, triangle1_neighbor)
#        triangle2.setAdjacentTriangle(vertex4, triangle1)

#    def registerPoints(self, points):
#        self.triangles = []
#        self.edges = []

#        p1, p2, p3 = points[:3]

#        triangle = Triangle([p1, p2, p3])
#        self.addTriangle(triangle)

#        for i in range(3, len(points)):
#            new_point = points[i]
#            triangle = self.addPointToLexTriangulation(new_point)
#            self.flipAllEdges(new_point, triangle)

#    def addPointToLexTriangulation(new_point):
#        new_triangles = []
#        for triangle in triangulation:
#            edge_pairs = [
#                (triangle[0], triangle[1]),
#                (triangle[1], triangle[2]),
#                (triangle[2], triangle[0])
#            ]
#            for edge_pair in edge_pairs:
#                edge = (edge_pair[0], edge_pair[1])
#                if not doesEdgeIntersectTriangle(edge, triangle, points):
#                    # Register the edge as part of the triangulation
#                    new_triangle = [edge_pair[0], edge_pair[1], point_index]
#                    new_triangles.append(new_triangle)
#        return new_triangles

#    def lexicographicTriangulate(self, points):
#        sorted_points = sorted(points, key=lambda p: (p[0], p[1]))
#        p1, p2, p3 = sorted_points[:3]

#        triangle = Triangle([p1, p2, p3])
#        self.addTriangle(triangle)

#        for i in range(3, len(sorted_points)):
#            new_point = sorted_points[i]
#            triangle = self.addPointToLexTriangulation(new_point)
#            self.flipAllEdges(new_point, triangle)

#    def flipAllEdges(self, new_vertex, triangle):
#        for edge in self.edges:
#            if self.isEdgeLegal(edge, new_vertex):
#                self.flipEdge(edge, new_vertex)

#    def isPointInCircumcircle(self, triangle, point):
#        vertices = triangle.vertices

#        ax, ay = vertices[0]
#        bx, by = vertices[1]
#        cx, cy = vertices[2]
#        dx, dy = point

#        ax -= dx
#        ay -= dy
#        bx -= dx
#        by -= dy
#        cx -= dx
#        cy -= dy

#        ab = ax * ax + ay * ay
#        bc = bx * bx + by * by
#        ca = cx * cx + cy * cy

#        det = ax * (by * ca - bc * cy) - ay * (bx * ca - bc * cx) + ab * (bx * cy - by * cx)

#        return det > 0

#    def isEdgeLegal(self, edge, new_vertex):
#        triangle1, triangle2 = self.getAdjacentTriangles(edge)

#        if triangle1 is None or triangle2 is None:
#            return True

#        vertex1 = edge[0]
#        vertex2 = edge[1]
#        vertex3 = triangle1.getThirdVertex(edge)
#        vertex4 = triangle2.getThirdVertex(edge)

#        old_angle_sum = self.calculateAngle(vertex1, new_vertex, vertex2) + self.calculateAngle(vertex2, new_vertex, vertex3)
#        new_angle_sum = self.calculateAngle(vertex1, vertex3, vertex2) + self.calculateAngle(vertex2, vertex3, new_vertex)

#        return new_angle_sum < old_angle_sum

#    def calculateAngle(self, vertex1, vertex2, vertex3):
#        a = vertex1 - vertex2
#        b = vertex3 - vertex2
#        dot_product = np.dot(a, b)
#        magnitude_product = np.linalg.norm(a) * np.linalg.norm(b)
#        angle_radians = np.arccos(dot_product / magnitude_product)
#        return np.degrees(angle_radians)


class Voronoi2DFlipping:
    def __init__(self, points):
        self.coords = []                            # basically a global vertex set which will be accessed trough indices inside the other fields
        self.triangles = {}                         # basically, face set, which contains set of vertex indices & neighbouring faces. all faces are a triangle
        self.circles = {}                           # set of circumcircles, given a triangle. Circumcenters & radius'es are actual values
        self.currently_iterated_point_index = 0     # helper variable to keep track of the current triangulation iteration
        self.current_convex_hull = {}               # helper variable to keep track of the convex hull to speed up the algorithm
        self.__registerPoints(points)
        self.__lexicographicalTriangulate()


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


    def __registerPoints(self, points):
        # Sort the points lexicographically
        self.coords = sorted(points, key=lambda p: (p[0], p[1]))


    def addPointToLexTriangulation(self, point):
       # register the vertex into coords

       # find the added triangles & register circles

       # update all neighbours

       # update the convex hull

       # go to next index
        self.currently_iterated_point_index += 1


    def __lexicographicalTriangulate(self):
        # Sort the points lexicographically to make sure
        self.coords = sorted(self.coords, key=lambda p: (p[0], p[1]))
        # Add all points one by one
        for i in range(len( self.coords)):
            self.addPointToLexTriangulation(self.coords[i])


    def flipAllEdges(self):
        """ Flips an illegal edge, until there is no illegal edge left"""
        # Find the triangles that share an edge with the newly added point
        result = self.getNextIllegalEdge()
        illegal_edge, diagonal, t1, t2 = result
        while (result != None):
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
                if x == illegal_edge[1]:
                    N1 = self.triangles[t1][i] # tri in front of ill_2 for t1
                    for j, (ja, jb, jc) in enumerate(self.triangles[N1]):
                        if (ja, jb, jc) == t1:
                            N1_index = j
            for i, x in enumerate(t1):
                if x == illegal_edge[0]:
                    N2 = self.triangles[t1][i] # tri in front of ill_1 for t1
                    for j, (ja, jb, jc) in enumerate(self.triangles[N2]):
                        if (ja, jb, jc) == t1:
                            N2_index = j
            for i, x in enumerate(t2):
                if x == illegal_edge[1]:
                    N3 = self.triangles[t2][i] # tri in front of ill_2 for t2
                    for j, (ja, jb, jc) in enumerate(self.triangles[N3]):
                        if (ja, jb, jc) == t2:
                            N3_index = j
            for i, x in enumerate(t1):
                if x == illegal_edge[0]:
                    N4 = self.triangles[t1][i] # tri in front of ill_1 for t2
                    for j, (ja, jb, jc) in enumerate(self.triangles[N4]):
                        if (ja, jb, jc) == t2:
                            N4_index = j
            self.triangles.pop(t1)
            self.triangles.pop(t2)
            self.circles.pop(t1)
            self.circles.pop(t2)
            t1_new = (illegal_edge[0], diagonal[0], diagonal[1])
            t2_new = (illegal_edge[1], diagonal[1], diagonal[0])
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
            result = self.getNextIllegalEdge()
            illegal_edge, diagonal, t1, t2 = result


    def getNextIllegalEdge(self):
        for (a, b, c) in self.triangles:
            neighbours = self.triangles[(a, b, c)]
            for (a_n, b_n, c_n) in neighbours:
                common_v1 = None
                common_v2 = None
                diag_1 = None
                diag_2 = None
                # find the common edge
                for u in (a, b, c):
                    for v in (a_n, b_n, c_n):
                        if u == v and common_v1 == None:
                            common_v1 = u
                        elif u == v and common_v1 != None and common_v2 == None:
                            common_v2 = u
                # find the diagonal edge
                for x in (a, b, c, a_n, b_n, c_n):
                    if x != common_v1 and x != common_v2 and diag_1 == None:
                        diag_1 = x
                    elif x != common_v1 and x != common_v2 and diag_1 != None and diag_2 == None:
                        diag_2 = x
                # if there is no common edge, there is an error in neighbour data structure
                assert(common_v1 != None and common_v2 != None) 
                circumcenter_current = self.circles[(a, b, c)][0]
                radius_current = self.circles[(a, b, c)][1]
                circumcenter_neighbour = self.circles[(a_n, b_n, c_n)][0]
                radius_neighbour = self.circles[(a_n, b_n, c_n)][1]
                for u in (a, b, c):
                    if self.inCircleTest((a_n, b_n, c_n), self.coords[u]):
                        return (common_v1, common_v2), (a, b, c), (a_n, b_n, c_n)
                for v in (a_n, b_n, c_n):
                    if self.inCircleTest((a, b, c), self.coords[v]):
                        return (common_v1, common_v2), (diag_1, diag_2), (a, b, c), (a_n, b_n, c_n)
        return None


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