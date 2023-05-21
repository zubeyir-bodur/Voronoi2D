import numpy as np
from math import sqrt

class Delaunay2D:
    def __init__(self, center=(0, 0), radius=36500):
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

    def inCircleTest(self, tri, p):
        center, radius = self.circles[tri]
        return np.sum(np.square(center - p)) <= radius

    # Bowyer-Watson increment
    def addPoint(self, p):
        p = np.asarray(p)
        idx = len(self.coords)
        self.coords.append(p)

        # Bad Triangles = triangles whose circumcircle contains p
        bad_triangles = []
        for T in self.triangles:
            if self.inCircleTest(T, p):
                bad_triangles.append(T)

        # Compute the boundary polygon of bad triangles
        boundary = []
        T = bad_triangles[0]
        edge = 0
        # get the opposite triangle of this edge
        while True:
            # The opposite triangle
            tri_op = self.triangles[T][edge]
            if tri_op not in bad_triangles:
                # Next edge in current triangle
                boundary.append((T[(edge + 1) % 3], T[(edge - 1) % 3], tri_op))
                edge = (edge + 1) % 3
                # Check for loop
                if boundary[0][0] == boundary[-1][1]:
                    break
            else:
                # Next edge in the opposite triangle
                edge = (self.triangles[tri_op].index(T) + 1) % 3
                T = tri_op

        # Delete illegal triangles
        for T in bad_triangles:
            del self.circles[T]
            del self.triangles[T]

        # Triangulate the polygon
        new_triangles = []
        for (e0, e1, tri_op) in boundary:
            T = (idx, e0, e1)
            self.circles[T] = self.circumcenter(T)
            self.triangles[T] = [tri_op, None, None]

            # Set T as neighbour of opposite edge if a neighbor is on the same edge
            if tri_op:
                for i, neigh in enumerate(self.triangles[tri_op]):
                    if neigh:
                        if e1 in neigh and e0 in neigh:
                            self.triangles[tri_op][i] = T

            new_triangles.append(T)
        N = len(new_triangles)
        for i, T in enumerate(new_triangles):
            self.triangles[T][1] = new_triangles[(i + 1) % N]  # Next neighbour
            self.triangles[T][2] = new_triangles[(i - 1) % N]  # Previous neighbour

    def exportTriangles(self):
        """Export the current list of Delaunay triangles
        """
        # Filter out triangles with any vertex in the extended BBox
        return [(a - 4, b - 4, c - 4)
                for (a, b, c) in self.triangles if a > 3 and b > 3 and c > 3]