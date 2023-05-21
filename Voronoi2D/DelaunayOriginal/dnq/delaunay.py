import numpy as np
import pygame

# global edge storage
edges = []


# DELAUNAY BACKEND INTERFACE FOR MAIN PROCESS

# Use for creating the whole mesh
def delaunay(S):
    if len(S) < 2:
        print("Invalid set of points")
        return

    global edges
    edges = []
    S = np.asarray(S, dtype=np.float64)

    # Sort points by x coordinate, y is a tiebreaker.
    S.view(dtype=[('f0', S.dtype), ('f1', S.dtype)]).sort(order=['f0', 'f1'], axis=0)

    # Remove duplicates.
    dupes = [i for i in range(1, len(S)) if S[i - 1][0] == S[i][0] and S[i - 1][1] == S[i][1]]
    if dupes:
        S = np.delete(S, dupes, 0)

    triangulate(S)
    edges = [e for e in edges if e.data is None]  # clean the garbage
    return edges


# Use for creating animation
def step(S, screen, surface, pic):
    if len(S) < 2:
        print("Invalid set of points")
        return

    global edges
    edges = []
    S = np.asarray(S, dtype=np.float64)

    # Sort points by x coordinate, y is a tiebreaker.
    S.view(dtype=[('f0', S.dtype), ('f1', S.dtype)]).sort(order=['f0', 'f1'], axis=0)

    # Remove duplicates.
    dupes = [i for i in range(1, len(S)) if S[i - 1][0] == S[i][0] and S[i - 1][1] == S[i][1]]
    if dupes:
        S = np.delete(S, dupes, 0)

    triangulateGUI(S, screen, surface, pic)
    edges = [e for e in edges if e.data is None]  # clean the garbage
    return edges


# Use for generating seeds
def genRandom(w, h, n):
    points_x = np.random.randint(0, w, n, dtype=np.int64)
    points_y = np.random.randint(0, h, n, dtype=np.int64)
    return np.asarray(list(zip(points_x, points_y)), dtype=np.float64)


# THE IMPLEMENTATION -- DO NOT IMPORT THIS PART

# Quad edge data structure.
# Direction from org -> dest
# Symmetric direction is from dest -> org
# Next is counter-clockwise traversal
# Prev is clockwise traversal
# Data is set to TRUE if deleted
class Edge:
    def __init__(self, org, dest):
        self.org = org
        self.dest = dest
        self.onext = None
        self.oprev = None
        self.sym = None  # symmetrical counterpart of this edge
        self.data = None  # can store anyting (e.g. tag), for external use

    def __str__(self):
        s = str(self.org) + ', ' + str(self.dest)
        if self.data is None:
            return s
        else:
            return s + ' ' + str(self.data)


# Whole mesh triangulation
def triangulate(S):
    if len(S) == 2:
        a = makeEdge(S[0], S[1])
        return a, a.sym

    elif len(S) == 3:
        # Create edges a(p1 -> p2) and b(p2 -> p3)
        p1, p2, p3 = S[0], S[1], S[2]
        a = makeEdge(p1, p2)
        b = makeEdge(p2, p3)
        splice(a.sym, b)

        # Create edge c(p3 -> p1)
        if rightOf(p3, a):
            connect(b, a)
            return a, b.sym
        elif leftOf(p3, a):
            c = connect(b, a)
            return c.sym, c
        else:  # the three points are collinear
            return a, b.sym

    else:
        # Recursive subdivision
        m = (len(S) + 1) // 2
        L, R = S[:m], S[m:]
        ldo, ldi = triangulate(L)
        rdi, rdo = triangulate(R)

        # Find the base LR-edge
        while True:
            if rightOf(rdi.org, ldi):
                ldi = ldi.sym.onext
            elif leftOf(ldi.org, rdi):
                rdi = rdi.sym.oprev
            else:
                break

        # Create a first base LR-edge base from rdi.org to ldi.org.
        base = connect(ldi.sym, rdi)

        # Adjust ldo and rdo
        if ldi.org[0] == ldo.org[0] and ldi.org[1] == ldo.org[1]:
            ldo = base
        if rdi.org[0] == rdo.org[0] and rdi.org[1] == rdo.org[1]:
            rdo = base.sym

        # Merge.
        while True:
            # Locate the first candidates
            rcand, lcand = base.sym.onext, base.oprev
            # If both candidates are invalid, then base LR is the lower common tangent.
            v_rcand, v_lcand = rightOf(rcand.dest, base), rightOf(lcand.dest, base)
            if not (v_rcand or v_lcand):
                break
            # Delete R edges out of base.dest that fail condition 2
            if v_rcand:
                while rightOf(rcand.onext.dest, base) and \
                        inCircleTest(base.dest, base.org, rcand.dest, rcand.onext.dest) == 1:
                    t = rcand.onext
                    deleteEdge(rcand)
                    rcand = t
            # Analogous for R
            if v_lcand:
                while rightOf(lcand.oprev.dest, base) and \
                        inCircleTest(base.dest, base.org, lcand.dest, lcand.oprev.dest) == 1:
                    t = lcand.oprev
                    deleteEdge(lcand)
                    lcand = t
            # Choose candidate to use for merge. Decide based on condition 2
            # This assumes unique triangulation exists
            if not v_rcand or \
                    (v_lcand and inCircleTest(rcand.dest, rcand.org, lcand.org, lcand.dest) == 1):
                # Add base LR-edge from rcand.dest to base.dest.
                base = connect(lcand, base.sym)
            else:
                # Add base LR-edge from base.org to lcand.dest
                base = connect(base.sym, rcand.sym)

        return ldo, rdo


# Animation triangulation
clock = pygame.time.Clock()
def triangulateGUI(S, screen, surface, pic):
    if len(S) == 2:
        a = makeEdge(S[0], S[1])
        # GUI
        pic.fill((160, 160, 160))
        for e in [e for e in edges if e.data is None]:
            pygame.draw.line(surface=pic, color="Green", start_pos=e.org, end_pos=e.dest, width=1)
        surface.blit(pic, (1, 1))
        screen.blit(pygame.transform.scale(surface, screen.get_rect().size), (0, 0))
        pygame.display.flip()
        clock.tick(90)
        pygame.time.wait(1000)
        return a, a.sym

    elif len(S) == 3:
        # Create edges a(p1 -> p2) and b(p2 -> p3)
        p1, p2, p3 = S[0], S[1], S[2]
        a = makeEdge(p1, p2)
        b = makeEdge(p2, p3)
        splice(a.sym, b)

        # Create edge c(p3 -> p1)
        if rightOf(p3, a):
            connect(b, a)
            # GUI
            pic.fill((160, 160, 160))
            for e in [e for e in edges if e.data is None]:
                pygame.draw.line(surface=pic, color="Green", start_pos=e.org, end_pos=e.dest, width=1)
            surface.blit(pic, (1, 1))
            screen.blit(pygame.transform.scale(surface, screen.get_rect().size), (0, 0))
            pygame.display.flip()
            clock.tick(90)
            pygame.time.wait(1000)

            return a, b.sym
        elif leftOf(p3, a):
            c = connect(b, a)
            # GUI
            pic.fill((160, 160, 160))
            for e in [e for e in edges if e.data is None]:
                pygame.draw.line(surface=pic, color="Green", start_pos=e.org, end_pos=e.dest, width=1)
            surface.blit(pic, (1, 1))
            screen.blit(pygame.transform.scale(surface, screen.get_rect().size), (0, 0))
            pygame.display.flip()
            clock.tick(90)
            pygame.time.wait(1000)

            return c.sym, c
        else:  # the three points are collinear
            # GUI
            pic.fill((160, 160, 160))
            for e in [e for e in edges if e.data is None]:
                pygame.draw.line(surface=pic, color="Green", start_pos=e.org, end_pos=e.dest, width=1)
            surface.blit(pic, (1, 1))
            screen.blit(pygame.transform.scale(surface, screen.get_rect().size), (0, 0))
            pygame.display.flip()
            clock.tick(90)
            pygame.time.wait(1000)
            return a, b.sym

    else:
        # Recursive subdivision
        m = (len(S) + 1) // 2
        L, R = S[:m], S[m:]
        ldo, ldi = triangulateGUI(L, screen, surface, pic)
        rdi, rdo = triangulateGUI(R, screen, surface, pic)

        # Find the base LR-edge
        while True:
            if rightOf(rdi.org, ldi):
                ldi = ldi.sym.onext
            elif leftOf(ldi.org, rdi):
                rdi = rdi.sym.oprev
            else:
                break

        # Create a first base LR-edge base from rdi.org to ldi.org.
        base = connect(ldi.sym, rdi)

        # Adjust ldo and rdo
        if ldi.org[0] == ldo.org[0] and ldi.org[1] == ldo.org[1]:
            ldo = base
        if rdi.org[0] == rdo.org[0] and rdi.org[1] == rdo.org[1]:
            rdo = base.sym

        # Merge.
        while True:
            # Locate the first candidates
            rcand, lcand = base.sym.onext, base.oprev
            # If both candidates are invalid, then base LR is the lower common tangent.
            v_rcand, v_lcand = rightOf(rcand.dest, base), rightOf(lcand.dest, base)
            if not (v_rcand or v_lcand):
                break
            # Delete R edges out of base.dest that fail condition 2
            if v_rcand:
                while rightOf(rcand.onext.dest, base) and \
                        inCircleTest(base.dest, base.org, rcand.dest, rcand.onext.dest) == 1:
                    t = rcand.onext
                    deleteEdge(rcand)
                    rcand = t
            # Analogous for R
            if v_lcand:
                while rightOf(lcand.oprev.dest, base) and \
                        inCircleTest(base.dest, base.org, lcand.dest, lcand.oprev.dest) == 1:
                    t = lcand.oprev
                    deleteEdge(lcand)
                    lcand = t
            # Choose candidate to use for merge. Decide based on condition 2
            # This assumes unique triangulation exists
            if not v_rcand or \
                    (v_lcand and inCircleTest(rcand.dest, rcand.org, lcand.org, lcand.dest) == 1):
                # Add base LR-edge from rcand.dest to base.dest
                base = connect(lcand, base.sym)
            else:
                # Add base LR-edge from base.org to lcand.dest
                base = connect(base.sym, rcand.sym)
            # GUI
            pic.fill((160, 160, 160))
            for e in [e for e in edges if e.data is None]:
                pygame.draw.line(surface=pic, color="Green", start_pos=e.org, end_pos=e.dest, width=1)
            surface.blit(pic, (1, 1))
            screen.blit(pygame.transform.scale(surface, screen.get_rect().size), (0, 0))
            pygame.display.flip()
            clock.tick(90)
            pygame.time.wait(1000)
        return ldo, rdo


# PREDICATE FUNCTIONS

def inCircleTest(a, b, c, d):
    a1, a2 = a[0] - d[0], a[1] - d[1]
    b1, b2 = b[0] - d[0], b[1] - d[1]
    c1, c2 = c[0] - d[0], c[1] - d[1]
    a3, b3, c3 = a1 ** 2 + a2 ** 2, b1 ** 2 + b2 ** 2, c1 ** 2 + c2 ** 2
    det = a1 * b2 * c3 + a2 * b3 * c1 + a3 * b1 * c2 - (a3 * b2 * c1 + a1 * b3 * c2 + a2 * b1 * c3)
    return det < 0

def rightOf(p, e):
    a, b = e.org, e.dest
    det = (a[0] - p[0]) * (b[1] - p[1]) - (a[1] - p[1]) * (b[0] - p[0])
    return det > 0

def leftOf(p, e):
    a, b = e.org, e.dest
    det = (a[0] - p[0]) * (b[1] - p[1]) - (a[1] - p[1]) * (b[0] - p[0])
    return det < 0

def makeEdge(org, dest):
    global edges
    e = Edge(org, dest)
    es = Edge(dest, org)
    e.sym, es.sym = es, e  # make edges mutually symmetrical
    e.onext, e.oprev = e, e
    es.onext, es.oprev = es, es
    edges.append(e)
    return e

# Join two edge chains together
def splice(a, b):
    if a == b:
        print("Splicing edge with itself, ignored: {}.".format(a))
        return

    a.onext.oprev, b.onext.oprev = b, a
    a.onext, b.onext = b.onext, a.onext


def connect(a, b):
    e = makeEdge(a.dest, b.org)
    splice(e, a.sym.oprev)
    splice(e.sym, b)
    return e


def deleteEdge(e):
    splice(e, e.oprev)
    splice(e.sym, e.sym.oprev)
    e.data, e.sym.data = True, True