import collections
import hashlib
import numpy as np
import numpy.linalg as la
import pyvista as pv
import scipy
import time
import lib


#######################
#
# BASIC
#

def argsort(seq):
    # http://stackoverflow.com/questions/3071415
    return sorted(range(len(seq)), key=seq.__getitem__)


flatten = lambda l: sum(map(flatten,l),[]) if isinstance(l,list) else [l]


def timeit(name, fun):
    t = time.time()
    value = fun()
    print(f"{name}: {time.time()-t:.3f} s")
    return value


def timefun(fun):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = fun(*args, **kwargs)
        end = time.time()
        print(f"function {fun.__name__} took {(end-start)*1000:.2f} ms")
        return result
    return wrapper


def timemany(n=1):
    def wrapper(fun):
        def wrapper(*args, **kwargs):
            start = time.time()
            for _ in range(n):
                result = fun(*args, **kwargs)
            end = time.time()
            print(f"function {fun.__name__} took {(end-start)*1000/n:.2f} ms")
            return result
        return wrapper
    return wrapper


def norm(v):
    return v / la.norm(v)


def digest(buf):
    return hashlib.md5(buf).hexdigest()


#######################
#
# IMPORT/EXPORT
#

def load(fn):
    print("load", fn)
    if fn.endswith(".npy"):
        return np.load(fn)
    elif fn.endswith(".step"):
        # TODO: this returned incomplete model on butterfly.step
        import build123d
        step = build123d.importers.import_step(fn)
        points, faces = step.tessellate(tolerance=0.1)
        points = [tuple(p) for p in points]
        print(f"{len(points)} points, {len(faces)} faces")
        mesh = pv.PolyData.from_regular_faces(points, faces)
        mesh = lib.orient(mesh)
        return mesh
    elif fn.endswith(".3mf"):
        import lib3mf
        wrapper = lib3mf.Wrapper()
        model = wrapper.CreateModel()
        model.QueryReader("3mf").ReadFromFile(fn)
        blocks = pv.MultiBlock()
        items = model.GetBuildItems()
        while items.MoveNext():
            item = items.GetCurrent()
            res = item.GetObjectResource()
            vertices = res.GetVertices()
            triangles = res.GetTriangleIndices()
            points = [v.Coordinates for v in vertices]
            faces = [t.Indices for t in triangles]
            blocks.append(pv.PolyData.from_regular_faces(points, faces))
        merged = pv.merge(blocks)
        return merged
    elif any((fn.endswith(ext) for ext in [".stl", ".obj", ".ply"])):
        return pv.read(fn)
    elif fn.endswith(".pickle"):
        import pickle
        with open(fn, "rb") as f:
            return pickle.load(f)
    else:
        raise Exception("unknown type: " + fn)


def save(x, fn):
    print("save", fn)
    if isinstance(x, np.ndarray) and fn.endswith(".npy"):
        np.save(fn, x)
    elif fn.endswith(".stl") or fn.endswith(".obj") or fn.endswith(".vtm"):
        x.save(fn, recompute_normals=False)
    elif fn.endswith(".step"):
        import build123d
        build123d.export_step(x, fn)
    elif fn.endswith(".3mf"):
        import lib3mf
        wrapper = lib3mf.Wrapper()
        model = wrapper.CreateModel()
        model.SetBuildUUID("00000000-0000-0000-0000-000000000000")
        for i, (name, pd) in enumerate(x.items()):
            mesh = model.AddMeshObject()
            mesh.SetName(name)
            mesh.SetUUID(f"00000000-0000-0000-0001-{i:012d}")
            # ugh
            def position(p):
                position = lib3mf.Position()
                for i in range(3):
                    position.Coordinates[i] = p[i]
                return position
            def triangle(face):
                triangle = lib3mf.Triangle()
                for i in range(3):
                    triangle.Indices[i] = face[i]
                return triangle
            positions = [position(p) for p in pd.points]
            triangles = [triangle(face) for face in pd.regular_faces]
            mesh.SetGeometry(positions, triangles)
            item = model.AddBuildItem(mesh, wrapper.GetIdentityTransform())
            item.SetUUID(f"00000000-0000-0000-0002-{i:012d}")
        writer = model.QueryWriter("3mf")
        writer.WriteToFile(fn)
    elif fn.endswith(".pickle"):
        with open(fn, "wb") as f:
            pickle.dump(x, f)
    else:
        raise Exception(f"don't recognize file type: {fn}")
        

def unpack(a):
    while len(a) > 0:
        n = a[0]
        yield a[1:n+1]
        a = a[n+1:]


def pack(a):
    return np.hstack([[len(aa), *aa] for aa in a])


#######################
#
# ANALYSIS
#

@timefun
def info(x, detail=False):
    if isinstance(x, pv.MultiBlock):
        for i, xx in enumerate(x):
            print("block", i)
            print("    ", info(xx))
    elif isinstance(x, pv.Cell):
        return f"cell {x.type.name} {x.faces} {x.points}"
    elif hasattr(x, "cell"):
        if detail:
            cell_counts = collections.defaultdict(lambda: 0)
            for cell in x.cell:
                cell_counts[cell.type.name] += 1
            base_info = f"{type(x).__name__}, {len(x.points)} points"
            cell_info = [f"{count} {name} cell(s)" for name, count in cell_counts.items()]
            info = ", ".join([base_info, *cell_info])
        else:
            info = f"{x.n_points} points, {x.n_cells} cells"
        if hasattr(x, "is_manifold"):
            info + f", mani {x.is_manifold}"
        return info
    else:
        raise Exception("info: don't understand", type(x))


def area(face, mesh):
    p1, p2, p3 = mesh.points[list(face)]
    return la.norm(np.cross(p2-p1, p3-p1))


def curvature(path, i, delta=1e-6):
    # TODO: don't assume closed - depend on polyline(...,closed=True) to signal that
    p1, p2, p3 = path[(i-1) % len(path)], path[i], path[(i+1) % len(path)]
    a, b, c = p2-p1, p3-p2, p1-p3
    k = 2 * la.norm(np.cross(a, c)) / (la.norm(a) * la.norm(b) * la.norm(c))
    return k


# max curvature
def max_k(path):
    return max(curvature(path.points, i) for i in range(len(path.points)))


def curvature_spline(spline, t, delta=1e-6):
    p1, p2, p3 = spline(t-delta), spline(t), spline(t+delta)
    a, b, c = p2-p1, p3-p2, p1-p3
    k = 2 * la.norm(np.cross(a, c)) / (la.norm(a) * la.norm(b) * la.norm(c))
    return k


def max_k_spline(spline, n=100):
    return max(curvature_spline(spline, t) for t in np.arange(0, 1, 1/n))


#######################
#
# TOPOLOGY
#

@timefun
class Mesh:

    """
    nv                      number of vertices
    nf                      number of faces
    f2e                     face number to oriented edges as vertex number tuples (v,w)
    e2f                     map of oriented edges as vertex number tuples (v,w) to associated face number
    f2n   (nf,3) float      face number to face normals
    v2p   (nv,3) float      vertex number to point coordinates
    f2v   (nf,3) int        face number to vertex numbers
    f2f   (nf,3) int        face number to adjacent face numbers
    v2f   (nv,ragged) int   vertex number to impinging face numbers
    """

    def __init__(self, pd, tolerant=False):

        # TODO: does this add more time if alread triangulated?
        self.pd = pd = pd.triangulate()

        self.nv = len(pd.points)
        self.nf = len(pd.regular_faces)
        print(f"nv {self.nv}, nf {self.nf}, is_manifold {pd.is_manifold}")

        self.v2p = pd.points
        self.f2v = pd.regular_faces
        self.f2n = pd.cell_normals
        assert(len(self.f2v[0]==3))

        # face number to oriented edges as vertex number tuples (v,w)
        self.f2e = [((f2v[0], f2v[1]), (f2v[1], f2v[2]), (f2v[2], f2v[0])) for f2v in self.f2v]

        # map oriented edges as tuples (v,w) to associated face
        # each oriented edge will map to only one face
        # if manifold and properly oriented the opposite edge (w,v)
        # will map to the neighboring face
        self.e2f = {}
        for f, f2v in enumerate(self.f2v):
            for v, w in self.f2e[f]:
                self.e2f[(v,w)] = f

        # if manifold and properly oriented our edges will appear in reverse order
        # of our neighbor's edges, so we use that to find our neighbors
        try:
            # TODO: should this be np.array?
            self.f2f = [
                [self.e2f[(w,v)] for v, w in self.f2e[f] if (w,v) in self.e2f or not tolerant]
                for f, f2v in enumerate(self.f2v)
            ]
        except KeyError:
            raise ValueError("mesh has open edges or is not properly oriented")

        # if manifold and properly oriented each edge will be enumerated twice in opposite
        # orders so v2v will have an entry for both directions of each edge
        self.v2f = [[] for _ in range(self.nv)]
        self.v2v = [[] for _ in range(self.nv)]
        for f, f2v in enumerate(self.f2v):
            for v in f2v:
                self.v2f[v].append(int(f))
            for v, w in self.f2e[f]:
                self.v2v[v].append(w)

        # checks
        v2f_degree = np.array([len(v2f) for v2f in self.v2f])
        v2v_degree = np.array([len(v2v) for v2v in self.v2v])
        assert all(v2f_degree==v2v_degree)
        print(f"vertex degree min {min(v2v_degree)} max {max(v2v_degree)}")


    # exclude isolated vs
    def connected_vs(self, vs=None):
        vs = vs or range(self.nv)
        vs = [v for v in vs if len(self.v2v[v]) > 0]
        return vs

    def local_minima(self, fun):
        funs = np.array([fun(v) for v in self.connected_vs()])
        return [
            v for v in range(self.nv)
            if all(funs[v] < funs[self.v2v[v]])
        ]

    # return maximal vertices: v such that it is higher than all of its neighbors
    def maximal(self, dirn, vs=None):
        v2p = self.v2p * dirn
        return [
            v for v in self.connected_vs(vs)
            if all(v2p[v,2] > v2p[self.v2v[v],2])
        ]
    
    # return upper vertices: v such that all incident faces point upwards
    def upper(self, dirn, vs=None):
        f2n = self.f2n * dirn
        return [
            v for v in self.connected_vs(vs)
            if all(f2n[self.v2f[v],2] > 0)
        ]

    # return upper vertices: v such that all incident faces point downward
    def lower(self, dirn, vs=None):
        f2n = self.f2n * dirn
        return [
            v for v in self.connected_vs(vs)
            if all(f2n[self.v2f[v],2] < 0)
        ]

    # return mixed vertices: v such that some incident faces point upward and some downward
    def mixed(self, dirn, vs=None):
        f2n = self.f2n * dirn
        return [
            v for v in self.connected_vs(vs)
            if any(f2n[self.v2f[v],2] > 0) and \
               any(f2n[self.v2f[v],2] < 0)
        ]

    # return bump vertices: v that are maximal and upward
    @timefun
    def bumps(self, dirn, maximal=None, vs=None):
        maximal = maximal or self.maximal(dirn, vs)
        return self.upper(dirn, maximal)

    # return pocket vertices: v that are maximal and downward
    @timefun
    def pockets(self, dirn, maximal=None, vs=None):
        maximal = maximal or self.maximal(dirn, vs)
        return self.lower(dirn, maximal)

    # return handles: v that are maximal and mixed
    @timefun
    def handles(self, dirn, maximal=None, vs=None):
        maximal = maximal or self.maximal(dirn, vs)
        return self.mixed(dirn, maximal)

    @timefun
    def maximal_types(self, dirn, vs=None):
        maximal = self.maximal(dirn, vs)
        bumps = self.bumps(dirn, maximal=maximal)
        pockets = self.pockets(dirn, maximal=maximal)
        handles = self.handles(dirn, maximal=maximal)
        print(f"{len(maximal)} maximal, {len(bumps)} bumps, {len(pockets)} pockets, {len(handles)} handles")
        return maximal, bumps, pockets, handles

    @timefun
    def connected_regions(self):
        unclaimed = set(range(self.nf))
        regions = []
        while unclaimed:
            region = [unclaimed.pop()]
            i = 0
            while i < len(region):
                f = region[i]
                for g in self.f2f[f]:
                    if g in unclaimed:
                        region.append(g)
                        unclaimed.remove(g)
                i += 1
            regions.append(region)
        return regions

    # not completeley tested
    def boundaries(self, fs=None):
        fs = fs or range(self.nf)
        unclaimed = {}
        for f in fs:
            for v, w in self.f2e[f]:
                if not (w, v) in self.e2f:
                    unclaimed[v] = w
        loops = []
        while unclaimed:
            loop = [unclaimed.pop(next(iter(unclaimed)))]
            i = 0
            while i < len(loop):
                v = loop[i]
                if v in unclaimed:
                    w = unclaimed.pop(v)
                    loop.append(w)
                i += 1
            loops.append(loop)
        return loops


    def area(self, *fs):
        area = 0.0
        for f in fs:
            p1, p2, p3 = self.v2p[self.f2v[f]]
            area += la.norm(np.cross(p2-p1, p3-p1))
        return area


@timefun
class Lines:

    """
    nv                      number of vertices
    nl                      number of lines
    v2p   (nv,3) float      vertex number to point coordinates
    l2v   (nl,2) int        face number to vertex numbers
    v2l   (nv,ragged) int   vertex number to impinging line numbers
    v2v   (nv,ragged) int   vertex number to adjacent vertex numbers
    """

    def __init__(self, pd):

        self.v2p = pd.points
        self.l2v = list(unpack(pd.lines))
        assert(len(self.l2v[0]==2))

        self.nv = len(self.v2p)
        self.nl = len(self.l2v)
        print(f"nv {self.nv}, nl {self.nl}")

        self.v2l = [[] for _ in range(self.nv)]
        for l, l2v in enumerate(self.l2v):
            for v in l2v:
                self.v2l[v].append(l)

        self.v2v = [[] for _ in range(self.nv)]
        for v, v2l in enumerate(self.v2l):
            for l in v2l:
                for w in self.l2v[l]:
                    if w != v:
                        self.v2v[v].append(w)

    def connected_vs(self):
        return [v for v in range(self.nv) if len(self.v2l) > 0]

    def local_minima(self, fun):
        vs = self.connected_vs()
        funs = np.array([fun(v) for v in vs])
        return [
            v for v in vs
            if all(funs[v] < funs[self.v2v[v]])
        ]

    def connected_loops(self):
        unclaimed = set(range(self.nv))
        loops = []
        while unclaimed:
            loop = [unclaimed.pop()]
            i = 0
            while i < len(loop):
                v = loop[i]
                neighbors = self.v2v[v]
                if len(neighbors) > 2:
                    raise Exception(f"vertex {v} has {len(neighbors)} neighbors which is > 2")
                for w in neighbors:
                    if w in unclaimed:
                        loop.append(w)
                        unclaimed.remove(w)
                        break
                    else:
                        pass
                i += 1
            loops.append(loop)
        return loops





#######################
#
# TRANSFORMS
#

def axis_angle_xform(axis, angle, deg=False):
    return pv.core.utilities.transformations.axis_angle_rotation(axis, angle, deg=deg)


# TODO: name?
# compute xform to move vector from to vector to
def xform_from_to(a, b=[0,0,1]):
    a, b = norm(a), norm(b)
    angle = np.arccos(np.dot(a, b) / la.norm(a))
    axis = np.cross(a, b) / la.norm(a)
    if la.norm(axis) < 1e-6:
        xform = np.identity(4)
    else:
        xform = axis_angle_xform(axis, angle, deg=False)
    return xform


def apply_xform(xform, o, inplace=False):
    if isinstance(o, np.ndarray):
        apply_xform = pv.core.utilities.transformations.apply_transformation_to_points
        return apply_xform(xform, o, inplace=inplace)
    else:
        x = o.transform(xform, inplace=inplace)
        if hasattr(o, "viz"):
            x.viz = o.viz
        return x


#######################
#
# BOUNDS
#

def bounds(pd):
    xmin, xmax, ymin, ymax, zmin, zmax = pd.bounds
    return np.array([xmin, ymin, zmin]), np.array([xmax, ymax, zmax])


@timefun
def bounding_ball(points):

    # doing convex hull first speeds up bball computation slightly
    # for 10k points: 157 ms total with convexhull, 167 ms total without
    # for 280k points: 1144 ms total with, 1185ms total without
    points = points[scipy.spatial.ConvexHull(points).vertices]

    # provide a seeded rng to miniball for repeatibility
    import miniball
    rng = np.random.default_rng(0)
    return miniball.get_bounding_ball(points, rng=rng)

def convex_hull(points):
    hull = scipy.spatial.ConvexHull(points)
    return pv.PolyData.from_regular_faces(points, hull.simplices)


# TODO: this
def boundary(pd):
    return pd.extract_feature_edges(boundary_edges=True)


#######################
#
# CSG
#

# so far pymeshlab by far does best booleans

"""
@timefun
def mesh_op(op, *args, **kwargs):

    import pymeshlab

    # load inputs
    mesh_set = pymeshlab.MeshSet()                
    for m in args:
        m = pymeshlab.Mesh(m.points, m.regular_faces)
        mesh_set.add_mesh(m)

    # perform op
    if len(args) == 2:
        result = mesh_set.apply_filter(op, first_mesh = 0, second_mesh = 1, **kwargs)
    else:
        result = mesh_set.apply_filter(op, **kwargs)

    # wrap output
    mesh = mesh_set.current_mesh()
    pd = pv.PolyData.from_regular_faces(mesh.vertex_matrix(), mesh.face_matrix())
    return pd
"""


def unary(op, m1, **kwargs):

    import pymeshlab

    mesh_set = pymeshlab.MeshSet()

    # set up inputs
    m1 = pymeshlab.Mesh(m1.points, m1.regular_faces)
    mesh_set.add_mesh(m1)

    # perform op
    result = mesh_set.apply_filter(op, **kwargs)

    # wrap output
    mesh = mesh_set.current_mesh()
    pd = pv.PolyData.from_regular_faces(mesh.vertex_matrix(), mesh.face_matrix())

    return pd
    

def binary(op, m1, m2):

    import pymeshlab

    mesh_set = pymeshlab.MeshSet()

    # set up inputs
    #m1 = m1.compute_normals(consistent_normals=True)
    #m1 = m1.compute_normals(consistent_normals=True)
    m1 = pymeshlab.Mesh(m1.points, m1.regular_faces)
    m2 = pymeshlab.Mesh(m2.points, m2.regular_faces)
    mesh_set = pymeshlab.MeshSet()                
    mesh_set.add_mesh(m1)
    mesh_set.add_mesh(m2)

    # perform op
    result = mesh_set.apply_filter(op, first_mesh = 0, second_mesh = 1)

    # wrap output
    mesh = mesh_set.current_mesh()
    pd = pv.PolyData.from_regular_faces(mesh.vertex_matrix(), mesh.face_matrix())

    return pd


@timefun
def difference(m1, m2):
    return binary("generate_boolean_difference", m1, m2)


@timefun
def union(m1, m2):
    if m1 is None:
        return m2
    if m2 is None:
        return m1
    return binary("generate_boolean_union", m1, m2)


@timefun
def all_union(*x):
    x = [(all_union(*y) if isinstance(y, (list,tuple)) else y) for y in x]
    if len(x) == 0:
        return None
    result = x[0]
    for y in x[1:]:
        if all_union.debug:
            import viz
            viz.show(
                "about to union", result, y.alpha(0.3),
                viz.normals(y.triangulate()), viz.normals(result.triangulate())
            )
        result = union(result, y)
    return result


# aid for debugging failing unions
all_union.debug = False


# sometimes segfaults, sometimes works ok, except has similar
# artifacts to pymeshlab version
"""
import vedo
def union(a, b):
    a = a.triangulate().compute_normals()
    b = b.triangulate().compute_normals()
    a, b = vedo.Mesh(a), vedo.Mesh(b)
    a.compute_normals()
    b.compute_normals()
    result = a.boolean("plus", b, method=0, tol=1e-3)
    result = pv.PolyData(result.dataset)
    return result
"""


def intersection(m1, m2):
    return binary("generate_boolean_intersection", m1, m2)


#######################
#
# SHAPES
#

def circle(r, n=50):
    points = [[np.sin(t*2*np.pi), np.cos(t*2*np.pi), 0] for t in np.arange(0, 1, 1/n)]
    points += [[0,0,0]]
    points = r * np.array(points)
    faces = [[n, i, (i+1)%n] for i in range(n)]
    return pv.PolyData.from_regular_faces(points, faces)


def rect(w, h):
    points = np.array([[-w/2,-h/2,0], [w/2,-h/2,0], [w/2,h/2,0], [-w/2,h/2,0]])
    return pv.PolyData.from_regular_faces(points, [[0, 1, 2, 3]])


# fibonacci spiral mapped from unit square to sphere using area-preserving mapping
def points_on_sphere(n):
    def angles(i, n):
        phi = np.arccos(1 - 2*i/n)
        theta = 2 * np.pi * i / scipy.constants.golden
        return phi, theta
    def angle_to_point(phi, theta):
        return np.array([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)])
    return np.array([angle_to_point(*angles(i,n)) for i in range(n)])


def sphere(r, n=500):
    points = points_on_sphere(n)
    hull = scipy.spatial.ConvexHull(points)
    sphere = pv.PolyData.from_regular_faces(points * r, hull.simplices)
    sphere = sphere.compute_normals()
    return sphere


@lib.timefun
class Directions:
        
    def __init__(self, n=100000):
        points = lib.points_on_sphere(n)
        self.tree = scipy.spatial.KDTree(points)

    def from_vector(self, vector):
        vector = lib.norm(vector)
        _, direction = self.tree.query(vector)
        return direction
        

def polyline(points, close=False):
    if close:
        lines = list(range(len(points))) + [0]
    else:
        lines = list(range(len(points)))
    lines = pack(zip(lines[:-1], lines[1:]))
    return pv.PolyData(points, lines=lines).c((0.0,0.0,0.0)).lw(3)

def polyface(points, close=False):
    if close:
        face = list(range(len(points))) + [0]
    else:
        face = list(range(len(points)))
    face = [len(face)] + face
    return pv.PolyData(points, faces=face)


def box(p0, p1):
    x0, y0, z0 = p0
    x1, y1, z1 = p1
    face0 = lib.polyface([[x0,y0,z0],  [x1,y0,z0],  [x1,y1,z0],  [x0,y1,z0]])
    return extrude(face0, (0,0,z1-z0))


# TODO: don't assume periodic
def spline(points):
    ys = np.array([*points, points[0]])
    xs = np.linspace(0, 1, len(ys))
    spline = scipy.interpolate.CubicSpline(xs, ys, bc_type="periodic")
    return spline


#######################
#
# CAD
#

# find closest points on mesh to the given points
# these are not necessarily mesh vertices
def find_closest(mesh, points):
    _, closest = mesh.find_closest_cell(points, return_closest_point=True)
    return closest


# embed 2d points in 3d with the the given z value
def embed(points, zvalue):
    zs = np.full((points.shape[0], 1), zvalue)
    return np.hstack([points, zs])


# smooth the n-d path defined
# UnivariateSpline doesn't have a periodic option so we simulate it by overlapping and trimming
def smooth_path(points, spline_param, out_n=None):

    # number of input and output points
    in_n = points.shape[0]
    out_n = out_n or in_n

    # ts are doubled to account for overlap below
    in_ts = range(2*in_n)
    out_ts = np.arange(0, 2*in_n, in_n/out_n)

    stack = []
    for i in range(points.shape[1]):
        xs = points[:,i]
        xs = np.concatenate([xs, xs]) # overlap
        xs = scipy.interpolate.UnivariateSpline(in_ts, xs, s=spline_param)(out_ts) # smooth
        xs = xs[out_n//2 : out_n//2+out_n] # trim
        stack.append(xs)

    result = np.vstack(stack).T
    assert len(result) == out_n
    return result

@timefun
def thicken(m, amount, cellsize_pct=1, **kwargs):
    import pymeshlab
    result = unary(
        "generate_resampled_uniform_mesh",
        m,
        offset=pymeshlab.PureValue(amount),
        cellsize=pymeshlab.PercentageValue(cellsize_pct),
    )
    return result


def extrude(pd, dirn, cap=True):
    res = pd.extrude(dirn, capping=cap).triangulate().compute_normals(auto_orient_normals=True)
    return res
    

# TODO: this gives the wrong answer if o is already fully or partially indiside up_to
def extrude_to(o, dirn, up_to, extra=0, through=False):

    # extrude o a long way in the specified direction
    o_min, o_max = bounds(o)
    u_min, u_max = bounds(up_to)
    d = la.norm(np.maximum(o_max, u_max) - np.minimum(o_min, u_min))
    dirn = norm(np.array(dirn)) * d
    ex = extrude(o, dirn, cap=True)

    # subtract the up_to shell and use the nearest resulting shell in specified direction
    parts = difference(ex, up_to)
    bodies = parts.split_bodies()

    # pick the ones we want
    if through:
        # eliminate anything that goes beyond up_to because these  will be
        # the original extrusion all the way up to dirn which we don't want
        limit = max(np.dot(p, dirn) for p in up_to.points)
        flt = lambda body: max(np.dot(p, dirn) for p in body.points) < limit
        through_bodies = list(filter(flt, bodies))
        result = pv.merge(through_bodies if len(through_bodies) > 0 else bodies).extract_surface()
    else:
        # only the first one
        key = lambda body: min([np.dot(p, dirn) for p in body.points])
        result = sorted(bodies, key=key)[0].extract_surface()

    # move the end faces an extra amount to avoid near coincident faces between result and up_to
    # end faces are the ones whose vertices are within abs(extra) of up_to
    # this is useful if you want to subsequently join result to up_to
    # if extra < 0 the end faces are moved back away from up_to, leaving a gap
    if extra != 0:
        _, closest = up_to.find_closest_cell(result.points, return_closest_point=True)
        ds = la.norm(result.points - closest, axis=1)
        px = 2 * extra * norm(dirn)
        extra = abs(extra)
        points = np.array([(p + px if d < extra else p) for p, d in zip(result.points, ds)])
        result = pv.PolyData.from_regular_faces(points, result.regular_faces)

    return result.compute_normals(auto_orient_normals=True)


# TODO: assumes edges and points match in all sections - can do better?
# ends is "open", "cap", or "close"
def loft(sections, ends="cap"):

    n_sections = len(sections)

    # points will include interior points of sections that are used for end caps,
    # but may go unused for interior sections
    # will remove these unused points before returning mesh
    result_points = np.concatenate([section.points for section in sections])
    n_points = len(sections[0].points)
    assert all(len(section.points) == n_points for section in sections)
    print(f"loft n_sections {n_sections}, n_points {n_points}")

    # edges
    # TODO: this assumes
    # 1) that boundary points are same as first n_points section points
    #    true for lib.rect and lib.circle, but not necessarily in general
    # 2) appropriate match between sections is just point i to point i
    boundaries = [boundary(section) for section in sections]
    edges = [list(unpack(boundary.lines)) for boundary in boundaries]
    assert all(len(e) == len(edges[0]) for e in edges)

    result_faces = []
    n_pairs = n_sections if ends == "close" else n_sections - 1
    for s in range(n_pairs):
        s1, s2 = s, (s+1)%n_sections
        for e1, e2 in zip(edges[s1], edges[s2]):
            a, b = e1[0] + s1*n_points, e1[1] + s1*n_points
            c, d = e2[0] + s2*n_points, e1[1] + s2*n_points
            result_faces.append([a, b, d, c])

    # cap ends
    if ends=="cap":
        for s, offset in [(sections[0],0), (sections[-1],(n_sections-1)*n_points)]:
            for face in unpack(s.faces):
                result_faces.append([f+offset for f in face])

    # form result and remove unused vertices
    result = pv.PolyData(result_points, pack(result_faces)).clean()

    # return a properly oriented mesh
    result = orient(result)

    return result


def frenet_frame(path, i):
    p1, p2, p3 = path[(i-1) % len(path)], path[i], path[(i+1) % len(path)]
    a, b, c = p2-p1, p3-p2, p1-p3
    z_axis = norm(c)
    x_axis = norm(np.cross(a, b))
    y_axis = norm(np.cross(z_axis, x_axis))
    xform = np.array([x_axis, y_axis, z_axis, p2])
    xform = np.concatenate([xform.T, [[0,0,0,1]]])
    return xform


# z axis points along direction of path
# x axis points as nearly as possible towards look_point while remaining perpendicular to z axis
# y axis is perpendicular to x and z axes
def lp_frame(look_point):
    def _(path, i):
        p1, p2, p3 = path[(i-1) % len(path)], path[i], path[(i+1) % len(path)]
        z_axis = norm(p3 - p1)
        look_direction = (look_point - p2)
        x_axis = norm(look_direction - np.dot(look_direction, z_axis) * z_axis)
        y_axis = norm(np.cross(z_axis, x_axis))
        xform = np.array([x_axis, y_axis, z_axis, p2])
        xform = np.concatenate([xform.T, [[0,0,0,1]]])
        return xform
    return _


def sweep(path, profile, frame=frenet_frame, ends="cap"):

    # TODO: this assumes path is simply the points of path in order
    # could be generalized to use edge connectivity
    profiles = ([
        profile.copy().transform(frame(path.points, i))
        for i in range(len(path.points))
    ])

    return loft(profiles, ends=ends)


# make part into a shell of the specified thickness
# assumes the bottom is flat to within leeway
# use the specified decimated version of part to form the cavity
# returns shelled versions of both part and decimated
def shell(part, decimated, shell, leeway=1e-1):

    # form a flat face at the bottom (+leeway) of decimated
    mn, mx = lib.bounds(decimated)
    slice = decimated.slice((0,0,1), [0, 0, mn[2]+leeway])
    loops = lib.Lines(slice).connected_loops()
    if len(loops) != 1:
        raise Exception(f"need 1 loop but got {len(loops)}")
    face = lib.polyface(slice.points[loops[0]])

    # extrude the face downwards and add it on to decimated
    extension = lib.extrude(face, (0,0,-2*shell-leeway))
    extended = lib.union(decimated, extension)

    # now thin (negative thicken) the extended shape and smooth it
    cavity =  lib.thicken(extended, -shell)
    smoothed = cavity.smooth(1000, feature_smoothing=False)
    smoothed = lib.fix(smoothed) # don't know why, but is needed

    # and subract it from part and decimated to get shelled versions
    part_shelled = lib.difference(part, smoothed)
    decimated_shelled = lib.difference(decimated, smoothed)

    # return both
    return part_shelled, decimated_shelled


#######################
#
# REPAIR
#

@timefun
def decimate(pd, target_faces):
    reduction = 1 - (target_faces / len(pd.regular_faces))
    return pd.decimate(reduction)


@timefun
def close_holes(m, **kwargs):
    return unary("meshing_close_holes", m, **kwargs)


@timefun
def fix(pd):
    import pymeshfix
    meshfix = pymeshfix.MeshFix(pd)
    meshfix.repair()
    return meshfix.mesh


# properly orient the faces of a closed mesh
@timefun
def orient(pd):
    return pd.compute_normals(auto_orient_normals=True, point_normals=False)


# input must have regular faces
# sort vertices into a (very likely) predictable order and renumber faces accordingly
# order each face with lowest vertex number first, preserving orientation
# sort faces by tuples of face numbers
# helps avoid spurious differences, e.g. for final save
# also makes textual diff of .obj files easier
@timefun
def canonicalize(pd):

    if pd is None:
        return None

    points = pd.points
    faces = pd.regular_faces

    # compute permutation that sorts points in a canonical order and its inverse
    # has a small chance of varying if there are ties; stable sort may reduce that chance
    new_to_old = np.array(argsort([tuple(p) for p in points]))
    old_to_new = np.empty_like(new_to_old)
    old_to_new[new_to_old] = np.arange(len(pd.points), dtype=int)
    
    # sort the points and renumber the faces accordingly
    new_points = points[new_to_old]
    new_faces = old_to_new[faces]

    # roll the face vertices so that the smallest vertex number is first, preserving orientation
    new_faces = [np.roll(face, -np.argmin(face)) for face in new_faces]

    # sort faces
    new_faces = np.array(sorted(new_faces, key=lambda face: tuple(face)))

    # construct new mesh
    result = pv.PolyData.from_regular_faces(new_points, new_faces)
    #viz.show_diff([pd, viz.normals(pd)], [result, viz.normals(result)], ("pd", "result"))
    return result


#######################
#
# MOLDS
#

def vent(loc, diameter, height, to):
    r = diameter / 2
    circle = lib.circle(r).translate(loc)
    upward = extrude(circle, (0,0,height))
    if to is None:
        downward = None
    elif isinstance(to, (np.ndarray,list,tuple)):
        downward = extrude(circle, to)
    else:
        # extrude a small extra distance to avoid coincident faces
        # this avoids artifacts from union
        downward = extrude_to(circle, (0,0,-1), to, extra=1e-4)
    return union(upward, downward)

def elbow(ps, d):
    r = d / 2
    result = None
    for i, (p1, p2) in enumerate(zip(ps[:-1], ps[1:])):
        if i != 0:
            result = union(result, sphere(r).translate(p1))
        xform = xform_from_to([0,0,1], p2-p1)
        circle = apply_xform(xform, lib.circle(r)).translate(p1)
        tube = extrude(circle, p2-p1)
        result = union(result, tube)
    return result

def mouth(pos, d, height, angle_deg=22.5):

    # always a list
    if not isinstance(pos, (tuple, list)):
        pos = [pos]

    # feed elbow, if any
    result = elbow(pos, d)

    # mouth itself
    r = d / 2
    pos1 = pos[-1]
    pos2 = pos1 + np.array([0, 0, height])
    r2 = r + height * np.sin(2 * np.pi * angle_deg / 360)
    circle1 = circle(r).translate(pos1)
    circle2 = circle(r2).translate(pos2)
    mouth = loft([circle1, circle2])
    
    # combine with elbow (if any)
    # TODO: subsequent union seems unhappy unless this is triangulated
    # maybe union needs to triangulate its inputs? performance issue?
    # TODO: this is resulting in two shells, not just one,
    # i.e. not actually unioning - not really a problem I guess?
    result = union(result, mouth.triangulate())
    return result


