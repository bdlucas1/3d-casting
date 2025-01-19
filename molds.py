import alphashape
import cache
import collections
import dataclasses
import itertools
import lib
import viz
import pymeshlab
import pyvista as pv
import numpy as np
import numpy.linalg as la
import ruamel.yaml
import scipy
import sys
import time

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(
        prog = "molds3d",
        description = "Design printable molds",
    )
    
    parser.add_argument("args", nargs="+")
    parser.add_argument("--test", "-t", choices = ["all", "changed", "none"], default = "changed")
    parser.add_argument("--debug", "-d", default = "")
    parser.add_argument("--force", "-f", default = "")
    parser.add_argument("--auto", "-a", action = "store_true")
    args = parser.parse_args()

    # yaml file
    yaml_fn = args.args.pop(0)
    with open(yaml_fn) as f:
        yaml = ruamel.yaml.YAML()
        yaml.indent(mapping=4, sequence=4, offset=2)
        yaml_info = yaml.load(f)

    # having successfuly read it, back it up
    with open(yaml_fn, "r") as f:
        s = f.read()
    with open(yaml_fn + "~", "w") as f:
        f.write(s)

    # dictionary whose items can be accessed as attributes
    class AttrDict(dict):
        def __init__(self, d):
            super().__init__(d)
            for n, v in d.items():
                setattr(self, n, v)

    # turn a ruamel tree back into a plain tree
    # this is better for pickling
    def plain(x):
        if isinstance(x,dict):
            return AttrDict({n: plain(v) for n, v in x.items()})
        elif isinstance(x,list):
            return [plain(y) for y in x]
        else:
            for t in [bool,int,float,str]:
                if isinstance(x,t):
                    return t(x)
            raise Exception("oops", type(x))

    # TODO: insert new things in an order defined by a standard set of keys?
    def save(path, value, comment=None):

        # leaf nested lists (only) are inline (have flow style)
        # TODO: can get same result in a simpler way?
        def style(value):
            if isinstance(value, (tuple,list,np.ndarray)):
                if not isinstance(value[0], (tuple,list,np.ndarray)):
                    value = ruamel.yaml.comments.CommentedSeq(value)
                    value.fa.set_flow_style()
                else:
                    value = ruamel.yaml.comments.CommentedSeq(style(v) for v in value)
            return value

        # leaf sequences are inserted with "flow" style (i.e. inline)
        value = style(value)

        path = path.split(".")
        node = mold_info

        # new things are inserted at the beginning because the comment
        # that precedes the next block is attached to the lat item of this block :(
        for name in path[:-1]:
            if not name in node:
                node.insert(0, name, ruamel.yaml.comments.CommentedMap())
            node = node[name]
        node.insert(0, path[-1], value, comment)
        
        # save it
        with open(yaml_fn, "w") as f:
            yaml.dump(yaml_info, f)


    # mold name and info
    mold_name = args.args.pop(0)
    mold_info = yaml_info[mold_name]

    # set up the cache
    c = cache.Cache(
        mold_name,
        **vars(plain(mold_info)),
    )

    c.debug = args.debug.split(",")
    c.force = args.force.split(",")
    c.auto = args.auto
        
    print("testing:", args.test)
    if args.test == "all":
        c.tester = viz.compare
        c.force = ["all"]
    elif args.test == "changed":
        c.tester = viz.compare
    elif args.test == "none":
        c.tester = None
        c.force = ["none"]

    print("debug:", c.debug)
    print("force:", c.force)


# TODO: apply these defaults only once, in an early step
default_mold_thickness = 2.5

default_feed_diameter = 3
default_feed_height = 25

default_vent_diameter = 3
default_vent_height = 15

#
#
#

# TODO: move to viz
def marker(p, label=None):
    marker = viz(lib.sphere(c.v.sphere_radius/30).translate(p))
    if label is not None:
        marker = marker.label(str(label))
    return marker

@c.step(debug=0, force=0)
def load():

    # our inputs
    part_fn = c.v.load.part_fn
    path_fn = c.get("load.path_fn", None)

    # processing inputs
    simplify = c.get("load.simplify", None)
    rotate = c.get("load.rotate", None)
    scale = c.get("load.scale", None)
    clip = c.get("load.clip", None)
    fix = c.get("load.fix", False)
    decimated_faces = c.get("load.decimated_faces", None)
    shell = c.get("load.shell", None)

    # load files
    part0 = lib.load(part_fn).triangulate()
    path0 = lib.load(path_fn) if path_fn is not None else np.empty((0,3))
    print(f"{part_fn}: {lib.info(part0)}")
    c.finish("load")

    # sanity check on size - probably forgot to scale from m to mm
    if np.max(part0.points) - np.min(part0.points) < 1 and scale is None:
        raise Exception("part is <1mm but scale is not specified")

    # simplify part itself
    if simplify is not None:
        part0 = part0.decimate(1 - simplify)
        print(f"after simplify: {lib.info(part0)}")
        c.finish("simplify")

    # rotate
    if rotate is not None:
        rx, ry, rz = rotate
        part0 = part0.rotate_x(rx).rotate_y(ry).rotate_z(rz)
        c.finish("rotate")

    # scale
    if scale is not None:
        part0 = part0.scale(scale)
        c.finish("scale")

    # clip as requested
    if clip is not None:
        lo, hi = lib.bounds(part0)
        part0 = part0.clip(origin=(0,0,lo[2]+clip), normal=(0,0,-1))
        bodies = part0.split_bodies()
        upper = np.argmax(np.max(part.points, axis=0)[2] for part in bodies)
        part0 = b2 = bodies[upper].extract_surface()
        c.finish("clip")

    # fix defects in mesh
    if fix:
        part0 = lib.fix(part0)
        print(f"after fix: {part0.n_points} points, {part0.n_cells} cells")
        c.finish("fix")
            
    # compute decimated model
    if decimated_faces is not None:
        decimated0 = lib.decimate(part0, target_faces=decimated_faces)
        c.finish("decimate")
        print(f"decimated: {lib.info(decimated0)}")
        c.dbg("decimated", decimated0)
    else:
        decimated0 = part0.copy()

    # do shell after decimating so we can shell the decimated model as well
    if shell is not None:
        if isinstance(shell, dict):
            part0, decimated0 = lib.shell(part0, decimated0, **shell)
        else:
            part0, decimated0 = lib.shell(part0, decimated0, shell)
        c.dbg("shelled", part0)

    # move origin to center of part0
    # fix is slightly non-deterministic so radius varies slightly,
    # so round it to make tests deterministic
    print("computing sphere")
    center, r2 = lib.bounding_ball(decimated0.points)
    c.v.sphere_radius = np.round(np.sqrt(r2), 4)
    c.finish("bounding ball")

    # position center at origin
    c.v.part0 = part0.translate(-center).c((0.70, 0.80, 1.0))
    c.v.decimated0 = decimated0.translate(-center).c((0.70, 0.80, 1.0))
    c.v.sphere = lib.sphere(c.v.sphere_radius).c((1,1,1)).alpha(0.05)
    c.v.path0 = path0 - center

    c.dbg("final load", c.v.part0, c.v.sphere)


# TODO: move this to lib?
# penalty for a set of face-normals defined by how horizontal the face is
max_good_deg = 45
min_bad_deg = 60
spline = scipy.interpolate.PchipInterpolator(
    [0,  np.sin(max_good_deg/180*np.pi),  np.sin(min_bad_deg/180*np.pi),  1],
    [0,  0.05,                            0.8,                            1])
def penalties_for_normals(normals):
    #return np.maximum(normals[:,2] ** c.v.power, 0)
    return spline(np.maximum(normals[:,2], 0))

# sum of per-normal penalties weight by area
def total_penalty_for_normals(pd, normals):
    # face areas for weighting in total penalty
    # remains constant as we rotate part, so can memo on part
    if not hasattr(pd, "face_areas"):
        def area(pd, face):
            ps = pd.points[face]
            return la.norm(np.cross(ps[2]-ps[0], ps[1]-ps[0]))
        pd.face_areas = [area(pd, face) for face in pd.regular_faces]
    return sum(penalties_for_normals(normals) * pd.face_areas)

# total penalty for original part in a given orientation
def total_penalty_for_orientation(pd, orientation):
    normals = pd.cell_normals
    xform = lib.xform_from_to(orientation)
    xnormals = lib.apply_xform(xform, normals)
    total_penalty = total_penalty_for_normals(pd, xnormals)
    return total_penalty


@c.step(debug=0)
def potential_orientations():

    # our parameters
    n_start = c.get("orient.n_start", 200)
    n_best = c.get("orient.n_best", 20)

    # work with the decimated part
    working = c.v.decimated0

    # points defining orientations of interest
    # an orientation defined by a point is the orientation that transforms point to (0,0,1)
    orientations = lib.points_on_sphere(n_start)

    # given an orientation, tweak it to minize total penalty
    # TODO: this optimizes wrt tweaks defined by a change in x and y for the orientation
    # this could fail or be ill-conditioned for orientations that are exactly horizontal
    # TODO: might be faster to quit early when we see that we're converging to a point we've alread seen
    def optimize_orientation(orientation):
        normals = np.array(working.cell_normals)
        def fun(xy):
            o = orientation + [*xy, 0]
            xform = lib.xform_from_to(o)
            xnormals = lib.apply_xform(xform, normals)
            total_penalty = total_penalty_for_normals(working, xnormals)
            return total_penalty
        initial_penalty = fun((0,0))
        result = scipy.optimize.minimize(fun, x0=[0,0], tol=1e-4)
        print(f"optimization called fun {result.nfev} times " +
              f"reducing penalty from {initial_penalty:.0f} to {result.fun:.0f}")
        return orientation + [*result.x, 0]

    # sort the orientations by total penalty and take the n_best
    print(f"finding {n_best} best orientations out of {len(orientations)}")
    penalties = [total_penalty_for_orientation(working, o) for o in orientations]
    orientations = orientations[np.argsort(penalties)][:n_best]
    c.finish(f"finding {n_best} best")

    # optimize the remaining orientations and sort them by penalty again
    print(f"optimizing {n_best} best")
    orientations = np.array([optimize_orientation(o) for o in orientations])
    penalties = np.array([total_penalty_for_orientation(working, o) for o in orientations])
    orientations = orientations[np.argsort(penalties)]
    orientations = [lib.norm(o) for o in orientations]
    c.finish(f"optimizing {n_best} best")

    # eliminate near-duplicates
    remaining = []
    for o in orientations:
        if all(la.norm(lib.norm(o)-lib.norm(r)) > 0.01 for r in remaining):
            remaining.append(o)
    print(f"{len(remaining)} unique orientations remain")

    # our outputs
    c.v.orientations = remaining


@c.step(debug=0, force=0)
def orient():

    # TODO: working with decimated part eliminates a lot of small pockets
    # it's a lot faster, and eliminating small pockets may be a good thing?
    working = c.v.decimated0
    orientations = np.array(c.v.orientations) # TODO: pass from previous step as np.array?

    # explicit plotter instead of viz.show so we can interact
    pl = viz.Plotter(axes=0)
    pl.translucent = False

    # initial orientation information
    pl.tab_index = None
    pl.base_orientation = c.get("orient.orientation", [0,0,1])
    pl.bump = [0, 0]
    print("base orientation starts at", pl.base_orientation)

    # pick orientation i
    def pick_orientation(i):
        pl.base_orientation = orientations[i] if i is not None else [0,0,1]
        pl.bump = np.array([0, 0])
        pl.tab_index = i

    # change bump
    def bump_by(x, y):
        pl.bump += np.array([x, y])

    # navigate orientations using keyboard
    def on_keypress(e):
        if e.keypress == "Tab":
            tab_index = (pl.tab_index+1) % len(orientations) if pl.tab_index is not None else 0
            pick_orientation(tab_index)
        elif e.keypress == "Up":
            bump_by(0, -5)
        elif e.keypress == "Down":
            bump_by(0, +5)
        elif e.keypress ==  "Left":
            bump_by(+5, 0)
        elif e.keypress == "Right":
            bump_by(-5, 0)
        elif e.keypress == "z":
            pick_orientation(None)
        elif e.keypress == "T":
            # TODO: not behaving as expected
            pl.translucent = not pl.translucent
        elif e.keypress and e.keypress in "0123456789":
            pick_orientation(int(e.keypress))
        render()
    pl.add_callback("on key press", on_keypress)

    # navigate orientations using markers
    def on_pick(i):
        pick_orientation(i)
        render()

    # xform to/from orientation vector
    o2t = lambda o: lib.xform_from_to(o, [0,0,1])
    t2o = lambda t: (la.inv(t) @ [0,0,1,0])[0:3]
    b2t = lambda b: lib.axis_angle_xform([0,1,0], b[1], deg=True) @ lib.axis_angle_xform([1,0,0], b[0], deg=True) 

    # show current state
    def render():

        # compute and apply xform for current orientation
        pl.xform = b2t(pl.bump) @ o2t(pl.base_orientation)
        pl.orientation = t2o(pl.xform)
        pl.tilt_deg = np.arccos(pl.orientation[2]) * 180 / np.pi # informational
        xworking = lib.apply_xform(pl.xform, working)
        xorientations = lib.apply_xform(pl.xform, orientations)

        # see if we've landed on one of the identified best orientations
        at_orientation = None
        for i, o in enumerate(orientations):
            if la.norm(o - pl.orientation) < 1e-4:
                at_orientation = i
                if pl.tab_index is None:
                    pl.tab_index = at_orientation
                break
        print("at orientation", at_orientation)

        # compute pockets
        # TODO: can probably take Mesh outside of loop and only xform normals as orientation changes
        mesh = lib.Mesh(xworking)
        upward_pockets = mesh.pockets(dirn=-1)
        upward_pockets = mesh.v2p[upward_pockets]

        # make the orientations pickable
        color = lambda i: (0,1,0) if i==at_orientation else (0,0.5,0)
        on_sphere = lambda o: np.array(o) * c.v.sphere_radius
        pickable_orientations = [
            marker(on_sphere(xo), i).c(color(i)).on_pick(on_pick, i)
            for i, xo in enumerate(xorientations)
        ]

        # if the current orientation is not one of the best ones display a yellow marker
        current_orientation = marker(on_sphere([0,0,1])).c(1,1,0) if at_orientation is None else None

        # compute penalty per face, apply to xworking
        # assigning to the "penalty" array makes it the current active array,
        # so it will be used for coloring when rendering
        xworking.cell_data["penalty"] = penalties_for_normals(xworking.cell_normals)
        xworking.cmap = [
            [0,0,0], # values below 0 shouldn't occur so make it black to stand out
            (0, [0.4,0.8,0.4]), (0.3, [1,1,0.3]), (1, [1,0.3,0.3]),
            [0,0,0] # values above 1  shouldn't occur so make it black to stand out
        ]

        # show it and interact
        pl.remove(pl.actors)
        pl.traverse([
            f"{len(upward_pockets)} upward facing pockets in white;",
            #f"bump {pl.bump[0]} {pl.bump[1]}" if pl.bump[0] or pl.bump[1] else None,
            f"tilt: {pl.tilt_deg:.1f} deg",
            xworking.alpha(0.5) if pl.translucent else xworking,
            c.v.sphere,
            pickable_orientations,
            current_orientation,
            viz(upward_pockets).c(1,1,1).ps(25),
        ])
        pl.add_axes(pl)
        pl.render()

    # show it
    render()
    if c.auto: pl.close_in(1000)
    pl.show(axes=0, title="Right click on blue dots to select orientation").clear().close()

    # pass it on
    c.v.xdecimated = lib.apply_xform(pl.xform, c.v.decimated0)
    c.v.xpart = lib.apply_xform(pl.xform, c.v.part0)
    c.v.xpath = lib.apply_xform(pl.xform, c.v.path0)
    
    # save it in yaml file
    save("orient.orientation", pl.orientation.tolist(), f"tilt: {pl.tilt_deg:.1f} deg")



# TODO: merge this into previous step b/c it's not that slow
# then don't need to carry decimated
@c.step(lib.thicken, debug=0, force=0)
def mold():

    # construct the mold (exterior) using the decimated part
    # since we don't care about the quality of the exterior
    part = c.v.xdecimated.triangulate()
    mold_thickness = c.get("mold.thickness", default_mold_thickness)
    cellsize_pct = c.get("mold.cellsize_pct", 1)

    # TODO: maybe resolution could be lower to speed up this and subsequent steps?
    xmold = lib.thicken(part, mold_thickness, cellsize_pct=cellsize_pct)

    # for subsequent steps
    if not hasattr(c.v.mold, "thickness"):
        c.v.mold.thickness = mold_thickness
    c.v.xmold = xmold.c((0.8, 0.8, 0.8, 0.3))
    c.dbg("xmold", lib.info(xmold), xmold.alpha(0.2), part)
    save("mold.thickness", mold_thickness)


def support1():

    # our inputs
    mold = c.v.xmold
    max_overhang_deg = c.v.support.max_overhang_deg
    min_support_area = c.v.support.min_support_area

    # calculate limit of z component of normal
    z_limit = -np.sin(np.pi * max_overhang_deg / 180)
    print("z_limit", z_limit)

    # plane just below bottom of mold to extrude to
    # TODO: calculate actual i_size, j_size
    mold_min, mold_max = lib.bounds(mold)
    bottom = mold_min[2] - 1 # TODO: prameterize?
    
    ids = [i for i, normal in enumerate(mold.cell_normals) if normal[2] < z_limit]
    downward = mold.extract_cells(ids).extract_surface()

    # find downward facing faces and split to connected regions
    downward_pieces = [piece for piece in downward.split_bodies() if piece.area > min_support_area]
    support_point_pieces = [piece.points for piece in downward_pieces]

    # alternative to above - use Mesh to find connected regions
    # this uses a stronger connectivity requirement, requiring edge adjacency, not just point
    # results in somewhat smaller supports
    """
    mesh = lib.Mesh(downward, tolerant=True)
    support_point_pieces = [
        mesh.v2p[list(itertools.chain.from_iterable(mesh.f2v[region]))]
        for region in mesh.connected_regions()
        if len(region) > 10 # TODO: area based
    ]
    """

    # tune these to tune smoothing
    # TODO: I think alpha_pct should be about same as mold cellsize_pct? make that a tunable also
    alpha_pct, spline_pct = 1, 40

    # use pct to compute absolute parameters
    mold_d = la.norm(mold_max - mold_min)
    alpha_param, spline_param = 1 / (mold_d*alpha_pct/100), mold_d * spline_pct/100

    # add a region, defined by a list of exterior points outlining it, to the support
    support = None
    all_exterior_points = []
    def process_poly(poly):

        import shapely
        if not isinstance(poly, shapely.geometry.polygon.Polygon):
            # TODO: in knot_3_1_tube if max_overhang is set to 46
            # we get back one of these but it has length 0, and otherwise nothing
            # save the interior points, try with different alpha_pct, maybe open issue
            print(f"processing {type(poly)} length {len(poly.geoms)}")
            for p in poly.geoms:
                process_poly(p)
            return
        elif isinstance(poly, shapely.geometry.polygon.Polygon):
            exterior_points = np.array(poly.exterior.coords).copy()
            print(f"processing poly with {len(exterior_points)} points")
        elif isinstance(poly, shapely.geometry.multipolygon.MultiPolygon):
            help(poly); exit()
        else:
            raise Exception(f"don't understand {type(poly)}")

        # smooth the curve
        exterior_points = lib.smooth_path(exterior_points, spline_param)
        exterior_points = np.vstack([exterior_points.T, np.full(len(exterior_points), bottom)]).T
        all_exterior_points.extend(exterior_points)

        # make it into a face, extrude, and add to support
        face = lib.polyface(exterior_points, close=True).triangulate()
        support_piece = lib.extrude_to(face, dirn=(0,0,1), up_to=mold, through=True)
        nonlocal support
        support = lib.union(support, support_piece)

    for support_points in support_point_pieces:

        # flatten points of piece and use alphashapes to find exterior "concave hull"
        interior_points = support_points[:,0:2]
        print(f"computing concave hull using alpha_param {alpha_param}")
        poly = lib.timeit("alphashape", lambda: alphashape.alphashape(interior_points, alpha_param))
        process_poly(poly)

    # add foot if requested
    if hasattr(c.v.support, "foot"):
        boundary = np.array([p[0:2] for p in all_exterior_points])
        hull_vertices = scipy.spatial.ConvexHull(boundary).vertices
        hull_points = np.array(all_exterior_points)[hull_vertices]
        face = lib.polyface(hull_points).triangulate()
        foot = lib.extrude(face, (0,0,c.v.support.foot), cap=True)
        support = lib.union(support, foot)
        
    # for debugging
    contours = mold.contour([z_limit], mold.copy().point_normals[:,2])
    all_interior_points = np.concatenate(support_point_pieces)
    c.dbg(
        "support", lib.info(support), support, mold, 
        contours.c(0,0,0).lw(5).c(1,0,0), viz(all_interior_points).c(0,0,1).ps(5)
    )

    c.v.final_support = support.c((.9,1.0,.9))


def support2():

    # our inputs
    mold = c.v.xmold
    max_overhang_deg = c.v.support.max_overhang_deg
    min_support_area = c.v.support.min_support_area

    # plane just below bottom of mold to extrude to
    mold_min, mold_max = lib.bounds(mold)
    bottom = mold_min[2] - 1 # TODO: prameterize?
    size = la.norm(mold_max - mold_min)

    # tunable smoothing parameters
    spline_pct = 20
    smooth_its = 500

    # extract contours using smoothed mold
    z_limit = -np.sin(np.pi * max_overhang_deg / 180)
    smoothed_mold = mold.smooth(smooth_its)
    contours = smoothed_mold.contour([z_limit], mold.copy().point_normals[:,2])

    # how to smooth the contour loops
    spline_param = size * spline_pct/100

    # add a support piece for each contour loop
    lines = lib.Lines(contours)
    support = None
    for loop in lines.connected_loops():
        points = lib.smooth_path(lines.v2p[loop], spline_param)
        points[:,2] = bottom
        face = lib.polyface(points).triangulate()
        if face.area > min_support_area:
            try:
                piece = lib.extrude_to(face, dirn=(0,0,1), up_to=mold, through=True)
            except:
                c.dbg("failed to extrude", points, face, mold)
            support = lib.union(support, lib.orient(piece))

    c.v.final_support = support.c((.9,1.0,.9))
    c.dbg("support", mold, contours.c(1,0,0).lw(5), support)


def support3():

    # our inputs
    mold = c.v.xmold
    max_overhang_deg = c.get("support.max_overhang_deg", 45)
    min_support_area = c.get("support.min_support_area", 50)
    margin = c.get("support.margin", 0)

    # tunable smoothing parameters
    spline_pct = c.get("support.spline_pct", 50)
    smooth_its = c.get("support.smooth_its", 500)
    out_n = c.get("support.out_n", 200)

    # derived
    z_limit = -np.sin(np.pi * max_overhang_deg / 180)
    mold_min, mold_max = lib.bounds(mold)
    bottom = mold_min[2] - 1 # TODO: prameterize?
    top = mold_max[2] + 1
    size = la.norm(mold_max - mold_min)
    spline_param = size * spline_pct/100

    # extract contours using smoothed mold for debugging
    smoothed = mold.smooth(smooth_its)
    contours = smoothed.contour([z_limit], smoothed.point_normals[:,2]).lw(3)

    # compute total region needing support based on normals of smoothed mold
    # TODO: contours does this also as "Contour data" - do we need both?
    smoothed.point_data.set_array(smoothed.point_normals[:,2], "nz")
    downward = smoothed.clip_scalar("nz", invert=True, value=z_limit)
    # TODO: use copy() above instead of the following
    downward = pv.PolyData.from_regular_faces(downward.points, downward.regular_faces) # remove color map :(
    #c.dbg(downward, contours)

    # separately extrude and the combine each  disconnected piece
    support = None
    for downward_piece in downward.split_bodies():
        
        # project the piece to the bottom plane
        downward_piece = downward_piece.extract_surface()
        bottom_points = downward_piece.points.copy()
        bottom_points[:,2] = bottom
        bottom_face = pv.PolyData.from_regular_faces(bottom_points, downward_piece.regular_faces)

        # ignore pieces that are too small
        print(f"support piece bottom_face area {bottom_face.area:.2f}")
        if bottom_face.area < min_support_area:
            print("    ignoring")
            continue

        # smooth the crinkled boundary edges by construction a smoothed path and moving each
        # boundary point to the nearest point on that path
        # a subtlety: this can leave behind points near but not on the boundary outside the new boundary
        # but that causes the faces to double back, and they correctly cancel out
        # (not sure if by extrude or union, probably the latter)
        c.dbg("pre-smooth", bottom_face, contours)
        mesh = lib.Mesh(bottom_face, tolerant=True)
        new_points = bottom_face.points.copy()
        for boundary in mesh.boundaries():
            boundary_points = new_points[boundary]
            smooth_points = lib.smooth_path(boundary_points, spline_param, out_n=out_n)
            if margin != 0:
                deltas = np.roll(smooth_points, 1, axis=0) - np.roll(smooth_points, -1, axis=0)
                deltas = np.array([[-d[1], d[0], 0] / la.norm(d) for d in deltas]) * margin
                smooth_points += deltas
            smooth_path = lib.polyline(smooth_points, close=True)
            new_boundary_points = lib.find_closest(smooth_path, boundary_points)
            new_points[boundary] = new_boundary_points
        bottom_face = pv.PolyData.from_regular_faces(new_points, bottom_face.regular_faces)
        c.dbg("post-smooth", bottom_face, contours)

        # extrude the piece to the top and subtract mold
        support_piece = lib.extrude(bottom_face, dirn=(0,0,top-bottom))
        support_piece = lib.difference(support_piece, mold)

        # this may result in multiple fragments, obtained by using split_bodies on support_piece
        # only one of the fragments will be the one needed to support downward_piece
        # this will be the fragment that touches downward_piece, which we determine by
        # finding how close the points of downward_piece are to the each fragment,
        # and choosing the one that is closest
        # note that since we used the smoothed mold to compute downward_piece but the unsmoothed mold
        # to compute the fragments, there may be a bit of a gap (more than machine precision)
        # so we look for smallest average distance
        fragments = []
        for fragment in support_piece.split_bodies():
            # determine how close this fragment is to the downward_piece and record
            # a (distance, fragment) tuple in fragments
            closest = lib.find_closest(fragment, downward_piece.points)
            distances = [la.norm(c-p) for c, p in zip(closest, downward_piece.points)]
            fragments.append((np.average(distances), fragment))
        support_piece = sorted(fragments, key = lambda distance_fragment: distance_fragment[0])[0][1]
        support_piece = support_piece.extract_surface()

        #c.dbg("to union", viz(support).alpha(0.2), support_piece, contours)
        support = lib.union(support, support_piece)

    # save it as not unlikely to want to change
    save("support.max_overhang_deg", max_overhang_deg)

    c.v.final_support = support.c((.9,1.0,.9))
    c.dbg("support", mold, contours, support)


@c.step(support1, support2, support3, lib.smooth_path, debug=0)
def support():

    if hasattr(c.v, "support"):
        style = c.get("support.style", default=3)
        if style == 1:
            support1()
        elif style == 2:
            support2()
        elif style == 3:
            support3()
    else:
        c.v.final_support = None


@c.step(debug=0, force=0)
def vents():

    # work with xdecimated
    # this gives fewer potential vents and is faster
    working = c.v.xdecimated
    xpart = c.v.xpart
    xmold = c.v.xmold
    mold_thickness = c.v.mold.thickness

    initial_selected = c.get("vents.position", [], np.array)
    vent_diameter = c.get("vents.diameter", default_vent_diameter)
    vent_height = c.get("vents.height", default_vent_height)

    pad = lambda x, y: x + [x[-1]] * (len(y) - len(x)) if isinstance(x, list) else[x] * len(y)
    vent_diameter = pad(vent_diameter, initial_selected)
    vent_height = pad(vent_height, initial_selected)


    # vents start just above highest point
    # TODO: we use high point of mold for this
    # avoids issue with extrude_to for mold when we start inside mold
    # fix that issue, or is this ok
    _, max = lib.bounds(xmold)
    c.v.top = max[2] + 1

    # calculate potential vent points
    print("calculating potential vent points")
    mesh = lib.Mesh(working)
    maximal, bumps, pockets, handles = mesh.maximal_types(dirn=1)
    hi_points = mesh.v2p[bumps + handles] # handles might be tilted mountains
    c.finish("Mesh")

    # vent info
    @dataclasses.dataclass
    class Vent:
        position: np.ndarray
        diameter: float
        height: float
        recommended: bool
        selected: bool
        vent: pv.PolyData = None
        
    # initial catalog of incomplete potential vents
    # calculated for display purposes - don't actually intersect object because that's a bit expensive to do
    # uses default diameter and height
    top = lambda pt: [pt[0], pt[1], c.v.top]
    vents = [
        Vent(hp, default_vent_diameter, default_vent_height, recommended=True, selected=False)
        for hp in hi_points
    ]

    # search in vents for an existing at hp
    def find_vent(hp):
        for i, vent in enumerate(vents):
            if la.norm(vent.position - hp) < 1e-4:
                return i
        return None

    # add ones recorded in yaml file as selected
    # may update diameter and height of initial catalog of potential vents
    for hp, diameter, height in zip(initial_selected, vent_diameter, vent_height):
        found = find_vent(hp)
        if found is not None:
            found = vents[found]
            found.selected, found.diameter, found.height = True, diameter, height
        else:
            vents.append(Vent(hp, diameter, height, recommended=False, selected=True))

    # now that we have desired height and diameter, generate incomplete vents for display
    make_vent = lambda vent: lib.vent(
        pt := top(vent.position),
        vent.diameter, vent.height,
        to = vent.position - pt
    )
    for vent in vents:
        vent.vent = make_vent(vent)

    # add x,y to position of currently active vent
    # may split or merge active vent with/from recommended vents
    # update active vent as needed by split/merge
    # generate the new vent
    def update_position(x, y):

        # remove the active vent while we decide what to do with it,
        # compute new position, and see if new position is for existing vent
        active_vent = vents[pl.active_vent_index]
        del vents[pl.active_vent_index]
        new_position = active_vent.position + [x, y, 0]
        found_index = find_vent(new_position)
        print("found index", found_index)

        if found_index is not None:

            # we've moved active vent to coincide with an existing vent
            # just update the existing vent with the characteristics of active vent
            # and don't put active vent back
            print("merging")
            vents[found_index].diameter = active_vent.diameter
            vents[found_index].selected = True
            active_vent = vents[found_index]
            pl.active_vent_index = found_index

        elif active_vent.recommended:

            # we're tweaking a recommended vent
            # put the recommended vent back, but deselect it
            print("separating")
            active_vent.selected = False
            vents.append(active_vent)

            # and split off the tweaked vent as a separate vent,
            # copying geometry of the vent we're moving
            print("moving new")
            active_vent = Vent(
                new_position, active_vent.diameter, active_vent.height,
                recommended=False, selected=True
            )
            vents.append(active_vent)
            pl.active_vent_index = len(vents) - 1

        else:

            # just update the vent and put it back
            print("moving existing")
            active_vent.position = new_position
            vents.append(active_vent)
            pl.active_vent_index = len(vents) - 1

        # generate new vent
        active_vent.vent = make_vent(active_vent)


    # update diameter of active vent
    def update_diameter(delta):
        active_vent = vents[pl.active_vent_index]
        active_vent.diameter += delta
        active_vent.vent = make_vent(active_vent)
        pl.message = f"diameter: {active_vent.diameter}"

    # find and active next selected vent after start
    def activate_next(start):
        if pl.active_vent_index is None:
            pl.active_vent_index = 0
        active_vent_index = None
        for i in range(len(vents)):
            j = (start + i) % len(vents)
            if vents[j].selected:
                active_vent_index = j
                break
        pl.active_vent_index = active_vent_index

    # selected/unselect vent i
    def on_pick(i):
        if pl.active_vent_index == i:
            vents[i].selected = False
            pl.active_vent_index = None
        else:
            vents[i].selected = True
            pl.active_vent_index = i
        render()

    # navigate orientations using keyboard
    def on_keypress(e, *args, **kwargs):

        print("keypress:", e.keypress)

        position_delta = 0.25
        diameter_delta = 0.25

        if e.keypress == "Tab":
            if pl.active_vent_index is None:
                pl.active_vent_index = 0
            activate_next(pl.active_vent_index + 1)
        elif pl.active_vent_index is not None:
            if e.keypress == "Up":
                update_position(0, -position_delta)
            elif e.keypress == "Down":
                update_position(0, +position_delta)
            elif e.keypress ==  "Left":
                update_position(+position_delta, 0)
            elif e.keypress == "Right":
                update_position(-position_delta, 0)
            elif e.keypress in ["PLUS", "equal"]:
                update_diameter(+diameter_delta)
            elif e.keypress in ["UNDERSCORE", "minus"]:
                update_diameter(-diameter_delta)
            elif e.keypress in ["x", "Backspace"]:
                if pl.active_vent_index is not None and not vents[pl.active_vent_index].recommended:
                    del vents[pl.active_vent_index]
                    activate_next(pl.active_vent_index)

        # make and render the new vent with updated geometry
        render()

    # set up plotter
    pl = viz.Plotter(axes=0)
    pl.active_vent_index = None
    pl.message = None
    pl.add_callback("on key press", on_keypress)

    # show currently state
    def render():

        # potential vents
        vent_display = []
        for i, vent in enumerate(vents):
            b = 1 if i == pl.active_vent_index else 0.8
            c = (0,b,0) if vent.recommended else (b,0.9*b,0.3)
            alpha = 1 if vent.selected else 0.2
            vent = viz(vent.vent).c(c).alpha(alpha)
            vent = vent.on_pick(on_pick, i).label(str(i))
            vent_display.append(vent)

        # show it and interact
        pl.remove(pl.actors)
        pl.traverse([
            #viz.Button(lambda *args: pl.break_interaction(), "Accept"),
            pl.message,
            xpart,
            xmold,
            vent_display,
        ])
        pl.add_axes(pl)
        pl.render()
        pl.message = None

    # show initial and interact
    render()
    if c.auto: pl.close_in(1000)
    pl.show(axes=0, title="Right click to select vents").close()

    # calculate actual vents
    #part_vents = [make_vent(vent.position, vent_diameter, xpart) for vent in selected]
    #mold_vents = [make_vent(vent.position, mold_diameter, xmold) for vent in selected]
    selected = [vent for vent in vents if vent.selected]
    part_vents = lib.all_union([
        lib.vent(top(vent.position), vent.diameter, vent.height, to=xpart)
        for vent in selected
    ])
    mold_vents = lib.all_union([
        lib.vent(top(vent.position), vent.diameter+2*mold_thickness, vent.height, to=xmold)
        for vent in selected
    ])

    # our outputs
    c.v.part_vents = lib.canonicalize(part_vents)
    c.v.mold_vents = lib.canonicalize(mold_vents)

    # save in yaml file
    save("vents.position", [vent.position.tolist() for vent in selected])
    save("vents.diameter", [vent.diameter for vent in selected])
    save("vents.height", [vent.height for vent in selected])


def multi_feed():

    # our inputs
    n_feeds = c.v.feed.n_feeds
    mold_thickness = c.v.mold.thickness
    feed_diameter = c.get("feed.diameter", default_feed_diameter)
    feed_height = c.get("feed.height", default_feed_height)
    aim_at = c.get("feed.aim_at", "path" if len(c.v.xpath) > 0 else "part")

    # define objective function that measures distance to center line
    hdist = lambda p, q: la.norm(np.array([p[0],p[1],0]) - np.array([q[0],q[1],0]))
    center = np.average(c.v.xpart.points, axis=0)
    fun = lambda p: hdist(p, center)

    # figure out what to aim at
    if aim_at == "path":
        # aim at path defined by c.v.xpath
        path = lib.polyline(c.v.xpath, close=True)
        mesh = lib.Lines(path)
    elif aim_at == "contour":
        # TODO: this is better, but we are aming too high
        # is meeting point too low, or is 45deg contour the wrong idea?
        contour_part = c.v.xpart # much smoother than decimated
        path = contour_part.copy().contour([0.707], contour_part.point_normals[:,2]).c(1,0,0).lw(8)
        mesh = lib.Lines(path)
    elif aim_at == "part":
        # aim at the part itself
        path = None
        mesh = lib.Mesh(c.v.xpart)
    else:
        raise Exception(f"don't understand aim_at {aim_at}")
    
    # find n_feeds locally closest points to center line
    closest = mesh.local_minima(lambda v: fun(mesh.v2p[v]))
    closest = sorted(closest, key = lambda v: -mesh.v2p[v][2])
    if n_feeds > 0:
        closest = closest[:n_feeds]
    else:
        closest = closest[n_feeds:]

    # compute meeting point and lines from meeting point to entry points
    c.v.eps = mesh.v2p[closest]
    slope = 1 # channels go upwards at 45 degrees
    meet_z = sum(ep[2] + slope * hdist(ep, center) for ep in c.v.eps) / len(c.v.eps)
    c.v.meet = np.array([center[0], center[1], meet_z])
    c.dbg("meeting point", c.v.xpart.alpha(0.1), c.v.meet, c.v.eps, path)

    # start with None then add in some order
    part_feed = None
    mold_feed = None

    # start with a sphere at meeting point
    # TODO: fr
    part_r = feed_diameter / 2
    mold_r = part_r + mold_thickness
    part_feed = lib.union(part_feed, lib.sphere(part_r).translate(c.v.meet))
    mold_feed = lib.union(mold_feed, lib.sphere(mold_r).translate(c.v.meet))

    # mouth
    top = np.array([c.v.meet[0], c.v.meet[1], c.v.top])
    feed = [c.v.meet, top]    
    #part_feed = lib.union(part_feed, lib.mouth(feed, 2 * part_r, c.v.top + c.v.feed_height - c.v.meet[2]))
    #mold_feed = lib.union(mold_feed, lib.mouth(feed, 2 * mold_r, c.v.top + c.v.feed_height - c.v.meet[2]))
    part_feed = lib.union(part_feed, lib.mouth(feed, 2 * part_r, feed_height))
    mold_feed = lib.union(mold_feed, lib.mouth(feed, 2 * mold_r, feed_height))

    # TODO: this assumes z is not None but x_axis and y_axis are None
    # generalize and move to lib
    def align(x=None, y=None, z=None):
        z_axis = lib.norm(z)
        x_axis = lib.norm((1,0,0) - np.dot((1,0,0), z_axis) * z_axis)
        y_axis = lib.norm(np.cross(z_axis, x_axis))
        xform = np.array([x_axis, y_axis, z_axis, np.array([0,0,0])])
        xform = np.concatenate([xform.T, [[0,0,0,1]]])
        return xform

    # add feed from meeting point to each ep
    for dirn in [ep - c.v.meet for ep in c.v.eps]:

        def add_feed(feed, r, target):
            c.start()
            circle = lib.circle(r).transform(align(z=dirn)).translate(c.v.meet)
            new_feed = lib.extrude_to(circle, dirn, target, extra=1e-4)
            return lib.union(feed, new_feed)

        part_feed = add_feed(part_feed, part_r, c.v.xpart)
        mold_feed = add_feed(mold_feed, mold_r, c.v.xmold)

    # done
    c.v.part_feed = part_feed
    c.v.mold_feed = mold_feed

    save("feed.diameter", feed_diameter)
    save("feed.height", feed_height)

    # debug
    c.dbg(
        "final",
        c.v.part_feed,
        viz(c.v.mold_feed).alpha(0.2),
        viz(c.v.xpart).alpha(0.5),
        c.v.xmold,
        c.v.meet,
        c.v.eps,
        path
    )


def manual_feed():

    # our inputs
    mold_thickness = c.v.mold.thickness
    feed_diameter = c.get("feed.diameter", default_feed_diameter)
    feed_height = c.get("feed.height", default_feed_height)
    feed_position = c.get("feed.position", [])
    contour_part = c.v.xdecimated.smooth(1000)
    pick_part = c.v.xpart
    
    # either a single point, or an array of points
    print("feed_position", feed_position)
    if len(feed_position) > 0:
        if isinstance(feed_position[0], list):
            feed_position = [np.array(p) for p in feed_position]
        else:
            feed_position = [np.array(feed_position)]
    print("feed_position", feed_position)

    # compute 45 deg contours to use as a visual guide
    contours = contour_part.contour([0.707], contour_part.point_normals[:,2]).c(1,0,0).lw(8)

    # diameter of mold feed
    mold_diameter = feed_diameter + 2 * mold_thickness

    # generate feed for a single point
    def generate_single_feed():
        position = pl.position[0]
        print("generating single feed at", position)
        picked_face = pick_part.find_closest_cell(position)
        picked_normal = pick_part.cell_normals[picked_face]
        p1 = position - feed_diameter * picked_normal
        p2 = position + feed_diameter * picked_normal
        p3 = np.array([p2[0], p2[1], c.v.top])
        pl.part_feed = lib.mouth([p1, p2, p3], feed_diameter, feed_height)
        pl.mold_feed = lib.mouth([p1, p2, p3], mold_diameter, feed_height)
    
    def generate_meet_points():
        pl.meet_points = []
        if len(pl.position) > 1:
            cx, cy, _ = np.average(pl.position, axis=0)
            for px, py, pz in pl.position:
                d = np.sqrt((px-cx)**2 + (py-cy)**2)
                pl.meet_points.append(np.array([cx, cy, pz + 0.707 * d]))

    def even_positions():

        def new_positions(x):
            a0, cx, cy, cz = x
            result = []
            for a in np.arange(a0, a0+2*np.pi, 2*np.pi/len(pl.position)):
                origin = [cx, cy, cz]
                end = [cx+1000*np.sin(a), cy+1000*np.cos(a), cz]
                p, _ = pick_part.ray_trace(origin, end, first_point=True)
                if len(p) == 0:
                    p = [100, 100, 100] # really far away
                result.append(p)
            return result

        def fun(x):
            return np.sum([np.sum((old - new) ** 2) for old, new in zip(pl.position, new_positions(x))])

        a0 = np.atan2(pl.position[0][0], pl.position[0][1])
        center = np.average(pl.position, axis=0)
        result = scipy.optimize.minimize(fun, x0=[a0, *center])
        pl.position = new_positions(result.x)
                

    # generate feed for multiple points
    def generate_multi_feed():
        # sort by ascending z
        order = np.argsort([p[2] for p in pl.position])
        for o in order:
            p2 = pl.meet_points[o]
            p1 = (p := pl.position[o]) - feed_diameter * lib.norm(p2 - p)
            if o == order[0]:
                # lowest gets an elbow and generates the feed
                p3 = np.array([p2[0], p2[1], c.v.top])
                pl.part_feed = lib.mouth([p1, p2, p3], feed_diameter, feed_height)
                pl.mold_feed = lib.mouth([p1, p2, p3], mold_diameter, feed_height)
            else:
                # others just generate a channel to the middle
                pl.part_feed = lib.union(pl.part_feed, lib.elbow([p1, p2], feed_diameter))
                pl.mold_feed = lib.union(pl.mold_feed, lib.elbow([p1, p2], mold_diameter))

    def generate_feed():
        if len(pl.position) > 1:
            generate_multi_feed()
        else:
            generate_single_feed()

    # pick a new position
    def on_part_pick(event):
        pl.position.append(event.picked3d)
        pl.part_feed = pl.mold_feed = None
        render()

    def on_position_pick(i):
        del pl.position[i]
        render()

    def on_feed_pick(_):
        pl.part_feed = pl.mold_feed = None
        render()

    def on_keypress(e):
        if e.keypress == "Return":
            generate_feed()
            render()
        elif e.keypress == "e":
            even_positions()
            render()

    # explicit plotter instead of viz.show so we can interact
    pl = viz.Plotter(axes=0)
    pl.position = feed_position
    pl.meet_points = []
    pl.part_feed = pl.mold_feed = None
    pl.add_callback("on key press", on_keypress)

    # show the picked point
    def render():

        generate_meet_points()
        position_markers = [marker(p).on_pick(on_position_pick, i).c((1,1,1)) for i, p in enumerate(pl.position)]
        meet_lines = [lib.polyline([p, q]) for p, q in zip(pl.position, pl.meet_points)]

        # show it
        pl.remove(pl.actors)
        pl.traverse([
            #viz.Button(lambda *args: pl.close(), "Accept"),
            pl.part_feed,
            viz(pl.mold_feed).on_pick(on_feed_pick).alpha(0.2),
            c.v.part_vents,
            contours.c(0.8,0,0),
            viz(pick_part).on_pick(on_part_pick), #.alpha(0.2),
            position_markers,
            viz(meet_lines).c(1,0,0).lw(5)
        ])
        pl.add_axes(pl)
        pl.render()

    # show it
    render()
    if c.auto: pl.close_in(1000)
    pl.show(axes=0, title="Right click to pick a feed entry point").close()

    # make sure we've generated something
    if pl.part_feed is None or pl.mold_feed is None:
        generate_feed()

    # TODO: above part just extends feed a little ways into the part
    # maybe we should generate a careful that is extruded just to the surface?
    #c.v.feed = elbow_feed(position, picked_normal, r, part)
    c.v.part_feed = pl.part_feed
    c.v.mold_feed = pl.mold_feed
    
    # show it
    c.dbg(
        c.v.part_feed,
        viz(c.v.mold_feed).alpha(0.2),
        c.v.xpart,
    )

    # save it
    save("feed.position", [p.tolist() for p in pl.position])
    save("feed.diameter", feed_diameter)
    save("feed.height", feed_height)


    
@c.step(debug=0, force=0)
def feed():

    if hasattr(c.v.feed, "n_feeds"):
        multi_feed()
    else:
        manual_feed()


@c.step()
def merge():

    # enable this to debug in case all_union fails
    #lib.all_union.debug = True

    # put the main part last
    # faster to union all the small non-overlapping pieces first then union w/ the big part
    print("calculating part")
    part = lib.all_union(c.v.part_vents, c.v.part_feed, c.v.xpart)
    c.finish("calculating part")

    print("calculating mold")
    mold = lib.all_union(c.v.mold_vents, c.v.mold_feed, c.v.xmold)    
    c.finish("calculating mold")

    print("subtracting part from mold")
    mold = lib.difference(mold, part)
    c.finish("subtracting part from mold")
    
    part.viz = c.v.xpart.viz
    mold.viz = c.v.xmold.viz

    c.v.final_part = part
    c.v.final_mold = mold

    c.dbg("final", c.v.final_part, c.v.final_mold)

@c.step(force=0)
def save():

    # canonicalize to help avoid spurious changes in output files
    # works for obj files,
    # but 3mf files have embedded uuids - maybe can avoid that?
    mold = lib.canonicalize(c.v.final_mold)
    support = lib.canonicalize(c.v.final_support)

    fn = f"molds/{c.cache_name}"
    # obj files are very big
    #lib.save(mold, f"{fn}-mold.obj")
    #lib.save(support, f"{fn}-support.obj")
    save = {"mold": mold}
    if support is not None:
        save["support"] = support
    lib.save(save, f"{fn}.3mf")

@c.step(force=1)
def info():

    part = c.v.xpart # without apparatus
    mold = c.v.final_mold # with apparatus

    alloy_g_per_cm3 = 8.56
    alloy_dollars_per_kg = 23.4

    part_area_cm2 = part.area / 100
    part_volume_cm3 = part.volume / 1000
    part_weight_g = part_volume_cm3 * alloy_g_per_cm3
    part_cost_dollars = part_weight_g / 1000 * alloy_dollars_per_kg

    print(f"part surface area: {part_area_cm2:.1f} cm²")
    print(f"part volume: {part_volume_cm3:.1f} cm³")
    print(f"part weight: {part_weight_g:.1f} g")
    print(f"part cost: ${part_cost_dollars:.2f}")

    pva_g_per_cm3 = 1.23
    pva_dollars_per_kg = 96.00

    mold_volume_cm3 = mold.volume / 1000
    mold_weight_g = mold_volume_cm3 * pva_g_per_cm3
    mold_cost_dollars = mold_weight_g / 1000 * pva_dollars_per_kg
    est_support_cost_dollars = 0.3 * mold_cost_dollars

    print(f"mold volume: {mold_volume_cm3:.1f} cm³")
    print(f"mold weight: {mold_weight_g:.1f} g")
    print(f"mold cost: ${mold_cost_dollars:.2f}")
    print(f"est support cost: ${est_support_cost_dollars:.2f}")

    print(f"est total cost: ${part_cost_dollars + mold_cost_dollars + est_support_cost_dollars:.2f}")

    close_in = 1000 if c.auto else None
    viz.show(c.v.final_part, c.v.final_mold, c.v.final_support, title="final", close_in=close_in)

