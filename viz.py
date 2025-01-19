import collections
import lib
import pyvista as pv
import numpy as np
import numpy.linalg as la
import os
import sys
import threading
import time
import types
import viz
import vedo

#
# I like the vedo fluent API for setting color, alpha, lw, etc.,
# so monkey patch DataObject to emulate that
#

# wrapper for a list or other object so we can set colors etc.
# because monkey() gives this class the appropriate methods
class _Viz(list):
    def copy(self):
        return(viz(self))
    def __str__(self):
        return str(self.__dict__)

# jump through a couple hoops to make the viz module calllable
# so users can write viz(x) to wrap x in an object with viz attributes
class CallableModule(types.ModuleType):
    def __call__(self, x=[]):
        if isinstance(x, (list,tuple)):
            return _Viz(x)
        else:
            return _Viz([x])
sys.modules[__name__].__class__ = CallableModule


# empty class to hold viz attributes
class EmptyViz:
    def __str__(self):
        return str(self.__dict__)

# we do a copy so we can change color without changing underlying object
# this is more consistent with the fluent api?
def copy(o):
    copy = o.copy()
    copy.viz = o.viz if hasattr(o, "viz") else EmptyViz()
    if hasattr(o, "picking"):
        copy.picking = o.picking
    return copy

def monkey(attr):
    def _(o, *value):
        if len(value) == 1:
            value = value[0]
        c = copy(o)
        setattr(c.viz, attr, value)
        return c
    setattr(EmptyViz, attr, None)
    # don't allow pd.on_pick because this interferes with pickling
    # so user must say viz(pd).on_pick to wrap pd in an object that won't be pickled
    if attr != "on_pick":
        setattr(pv.DataObject, attr, _)
    setattr(_Viz, attr, _)

viz_attrs = ["c", "alpha", "lw", "ps", "on_pick", "label"]

# e.g.: gives pv.DataObject a method c that takes an argument and assigns it to self.viz.c
for attr in viz_attrs:
    monkey(attr)

def merge_viz(upper, lower):
    result = EmptyViz()
    for attr in viz_attrs:
        merged = getattr(upper, attr) or getattr(lower, attr)
        setattr(result, attr, merged)
    return result
    
#
#
#

# TODO: is this the right place for this?
vedo.settings.enable_default_keyboard_callbacks = False

Button = collections.namedtuple("Button", "fun label")

# configure a plotter the way we like it
class Plotter(vedo.Plotter):

    window_pos = (100,100)    
    window_size = (1500,1500)

    camera_view_up = (0, 0, 1)
    camera_position = (0, -100, 30)
    axes_options = dict(
        xlabel_rotation=180, xtitle_rotation=180,
        ylabel_rotation=180, ytitle_rotation=180
    )

    def __init__(self, title="viz", axes=1, *args, **kwargs):

        super().__init__(
            *args, **kwargs,
            pos=self.window_pos, size=self.window_size, title=title, axes=self.axes_options.copy()
        )

        # handle picking via right mouse clicks
        def on_click(event):
            if event.actor is not None and hasattr(event.actor, "picking"):
                picking = event.actor.picking
                if isinstance(picking, (list,tuple)) and len(picking) > 1:
                    picking[0](picking[1])
                else:
                    picking(event)
        self.add_callback("RightButtonPress", on_click)

        # configure camera
        self.camera.SetViewUp(*self.camera_view_up)
        self.camera.SetPosition(*self.camera_position)
        #self.parallel_projection()

        # TODO: check out user_mode in vedo.Plotter
        # terrain interaction style, plus scroll wheel zoom
        # TODO: need support for shift to pan
        import vtk
        style = vtk.vtkInteractorStyleTerrain()
        self.interactor.SetInteractorStyle(style)

        # scroll wheel zooms
        def on_scroll(obj, event):
            factor = 1.02
            zoom = factor if event=="MouseWheelForwardEvent" else 1/factor
            self.zoom(zoom) 
            self.render()
        self.interactor.AddObserver("MouseWheelForwardEvent", on_scroll)
        self.interactor.AddObserver("MouseWheelBackwardEvent", on_scroll)
 
        # pan with shift left mouse button
        # TODO: somehow causing segfaults
        """
        def pan_start(obj, event):
            if self.interactor.GetShiftKey():
                style.StartPan()
            style.OnLeftButtonDown()
        def pan_end(obj, event):
            style.EndPan()
            style.OnLeftButtonUp()
        style.AddObserver("LeftButtonPressEvent", pan_start)
        style.AddObserver("LeftButtonReleaseEvent", pan_end)
        style.AddObserver("DragLeaveEvent", pan_end) # not sure if this helps
        """

    def add_axes(self, model):
        self.add(vedo.Axes(model, **self.axes_options))

    # top-level traverse
    # TODO: for now we accumulate text objects into a single string and display that
    def traverse(self, *args):

        self.accumulated_texts = []
        self._traverse(args)
    
        # add the accumulated texts
        if self.accumulated_texts != "":
            text = " ".join(self.accumulated_texts)
            self.add(vedo.Text2D(text, s=1.2, font="ComicMono"))


    # recursive traversal
    # TODO: similar to viz() - do we need both?
    def _traverse(self, o, upper_viz=EmptyViz()):

        actor = viz = None

        if isinstance(o, (str,int,float,np.float32,np.float64)):

            # TODO: for now just accumulate into a single string and display that
            self.accumulated_texts.append(str(o))

        elif isinstance(o, Button):

            # TODO: accumulate and position
            actor = button = self.add_button(
                o.fun, [o.label],
                c="w", bc="gray", font="ComicMono", pos=(0.07,0.97)
            )
            button.plotter = self # TODO: is this still needed?

        # TODO: this is also in viz - do we need it here?
        elif isinstance(o, np.ndarray) and o.shape[-1] == 3:
            self._traverse(points(o), upper_viz)

        elif isinstance(o, (list,tuple)):
            if hasattr(o, "viz"):
                upper_viz = merge_viz(upper_viz, o.viz)
            for oo in o:
                self._traverse(oo, upper_viz)

        elif "vedo" in str(type(o)):
            self.add(o)

        elif "pyvista" in str(type(o)):

            actor = mesh = vedo.Mesh(o)

            # pick up colors etc. assigned by .c(), .alpha()
            viz = o.viz if hasattr(o, "viz") else EmptyViz()
            viz = merge_viz(upper_viz, viz)
            color, alpha, lw, ps, = viz.c, viz.alpha, viz.lw, viz.ps
            if color is not None:
                if len(color) > 3:
                    alpha = color[3]
                    color = color[0:3]

            # TODO: .alpha() returns float - ???
            # apply colors etc.
            mesh.flat()
            if color is not None: mesh.c(color)
            if alpha is not None: mesh.alpha(alpha)
            if lw is not None: mesh.lw(lw)
            if ps is not None: mesh.ps(ps)

            # apply a color map if one is requested
            if hasattr(o, "cmap"):
                args = dict(
                    below_color = o.cmap[0],
                    colorlist = o.cmap[1:-1],
                    above_color = o.cmap[-1],
                )
                lut = vedo.build_lut(**args, interpolate=True)
                mesh.cmap(lut)

            # enable picking iff we have a pick callback
            mesh.pickable(viz.on_pick is not None)
                
            # add pop-up label if requested
            if viz.label is not None:
                self.add_hint(mesh, str(viz.label), size=36, delay=0)

            # show the mesh
            self.add(mesh)

        elif o is None:
            pass

        else:
            print(f"traverse: don't understand {type(o)}; skipping")

        if actor is not None and viz is not None and viz.on_pick is not None:
            actor.pickable(True)
            actor.picking = viz.on_pick


    # TODO: calling close from a callback may be unsafe
    # but it closes the window, whereas self.break_interaction followed by a close doesn't
    def close_in(self, dt):
        self.timer_callback("create", dt=dt, one_shot=True)
        self.add_callback("timer", lambda *_: self.close())


"""
pv.global_theme.allow_empty_mesh = True
pv.global_theme.color = (1.0, 0.97, 0)
pv.global_theme.edge_color = (0.5,0.5,0.5)
pv.global_theme.line_width = 3
pv.global_theme.point_size = 8
pv.global_theme.window_size = (1500, 1500)
"""

def show(*args, axes=1, title="viz", close_in=None, **kwargs):
    pl = Plotter(title=title, axes=axes)
    pl.traverse(args)
    if close_in is not None: pl.close_in(close_in)
    return pl.show(interactive=True, axes=axes)


def diff(o1, o2, title="diff", scheme=0, alpha=0.25):

    schemes = [
        (0, 250, 0), # green/red(dish) https://davidmathlogic.com/colorblind/#%2300FA00-%23FA00FA
        (0, 100, 250), # blue/orange https://davidmathlogic.com/colorblind/#%23FA9600-%230064FA
        (100, 75, 250), # purple/yellow https://davidmathlogic.com/colorblind/#%23E1FA4B-%23644BFA
    ]
    
    # complementary colors
    c1 = schemes[scheme]
    c2 = tuple(250 + min(c1) - c for c in c1)

    # completely opaque doesn't work
    alpha = min(alpha, 0.99)

    return [viz(o1).c(c1).alpha(alpha), viz(o2).c(c2).alpha(alpha)]


def show_diff(o1, o2, title="diff"):

    if isinstance(title, tuple):
        title = f"green: {title[0]}    |    red: {title[1]}"

    show(diff(o1, o2), title=title)

             
def compare(name, old, new):

    # TODO: this seems to avoid a segfault
    # maybe a race condition related to window closing from the step that is calling us
    # and then this function opening a window?
    time.sleep(1)

    def render(o):
        pl = pv.Plotter(off_screen=True)
        flatten = lambda l: sum(map(flatten,l),[]) if isinstance(l,list) else [l]
        for oo in flatten(o):
            if isinstance(oo, pv.PolyData):
                pl.add_mesh(oo, color=(0.5,0.5,0.5), opacity=0.25)
            else:
                # TODO: need something better for non-mesh objects
                pl.add_text(str(oo), color=(0.5,0.5,0.5)) #, opacity=0.25)
        img_array = pl.screenshot(return_img=True, window_size=(1000,1000))[:,:,0]
        return img_array
    img_old, img_new = render(old), render(new)

    if not (img_old==img_new).all():

        histogram, _ = np.histogram(abs(img_old-img_new), range=(0,255), bins=255)
        summary = "change histogram " + " ".join([str(h) for h in histogram[0:5]]) + " ... " + str(sum(histogram[6:]))
        print(summary)

        accepted = False
        def accept(button, *args):
            nonlocal accepted
            accepted = True
            print("change accepted")
            # TODO: this may be unsafe
            # it seemed to cause segfaults in molds.py
            # to fix changed to break_interaction
            # but then window never gets closed...
            button.plotter.close()

        viz.show(
            #viz.diff(diff_value(old_value), diff_value(new_value)),
            #viz.diff(diff_value("ab1"), diff_value("ab2")),
            diff(old, new),
            viz.Button(accept, "Accept"),
            title = f"{name}: new (red) differs from old (green)",
        )

        if not accepted:
            raise Exception(f"TEST FAILS: {name} changed")



#
# Tried reusing the Plotter, doing a clear between each use
# but did not understand behavior so backed out
#

# for debugging
digest_count = 0

@lib.timefun
def polydata_digest(o):
    digestive = pv.Plotter(off_screen=True)
    for oo in lib.flatten(o):
        digestive.add_mesh(oo, color=(0.5,0.5,0.5), opacity=0.25)
    img_array = digestive.screenshot(return_img=True, window_size=(1000,1000))[:,:,0]
    if True:
        import PIL
        img = PIL.Image.fromarray(img_array)
        global digest_count
        img.save(f"/tmp/digest{digest_count}.png")
        digest_count += 1
    img_bytes = img_array.tobytes()
    digest = lib.digest(img_bytes)
    return digest

pv.PolyData.digest = polydata_digest


def normals(o, scale=1):

    # only if triangulated
    # this computes normals based on point order in triangle
    # whereas o.cell_normals computes normals based on a consistent ordering,
    # and also modifies o (I think)
    def cell_normals(o):
        for face in o.regular_faces:
            p1, p2, p3 = o.points[face]
            cross = np.cross(p2-p1, p3-p1)
            yield cross / la.norm(cross)

    try:
        p0s = np.array([np.average(cell.points, axis=0) for cell in o.cell])
        p1s = np.array([p0 + scale * n for n, p0 in zip(cell_normals(o), p0s)])
        ps = np.concatenate([p0s, p1s], axis=0)
        lines = np.hstack([[2, i, i+len(p0s)] for i in range(len(p0s))])
        verts = [len(p0s)] + list(range(len(p0s)))
        normals = pv.PolyData(ps, lines=lines, verts=verts).c((0.0,0.0,1.0)).lw(3)
        return normals
    except Exception as e:
        print("can't get normals", e)
        return pv.PolyData()

def points(points):
    #return pv.PolyData(points, verts=range(len(points))).c((0.0,0.0,1.0))
    if len(points.shape) == 1:
        points = [points]
    verts = [len(points)] + list(range(len(points)))
    return pv.PolyData(points, verts=verts).c((0.0,0.0,1.0)).ps(12)


#
#
#

class Animation:

    def __init__(self, run, fps=10):

        import vedo
        import vtk

        self.data = None
        self.stopping = False
        self.closing = False
        self.model = None

        self.plotter = Plotter(title="animation")

        # starting a separate thread to generate the sequence of models to be displayed
        threading.Thread(target = lambda: run(self)).start()

        # we can't call underlying vtk from within anything but the main thread
        # (get deadlocks, segfaults, illegal instruction, etc.)
        # so we arrange to do all state changes from within a callback that runs periodically

        self.last_time = None
        self.actual_fps = fps

        def callback(_):

            # while user has provided something to render
            if self.model is not None:

                # calculate actual_fps for display
                if self.last_time is not None:
                    actual_fps = 0.8 * self.actual_fps + 0.2 / (time.time() - self.last_time)
                self.last_time = time.time()

                # add the model to the plotter
                self.plotter.clear(deep=True)
                self.plotter.traverse(viz(self.model()), f"{self.actual_fps:.1f} fps")
                self.plotter.reset_camera()

                # finish up if requested
                if self.stopping:
                    self.plotter.timer_callback("stop", timer_id=self.timer_id)
                    self.plotter.add_axes(self.model)
                    self.model = None
                    self.stopping = False

                #self.plotter.zoom(self.zoom)
                self.plotter.render()

                # give the data generator a chance to run                
                time.sleep(1/fps)

            else:

                # model is no longer updating, so just poll frequently for smooth interaction
                time.sleep(0.01)

            if self.closing:
                print("callback closing")
                self.plotter.close()
                self.closing = False

        # arrange for callback to be called periodically
        self.plotter.add_callback("timer", callback)
        self.timer_id = self.plotter.timer_callback("start", dt=0)

        # show window
        print("showing")
        self.plotter.show()
        time.sleep(1) # TODO: needed?


    def stop(self, close=False):
        print(f"display stopping, close={close}")
        self.stopping = True
        if close:
            self.closing = True
if __name__ == "__main__":

    import lib
    import sys
    pd = lib.load(sys.argv[1])
    show(pd, title=sys.argv[1])
    
