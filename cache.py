import enum
import inspect
import lib
import numpy as np
import os
import pickle
import pprint
import ruamel.yaml
import time

yaml = ruamel.yaml.YAML(typ='safe', pure=True)

class Cache:

    # wrap cache values in a separate object Cache.v so we can keep value namespace
    # clear of other cache method like Cache.step, start, finish, etc.
    class Value:

        def __init__(self, cache):
            self._cache = cache

        def __setattr__(self, name, value):
            if not name.startswith("_"):
                if self._cache.outs is not None:
                    self._cache.outs[name] = value
            super().__setattr__(name, value)

        def __getattribute__(self, name):
            if not name.startswith("_"):
                if self._cache.ins is not None and not name in self._cache.outs:
                    self._cache.ins.add(name)
            return super().__getattribute__(name)

    # useful for debugging changes
    class Old:

        def __init__(self, cache):
            self._cache = cache

        def __getattribute__(self, name):
            if not name.startswith("_"):
                old_fn = self._cache.full(self._cache.pickle_name(name))
                with open(old_fn, "rb") as f:
                    return pickle.load(f)
            else:
                return super().__getattribute__(name)


    def __init__(self, cache_name, **kwargs):

        self.cache_name = cache_name
        self.cache_dn = f".cache/{self.cache_name}"
        self.info_fn = self.full("_info")
        self.digests = {}
        self.ins = self.outs = None
        self.previous_outs = set()
        self.v = self.Value(self)
        self.old = self.Old(self)
        self.force = []
        self.debug = []
        self.auto = False
        self.tester = None

        # ensure cache dir exists
        os.makedirs(self.cache_dn, exist_ok=True)

        # read current info (if any)
        # TODO: avoid eval?
        if os.path.exists(self.info_fn):
            with open(self.info_fn, "r") as f:
                self.info = yaml.load(f)
        else:
            print("new cache")
            self.info = {}

        # process initital values passed in as kwargs
        if len(kwargs):
            @self.step(force=True)
            def _init():
                for name, value in kwargs.items():
                    setattr(self.v, name, value)

        self.clean()

    def get(self, path, default, typ=None):
        x = self.v
        path = path.split(".")
        for name in path:
            if not hasattr(x, name):
                return default
            x = getattr(x, name)
        return typ(x) if typ is not None else x

        

    def full(self, name):
        return f"{self.cache_dn}/{name}"

    def clean(self):
        known = set()
        for step in self.info.values():
            for out in step["outputs"]:
                known.add(out + ".pickle")
        existing = {e.name for e in os.scandir(self.cache_dn) if e.name.endswith(".pickle")}
        for name in existing - known:
            print(f"{name} not known; removing")
            os.remove(self.full(name))

    def step(self, *args, **kwargs):
        def wrapper(fun):
            self.step_name = fun.__name__
            self.step_debug = \
                (("debug" in kwargs and kwargs["debug"]) or "all" in self.debug or self.step_name in self.debug) \
                and not "none" in self.debug
            self.step_force = \
                (("force" in kwargs and kwargs["force"]) or "all" in self.force or self.step_name in self.force) \
                and not "none" in self.force
            self.skip_test = "skip_test" in kwargs and kwargs["skip_test"]
            self.process_step(fun, extra_funs=list(args))
        return wrapper

    def start(self):
        self.start_time = time.time()

    def finish(self, msg):
        print(f"{msg} took {(time.time()-self.start_time)*1000:.1f} ms")
        self.finished = True
        self.start_time = time.time()

    def pickle_name(self, name):
        return f"{name}.pickle"

    def process_step(self, fun, extra_funs):

        # "digest" marker for undefined values
        # e.g. we checked if a value exists so our step depends on it,
        # but it didn't exist so it didn't have an actual digest
        UNDEFINED = "UNDEFINED"

        # get source
        fun_source = inspect.getsource(fun).split("\n", 1)[1]
        fun_source = "\n".join((fun_source, *(inspect.getsource(f) for f in extra_funs)))
        fun_source = fun_source.encode("utf-8")
        fun_source = lib.digest(fun_source)

        # get step info
        ok = True
        step_info = self.info[self.step_name] if self.step_name in self.info else {}
        if self.step_force:
            reason = "forced"
            ok = False
        elif self.step_name in self.info:
            if fun_source != step_info["source"]:
                reason = "source code changed"
                ok = False
            for inp, old_sig in step_info["inputs"].items():
                if inp not in self.digests:
                    if old_sig != UNDEFINED:
                        reason = f"input {inp} disappeared"
                        ok = False
                elif (new_sig := self.digests[inp]) != old_sig:
                    reason = f"{inp} changed"
                    ok = False
            for name in step_info["outputs"]:
                fn = self.full(self.pickle_name(name))
                if not os.path.exists(fn):
                    reason = f"output {name} disappeared"
                    ok = False
        else:
            reason = "no info"
            ok = False

        if ok:

            # inputs and code didn't change since last time we called fun
            # and we have step_info, so just load cached outputs
            print(f"\n=== not executing {self.step_name}")
            for name, digest in step_info["outputs"].items():
                fn = self.pickle_name(name)
                print(f"loading {fn}")
                with open(self.full(fn), "rb") as f:
                    value = pickle.load(f)
                    setattr(self.v, name, value)
                self.digests[name] = digest
                self.previous_outs.add(name)

        else:

            # something changed so we have to re-run, recording our inputs and outputs
            print(f"\n=== executing step {self.step_name} ({reason})")
            self.ins = set()
            self.outs = dict()
            self.finished = False
            self.start()
            try:
                fun()
            except SystemExit:
                print("exit called")
                raise
            except Exception as e:
                print(f"ERROR: step {self.step_name} threw {e}")
                raise
            if not self.finished:
                self.finish(f"step {self.step_name}")

            # for each output, compute digest, compare to old if requested, save pickle if ok
            for name, value in self.outs.items():

                # check to make sure we're not trying to modify a value defined in a previous step
                if name in self.previous_outs:
                    raise Exception(f"step {self.step_name} attempting to change {name}")

                # compute pickled output and digest
                # if the object has a .digest() method use that,
                # else digest the pickle
                pickled = pickle.dumps(value)
                if hasattr(value, "digest"):
                    digest = value.digest()
                else:
                    digest = lib.digest(pickled)
                self.digests[name] = digest
                #print(f"{name} digest: {digest}")
                    
                # compare digests for change, and if available
                # call self.tester for a closer look if digests have changed
                if "outputs" in step_info and name in step_info["outputs"]:
                    old_digest = step_info["outputs"][name] 
                    if digest != old_digest:
                        print(f"{name} changed")
                        if self.tester is not None:
                            fn = self.pickle_name(name)
                            print(f"loading {fn} for comparison")
                            with open(self.full(fn), "rb") as f:
                                old_value = pickle.load(f)
                            print("calling self.tester to take a closer look")
                            self.tester(name, old_value, value)

                # write pickle file if no complaint from tester
                pickle_fn = self.pickle_name(name)
                print(f"saving {pickle_fn}")
                with open(self.full(pickle_fn), "wb") as f:
                    f.write(pickled)

                # caution against changing down the line
                self.previous_outs.add(name)

            # having successfuly checked and saved our outputs,
            # now we can record input digests so we don't run this step again
            # if inputs don't change
            self.info[self.step_name] = {
                "source": fun_source,
                "inputs": {
                    # 0 means we checked to see if attr existed but it didn't
                    # by emitting 0 as the digest we will be sure to be executed if it does exist in the future
                    name: (self.digests[name] if name in self.digests else UNDEFINED) for name in self.ins
                },
                "outputs": {
                    name: self.digests[name] for name in self.outs.keys()
                }
            }
            with open(self.info_fn, "w") as f:
                yaml.dump(self.info, f)

            # reset
            self.ins = self.outs = None

    def dbg(self, name, *args, **kwargs):
        if not self.step_debug:
            print(f"not showing {self.step_name} because debug mode is not enabled")
        else:
            import viz
            title = f"debug: {self.step_name}: {name}"
            print("showing", title)
            viz.show(title = title, *args, **kwargs)

