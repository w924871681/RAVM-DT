"""Bounded immutable input reconstruction cache; never caches a scientific score."""
from collections import OrderedDict
import threading
LOCK=threading.RLock();CACHE=OrderedDict();CAP=32*2**20
class InputCacheView(dict):
    def key(self,k):
        import bridge
        return (str(bridge.current_root()),k)
    def __contains__(self,k):
        if super().__contains__(k):return True
        with LOCK:
            key=self.key(k)
            if key in CACHE:
                value,_=CACHE[key];CACHE.move_to_end(key);super().__setitem__(k,value);return True
        return False
    def __setitem__(self,k,value):
        n=0;seen=set()
        for u,v,rb,hb in value.values():
            for s in [u,v,*rb,*hb]:
                if id(s) in seen:continue
                seen.add(id(s))
                for a in (s.x,s.y,s.ts):a.flags.writeable=False;n+=a.nbytes
        super().__setitem__(k,value)
        with LOCK:
            CACHE[self.key(k)]=(value,n);CACHE.move_to_end(self.key(k))
            while sum(x[1] for x in CACHE.values())>CAP:CACHE.popitem(last=False)
