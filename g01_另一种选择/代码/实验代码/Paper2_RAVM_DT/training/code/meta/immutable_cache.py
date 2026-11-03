"""Bounded decoding only; no prediction, score, action, or audit verdict cache."""
from pathlib import Path
from collections import OrderedDict
from io import BytesIO
import ctypes,hashlib,json,threading
from ctypes import wintypes
import numpy as np
LOCK=threading.RLock();WEIGHTS=OrderedDict();ARRAYS=OrderedDict();WEIGHT_CAP=32*2**20;ARRAY_CAP=16*2**20
HASHES=OrderedDict()
K=ctypes.WinDLL('kernel32',use_last_error=True)
K.CreateFileW.argtypes=[wintypes.LPCWSTR,wintypes.DWORD,wintypes.DWORD,wintypes.LPVOID,wintypes.DWORD,wintypes.DWORD,wintypes.HANDLE];K.CreateFileW.restype=wintypes.HANDLE
K.CloseHandle.argtypes=[wintypes.HANDLE]
def pin(path):
    h=K.CreateFileW(str(Path(path).resolve()),0x80000000,1,None,3,0x80,None)
    if h==ctypes.c_void_p(-1).value:raise ctypes.WinError(ctypes.get_last_error())
    return h
def evict(cache,cap):
    while sum(v['bytes'] for v in cache.values())>cap:
        _,old=cache.popitem(last=False);K.CloseHandle(old['pin'])
def state_dict(path,expected):
    import torch
    key=str(Path(path).resolve())
    with LOCK:
        if key not in WEIGHTS:
            h=pin(path)
            try:
                wire=Path(path).read_bytes();assert hashlib.sha256(wire).hexdigest()==expected
                data=torch.load(BytesIO(wire),map_location='cpu',weights_only=True)
                WEIGHTS[key]=dict(pin=h,bytes=len(wire),hash=expected,data=data)
                evict(WEIGHTS,WEIGHT_CAP)
            except BaseException:K.CloseHandle(h);raise
        value=WEIGHTS[key];assert value['hash']==expected;WEIGHTS.move_to_end(key)
        # load_state_dict(assign=False) copies into the caller's independent model.
        return {name:tensor.clone() for name,tensor in value['data'].items()}
def verified(path,expected):
    key=str(Path(path).resolve())
    with LOCK:
        if key in WEIGHTS:return WEIGHTS[key]['hash']==expected
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()==expected

def hash_file(path):
    """Reuse a digest only while Windows denies writes/deletion on that exact file."""
    key=str(Path(path).resolve())
    with LOCK:
        if key not in HASHES:
            handle=pin(path)
            try:
                digest=hashlib.sha256()
                with Path(path).open('rb')as f:
                    for chunk in iter(lambda:f.read(4*2**20),b''):digest.update(chunk)
                HASHES[key]=(handle,digest.hexdigest())
                while len(HASHES)>512:_,old=HASHES.popitem(last=False);K.CloseHandle(old[0])
            except BaseException:K.CloseHandle(handle);raise
        HASHES.move_to_end(key);return HASHES[key][1]
def arrays(path):
    key=str(Path(path).resolve())
    with LOCK:
        if key not in ARRAYS:
            h=pin(path)
            try:
                with np.load(path,allow_pickle=False) as z:
                    x,y,ts=z['x'],z['y'],z['ts'];meta=str(z['meta'])
                for a in (x,y,ts):a.flags.writeable=False
                data=(x,y,ts,meta);ARRAYS[key]=dict(pin=h,bytes=sum(a.nbytes for a in (x,y,ts)),data=data)
                evict(ARRAYS,ARRAY_CAP)
            except BaseException:K.CloseHandle(h);raise
        value=ARRAYS[key];ARRAYS.move_to_end(key)
        x,y,ts,meta=value['data'];return x,y,ts,json.loads(meta)
