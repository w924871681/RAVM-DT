"""Verify immutable training assets once and transfer OS read locks to workers.

Windows deny-write/delete handles keep verified bytes immutable. Workers duplicate
the exact handles from the identity-bound parent, validate resolved path and size,
and retain their own locks. No timestamp-only hash cache or unpinned shortcut.
"""
import ctypes,hashlib,json,os
from ctypes import wintypes as w
from pathlib import Path
B=Path(__file__).resolve().parent/'training'
K=ctypes.WinDLL('kernel32',use_last_error=True)
K.CreateFileW.argtypes=[w.LPCWSTR,w.DWORD,w.DWORD,w.LPVOID,w.DWORD,w.DWORD,w.HANDLE];K.CreateFileW.restype=w.HANDLE
K.GetCurrentProcess.restype=w.HANDLE
K.OpenProcess.argtypes=[w.DWORD,w.BOOL,w.DWORD];K.OpenProcess.restype=w.HANDLE
K.CloseHandle.argtypes=[w.HANDLE]
K.GetProcessTimes.argtypes=[w.HANDLE,*([ctypes.c_void_p]*4)]
K.GetFileSizeEx.argtypes=[w.HANDLE,ctypes.POINTER(ctypes.c_longlong)]
K.GetFinalPathNameByHandleW.argtypes=[w.HANDLE,w.LPWSTR,w.DWORD,w.DWORD]
K.DuplicateHandle.argtypes=[w.HANDLE,w.HANDLE,w.HANDLE,ctypes.POINTER(w.HANDLE),w.DWORD,w.BOOL,w.DWORD]
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def stamp(handle):
    times=[ctypes.c_ulonglong() for _ in range(4)]
    if not K.GetProcessTimes(handle,*[ctypes.byref(t) for t in times]):raise ctypes.WinError(ctypes.get_last_error())
    return str(times[0].value)
def pin(path):
    handle=K.CreateFileW(str(path.resolve()),0x80000000,1,None,3,0x80,None)
    if handle==ctypes.c_void_p(-1).value:raise ctypes.WinError(ctypes.get_last_error())
    return handle
def identity(handle,path,size):
    name=ctypes.create_unicode_buffer(32768);n=K.GetFinalPathNameByHandleW(handle,name,len(name),0)
    actual=name.value
    if actual.startswith('\\\\?\\'):actual=actual[4:]
    length=ctypes.c_longlong()
    if not n or n>=len(name) or not K.GetFileSizeEx(handle,ctypes.byref(length)):raise ctypes.WinError(ctypes.get_last_error())
    if os.path.normcase(str(path.resolve()))!=os.path.normcase(actual) or length.value!=size:raise RuntimeError('Pinned asset path/size differs')

class AssetPins:
    def __init__(self,receipt):
        self.handles=[];manifest=B/'ASSET_MANIFEST.json';h=pin(manifest);self.handles.append(h)
        self.manifest=json.loads(manifest.read_text('utf-8-sig'));records=[]
        for row in self.manifest['files']:
            path=B/row['file']
            if not path.resolve().is_relative_to(B.resolve()):raise RuntimeError('Asset outside package')
            h=pin(path);self.handles.append(h);identity(h,path,row['bytes'])
            if sha(path)!=row['sha256']:raise RuntimeError('Immutable asset changed: '+row['file'])
            records.append(dict(row,handle=int(h)))
        result=dict(status='PASS_FULL_ASSET_SHA256_WITH_HELD_OS_READ_LOCKS',pid=os.getpid(),creation_filetime=stamp(K.GetCurrentProcess()),
                    manifest_sha256=sha(manifest),manifest_handle=int(self.handles[0]),manifest_bytes=manifest.stat().st_size,
                    implementation_sha256=sha(Path(__file__)),files=records)
        receipt.parent.mkdir(parents=True,exist_ok=True);receipt.write_text(json.dumps(result),'utf-8')
    def close(self):
        for h in self.handles:K.CloseHandle(h)
        self.handles=[]

class InheritedAssetPins:
    def __init__(self,receipt):
        self.handles=[];r=json.loads(receipt.read_text('utf-8-sig'));parent=K.OpenProcess(0x40|0x1000,False,r['pid'])
        if not parent:raise ctypes.WinError(ctypes.get_last_error())
        try:
            if r['status']!='PASS_FULL_ASSET_SHA256_WITH_HELD_OS_READ_LOCKS' or stamp(parent)!=r['creation_filetime']:raise RuntimeError('Asset verifier process identity differs')
            if sha(Path(__file__))!=r['implementation_sha256']:raise RuntimeError('Asset verifier implementation differs')
            def duplicate(handle,path,size):
                dest=w.HANDLE()
                if not K.DuplicateHandle(parent,handle,K.GetCurrentProcess(),ctypes.byref(dest),0,False,2):raise ctypes.WinError(ctypes.get_last_error())
                self.handles.append(dest.value);identity(dest.value,path,size)
            manifest=B/'ASSET_MANIFEST.json';duplicate(r['manifest_handle'],manifest,r['manifest_bytes'])
            if sha(manifest)!=r['manifest_sha256']:raise RuntimeError('Asset manifest changed')
            self.manifest=json.loads(manifest.read_text('utf-8-sig'))
            if len(self.manifest['files'])!=len(r['files']):raise RuntimeError('Pinned asset coverage differs')
            for original,record in zip(self.manifest['files'],r['files']):
                if any(record[k]!=v for k,v in original.items()):raise RuntimeError('Pinned asset identity differs')
                duplicate(record['handle'],B/original['file'],original['bytes'])
        finally:K.CloseHandle(parent)
    def close(self):
        for h in self.handles:K.CloseHandle(h)
        self.handles=[]
