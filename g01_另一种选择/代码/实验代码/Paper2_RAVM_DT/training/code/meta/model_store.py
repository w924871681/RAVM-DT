"""Exact-byte global CAS with hash deduplication and explicit disk reserves."""
from common import ROOT as META,STORAGE,sha
from pathlib import Path
import hashlib,io,os,shutil,threading,time
MUTEX=threading.RLock()
BLOB_ROOTS=[(STORAGE/'CAS',8*2**30)]
class ModelStore:
    def __init__(self,path):
        self.root=Path(path)
        if self.root.is_relative_to(STORAGE):self.root.mkdir(parents=True,exist_ok=True)
    def find(self,h):
        if not isinstance(h,str) or len(h)!=64 or any(c not in '0123456789abcdef' for c in h):raise ValueError('Invalid blob hash')
        for path in [self.root/h]+[root/h[:2]/h for root,_ in BLOB_ROOTS]:
            if path.is_file():return path
        raise FileNotFoundError(h)
    def put(self,model):
        import torch
        from core.models import state_hash
        assert self.root.is_relative_to(STORAGE),'Old source CAS is read-only'
        stream=io.BytesIO();torch.save({k:v.detach().cpu().contiguous() for k,v in model.state_dict().items()},stream)
        wire=stream.getvalue();h=hashlib.sha256(wire).hexdigest()
        with MUTEX:
            try:path=self.find(h)
            except FileNotFoundError:
                choices=[]
                while not choices:
                    for root,reserve in BLOB_ROOTS:
                        drive=Path(root.anchor)
                        if drive.exists() and shutil.disk_usage(drive).free>=reserve+len(wire)+64*2**20:choices.append(root)
                    if not choices:
                        from common import atomic
                        atomic(META/'progress/STORAGE_WAIT.json',dict(status='WAITING_FOR_STORAGE',candidate_kept_in_memory=True,no_optimizer_replay=True,at_unix=time.time()))
                        time.sleep(3)
                path=choices[0]/h[:2]/h;path.parent.mkdir(parents=True,exist_ok=True)
                with path.open('xb') as f:f.write(wire);f.flush();os.fsync(f.fileno())
            assert path.stat().st_size==len(wire) and sha(path)==h
            (META/'progress/STORAGE_WAIT.json').unlink(missing_ok=True)
        return dict(file_sha256=h,state_hash=state_hash(model),bytes=len(wire),storage_path=str(path))
    def verify(self,h):
        from immutable_cache import verified
        try:return verified(self.find(h),h)
        except (OSError,ValueError):return False
    def load(self,entry):
        import torch
        from core.models import DEVICE,state_hash
        import bridge
        from immutable_cache import state_dict
        path=self.find(entry['file_sha256']);assert path.stat().st_size==entry['bytes']
        if 'storage_path' in entry:assert Path(entry['storage_path']).resolve()==path.resolve()
        model=bridge.Forecast().to(DEVICE);model.load_state_dict(state_dict(path,entry['file_sha256']))
        assert state_hash(model)==entry['state_hash'];return model
