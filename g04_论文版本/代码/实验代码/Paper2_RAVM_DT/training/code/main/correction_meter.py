from r_common import *
import ast,inspect,contextlib,threading
def call_key(kind, identity):
    return json.dumps(dict(kind=kind, identity=identity), sort_keys=True)

def load_accounting(path):
    begins, ends = {}, {}
    for line in path.read_text(encoding='utf8').splitlines():
        e = json.loads(line)
        assert e['kind'] in ('warmup', 'candidate')
        key = e['call_id']
        assert json.loads(key)['kind'] == e['kind']
        if e['event'] == 'BEGIN':
            assert key not in begins and key not in ends
            assert e['reserved_steps'] == 50
            assert call_key(e['kind'], e['identity']) == key
            begins[key] = e
        else:
            assert e['event'] == 'DONE' and key in begins and key not in ends
            assert e['steps'] == 50 and begins[key]['kind'] == e['kind']
            ends[key] = e
    assert set(begins) == set(ends), 'Unmatched optimizer reservation: no automatic replay'
    done = {k: sum(e['steps'] for e in ends.values() if e['kind'] == k)
            for k in ('warmup', 'candidate')}
    return begins, done

class RecoveryMeter:
    """Original step instrumentation, initialized from a fully reconciled durable ledger."""
    def __init__(self, api, phase, restored_calls, restored_steps):
        self.phase=phase;self.mutex=threading.RLock();self.tls=threading.local();self.done=dict(restored_steps);self.reserved=self.done.copy();self.used=set(restored_calls)
        self.path=ROOT/f'process/{phase}_SGD_CALL_ACCOUNTING.jsonl'
        self.limits={'warmup':800 if phase=='Smoke' else 4800,'candidate':403200 if phase=='Smoke' else 4147200}
        self.total_limit=404000 if phase=='Smoke' else 4152000;self.original=api.optim.SGD.step;owner=self
        def step(optimizer,*a,**kw):
            kind=owner.tls.kind
            with owner.mutex:
                if owner.tls.count%50==0:
                    identity={**owner.tls.identity,'phase':phase,'stage':owner.tls.count//50+1} if kind=='candidate' else {**owner.tls.identity,'phase':phase}
                    key=json.dumps(dict(kind=kind,identity=identity),sort_keys=True);assert key not in owner.used
                    assert owner.reserved[kind]+50<=owner.limits[kind] and sum(owner.reserved.values())+50<=owner.total_limit
                    owner.used.add(key);owner.reserved[kind]+=50;owner.tls.key=key;owner.append(dict(event='BEGIN',call_id=key,kind=kind,reserved_steps=50,identity=identity))
            answer=owner.original(optimizer,*a,**kw)
            with owner.mutex:
                owner.tls.count+=1;owner.done[kind]+=1
                if owner.tls.count%50==0:owner.append(dict(event='DONE',call_id=owner.tls.key,kind=kind,steps=50))
            return answer
        api.optim.SGD.step=step

    def append(self,event):
        with self.path.open('a',encoding='utf8') as f:f.write(json.dumps(dict(**event,at_unix=time.time()))+'\n');f.flush();os.fsync(f.fileno())

    @contextlib.contextmanager
    def scope(self,kind,identity):
        self.tls.kind=kind;self.tls.identity=identity;self.tls.count=0
        try:yield
        finally:self.tls.kind=None

def check_instrumentation():
    import textwrap
    original = ast.parse((ROOT / 'new_runner.py').read_text(encoding='utf8'))
    original_step = next(n for n in ast.walk(original) if isinstance(n, ast.FunctionDef) and n.name == 'step')
    recovery = ast.parse(textwrap.dedent(inspect.getsource(RecoveryMeter.__init__)))
    recovered_step = next(n for n in ast.walk(recovery) if isinstance(n, ast.FunctionDef) and n.name == 'step')
    assert ast.dump(original_step, include_attributes=False) == ast.dump(recovered_step, include_attributes=False)
    for name in ('append', 'scope'):
        cls = next(n for n in original.body if isinstance(n, ast.ClassDef) and n.name == 'Meter')
        old = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
        new = ast.parse(textwrap.dedent(inspect.getsource(getattr(RecoveryMeter, name)))).body[0]
        assert ast.dump(old, include_attributes=False) == ast.dump(new, include_attributes=False)
    return True
