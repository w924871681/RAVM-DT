"""Validate the original engine's float32 reporting without changing science.

The native engine reports float(can.mean()-cur.mean()) on NumPy float32
arrays. Portable result files intentionally retain the frozen scalar difference
float(can.mean())-float(cur.mean()). These are two distinct representations.
Do not compare them using an arbitrary inter-representation tolerance.
"""
import ast,hashlib,math
from pathlib import Path
import numpy as np


def validate_native_reporting(row):
    parent=row['current_service_mse'];candidate=row['candidate_service_mse']
    assert math.isfinite(parent) and math.isfinite(candidate)
    assert float(np.float32(parent))==parent and float(np.float32(candidate))==candidate
    delta=float(np.float32(candidate)-np.float32(parent))
    assert math.isfinite(delta)
    assert row['delta_srv']==delta,'Native delta differs from original float32 operation'
    expected=max(delta,0.) if row['commit'] else 0.
    assert row['downside']==expected,'Native downside differs from original exact operation'
    assert bool(row['harmful_commit'])==bool(row['commit'] and delta>0)
    return True


def install(b):
    if getattr(b,'native_reporting_precision_installed',False):return
    src=b.PRIOR/'legacy_adapter.py';tree=ast.parse(src.read_text('utf-8'))
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='persist')
    target="abs(row['downside'] - (max(delta, 0.0) if row['commit'] else 0.0)) <= 1e-09"
    found=[n for n in node.body if isinstance(n,ast.Assert) and ast.unparse(n.test)==target]
    assert len(found)==1,'Frozen reporting guard changed; review required'
    found[0].test=ast.parse('validate_native_reporting(row)',mode='eval').body
    # Retain all existing adapter transformations and every downstream output expression.
    node=b.LegacyTransform().visit(node)
    b.validate_native_reporting=validate_native_reporting
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(src),'exec'),b.__dict__)
    original_run=b.run_legacy
    def run_legacy(*args,**kw):
        rows=original_run(*args,**kw)
        phase,center,seed,stream,arm=args[:5]
        rp=b.ROOT/f'science/{phase}/{center}/{seed}/{stream}/{arm["id"]}/NATIVE_EXACTNESS_RECEIPT.json'
        receipt=b.read(rp)
        receipt['common_measurement']='Frozen scalar mean difference retained; native float32 delta/downside reconstructed exactly in original dtype. No cross-representation tolerance.'
        receipt['reporting_precision_adapter_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        b.atomic(rp,receipt)
        return rows
    b.run_legacy=run_legacy
    b.native_reporting_precision_installed=True
