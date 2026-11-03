"""Read-only checks for the alternative Alibaba workflow; never trains or changes files."""
import argparse
import ast
import hashlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PAPER = HERE.parent / 'g04_论文版本' / '代码' / '实验代码' / 'Paper2_RAVM_DT'
T = PAPER / 'training'
EXPECTED_ARCHIVE = '3e6ee87fd204bb85b9e234c5c75a5096580fdabc8f085b224033080090753a7a'
SOURCE_FILES = [HERE / name for name in ('select_and_prepare.py', 'fit_source_models.py',
                'prepare_initial_models.py', 'run_new_machines.py')] + [
    T / 'run.py', T / 'code/recipes/historical_pretrain.py',
    T / 'code/meta/bridge.py', T / 'code/main/runtime/cmc_original_gate_adapter.py',
    T / 'code/main/runtime/core/models.py', T / 'code/main/runtime/core/protocol.py',
    T / 'code/main/runtime/core/v2_engine.py', T / 'code/main/runtime/core/engine.py',
    PAPER / 'legacy_reporting_precision.py']


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def metadata(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--role', choices=('source', 'lifecycle'), required=True)
    p.add_argument('--archive', type=Path, help='Optional official archive path; existence checked only')
    p.add_argument('--check-archive-sha256', action='store_true', help='Read the whole archive to verify SHA-256')
    p.add_argument('--selection', type=Path)
    p.add_argument('--sources', type=Path)
    p.add_argument('--initials', type=Path)
    a = p.parse_args()
    if a.check_archive_sha256 and not a.archive:
        p.error('--check-archive-sha256 requires --archive')
    issues = []
    for f in SOURCE_FILES:
        if not f.is_file():
            issues.append('Missing code: ' + str(f.relative_to(HERE.parent)))
            continue
        try:
            ast.parse(f.read_text(encoding='utf-8-sig'), filename=str(f))
        except (SyntaxError, UnicodeError) as exc:
            issues.append('Invalid source syntax: ' + f.name + ': ' + str(exc))
    deps = ['numpy', 'pandas', 'scipy', 'torch']
    if a.role == 'lifecycle':
        deps += ['cryptography', 'PyYAML']
    versions = {name: metadata(name) for name in deps}
    issues.extend('Missing dependency: ' + name for name, value in versions.items() if value is None)
    expected_python = '3.11.15' if a.role == 'source' else '3.13.5'
    expected_torch = '2.5.1+cu121' if a.role == 'source' else '2.13.0+cu126'
    warnings = []
    if platform.python_version() != expected_python:
        warnings.append('Python differs from recorded version ' + expected_python)
    if versions['torch'] and versions['torch'] != expected_torch:
        warnings.append('PyTorch differs from recorded version ' + expected_torch)
    if a.archive:
        if not a.archive.is_file():
            issues.append('Archive path not found')
        elif a.check_archive_sha256 and digest(a.archive) != EXPECTED_ARCHIVE:
            issues.append('Archive SHA-256 differs from the registered value')
    try:
        sel_path = a.selection / 'selection.json' if a.selection else None
        src_path = a.sources / 'source_models.json' if a.sources else None
        ini_path = a.initials / 'initial_models.json' if a.initials else None
        sel = json.loads(sel_path.read_text(encoding='utf-8')) if sel_path else None
        src = json.loads(src_path.read_text(encoding='utf-8')) if src_path else None
        ini = json.loads(ini_path.read_text(encoding='utf-8')) if ini_path else None
        if sel is not None:
            rows = sel['machines']
            if len(rows) != 100 or len({r['center_id'] for r in rows}) != 100:
                issues.append('Selection must contain 100 distinct center IDs')
            missing = sum(not (a.selection / 'data' / ('center_%s.npz' % r['center_id'])).is_file() for r in rows)
            if missing:
                issues.append('Missing selected input arrays: ' + str(missing))
        if src is not None:
            if src.get('status') != 'NEW_SELECTION_SOURCE_MODELS_TRAINED':
                issues.append('Source manifest is not complete')
            if sel_path and src.get('selection_sha256') != digest(sel_path):
                issues.append('Source manifest does not match selection')
            if any(not (a.sources / row['checkpoint']).is_file() for row in src['models']):
                issues.append('One or more source checkpoints missing')
        if ini is not None:
            if ini.get('status') != 'NEW_SELECTION_INITIAL_MODELS_TRAINED':
                issues.append('Initial manifest is not complete')
            if sel_path and ini.get('selection_sha256') != digest(sel_path):
                issues.append('Initial manifest does not match selection')
            if src_path and ini.get('source_manifest_sha256') != digest(src_path):
                issues.append('Initial manifest does not match source manifest')
            missing = sum(not (a.initials / ('%s_%s.pt' % (r['center'], r['seed']))).is_file()
                          or not (a.initials / ('%s_%s.json' % (r['center'], r['seed']))).is_file()
                          for r in ini['models'])
            if missing:
                issues.append('Missing initial checkpoints or sidecars: ' + str(missing))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        issues.append('Could not read supplied manifest: ' + str(exc))
    print(json.dumps({'status': 'ISSUES_FOUND' if issues else 'BASIC_CHECKS_PASSED',
          'role': a.role, 'python': platform.python_version(), 'dependencies': versions,
          'issues': issues, 'warnings': warnings,
          'scope': 'Read-only presence/syntax/manifest checks only; not a training run or scientific validation.',
          'archive_sha256_checked': bool(a.check_archive_sha256)}, indent=2))
    return 1 if issues else 0


if __name__ == '__main__':
    sys.exit(main())
