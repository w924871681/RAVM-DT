"""Validate ZIP names first; resolve only stable, precreated directories."""
import os
import re
import stat
from pathlib import Path, PureWindowsPath


def member_parts(name):
    if not isinstance(name, str) or not name or '\\' in name or ':' in name:
        raise ValueError(f'Non-portable ZIP member: {name!r}')
    parts = name.split('/')
    if PureWindowsPath(name).drive or any(p in ('', '.', '..') for p in parts):
        raise ValueError(f'Non-relative ZIP member: {name!r}')
    for part in parts:
        stem = part.split('.')[0].upper()
        if part != part.rstrip(' .') or any(ord(c) < 32 for c in part):
            raise ValueError(f'Ambiguous Windows member: {name!r}')
        if re.fullmatch(r'CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9]', stem):
            raise ValueError(f'Reserved Windows member: {name!r}')
        if any(c in part for c in '<>"|?*'):
            raise ValueError(f'Invalid Windows member: {name!r}')
    return tuple(parts)


def prepare_paths(root, names):
    """Caller supplies a new root; no worker may create or change directories."""
    root = Path(root)
    if root.exists():
        raise FileExistsError(root)
    root.mkdir()
    canonical = root.resolve(strict=True)
    parsed = {name: member_parts(name) for name in names}
    if len(parsed) != len(names) or len({n.casefold() for n in names}) != len(names):
        raise ValueError('Duplicate or case-colliding ZIP members')
    dirs = {()}
    for parts in parsed.values():
        dirs.update(parts[:i] for i in range(1, len(parts)))
    canonical_dirs = {}
    for parts in sorted(dirs, key=lambda p: (len(p), p)):
        path = canonical.joinpath(*parts)
        if parts:
            path.mkdir(exist_ok=True)
        # Directory creation is serial and completed before parallel file writes.
        real = path.resolve(strict=True)
        if not real.is_relative_to(canonical) or path.is_symlink() or path.is_junction():
            raise ValueError(f'Unsafe extraction directory: {path} -> {real}')
        canonical_dirs[parts] = real
    targets = {n: canonical_dirs[p[:-1]]/p[-1] for n, p in parsed.items()}
    for name, path in targets.items():
        if parsed[name] in dirs or not path.parent.is_relative_to(canonical) or path.exists():
            raise ValueError(f'Unsafe extraction target: {name}')
    return canonical, targets


def validate_infos(infos):
    names = [i.filename for i in infos]
    for info in infos:
        member_parts(info.filename)
        if info.is_dir() or stat.S_ISLNK(info.external_attr >> 16) or info.flag_bits & 1:
            raise ValueError(f'Unsupported ZIP member: {info.filename}')
    if len(names) != len(set(names)) or len(names) != len({n.casefold() for n in names}):
        raise ValueError('Duplicate or case-colliding ZIP members')
    return names
