"""Preserve the native Main attack return tuple, including provenance fields."""
def install(b):
    if getattr(b, 'main_attack_metadata_installed', False):
        return
    previous = b.native.attack
    def attack(seg, center, t, kind, alpha, location, seed=9417):
        if b.ctx().get('group') == 'Main' and kind in ('clean', 'mixed'):
            spec = b.ctx()['stream_specs'][b.TLS.ctx['stream']]
            assert (kind, alpha, location) == (spec['kind'], spec['alpha'], spec['location'])
            assert seed == 9417
            return b.NATIVE_PROTOCOL_ATTACK(seg, center, t, kind, alpha, location, seed)
        return previous(seg, center, t, kind, alpha, location, seed)
    b.native.attack = attack
    b.main_attack_metadata_installed = True
