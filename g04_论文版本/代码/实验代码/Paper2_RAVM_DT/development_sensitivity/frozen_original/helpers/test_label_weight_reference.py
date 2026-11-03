"""Independent deterministic smoke tests; no model training and no Service."""
from __future__ import annotations
import numpy as np
from label_weight_reference import (
    label_weights_from_update_raw,
    target_weights_for_windows,
    weighted_mse_numpy,
)

def test_constant():
    w = label_weights_from_update_raw(np.full(400, .4))
    assert w.shape == (400,)
    assert np.all(w == 1)

def test_anomaly_soft_weight():
    y = np.full(400, .4)
    y[140] += .2
    w = label_weights_from_update_raw(y)
    assert .2 <= w[140] < .5
    assert w[139] == 1 and w[141] == 1
    assert np.all((w >= .2) & (w <= 1))

def test_target_alignment():
    raw_w = np.arange(400, dtype=np.float32)
    target_w = target_weights_for_windows(raw_w, L=96, H=1)
    assert target_w.shape == (304,)
    assert target_w[0] == 96 and target_w[-1] == 399

def test_weighted_equals_mse_if_all_one():
    pred = np.array([1., 2., 4.])
    targ = np.array([.7, 2.4, 2.0])
    got = weighted_mse_numpy(pred, targ, np.ones_like(targ))
    assert np.isclose(got, np.mean((pred-targ)**2), atol=1e-12)

def test_sample_weighting_effect():
    pred = np.array([.0, .0, .0])
    targ = np.array([0., 0., 1.])
    assert weighted_mse_numpy(pred, targ, np.array([1.,1.,.2])) < weighted_mse_numpy(pred, targ, np.ones(3))

def test_torch_broadcast_is_column_weights():
    try:
        import torch
    except ImportError:
        return  # Torch-specific implementation validated by Codex GPU runner
    pred=torch.tensor([[0.],[0.],[0.]],dtype=torch.float64)
    target=torch.tensor([[0.],[0.],[1.]],dtype=torch.float64)
    w=torch.tensor([1.,1.,.2],dtype=torch.float64)
    loss=(w.view(-1,1)*(pred-target).square()).sum()/w.sum()
    truth=weighted_mse_numpy(pred.numpy(),target.numpy(),w.numpy())
    assert np.isclose(loss.item(),truth,atol=1e-12)
    bad=(w*(pred-target).square()).sum()/w.sum()
    assert not np.isclose(bad.item(),truth), "test must detect accidental NxN broadcasting"

def test_no_future_oracle_api():
    # Reference API accepts one raw Update y array and cannot take a service, mask or regime.
    import inspect
    params = set(inspect.signature(label_weights_from_update_raw).parameters)
    assert params == {'raw_y','half_window','additive_scale','transition','floor'}

def test_nonfinite_fails():
    y = np.full(400, .3); y[20] = np.nan
    try:
        label_weights_from_update_raw(y)
    except ValueError:
        pass
    else:
        raise AssertionError('non-finite Update silently accepted')

def test_invalid_future_horizon_fails():
    try:
        target_weights_for_windows(np.ones(400), L=96, H=4)
    except ValueError:
        pass
    else:
        raise AssertionError('unverified H=4 mapping accepted')

if __name__ == '__main__':
    import inspect
    fn = [f for n, f in sorted(globals().items()) if n.startswith('test_') and inspect.isfunction(f)]
    for f in fn:
        f()
        print('PASS', f.__name__)
    print('PASS_ALL', len(fn))
