"""PAT-304 Gate 0 (whole-model, A1 = memory OFF): time-major patch-state scan + readout G + state injection.
usage (GPU node): python patchmem/test_tm_model.py"""
import sys
from argparse import Namespace
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from models.MambaTS import Model
from patchmem.tm_model import TMMambaTS, patch_targets, a1_loss
from patchmem.ref_scan import mamba_step_noconv

dev = 'cuda'
torch.manual_seed(0)
C, P, NP, H = 7, 48, 5, 96


def build(d_model=16, e_layers=3, dropout=0.0):
    cfg = Namespace(task_name='long_term_forecast', seq_len=P * NP, pred_len=H, patch_len=P, stride=P, d_model=d_model, e_layers=e_layers,
                    enc_in=C, dropout=dropout, VPT_mode=1, ATSP_solver='SA', use_casual_conv=False)
    return TMMambaTS(Model(cfg)).to(dev)


def close(a, b, name, rtol=2e-3, atol=2e-4):
    err = (a - b).abs().max().item(); ok = torch.allclose(a, b, rtol=rtol, atol=atol)
    print(f'  {"ok  " if ok else "FAIL"} {name}: max abs err {err:.2e}'); assert ok, name


def run_tests():
    m = build().eval(); b = 4
    x = torch.randn(b, P * NP, C, device=dev, requires_grad=True)
    perm = torch.stack([torch.randperm(C) for _ in range(b)]).to(dev)
    print('T0 shapes of the real SSM cache (per sample): (d_inner, d_state) per TMB layer')
    o = m(x, perm); print('  ', [tuple(s.shape) for s in o['states']], '= (b, n_patch, d_inner, d_state) per layer; pred', tuple(o['pred'].shape))

    print('T1 full forward == token-by-token step through all layers (boundary states of every layer, final residual stream)')
    with torch.no_grad():
        xn, stats = m.norm(x); hs, nb = m.embed(xn, perm); L = hs.shape[1]
        lay = m.base.encoder.layers; sts = [torch.zeros(b, l.mixer.d_inner, l.mixer.d_state, device=dev) for l in lay]
        outs = []
        for t in range(L):
            h, res = hs[:, t:t + 1], None
            for li, blk in enumerate(lay):
                res = h if res is None else h + res
                h, sts[li] = mamba_step_noconv(blk.mixer, blk.norm(res), sts[li])
            outs.append(h + res)
            if (t + 1) % C == 0:
                j = (t + 1) // C - 1
                for li in range(len(lay)): close(sts[li], o['states'][li][:, j], f'layer {li} state after patch {j}')
        close(torch.cat(outs, 1), o['out'] + o['res'], 'final stream (mixer out + residual)', rtol=5e-3, atol=5e-4)

    print('T2 causality with window stats held fixed: perturbing X from patch k on leaves h_j^- and G(h_j^-) unchanged for j<k')
    k = 3
    with torch.no_grad():
        x2 = x.detach().clone(); x2[:, P * k:] += torch.randn_like(x2[:, P * k:])
        o2 = m(x2, perm, stats=o['stats'])
        for li in range(len(lay)): close(o2['states'][li][:, :k], o['states'][li][:, :k], f'layer {li} states j<{k}')
        close(o2['pred'][:, :k], o['pred'][:, :k], f'pred_j j<{k}')
        assert (o2['pred'][:, k:] - o['pred'][:, k:]).abs().max() > 1e-4; print('  ok   later predictions do change')

    L_ = len(lay) - 1
    dm = lay[-1].mixer.d_inner, lay[-1].mixer.d_state
    Jr = torch.nn.Parameter(torch.randn(*dm, device=dev) * 0.5); Jp = torch.nn.Parameter(torch.randn(*dm, device=dev) * 0.5)
    inj_at = 1
    hk = lambda at, r=None, p=None: {L_: lambda i, h: ((r.expand_as(h) if r is not None else None), (p.expand_as(h) if p is not None else None)) if i == at else None}

    print('T3 no hook == zero-delta hook (pred, pred_prior, prior states, reads)')
    with torch.no_grad():
        assert torch.equal(o['pred'], o['pred_prior']) and torch.equal(o['reads'], o['states'][-1]); print('  ok   without a hook read == prior, pred == pred_prior')
        oz = m(x, perm, hooks={L_: lambda i, h: (torch.zeros_like(h), torch.zeros_like(h))})
        for li in range(len(lay)): close(oz['states'][li], o['states'][li], f'zero-inject layer {li} states')
        close(oz['pred'], o['pred'], 'pred'); close(oz['pred_prior'], o['pred_prior'], 'pred_prior')

    print('T4a post-label correction d_post at patch k: current-patch pred and h^- up to k unchanged; later patches change')
    with torch.no_grad():
        op = m(x, perm, hooks=hk(inj_at, p=Jp))
        close(op['states'][-1][:, :inj_at + 1], o['states'][-1][:, :inj_at + 1], 'prior h^- up to the injection patch'); close(op['pred'][:, :inj_at + 1], o['pred'][:, :inj_at + 1], 'pred up to the injection patch')
        assert (op['pred'][:, inj_at + 1:] - o['pred'][:, inj_at + 1:]).abs().max() > 1e-4 and (op['pred_prior'][:, inj_at + 1:] - o['pred_prior'][:, inj_at + 1:]).abs().max() > 1e-4
        print('  ok   d_post changes later states and predictions')

    print('T4b read-time correction d_read at patch k: pred_k changes, pred_prior / prior h^- / every other patch unchanged (incl. k = last origin)')
    with torch.no_grad():
        for k in (inj_at, NP - 1):
            orr = m(x, perm, hooks=hk(k, r=Jr))
            for li in range(len(lay)): close(orr['states'][li], o['states'][li], f'k={k} layer {li} prior states')
            close(orr['pred_prior'], o['pred_prior'], f'k={k} pred_prior')
            diff = (orr['pred'] - o['pred']).abs().amax((0, 2, 3))
            assert diff[k] > 1e-4 and torch.all(diff[torch.arange(NP) != k] == 0), diff
            print(f'  ok   k={k}: only pred[{k}] changes (max diff {diff[k].item():.3e})')

    print('T4c d_read and d_post together at patch k: pred_k sees d_read only; later patches see d_post only')
    with torch.no_grad():
        ob = m(x, perm, hooks=hk(inj_at, r=Jr, p=Jp))
        close(ob['pred'][:, inj_at + 1:], op['pred'][:, inj_at + 1:], 'later pred == d_post-only run'); close(ob['pred'][:, :inj_at], o['pred'][:, :inj_at], 'earlier pred')
        orr = m(x, perm, hooks=hk(inj_at, r=Jr)); close(ob['pred'][:, inj_at], orr['pred'][:, inj_at], 'pred_k == d_read-only run')

    print('T4d d_read on a lower layer is rejected')
    try:
        m(x, perm, hooks={0: lambda i, h: (torch.zeros_like(h), None)}); raise SystemExit('FAIL: should have asserted')
    except AssertionError:
        print('  ok   assertion raised')

    print('T5 gradients: loss on the last-origin pred reaches d_read, d_post (via later patches), early X, all layer / G params')
    m.zero_grad(); x.grad = None; Jr.grad = None; Jp.grad = None
    oi = m(x, perm, hooks=hk(NP - 1, r=Jr)); oi['pred'][:, -1].square().mean().backward()
    assert Jr.grad.abs().max() > 0 and x.grad[:, :P].abs().max() > 0; print(f'   last-origin d_read: |grad Jr| {Jr.grad.norm().item():.3e}  |grad x[:, :P]| {x.grad[:, :P].norm().item():.3e}  |grad G| {m.G.weight.grad.norm().item():.3e}')
    m.zero_grad(); x.grad = None; Jp.grad = None
    oi = m(x, perm, hooks=hk(inj_at, p=Jp)); oi['pred'][:, -1].square().mean().backward()
    assert Jp.grad.abs().max() > 0 and x.grad[:, :P].abs().max() > 0; print(f'   d_post at patch {inj_at}: |grad Jp| {Jp.grad.norm().item():.3e}')
    for n, p_ in m.named_parameters():
        if p_.requires_grad and p_.grad is None and 'adjacency' not in n and 'count_matrix' not in n and 'ending' not in n and 'norm_f' not in n:
            print('   no grad:', n)

    print('T5b detached restart state (what a naive h.detach() rollout does) kills the history gradient; non-detached keeps it')
    m.zero_grad(); x.grad = None
    od = m(x, perm, hooks={L_: lambda i, h: (None, Jp.expand_as(h) + (h.detach() - h)) if i == inj_at else None})
    od['pred'][:, -1].square().mean().backward()
    print(f'   |grad x[:, :P]| with detached restart state {x.grad[:, :P].norm().item():.3e} (last layer cut only; lower layers still carry gradient)')

    print('T6 dropout: eval deterministic, train differs')
    mt = build(dropout=0.3); mt.eval(); a, b2 = mt(x[:, :], perm)['pred'], mt(x, perm)['pred']; close(a, b2, 'eval deterministic', 1e-6, 1e-7)
    mt.train(); assert (mt(x, perm)['pred'] - a).abs().max() > 1e-5; print('  ok   train-mode dropout active in x_proj')

    print('T7 VPT order: output channel order stays the original variable order under different perms')
    with torch.no_grad():
        ids = torch.arange(C, device=dev)[None].expand(b, -1)
        p0 = m(x, ids)['pred']; p1 = m(x, perm)['pred']
        assert p0.shape == p1.shape == (b, NP, H, C); assert (p0 - p1).abs().max() > 1e-5; print('  ok   shapes (b, n_patch, H, C); perm changes the state (expected: scan order differs)')

    print('T8 per-patch targets: Y_{last} == official future block; Y_j (j<=n-3) lies inside the window; loss is finite and trains G')
    yf = torch.randn(b, H, C, device=dev); tg = patch_targets(x.detach(), yf, P, H)
    assert tg.shape == (b, NP, H, C) and torch.equal(tg[:, -1], yf)
    assert torch.equal(tg[:, 0], x.detach()[:, P:P + H]) and torch.equal(tg[:, NP - 3], x.detach()[:, P * (NP - 2):P * (NP - 2) + H])
    assert torch.equal(tg[:, NP - 2, :P], x.detach()[:, -P:]) and torch.equal(tg[:, NP - 2, P:], yf[:, :P])
    m.zero_grad(); loss, per = a1_loss(m(x, perm)['pred'], x, yf, P, H); loss.backward()
    print(f'  ok   targets aligned; loss {loss.item():.3f}, per-origin {[round(v, 2) for v in per.tolist()]}, |grad G| {m.G.weight.grad.norm().item():.3e}')


if __name__ == '__main__':
    run_tests(); print('ALL PASS')
