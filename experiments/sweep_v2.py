"""Sweep v2: Multi-seed runs with best architectures + training improvements.

Key learnings from sweep v1:
- G[256,256] D[1024] is the best architecture class
- noise_dim=20 helps (v17/v18 both better)
- Deeper D causes collapse (v15, v18 had D_loss→0.3)
- Seed sensitivity is huge (v14: 44% on cfx002 vs 37% on GPU)
- Cu conditioning is the bottleneck (16-27% match)
- Fe is easy (77-87% match)

Changes in v2:
1. Multiple seeds per config (3 seeds, report mean±std)
2. Focus on G[256,256] D[1024] with noise_dim variations
3. Try label smoothing for D (0.9 instead of 1.0)
4. Try spectral normalization on D to stabilize
5. Try longer training (4000 epochs) for best config
"""
import importlib.util
import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import paddle
import paddle.nn as nn
from scipy.stats import wasserstein_distance

def _import(name, filepath):
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_dataset = _import('alloy_dataset', os.path.join(BASE, 'ppmat/datasets/alloy_dataset.py'))

ELEMS = ['Cu','Zr','Al','Ni','Ti','Ag','Fe','Mg','B','Si','Nb','Y','Ca','La','Co',
         'Be','C','Mo','Pd','P','Sn','Cr','Hf','Zn','Gd','Ce','Er','Ga','Au','Nd',
         'Dy','W','Pr','Ta','Sc','Li','Sm','S','Pt','Mn']

def make_generator(layers_config, noise_dim=5, cond_dim=26, comp_dim=40):
    modules = []
    in_dim = noise_dim + cond_dim
    for h in layers_config:
        modules.extend([nn.Linear(in_dim, h), nn.LeakyReLU(0.2)])
        in_dim = h
    modules.extend([nn.Linear(in_dim, comp_dim), nn.Sigmoid()])
    return nn.Sequential(*modules)

def make_discriminator(layers_config, input_dim=66, spectral_norm=False):
    modules = []
    in_dim = input_dim
    for h in layers_config:
        linear = nn.Linear(in_dim, h)
        if spectral_norm:
            linear = nn.utils.spectral_norm(linear)
        modules.extend([linear, nn.LeakyReLU(0.2)])
        in_dim = h
    final = nn.Linear(in_dim, 1)
    if spectral_norm:
        final = nn.utils.spectral_norm(final)
    modules.extend([final, nn.Sigmoid()])
    return nn.Sequential(*modules)

def train_and_eval(g_layers, d_layers, epochs=2000, lr=0.0002, batch_size=64,
                   noise_dim=5, weight_decay=1e-5, device='gpu', seed=42,
                   label_smooth=False, spectral_norm=False, lr_schedule=None):
    """Train a CGAN variant and return metrics."""
    paddle.seed(seed)
    np.random.seed(seed)
    paddle.set_device(device)
    
    ds = _dataset.AlloyDataset(
        os.path.join(BASE, 'data/alloy/Alloy_train.csv'),
        categories=['Cu','Fe','Ti','Zr'], normalize=True
    )
    from paddle.io import DataLoader, BatchSampler
    loader = DataLoader(ds, batch_sampler=BatchSampler(ds, batch_size=batch_size, shuffle=True, drop_last=False))
    
    G = make_generator(g_layers, noise_dim=noise_dim)
    D = make_discriminator(d_layers, spectral_norm=spectral_norm)
    
    g_params = sum(p.numpy().size for p in G.parameters())
    d_params = sum(p.numpy().size for p in D.parameters())
    
    loss_fn = nn.BCELoss()
    EPS = 1e-7
    
    # Learning rate scheduling
    if lr_schedule == 'cosine':
        g_lr = paddle.optimizer.lr.CosineAnnealingDecay(learning_rate=lr, T_max=epochs)
        d_lr = paddle.optimizer.lr.CosineAnnealingDecay(learning_rate=lr, T_max=epochs)
    else:
        g_lr = lr
        d_lr = lr
    
    opt_g = paddle.optimizer.Adam(parameters=G.parameters(), learning_rate=g_lr, beta1=0.5, beta2=0.999, weight_decay=weight_decay)
    opt_d = paddle.optimizer.Adam(parameters=D.parameters(), learning_rate=d_lr, beta1=0.5, beta2=0.999, weight_decay=weight_decay)
    
    # Label smoothing targets
    real_label = 0.9 if label_smooth else 1.0
    fake_label = 0.1 if label_smooth else 0.0
    
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        for batch_data in loader:
            data = batch_data['data']
            cur_bs = data.shape[0]
            if cur_bs < 2:
                break
            real_comp = data[:, :40]
            conditions = data[:, 40:]
            ones = paddle.full([cur_bs, 1], real_label)
            zeros = paddle.full([cur_bs, 1], fake_label)
            
            # D step
            z = paddle.rand([cur_bs, noise_dim])
            fake_comp = G(paddle.concat([z, conditions], axis=1)).detach()
            d_real = paddle.clip(D(paddle.concat([real_comp, conditions], axis=1)), EPS, 1-EPS)
            d_fake = paddle.clip(D(paddle.concat([fake_comp, conditions], axis=1)), EPS, 1-EPS)
            d_loss = loss_fn(d_real, ones) + loss_fn(d_fake, zeros)
            opt_d.clear_grad()
            d_loss.backward()
            opt_d.step()
            
            # G step
            z = paddle.randn([cur_bs, noise_dim])
            fake_comp = G(paddle.concat([z, conditions], axis=1))
            d_out = paddle.clip(D(paddle.concat([fake_comp, conditions], axis=1)), EPS, 1-EPS)
            g_loss = loss_fn(d_out, paddle.full([cur_bs, 1], real_label))
            opt_d.clear_grad()
            opt_g.clear_grad()
            g_loss.backward()
            opt_g.step()
        
        if lr_schedule == 'cosine':
            g_lr.step()
            d_lr.step()
        
        if epoch % 500 == 0:
            print(f'    Epoch {epoch}: D={d_loss.item():.4f} G={g_loss.item():.4f}', flush=True)
    
    train_time = time.time() - t0
    
    # Eval
    G.eval()
    D.eval()
    loader_eval = DataLoader(ds, batch_sampler=BatchSampler(ds, batch_size=len(ds), shuffle=False))
    batch = next(iter(loader_eval))
    data = batch['data']
    real_comp = data[:, :40].numpy()
    conditions = data[:, 40:]
    
    # Average over 5 eval seeds for stability
    dom_matches = []
    wds = []
    sum_means = []
    sum_stds = []
    cat_match_list = []
    for eval_seed in range(5):
        paddle.seed(eval_seed + 1000)
        with paddle.no_grad():
            z = paddle.randn([conditions.shape[0], noise_dim])
            fake_comp_np = G(paddle.concat([z, conditions], axis=1)).numpy()
        
        wd = np.mean([wasserstein_distance(real_comp[:, i], fake_comp_np[:, i]) for i in range(40)])
        wds.append(wd)
        sums = fake_comp_np.sum(axis=1) * 100
        sum_means.append(sums.mean())
        sum_stds.append(sums.std())
        real_dom = real_comp.argmax(axis=1)
        fake_dom = fake_comp_np.argmax(axis=1)
        dom_matches.append((real_dom == fake_dom).mean())
        
        cm = {}
        for cat_name, cat_idx in [('Cu', 0), ('Zr', 1), ('Ti', 4), ('Fe', 6)]:
            mask = real_dom == cat_idx
            if mask.sum() > 0:
                cm[cat_name] = (fake_dom[mask] == cat_idx).mean()
        cat_match_list.append(cm)
    
    # Average cat matches
    avg_cat = {}
    for cat in ['Cu', 'Zr', 'Ti', 'Fe']:
        vals = [cm[cat] for cm in cat_match_list if cat in cm]
        if vals:
            avg_cat[cat] = np.mean(vals)
    
    # Top-5 using last fake_comp
    top5 = {}
    for idx in np.argsort(-real_comp.mean(axis=0))[:5]:
        top5[ELEMS[idx]] = (real_comp[:, idx].mean() * 100, fake_comp_np[:, idx].mean() * 100)
    
    return {
        'wd': np.mean(wds),
        'wd_std': np.std(wds),
        'sum_mean': np.mean(sum_means),
        'sum_std': np.mean(sum_stds),
        'dom_match': np.mean(dom_matches),
        'dom_match_std': np.std(dom_matches),
        'cat_matches': avg_cat,
        'top5': top5,
        'g_params': g_params,
        'd_params': d_params,
        'train_time': train_time,
        'd_loss': d_loss.item(),
        'g_loss': g_loss.item(),
    }

def print_results(name, r):
    print(f'\n{"="*60}', flush=True)
    print(f'{name}', flush=True)
    print(f'  G params: {r["g_params"]:,}  D params: {r["d_params"]:,}  Time: {r["train_time"]:.1f}s', flush=True)
    print(f'  D_loss={r["d_loss"]:.4f}  G_loss={r["g_loss"]:.4f}', flush=True)
    print(f'  WD={r["wd"]:.4f}±{r["wd_std"]:.4f}  sum={r["sum_mean"]:.1f}±{r["sum_std"]:.1f}  dom_match={r["dom_match"]:.1%}±{r["dom_match_std"]:.1%}', flush=True)
    print(f'  Per-category dom match:', flush=True)
    for cat, m in r['cat_matches'].items():
        print(f'    {cat}: {m:.1%}', flush=True)
    print(f'  Top-5 elements (real vs fake at%):', flush=True)
    for elem, (real, fake) in r['top5'].items():
        diff = abs(real - fake)
        marker = '✓' if diff < 3 else '✗'
        print(f'    {elem:3s}: real={real:.1f} fake={fake:.1f} {marker}', flush=True)


if __name__ == '__main__':
    SEEDS = [42, 123, 777]
    
    configs = [
        # (name, g_layers, d_layers, epochs, noise_dim, kwargs)
        # Best arch + noise dim exploration
        ('A: G[256,256] D[1024] n=10', [256, 256], [1024], 2000, 10, {}),
        ('B: G[256,256] D[1024] n=20', [256, 256], [1024], 2000, 20, {}),
        ('C: G[256,256] D[1024] n=10 smooth', [256, 256], [1024], 2000, 10, {'label_smooth': True}),
        # Longer training with best config
        ('D: G[256,256] D[1024] n=10 4Kep', [256, 256], [1024], 4000, 10, {}),
        # Wider G
        ('E: G[512,256] D[1024] n=10', [512, 256], [1024], 2000, 10, {}),
    ]
    
    all_results = {}
    
    for name, g_layers, d_layers, epochs, noise_dim, kwargs in configs:
        print(f'\n{"#"*60}', flush=True)
        print(f'Config: {name}', flush=True)
        print(f'  G layers: {g_layers}  D layers: {d_layers}  noise_dim: {noise_dim}  epochs: {epochs}  kwargs: {kwargs}', flush=True)
        
        seed_results = []
        for seed in SEEDS:
            print(f'\n  --- Seed {seed} ---', flush=True)
            r = train_and_eval(g_layers, d_layers, epochs=epochs, noise_dim=noise_dim, seed=seed, **kwargs)
            seed_results.append(r)
            print(f'    WD={r["wd"]:.4f}  dom_match={r["dom_match"]:.1%}  Cu={r["cat_matches"].get("Cu",0):.1%}  Fe={r["cat_matches"].get("Fe",0):.1%}', flush=True)
        
        # Average across seeds
        avg = {
            'wd': np.mean([r['wd'] for r in seed_results]),
            'wd_std': np.std([r['wd'] for r in seed_results]),
            'sum_mean': np.mean([r['sum_mean'] for r in seed_results]),
            'sum_std': np.mean([r['sum_std'] for r in seed_results]),
            'dom_match': np.mean([r['dom_match'] for r in seed_results]),
            'dom_match_std': np.std([r['dom_match'] for r in seed_results]),
            'cat_matches': {},
            'top5': seed_results[0]['top5'],  # use first seed's top5
            'g_params': seed_results[0]['g_params'],
            'd_params': seed_results[0]['d_params'],
            'train_time': np.mean([r['train_time'] for r in seed_results]),
            'd_loss': np.mean([r['d_loss'] for r in seed_results]),
            'g_loss': np.mean([r['g_loss'] for r in seed_results]),
        }
        for cat in ['Cu', 'Zr', 'Ti', 'Fe']:
            vals = [r['cat_matches'].get(cat, 0) for r in seed_results]
            avg['cat_matches'][cat] = np.mean(vals)
        
        all_results[name] = avg
        print_results(f'{name} (avg over {len(SEEDS)} seeds)', avg)
    
    # Summary table
    print(f'\n\n{"="*80}', flush=True)
    print('SUMMARY TABLE (averaged over 3 seeds)', flush=True)
    print(f'{"Config":<45} {"WD":>10} {"Sum":>10} {"DomM":>10} {"Cu":>6} {"Fe":>6} {"Time":>6}', flush=True)
    print('-' * 95, flush=True)
    for name, r in all_results.items():
        short = name.split(':')[0]
        cu = r['cat_matches'].get('Cu', 0)
        fe = r['cat_matches'].get('Fe', 0)
        print(f'{short:<45} {r["wd"]:.4f}±{r["wd_std"]:.3f} {r["sum_mean"]:>5.1f}±{r["sum_std"]:<4.1f} {r["dom_match"]:>5.1%}±{r["dom_match_std"]:.1%} {cu:>5.1%} {fe:>5.1%} {r["train_time"]:>5.0f}s', flush=True)
    
    best = max(all_results.items(), key=lambda x: x[1]['dom_match'])
    print(f'\nBest by dom_match: {best[0]} = {best[1]["dom_match"]:.1%}', flush=True)
    best_wd = min(all_results.items(), key=lambda x: x[1]['wd'])
    print(f'Best by WD: {best_wd[0]} = {best_wd[1]["wd"]:.4f}', flush=True)
