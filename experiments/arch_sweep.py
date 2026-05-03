"""Architecture sweep: test deeper G/D variants on GPU."""
import importlib.util
import sys
import os
import time
import numpy as np

# Add parent to path
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
    """Build G from layer config like [256, 256] -> Linear(31,256), LReLU, Linear(256,256), LReLU, Linear(256,40), Sigmoid."""
    modules = []
    in_dim = noise_dim + cond_dim
    for h in layers_config:
        modules.extend([nn.Linear(in_dim, h), nn.LeakyReLU(0.2)])
        in_dim = h
    modules.extend([nn.Linear(in_dim, comp_dim), nn.Sigmoid()])
    return nn.Sequential(*modules)

def make_discriminator(layers_config, input_dim=66):
    """Build D from layer config like [512, 256] -> Linear(66,512), LReLU, Linear(512,256), LReLU, Linear(256,1), Sigmoid."""
    modules = []
    in_dim = input_dim
    for h in layers_config:
        modules.extend([nn.Linear(in_dim, h), nn.LeakyReLU(0.2)])
        in_dim = h
    modules.extend([nn.Linear(in_dim, 1), nn.Sigmoid()])
    return nn.Sequential(*modules)

def train_and_eval(g_layers, d_layers, epochs=2000, lr=0.0002, batch_size=64, 
                   noise_dim=5, weight_decay=1e-5, device='gpu'):
    """Train a CGAN variant and return metrics."""
    paddle.set_device(device)
    
    ds = _dataset.AlloyDataset(
        os.path.join(BASE, 'data/alloy/Alloy_train.csv'),
        categories=['Cu','Fe','Ti','Zr'], normalize=True
    )
    from paddle.io import DataLoader, BatchSampler
    loader = DataLoader(ds, batch_sampler=BatchSampler(ds, batch_size=batch_size, shuffle=True, drop_last=False))
    
    G = make_generator(g_layers, noise_dim=noise_dim)
    D = make_discriminator(d_layers)
    
    g_params = sum(p.numpy().size for p in G.parameters())
    d_params = sum(p.numpy().size for p in D.parameters())
    
    loss_fn = nn.BCELoss()
    EPS = 1e-7
    opt_g = paddle.optimizer.Adam(parameters=G.parameters(), learning_rate=lr, beta1=0.5, beta2=0.999, weight_decay=weight_decay)
    opt_d = paddle.optimizer.Adam(parameters=D.parameters(), learning_rate=lr, beta1=0.5, beta2=0.999, weight_decay=weight_decay)
    
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        for batch_data in loader:
            data = batch_data['data']
            cur_bs = data.shape[0]
            if cur_bs < batch_size:
                break
            real_comp = data[:, :40]
            conditions = data[:, 40:]
            ones = paddle.ones([cur_bs, 1])
            zeros = paddle.zeros([cur_bs, 1])
            
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
            g_loss = loss_fn(d_out, ones)
            opt_d.clear_grad()
            opt_g.clear_grad()
            g_loss.backward()
            opt_g.step()
        
        if epoch % 200 == 0:
            print(f'  Epoch {epoch}: D={d_loss.item():.4f} G={g_loss.item():.4f}', flush=True)
    
    train_time = time.time() - t0
    
    # Eval
    G.eval()
    D.eval()
    loader_eval = DataLoader(ds, batch_sampler=BatchSampler(ds, batch_size=len(ds), shuffle=False))
    batch = next(iter(loader_eval))
    data = batch['data']
    real_comp = data[:, :40].numpy()
    conditions = data[:, 40:]
    
    with paddle.no_grad():
        z = paddle.randn([conditions.shape[0], noise_dim])
        fake_comp = G(paddle.concat([z, conditions], axis=1)).numpy()
    
    wd = np.mean([wasserstein_distance(real_comp[:, i], fake_comp[:, i]) for i in range(40)])
    sums = fake_comp.sum(axis=1) * 100
    real_dom = real_comp.argmax(axis=1)
    fake_dom = fake_comp.argmax(axis=1)
    dom_match = (real_dom == fake_dom).mean()
    
    # Per-category dom match
    cat_matches = {}
    for cat_name, cat_idx in [('Cu', 0), ('Zr', 1), ('Ti', 4), ('Fe', 6)]:
        mask = real_dom == cat_idx
        if mask.sum() > 0:
            cat_matches[cat_name] = (fake_dom[mask] == cat_idx).mean()
    
    # Top-5 element comparison
    top5 = {}
    for idx in np.argsort(-real_comp.mean(axis=0))[:5]:
        top5[ELEMS[idx]] = (real_comp[:, idx].mean() * 100, fake_comp[:, idx].mean() * 100)
    
    return {
        'wd': wd,
        'sum_mean': sums.mean(),
        'sum_std': sums.std(),
        'dom_match': dom_match,
        'cat_matches': cat_matches,
        'top5': top5,
        'g_params': g_params,
        'd_params': d_params,
        'train_time': train_time,
        'd_loss': d_loss.item(),
        'g_loss': g_loss.item(),
    }

def print_results(name, r):
    print(f'\n{"="*60}')
    print(f'{name}')
    print(f'  G params: {r["g_params"]:,}  D params: {r["d_params"]:,}  Time: {r["train_time"]:.1f}s')
    print(f'  D_loss={r["d_loss"]:.4f}  G_loss={r["g_loss"]:.4f}')
    print(f'  WD={r["wd"]:.4f}  sum={r["sum_mean"]:.1f}±{r["sum_std"]:.1f}  dom_match={r["dom_match"]:.1%}')
    print(f'  Per-category dom match:')
    for cat, m in r['cat_matches'].items():
        print(f'    {cat}: {m:.1%}')
    print(f'  Top-5 elements (real vs fake at%):')
    for elem, (real, fake) in r['top5'].items():
        diff = abs(real - fake)
        marker = '✓' if diff < 3 else '✗'
        print(f'    {elem:3s}: real={real:.1f} fake={fake:.1f} {marker}')


if __name__ == '__main__':
    configs = [
        # (name, g_layers, d_layers, epochs, noise_dim)
        ('v14-baseline: G[256,256] D[1024]', [256, 256], [1024], 2000, 5),
        ('v15: G[256,256,128] D[512,256]', [256, 256, 128], [512, 256], 2000, 5),
        ('v17: G[256,256] D[1024] noise=20', [256, 256], [1024], 2000, 20),
        ('v18: G[512,256,128] D[512,256] noise=20', [512, 256, 128], [512, 256], 2000, 20),
    ]
    
    results = {}
    for name, g_layers, d_layers, epochs, noise_dim in configs:
        print(f'\n{"#"*60}', flush=True)
        print(f'Training: {name}', flush=True)
        print(f'  G layers: {g_layers}  D layers: {d_layers}  noise_dim: {noise_dim}', flush=True)
        r = train_and_eval(g_layers, d_layers, epochs=epochs, noise_dim=noise_dim)
        results[name] = r
        print_results(name, r)
    
    # Summary table
    print(f'\n\n{"="*80}')
    print('SUMMARY TABLE')
    print(f'{"Config":<45} {"WD":>6} {"Sum":>10} {"DomM":>6} {"Time":>6}')
    print('-' * 80)
    for name, r in results.items():
        short = name.split(':')[0]
        print(f'{short:<45} {r["wd"]:>6.4f} {r["sum_mean"]:>5.1f}±{r["sum_std"]:<4.1f} {r["dom_match"]:>5.1%} {r["train_time"]:>5.0f}s')
    
    # Best
    best = max(results.items(), key=lambda x: x[1]['dom_match'])
    print(f'\nBest by dom_match: {best[0].split(":")[0]} = {best[1]["dom_match"]:.1%}')
    best_wd = min(results.items(), key=lambda x: x[1]['wd'])
    print(f'Best by WD: {best_wd[0].split(":")[0]} = {best_wd[1]["wd"]:.4f}')
