"""Entrena el VERIFICADOR de segunda etapa (29/09). Ver src/verif_datos.py.

Entrada: recorte 64x64 con contexto + mascara de la caja propuesta (2 canales). Salida: probabilidad de que la
caja sea UN componente bien encuadrado. CNN chica desde cero (sin pesos preentrenados: no hace falta descargar
nada y los dibujos de linea son muy distintos a ImageNet).

Contra el sobreajuste: validacion separada por TILE de origen (10%), aumentos (flip, rot90, erosion/dilatacion,
brillo, jitter de la caja con la etiqueta recalculada solo si el jitter es chico), early stopping por la perdida de
validacion. Los 7 planos de test NO se usan.

    venv_rfdetr/Scripts/python.exe -u src/verif_train.py work/verif/rf4_ds21.npz work/verif/verif_v1.pt
"""
import os, sys, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

R = 64


class Verif(nn.Module):
    def __init__(self):
        super().__init__()
        def blk(i, o): return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(True),
                                            nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(True))
        self.b1, self.b2, self.b3, self.b4 = blk(2, 32), blk(32, 64), blk(64, 128), blk(128, 192)
        self.fc = nn.Sequential(nn.Dropout(.3), nn.Linear(192 * 2, 1))

    def forward(self, x):
        x = F.max_pool2d(self.b1(x), 2); x = F.max_pool2d(self.b2(x), 2)
        x = F.max_pool2d(self.b3(x), 2); x = self.b4(x)
        return self.fc(torch.cat([F.adaptive_avg_pool2d(x, 1).flatten(1), F.adaptive_max_pool2d(x, 1).flatten(1)], 1)).squeeze(1)


def mascara(M, n=R):
    """(N,4) caja relativa -> (N,1,n,n) rectangulo relleno."""
    g = torch.arange(n, device=M.device).float() / n
    x0, y0, x1, y1 = [M[:, i].view(-1, 1, 1) for i in range(4)]
    gx, gy = g.view(1, 1, n), g.view(1, n, 1)
    return (((gx >= x0) & (gx <= x1)) & ((gy >= y0) & (gy <= y1))).float().unsqueeze(1)


def entrada(X, M):
    img = 1.0 - X.float().unsqueeze(1) / 255.0          # tinta = 1
    return torch.cat([img, mascara(M)], 1)


def aumentar(X, M):
    """Aumentos en GPU, coherentes entre imagen y caja."""
    N = X.shape[0]
    X = X.float()
    # brillo / contraste del trazo
    X = 255 - (255 - X) * torch.empty(N, 1, 1, device=X.device).uniform_(.6, 1.4)
    X = X.clamp(0, 255)
    # erosion o dilatacion (engrosa/afina trazos)
    k = torch.rand(N, device=X.device)
    t = 1.0 - X.unsqueeze(1) / 255.0
    grueso = F.max_pool2d(t, 3, 1, 1); fino = -F.max_pool2d(-t, 3, 1, 1)
    t = torch.where((k < .2).view(-1, 1, 1, 1), grueso, torch.where((k > .9).view(-1, 1, 1, 1), fino, t))
    X = (1.0 - t.squeeze(1)) * 255
    # flip horizontal / vertical y rot90 (la caja acompana)
    M = M.clone()
    fh = torch.rand(N, device=X.device) < .5
    X[fh] = X[fh].flip(2); M[fh] = torch.stack([1 - M[fh, 2], M[fh, 1], 1 - M[fh, 0], M[fh, 3]], 1)
    fv = torch.rand(N, device=X.device) < .2
    X[fv] = X[fv].flip(1); M[fv] = torch.stack([M[fv, 0], 1 - M[fv, 3], M[fv, 2], 1 - M[fv, 1]], 1)
    rt = torch.rand(N, device=X.device) < .2
    X[rt] = X[rt].transpose(1, 2).flip(2)   # rot90 horario
    M[rt] = torch.stack([1 - M[rt, 3], M[rt, 0], 1 - M[rt, 1], M[rt, 2]], 1)
    # jitter chico de la caja (<= 4% del recorte): no cambia la etiqueta
    M = (M + torch.empty_like(M).uniform_(-.04, .04)).clamp(0, 1)
    return X, M


def main():
    datos, salida = sys.argv[1], sys.argv[2]
    # v3: varios npz separados por coma
    ds = [np.load(f) for f in datos.split(',')]
    X = np.concatenate([d['X'] for d in ds]); M = np.concatenate([d['M'] for d in ds]).astype(np.float32)
    y = np.concatenate([d['y'] for d in ds]).astype(np.float32)
    N = len(y)
    # validacion por bloques contiguos (las muestras de un mismo tile quedan juntas): 10%
    rng = np.random.RandomState(0)
    bloques = np.arange(N) // 200
    vb = set(rng.choice(bloques.max() + 1, size=max(1, (bloques.max() + 1) // 10), replace=False).tolist())
    va = np.array([b in vb for b in bloques]); tr = ~va
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    Xt, Mt, yt = torch.tensor(X[tr], device=dev), torch.tensor(M[tr], device=dev), torch.tensor(y[tr], device=dev)
    Xv, Mv, yv = torch.tensor(X[va], device=dev), torch.tensor(M[va], device=dev), torch.tensor(y[va], device=dev)
    pos = float(yt.mean()); pw = torch.tensor((1 - pos) / max(pos, 1e-6), device=dev)
    print('[verif] train %d (pos %.2f) | val %d' % (len(yt), pos, len(yv)))
    net = Verif().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=5e-4)
    EP = int(os.environ.get('VERIF_EPOCHS', '30')); B = 512
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=EP * ((len(yt) + B - 1) // B))
    mejor, paciencia = 1e9, 0
    for ep in range(EP):
        net.train(); perm = torch.randperm(len(yt), device=dev); t0 = time.time(); L = 0
        for i in range(0, len(yt), B):
            idx = perm[i:i + B]
            xa, ma = aumentar(Xt[idx], Mt[idx])
            loss = F.binary_cross_entropy_with_logits(net(entrada(xa, ma)), yt[idx], pos_weight=pw)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step(); L += float(loss) * len(idx)
        net.eval(); pv = []
        with torch.no_grad():
            for i in range(0, len(yv), 2048):
                pv.append(torch.sigmoid(net(entrada(Xv[i:i + 2048], Mv[i:i + 2048]))))
        pv = torch.cat(pv); lv = float(F.binary_cross_entropy(pv.clamp(1e-6, 1 - 1e-6), yv))
        acc = float(((pv > .5).float() == yv).float().mean())
        # recall de positivos y rechazo de negativos a umbral 0,5
        rp = float((pv[yv == 1] > .5).float().mean()); rn = float((pv[yv == 0] <= .5).float().mean())
        print('[verif] ep %2d  loss %.4f  val %.4f  acc %.3f  pos>0,5 %.3f  neg<=0,5 %.3f  (%.0f s)'
              % (ep, L / len(yt), lv, acc, rp, rn, time.time() - t0), flush=True)
        if lv < mejor - 1e-4:
            mejor, paciencia = lv, 0; torch.save(net.state_dict(), salida)
        else:
            paciencia += 1
            if paciencia >= 6: print('[verif] early stopping'); break
    print('[verif] mejor val loss %.4f -> %s' % (mejor, salida))


if __name__ == '__main__':
    main()
