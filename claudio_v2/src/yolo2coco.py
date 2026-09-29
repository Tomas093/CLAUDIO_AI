"""Convierte un dataset YOLO (work/dsNN) al formato COCO que pide RF-DETR (work/dsNN_coco).

24/09, pedido de Tomas: probar RF-DETR con el mismo dataset que los YOLO. RF-DETR lee
<dir>/{train,valid,test}/_annotations.coco.json con las imagenes al lado. Las imagenes se
enlazan (hardlink), no se copian: son 800 MB. `test` es el mismo `valid` (la libreria lo exige;
la evaluacion de verdad es la de siempre, con los DXF).

Convencion Roboflow: categoria 0 = padre (supercategory 'none'), 1 = componente.

    py -3 src/yolo2coco.py work/ds18
"""
import sys, os, json, shutil
import cv2


def convertir(ds, out):
    for split, dst in (('train', 'train'), ('val', 'valid'), ('val', 'test')):
        d = os.path.join(out, dst); os.makedirs(d, exist_ok=True)
        imgs, anns = [], []
        carpeta = os.path.join(ds, 'images', split)
        for k, f in enumerate(sorted(os.listdir(carpeta))):
            src = os.path.join(carpeta, f); tgt = os.path.join(d, f)
            if not os.path.exists(tgt):
                try: os.link(src, tgt)
                except OSError: shutil.copy2(src, tgt)
            H, W = cv2.imread(src, cv2.IMREAD_GRAYSCALE).shape
            imgs.append(dict(id=k, file_name=f, width=W, height=H))
            lp = os.path.join(ds, 'labels', split, os.path.splitext(f)[0] + '.txt')
            if os.path.exists(lp):
                for ln in open(lp):
                    p = ln.split()
                    if len(p) < 5: continue
                    cx, cy, w, h = float(p[1]) * W, float(p[2]) * H, float(p[3]) * W, float(p[4]) * H
                    anns.append(dict(id=len(anns), image_id=k, category_id=1, iscrowd=0, area=w * h,
                                     bbox=[cx - w / 2, cy - h / 2, w, h]))
        cats = [dict(id=0, name='componentes', supercategory='none'),
                dict(id=1, name='componente', supercategory='componentes')]
        # 28/09: escritura atomica (.tmp + replace): un corte de luz a mitad no deja un json truncado
        fj = os.path.join(d, '_annotations.coco.json')
        with open(fj + '.tmp', 'w') as h:
            json.dump(dict(images=imgs, annotations=anns, categories=cats), h)
        os.replace(fj + '.tmp', fj)
        print('%s: %d imagenes, %d cajas' % (dst, len(imgs), len(anns)))


if __name__ == '__main__':
    ds = sys.argv[1] if len(sys.argv) > 1 else os.path.join('work', 'ds18')
    out = ds.rstrip('/\\') + '_coco'
    convertir(ds, out)
    open(os.path.join(out, 'COMPLETO'), 'w').write('ok')   # 28/09: marca de conversion terminada (la mira la cadena)
