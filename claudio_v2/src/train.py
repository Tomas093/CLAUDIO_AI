"""Uso: python src/train.py --model yolo11n.pt --name exp --epochs 100 [--hours 4.5] [--imgsz 640] [--batch 16]"""
import sys, os, argparse, torch
sys.path.insert(0, os.path.dirname(__file__))
from paths import BASE, WORK, DS, PACK
from ultralytics import YOLO

def clean(src, dst):
    ck = torch.load(src, map_location='cpu', weights_only=False)
    ck['optimizer'] = None
    for k in ('train_args', 'train_metrics', 'train_results', 'git', 'date', 'version', 'license', 'docs'):
        ck.pop(k, None)
    torch.save(ck, dst)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='yolo11n.pt'); ap.add_argument('--name', default='v2')
    ap.add_argument('--epochs', type=int, default=100); ap.add_argument('--hours', type=float, default=0)
    ap.add_argument('--imgsz', type=int, default=640); ap.add_argument('--batch', type=int, default=16)
    a = ap.parse_args()
    cargar = None  # pesos a transferir cuando el modelo se construye desde un yaml
    if a.name == 'M11_20':
        # 26/09, pedido de Tomas: yolo11m sobre ds20, para comparar en igualdad de tamanio contra
        # RF-DETR Nano (30 M parametros; yolo11m ~20 M, yolo11n 2,6 M). Sin esto no se sabe si RF-DETR
        # gana por ser transformer o solo por ser mas grande. Mismo dataset que RF4/RF5.
        DS = os.path.join(WORK, 'ds20')
        a.model = 'yolo11m.pt'
        a.batch = int(os.environ.get('BATCH', '8'))
        a.epochs = int(os.environ.get('M_EPOCHS', '200'))
        a.hours = 0.0
        print('[train] plan M11_20: yolo11m | dataset %s | epocas %d | batch %d' % (DS, a.epochs, a.batch))
    if a.name == 'N21':
        # 28/09, pedido de Tomas: dataset ds21 con lo que faltaba segun las fallas de RF4 (ver compose):
        # letra en recuadro (amperimetro A$C45664E16) con su par negativo de texto, rotulos verticales
        # grandes (test1), negativos nuevos (circulos numerados grandes, circulos que se cortan, flecha
        # de alimentacion) y VIS_MIN 0,85 (antes un simbolo visible al 50% por lado era positivo: la
        # mitad de los FP >= 0,5 de RF4 son pedazos de componentes). Base = receta de ds20 (U + fusible
        # DXF 0,15). P_LETRA 0,15: con 0,30 las cajas < 20 px subian a 15,3% (el plan O fallo con 16,3%);
        # con 0,15 queda 14,5% contra 13,5% de ds20. Nano primero, despues RF6 sobre el mismo ds21.
        import subprocess
        from receta_ds21 import RECETA
        ds21 = os.path.join(WORK, 'ds21')
        if not os.path.exists(os.path.join(ds21, 'data.yaml')):
            env = dict(os.environ, NP=os.environ.get('L_NP', '12000'), NN=os.environ.get('L_NN', '3500'),
                       NR=os.environ.get('L_NR', '900'), USAR_PSEUDO='1',
                       PSEUDO_ALTA=os.environ.get('PSEUDO_ALTA', '0.80'), CLAUDIO_DS=ds21)
            env.update({k: os.environ.get(k, v) for k, v in RECETA.items()})
            subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), 'build_all.py')], env=env, check=True)
        DS = ds21
        a.model = 'yolo11n.pt'
        a.epochs = int(os.environ.get('N21_EPOCHS', '200'))
        a.hours = 0.0
        print('[train] plan N21: yolo11n | dataset %s | epocas %d' % (DS, a.epochs))
    if a.name == 'Y26s_ds23':
        # 30/09 (goal 0 FN < 350 FP; informe de modelos): YOLO26s (STAL: asignacion pensada para objetos chicos) con
        # CLASE AUXILIAR de negativos duros (ds23: 1 = PAT/flecha/rotulo). A la salida solo clase 0
        # (evaluate.py con EVAL_CLASES_FUERA=1). Pesos yolo26s.pt los baja ultralytics.
        # 30/09: corre sobre ds23 (misma base que RF8); Y26_DS=ds24 para la variante con suciedad
        DS = os.path.join(WORK, os.environ.get('Y26_DS', 'ds23')); a.model = 'yolo26s.pt'; a.batch = int(os.environ.get('Y26_BATCH', '8'))
        a.epochs = int(os.environ.get('Y26_EPOCHS', '60')); a.hours = 0.0
        print('[train] plan Y26s_ds23: yolo26s | dataset %s | epocas %d' % (DS, a.epochs))
    if a.name in ('X_26p2', 'Y_11s', 'Z_26n'):
        # 22/09. Pedido de Tomas: otras arquitecturas con EXACTAMENTE el mismo dataset, para
        # atribuir la diferencia solo al modelo. Tomas pidio corregir primero el dataset de R:
        # corren sobre ds18 (el de R2_full, la comparacion justa es contra R2). XYZ_DS=ds13 vuelve.
        #   X: yolo26n-p2. Cabeza extra de stride 4 para lo chico (amperimetro ~17 px, C7AFF483F
        #      angosto). No hay pesos p2: se arma del yaml y se transfiere yolo26n.pt (360/902).
        #   Y: yolo11s. Comparacion; el proyecto es nano.
        #   Z: yolo26n. Ojo: YOLO26 es NMS-free (uno a uno), puede suprimir de mas en filas pegadas.
        # p2 y small van con batch 8: con 16 no entran en los ~5 GB que dejan libres las apps.
        DS = os.path.join(WORK, os.environ.get('XYZ_DS', 'ds18'))
        a.model, cargar, lote = {'X_26p2': ('yolo26n-p2.yaml', 'yolo26n.pt', 8),
                                 'Y_11s': ('yolo11s.pt', None, 8),
                                 'Z_26n': ('yolo26n.pt', None, 16)}[a.name]
        a.batch = int(os.environ.get('BATCH', lote))
        a.epochs = int(os.environ.get('XYZ_EPOCHS', '200'))
        a.hours = 0.0
        print('[train] plan %s: %s%s | dataset %s | epocas %d | batch %d'
              % (a.name, a.model, ' + ' + cargar if cargar else '', DS, a.epochs, a.batch))
    if a.name == 'E_largo':
        # Corrida larga nocturna: sigue desde el mejor de D, mismo dataset ds2, sin tope de horas (para sola por paciencia)
        DS = os.path.join(WORK, 'ds2')
        wd = os.path.join(WORK, 'runs', 'B_s_largo', 'weights', 'best.pt')
        a.model = wd if os.path.exists(wd) else os.path.join(WORK, 'runs', 'A_n_largo', 'weights', 'best.pt')
        a.epochs, a.hours = int(os.environ.get('E_EPOCHS', '250')), float(os.environ.get('E_HOURS', '0'))
        print('[train] plan E (largo): nano desde', a.model)
    if a.name == 'W_ft':
        # 22/09. Fine-tuning CORTO, segundo intento. El primero (V_ft, 30 epocas, lr 0,0015,
        # mosaic activo) SI aprendio el `PLC` pero costo 7 componentes: la precision cayo de
        # 0,442 a 0,401 y los falsos positivos subieron de 3968 a 4680. Con matching 1-a-1 esos
        # FP de mas compiten por las asignaciones y le roban pareja a componentes reales -el
        # mismo mecanismo que ya habia pasado con el plan Q y los contactores-.
        #
        # O sea: 30 epocas con mosaic sigue siendo re-entrenar, no ajustar. Este intento va al
        # otro extremo: 8 epocas, lr 3 veces mas baja y SIN mosaic, para que el modelo mueva el
        # sesgo de salida sin reorganizar lo que ya aprendio.
        DS = os.path.join(WORK, 'ds17')
        wp = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          'best_componente_v13_R.pt')
        if not os.path.exists(wp):
            raise SystemExit('falta el modelo de R: %s' % wp)
        a.model = wp
        a.epochs = int(os.environ.get('W_EPOCHS', '8'))
        a.hours = 0.0
        os.environ['LR0'] = os.environ.get('W_LR0', '0.0005')
        os.environ['LRF'] = '0.2'
        os.environ['WARMUP'] = '0.5'
        os.environ['MOSAIC'] = '0'
        os.environ['CLOSE_MOSAIC'] = '0'
        os.environ['PATIENCE'] = '20'
        print('[train] fine-tuning W (corto): desde %s | %d epocas | lr0 %s | sin mosaic'
              % (os.path.basename(a.model), a.epochs, os.environ['LR0']))
    if a.name == 'V_ft':
        # 22/09 (idea de Tomas): en vez de otro entrenamiento de 9 horas desde cero, un
        # FINE-TUNING corto partiendo del plan R, que ya esta en 1 FN de 3143.
        #
        # R falla en UNA sola cosa: el `PLC` en marco punteado de nyw-un-01, que nunca vio. Los
        # planes T y U demostraron que `compose.marco_punteado` lo resuelve, pero entrenados
        # desde cero perdieron otros componentes que R si encuentra. Partir de R y darle pocas
        # epocas con lr baja apunta a sumar el simbolo nuevo SIN pisar lo que ya sabe.
        #
        # Dataset: ds17, que es el de R mas los marcos punteados (ya construido por el plan U).
        DS = os.path.join(WORK, 'ds17')
        wp = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          'best_componente_v13_R.pt')
        if not os.path.exists(wp):
            raise SystemExit('falta el modelo de R: %s' % wp)
        a.model = wp
        a.epochs = int(os.environ.get('V_EPOCHS', '30'))
        a.hours = 0.0
        os.environ.setdefault('LR0', '0.0015')      # 1/7 de la lr normal
        os.environ.setdefault('LRF', '0.05')
        os.environ.setdefault('WARMUP', '1.0')
        os.environ.setdefault('PATIENCE', '30')
        os.environ.setdefault('CLOSE_MOSAIC', '8')
        print('[train] fine-tuning V: desde %s | dataset %s | epocas %d | lr0 %s'
              % (os.path.basename(a.model), DS, a.epochs, os.environ['LR0']))
    if a.name == 'U_full':
        # 22/09. Plan U = plan R + `marco_punteado`, Y NADA MAS.
        #
        # El plan T demostro que `compose.marco_punteado` SI resuelve el marco punteado del PLC
        # de nyw-un-01, que era el unico falso negativo de R. Pero T perdio 5 componentes que R
        # encuentra: tres `C4C456AEF` (recuadros chicos con una "R" en cursiva y "1300W"), un
        # `C7AFF483F` y un `Seccionador-Rotativo-01`.
        #
        # T traia DOS cambios sobre R -`marco_punteado` y `pat_negativo`- y con los datos que
        # hay no se puede atribuir la perdida a uno u otro: R (sin pat) pierde 0 de esos, S
        # (con pat, sin marco) pierde 1, T (con los dos) pierde 3. Sugiere el pat, pero el plan
        # M, que no lo tiene, pierde 4: la correlacion no cierra.
        #
        # Por eso U aisla UNA variable: R + marco_punteado, sin `pat_negativo`. Si U queda en 0
        # FN, listo. Si tambien pierde los `C4C456AEF`, el culpable es `marco_punteado` y hay
        # que buscarle otra forma al PLC. Es la regla de no tocar dos cosas a la vez (§7.4).
        #
        # Costo asumido: se resigna la mejora de precision del negativo del PAT (test1 de 0,639
        # a 0,706 en el plan S). El objetivo de Tomas prioriza el recall sobre todo lo demas.
        import subprocess
        ds17 = os.path.join(WORK, 'ds17')
        if not os.path.exists(os.path.join(ds17, 'data.yaml')):
            env = dict(os.environ, NP=os.environ.get('L_NP', '12000'), NN=os.environ.get('L_NN', '3500'),
                       NR=os.environ.get('L_NR', '900'), P_TABLERO='0.32',
                       P_SPM=os.environ.get('P_SPM', '0.45'), P_FUSIBLE=os.environ.get('P_FUSIBLE', '0.35'),
                       P_POLO=os.environ.get('P_POLO', '0.25'), P_DENSA=os.environ.get('P_DENSA', '0.28'),
                       P_BARRA=os.environ.get('P_BARRA', '0.10'), P_TRAFO=os.environ.get('P_TRAFO', '0.15'),
                       P_CELDAS=os.environ.get('P_CELDAS', '0.25'), P_COMPUESTO='0',
                       P_TERNA=os.environ.get('P_TERNA', '0.35'),
                       P_PULS=os.environ.get('P_PULS', '0.30'),
                       P_PAT='0',                      # <- la unica diferencia con el plan T
                       P_MARCO=os.environ.get('P_MARCO', '0.20'),
                       GENS_SIN_PUNTEADA='1', GENS_NOMBRE_SIMPLE='1',
                       PESO_TOMAS=os.environ.get('PESO_TOMAS', '6'),
                       USAR_PSEUDO='1', PSEUDO_ALTA=os.environ.get('PSEUDO_ALTA', '0.80'),
                       CLAUDIO_DS=ds17)
            subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), 'build_all.py')], env=env, check=True)
        DS = ds17
        a.model = 'yolo11n.pt'
        a.epochs = int(os.environ.get('U_EPOCHS', '200'))
        a.hours = 0.0
        print('[train] plan U: desde %s | dataset %s | epocas %d' % (a.model, ds17, a.epochs))
    if a.name == 'T_full':
        # 21/09. Plan T = plan R + las dos cosas que la medicion dejo demostradas.
        #
        # 1) `marco_punteado` (P_MARCO). El unico falso negativo que le queda a R en los 7
        #    planos es el `PLC / LOGICAS: / - TRANSF AUT` de nyw-un-01, y esta verificado que
        #    NO lo detecta: sus dos detecciones mas cercanas caen fuera del marco. El plan S
        #    intento arreglarlo con `procsym.caja_punteada` y fallo, y se midio por que: ese
        #    sprite pasa por `sym_instance`, que lo escala al tamano de un componente comun,
        #    mientras que el marco real mide 2,778 x 1,320 cuando el componente tipico de ese
        #    plano mide 0,913. Es TRES VECES mas ancho y el modelo nunca vio uno asi. Este
        #    generador lo dibuja a nivel de tile, con el tamano y la proporcion reales.
        #
        # 2) `pat_negativo` (P_PAT). Tomas decidio que el PAT no es un componente. En el plan S
        #    el negativo funciono: la precision de test1 subio de 0,639 a 0,706.
        #
        # NO lleva `caja_punteada` en GENS: no cumplio su proposito y `marco_punteado` cubre el
        # caso a la escala correcta.
        import subprocess
        ds16 = os.path.join(WORK, 'ds16')
        if not os.path.exists(os.path.join(ds16, 'data.yaml')):
            env = dict(os.environ, NP=os.environ.get('L_NP', '12000'), NN=os.environ.get('L_NN', '3500'),
                       NR=os.environ.get('L_NR', '900'), P_TABLERO='0.32',
                       P_SPM=os.environ.get('P_SPM', '0.45'), P_FUSIBLE=os.environ.get('P_FUSIBLE', '0.35'),
                       P_POLO=os.environ.get('P_POLO', '0.25'), P_DENSA=os.environ.get('P_DENSA', '0.28'),
                       P_BARRA=os.environ.get('P_BARRA', '0.10'), P_TRAFO=os.environ.get('P_TRAFO', '0.15'),
                       P_CELDAS=os.environ.get('P_CELDAS', '0.25'), P_COMPUESTO='0',
                       P_TERNA=os.environ.get('P_TERNA', '0.35'),
                       P_PULS=os.environ.get('P_PULS', '0.30'),
                       P_PAT=os.environ.get('P_PAT', '0.30'),
                       P_MARCO=os.environ.get('P_MARCO', '0.20'),
                       GENS_SIN_PUNTEADA='1', GENS_NOMBRE_SIMPLE='1',
                       PESO_TOMAS=os.environ.get('PESO_TOMAS', '6'),
                       USAR_PSEUDO='1', PSEUDO_ALTA=os.environ.get('PSEUDO_ALTA', '0.80'),
                       CLAUDIO_DS=ds16)
            subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), 'build_all.py')], env=env, check=True)
        DS = ds16
        a.model = 'yolo11n.pt'
        a.epochs = int(os.environ.get('T_EPOCHS', '200'))
        a.hours = 0.0
        print('[train] plan T: desde %s | dataset %s | epocas %d' % (a.model, ds16, a.epochs))
    if a.name == 'S_full':
        # 21/09. Plan S = plan R + el ultimo falso negativo.
        #
        # Con el GT v4 corregido R queda en 2 FN de 3117 (recall 0,9994) y de esos dos, uno es
        # un duplicado del propio GT. El unico fallo real del modelo es el `PLC / LOGICAS: /
        # TRANSF AUT` de nyw-un-01, que NINGUNO de los 13 modelos detecta: sus detecciones mas
        # cercanas llegan a confianza 0,087.
        #
        # La causa es del dataset, no del modelo. El recuadro de linea PUNTEADA aparece solo
        # como negativo -los marcos punteados vacios no son componentes, y esta bien que no lo
        # sean-, asi que el modelo generalizo "punteado = ignorar". Pero un recuadro punteado
        # CON el nombre de un equipo adentro si es un componente, igual que su equivalente de
        # linea llena. `procsym.caja_punteada` cubre ese segundo caso y entra en GENS.
        #
        # Segundo cambio, 21/09, decidido por Tomas: "el PAT no es un componente". Es el falso
        # positivo mas confiado que queda (8 de los 9 FP sobre 0,9 en test1, a 0,93-0,94) y va
        # directo contra la segunda mitad del objetivo: confianza baja en los falsos. Entra
        # como NEGATIVO explicito (`compose.pat_negativo`, P_PAT), porque el sprite ya estaba
        # fuera de la biblioteca y `pat` fuera de GENS: lo que lo mantiene vivo son las
        # pseudo-etiquetas, generadas con un modelo que ya lo detectaba.
        #
        # Los dos cambios se pueden atribuir por separado aunque vayan juntos, porque cada uno
        # tiene su propia metrica: el PLC es un FN y el PAT son FP de alta confianza.
        import subprocess
        ds14 = os.path.join(WORK, 'ds15')
        if not os.path.exists(os.path.join(ds14, 'data.yaml')):
            env = dict(os.environ, NP=os.environ.get('L_NP', '12000'), NN=os.environ.get('L_NN', '3500'),
                       NR=os.environ.get('L_NR', '900'), P_TABLERO='0.32',
                       P_SPM=os.environ.get('P_SPM', '0.45'), P_FUSIBLE=os.environ.get('P_FUSIBLE', '0.35'),
                       P_POLO=os.environ.get('P_POLO', '0.25'), P_DENSA=os.environ.get('P_DENSA', '0.28'),
                       P_BARRA=os.environ.get('P_BARRA', '0.10'), P_TRAFO=os.environ.get('P_TRAFO', '0.15'),
                       P_CELDAS=os.environ.get('P_CELDAS', '0.25'), P_COMPUESTO='0',
                       P_TERNA=os.environ.get('P_TERNA', '0.35'),
                       P_PULS=os.environ.get('P_PULS', '0.30'),
                       P_PAT=os.environ.get('P_PAT', '0.30'),
                       GENS_NOMBRE_SIMPLE='1',
                       PESO_TOMAS=os.environ.get('PESO_TOMAS', '6'),
                       USAR_PSEUDO='1', PSEUDO_ALTA=os.environ.get('PSEUDO_ALTA', '0.80'),
                       CLAUDIO_DS=ds14)
            subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), 'build_all.py')], env=env, check=True)
        DS = ds14
        a.model = 'yolo11n.pt'
        a.epochs = int(os.environ.get('S_EPOCHS', '200'))
        a.hours = 0.0
        print('[train] plan S: desde %s | dataset %s | epocas %d' % (a.model, ds14, a.epochs))
    if a.name in ('R_full', 'R2_full'):
        # 22/09. R2_full: la MISMA receta que R pero con el dataset corregido (ds18): sin los 6
        # PAT de real_labels.json, sin las 492 etiquetas sobre simbolos blanqueados de los zips y
        # con los tiles reales repartidos por cantidad de cajas (ver build_all.py).
        # 20/09. Plan R: junta lo mejor de N y de Q, que fallan en cosas distintas.
        #
        # Medido con el GT v4 (sin duplicados ni cajas inventadas), al mismo umbral 0,05:
        #   N: 11 FN. Se le escapan 7 amperimetros `A$C45664E16` y ningun `PULS`.
        #   Q: 16 FN. Se le escapan 2 amperimetros y 7 `PULS`.
        # Son complementarios: un modelo con las dos virtudes quedaria en ~4 FN.
        #
        #  - El amperimetro lo arreglo Q y NO estaba medido: el arreglo de `procsym.caja_letras`
        #    (una sola letra, cursiva, cable pegado, §5.9) le hace localizar los 16 de 16, pero
        #    con confianza mediana 0,23, y los planos de Marcelo se evaluaban a conf 0,20. La
        #    conclusion vieja ("no rindio: 15 -> 14") era un artefacto del umbral. Se MANTIENE.
        #  - Los `PULS` Q no los pierde: se los TRAGA. De los 25 del GT saca caja propia para 6
        #    y mete 18 adentro de la caja del contactor (N saca 21 propias). Eso es una caja con
        #    dos componentes, que es lo que mas complica la etapa 2.
        #
        # Cambios sobre Q, todos apuntando a que no junte simbolos vecinos:
        #   - `puls_contactor` (P_PULS): dibuja el par contactor+pulsador como esta en el plano
        #     y lo etiqueta con DOS cajas. Ataca el defecto en el caso exacto donde se mide.
        #   - `caja_nombre` vuelve a peso simple (GENS_NOMBRE_SIMPLE=1). Con peso doble ensena
        #     "region con trazos y texto adentro = una caja", que es el mecanismo de la fusion.
        #     `caja_letras` queda en doble: es el generador del amperimetro y no es el que fallo.
        #   - `P_CELDAS` 0,18 -> 0,25 y vocabulario real de planilla en `tabla_celdas`. Ojo: esto
        #     es lo de MENOS. La metrica propia (`src/metrica_tablas.py`) dice que Q pone 25
        #     detecciones sobre celdas y N 17: 0,9% de las detecciones, no "casi todas las
        #     celdas" como se creia. La diferencia real entre N y Q estaba en los PULS.
        # Todo lo demas es identico a Q para poder atribuir la diferencia.
        import subprocess
        ds13 = os.path.join(WORK, 'ds18' if a.name == 'R2_full' else 'ds13')
        if not os.path.exists(os.path.join(ds13, 'data.yaml')):
            env = dict(os.environ, NP=os.environ.get('L_NP', '12000'), NN=os.environ.get('L_NN', '3500'),
                       NR=os.environ.get('L_NR', '900'), P_TABLERO='0.32',
                       P_SPM=os.environ.get('P_SPM', '0.45'), P_FUSIBLE=os.environ.get('P_FUSIBLE', '0.35'),
                       P_POLO=os.environ.get('P_POLO', '0.25'), P_DENSA=os.environ.get('P_DENSA', '0.28'),
                       P_BARRA=os.environ.get('P_BARRA', '0.10'), P_TRAFO=os.environ.get('P_TRAFO', '0.15'),
                       P_CELDAS=os.environ.get('P_CELDAS', '0.25'), P_COMPUESTO='0',
                       P_TERNA=os.environ.get('P_TERNA', '0.35'),
                       P_PULS=os.environ.get('P_PULS', '0.30'),
                       GENS_NOMBRE_SIMPLE='1',
                       PESO_TOMAS=os.environ.get('PESO_TOMAS', '6'),
                       USAR_PSEUDO='1', PSEUDO_ALTA=os.environ.get('PSEUDO_ALTA', '0.80'),
                       CLAUDIO_DS=ds13)
            subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), 'build_all.py')], env=env, check=True)
        DS = ds13
        a.model = 'yolo11n.pt'
        a.epochs = int(os.environ.get('R_EPOCHS', '200'))
        a.hours = 0.0
        print('[train] plan %s: desde %s | dataset %s | epocas %d' % (a.name, a.model, ds13, a.epochs))
    if a.name == 'Q_full':
        # 19/09. Un revisor de vision audito las 2106 detecciones sobre los 11 PDF de Marcelo
        # (material en revision_ia/, lo lee src/leer_revision.py). Dos resultados:
        #
        #  1. 340 de los 536 componentes perdidos estaban en cuatro planos que `detectar.py`
        #     analizaba DE CABEZA, por un error de sentido en el auto-enderezado. Eso se
        #     arreglo en detectar.py y no es cosa del dataset.
        #  2. De los 196 que quedan en planos bien orientados, el 73% son fusibles (92) y
        #     lamparas (51), y las cajas que abarcaban varios componentes eran justo ternas
        #     R/S/T con los tres adentro de una sola caja. Es el mismo problema visto de los
        #     dos lados, y es lo que Tomas pidio resolver desde el principio.
        #
        # De ahi sale `terna_rst` (P_TERNA): pocas columnas del mismo simbolo colgando de una
        # barra, con la letra de fase debajo y un segundo piso, cada simbolo con su caja.
        # Lleva ademas lo del plan P (rotulos de varias lineas).
        import subprocess
        ds12 = os.path.join(WORK, 'ds12')
        if not os.path.exists(os.path.join(ds12, 'data.yaml')):
            env = dict(os.environ, NP=os.environ.get('L_NP', '12000'), NN=os.environ.get('L_NN', '3500'),
                       NR=os.environ.get('L_NR', '900'), P_TABLERO='0.32',
                       P_SPM=os.environ.get('P_SPM', '0.45'), P_FUSIBLE=os.environ.get('P_FUSIBLE', '0.35'),
                       P_POLO=os.environ.get('P_POLO', '0.25'), P_DENSA=os.environ.get('P_DENSA', '0.28'),
                       P_BARRA=os.environ.get('P_BARRA', '0.10'), P_TRAFO=os.environ.get('P_TRAFO', '0.15'),
                       P_CELDAS=os.environ.get('P_CELDAS', '0.18'), P_COMPUESTO='0',
                       P_TERNA=os.environ.get('P_TERNA', '0.35'),
                       PESO_TOMAS=os.environ.get('PESO_TOMAS', '6'),
                       USAR_PSEUDO='1', PSEUDO_ALTA=os.environ.get('PSEUDO_ALTA', '0.80'),
                       CLAUDIO_DS=ds12)
            subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), 'build_all.py')], env=env, check=True)
        DS = ds12
        a.model = 'yolo11n.pt'
        a.epochs = int(os.environ.get('Q_EPOCHS', '200'))
        a.hours = 0.0
        print('[train] plan Q: desde %s | dataset %s | epocas %d' % (a.model, ds12, a.epochs))
    if a.name == 'P_full':
        # 19/09. Tomas marco sobre los PDF de Marcelo que hay muchas cajas mal puestas. Mirando
        # los 12 peores casos de las 2106 detecciones de N, casi todas caen sobre lo mismo:
        # recuadros con texto de 2-3 lineas ('Iso-Gard IG6', 'Fuente 24 VCC 2 A', 'UPS 6 kVA
        # 15 min', 'VigilOhm IM400'), donde el modelo pone 3 o 4 cajas superpuestas y
        # desalineadas en vez de una sola. El generador ya tenia `caja_nombre` para esto pero
        # con 8 rotulos fijos, ninguno parecido a los de estos planos, y pesaba 1 de 9.
        # Aca va con 20 rotulos (incluidos los reales) y peso doble, mas el negativo que
        # faltaba: notas de 2-4 lineas alineadas SIN recuadro, que no llevan ninguna caja.
        # Todo lo demas es igual a O (simbolos chicos) mas real_labels.json ya cenido.
        import subprocess
        ds11 = os.path.join(WORK, 'ds11')
        if not os.path.exists(os.path.join(ds11, 'data.yaml')):
            env = dict(os.environ, NP=os.environ.get('L_NP', '12000'), NN=os.environ.get('L_NN', '3500'),
                       NR=os.environ.get('L_NR', '900'), P_TABLERO='0.32',
                       P_SPM=os.environ.get('P_SPM', '0.45'), P_FUSIBLE=os.environ.get('P_FUSIBLE', '0.35'),
                       P_POLO=os.environ.get('P_POLO', '0.25'), P_DENSA=os.environ.get('P_DENSA', '0.28'),
                       P_BARRA=os.environ.get('P_BARRA', '0.10'), P_TRAFO=os.environ.get('P_TRAFO', '0.15'),
                       P_CELDAS=os.environ.get('P_CELDAS', '0.18'), P_COMPUESTO='0',
                       PESO_TOMAS=os.environ.get('PESO_TOMAS', '6'),
                       USAR_PSEUDO='1', PSEUDO_ALTA=os.environ.get('PSEUDO_ALTA', '0.80'),
                       CLAUDIO_DS=ds11)
            subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), 'build_all.py')], env=env, check=True)
        DS = ds11
        a.model = 'yolo11n.pt'
        a.epochs = int(os.environ.get('P_EPOCHS', '200'))
        a.hours = 0.0
        print('[train] plan P: desde %s | dataset %s | epocas %d' % (a.model, ds11, a.epochs))
    if a.name == 'O_full':
        # 19/09. N dejo 10 falsos negativos y 7 son el mismo componente: el amperimetro
        # A$C45664E16 de LU-UN-01 y nyw-un-01. Mide 0.22 unidades CAD, o sea 13 px a la escala
        # del detector, y N lo encuentra bien (IoU 0.82-0.93) pero con confianza 0.06-0.17.
        # La causa es el dataset: tenia 13% de cajas de menos de 20 px cuando en los planos
        # reales son 27-32%. Los simbolos chicos estaban subrepresentados. Aca el tramo chico
        # de `sym_instance` es explicito (1.10-1.95 veces la altura de texto, 45% de las veces)
        # y el margen del chequeo de solapamiento es proporcional al simbolo: 22% de cajas de
        # menos de 20 px y 9.2% de menos de 16.
        import subprocess
        ds10 = os.path.join(WORK, 'ds10')
        if not os.path.exists(os.path.join(ds10, 'data.yaml')):
            env = dict(os.environ, NP=os.environ.get('L_NP', '12000'), NN=os.environ.get('L_NN', '3500'),
                       NR=os.environ.get('L_NR', '900'), P_TABLERO='0.32',
                       P_SPM=os.environ.get('P_SPM', '0.45'), P_FUSIBLE=os.environ.get('P_FUSIBLE', '0.35'),
                       P_POLO=os.environ.get('P_POLO', '0.25'), P_DENSA=os.environ.get('P_DENSA', '0.28'),
                       P_BARRA=os.environ.get('P_BARRA', '0.10'), P_TRAFO=os.environ.get('P_TRAFO', '0.15'),
                       P_CELDAS=os.environ.get('P_CELDAS', '0.18'), P_COMPUESTO='0',
                       PESO_TOMAS=os.environ.get('PESO_TOMAS', '6'),
                       USAR_PSEUDO='1', PSEUDO_ALTA=os.environ.get('PSEUDO_ALTA', '0.80'),
                       CLAUDIO_DS=ds10)
            subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), 'build_all.py')], env=env, check=True)
        DS = ds10
        a.model = 'yolo11n.pt'
        a.epochs = int(os.environ.get('O_EPOCHS', '200'))
        a.hours = 0.0
        print('[train] plan O: desde %s | dataset %s | epocas %d' % (a.model, ds10, a.epochs))
    if a.name == 'N_full':
        # 19/09. M mejoro el recall (LU-UN-01 de 21 a 7 falsos negativos) pero empeoro el
        # encuadre: IoU global 0.832 contra 0.861 de L, y volvieron las cajas que abarcan dos
        # componentes (3 y 2 a conf 0.25, cuando L tenia 0). La causa es que las pseudo
        # etiquetas se generaron con K, que encuadra peor (IoU 0.810), asi que el modelo
        # heredo su forma de encajonar junto con la ubicacion. Aca se regeneran con L, que es
        # el que mejor encuadra, para quedarse con el recall de M y el encuadre de L.
        # Ademas entra `real_labels.json` cenido (24.2% -> 0.7% de cajas con aire) y las zonas
        # neutras ya no borran simbolos etiquetados.
        import subprocess
        ds9 = os.path.join(WORK, 'ds9')
        if not os.path.exists(os.path.join(ds9, 'data.yaml')):
            env = dict(os.environ, NP=os.environ.get('L_NP', '12000'), NN=os.environ.get('L_NN', '3500'),
                       NR=os.environ.get('L_NR', '900'), P_TABLERO='0.32',
                       P_SPM=os.environ.get('P_SPM', '0.45'), P_FUSIBLE=os.environ.get('P_FUSIBLE', '0.35'),
                       P_POLO=os.environ.get('P_POLO', '0.25'), P_DENSA=os.environ.get('P_DENSA', '0.28'),
                       P_BARRA=os.environ.get('P_BARRA', '0.10'), P_TRAFO=os.environ.get('P_TRAFO', '0.15'),
                       P_CELDAS=os.environ.get('P_CELDAS', '0.18'), P_COMPUESTO='0',
                       PESO_TOMAS=os.environ.get('PESO_TOMAS', '6'),
                       USAR_PSEUDO='1', PSEUDO_ALTA=os.environ.get('PSEUDO_ALTA', '0.80'),
                       CLAUDIO_DS=ds9)
            subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), 'build_all.py')], env=env, check=True)
        DS = ds9
        a.model = 'yolo11n.pt'
        a.epochs = int(os.environ.get('N_EPOCHS', '200'))
        a.hours = 0.0
        print('[train] plan N: desde %s | dataset %s | epocas %d' % (a.model, ds9, a.epochs))
    if a.name == 'M_full':
        # 18/09. Igual que L, mas la correccion de los zips anotados a mano.
        # Hallazgo: cada zip estaba anotado para UN tipo de componente y el resto del tablero
        # quedaba sin etiqueta, o sea miles de negativos falsos justo donde hay componentes
        # reales (hasta 6,6 veces mas sin etiquetar que etiquetados). Se completan con
        # src/pseudo_zips.py: lo firme entra como etiqueta y lo dudoso se blanquea.
        import subprocess
        ds8 = os.path.join(WORK, 'ds8')
        if not os.path.exists(os.path.join(ds8, 'data.yaml')):
            env = dict(os.environ, NP=os.environ.get('L_NP', '12000'), NN=os.environ.get('L_NN', '3500'),
                       NR=os.environ.get('L_NR', '900'), P_TABLERO='0.32',
                       P_SPM=os.environ.get('P_SPM', '0.45'), P_FUSIBLE=os.environ.get('P_FUSIBLE', '0.35'),
                       P_POLO=os.environ.get('P_POLO', '0.25'), P_DENSA=os.environ.get('P_DENSA', '0.28'),
                       P_BARRA=os.environ.get('P_BARRA', '0.10'), P_TRAFO=os.environ.get('P_TRAFO', '0.15'),
                       P_CELDAS=os.environ.get('P_CELDAS', '0.18'), P_COMPUESTO='0',
                       PESO_TOMAS=os.environ.get('PESO_TOMAS', '6'),
                       USAR_PSEUDO='1', PSEUDO_ALTA=os.environ.get('PSEUDO_ALTA', '0.80'),
                       CLAUDIO_DS=ds8)
            subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), 'build_all.py')], env=env, check=True)
        DS = ds8
        a.model = 'yolo11n.pt'
        a.epochs = int(os.environ.get('M_EPOCHS', '200'))
        a.hours = 0.0
        print('[train] plan M: desde %s | dataset %s | epocas %d' % (a.model, ds8, a.epochs))
    if a.name == 'L_full':
        # 18/09. Mismo criterio que K, con el generador y la biblioteca saneados:
        #   - ninguna etiqueta puede quedar adentro de otra (eran 58 pares cada 1500 tiles).
        #     `fila_densa` se encimaba con `spm_branch` y `tablero_row`, y `trafo_o_pareja`
        #     elegia lugar sin mirar nada. Motivo: las cajas que abarcan dos componentes le
        #     ensucian a Tomas la etapa 2 de clasificacion.
        #   - `fila_densa` ahora tambien hace filas de simbolos DISTINTOS pegados.
        #   - la biblioteca perdio los sprites que no son componentes (el texto suelto y los
        #     recortes de plano que Tomas marco uno por uno) y los que estaban vacios.
        import subprocess
        ds7 = os.path.join(WORK, 'ds7')
        if not os.path.exists(os.path.join(ds7, 'data.yaml')):
            env = dict(os.environ, NP=os.environ.get('L_NP', '12000'), NN=os.environ.get('L_NN', '3500'),
                       NR=os.environ.get('L_NR', '900'), P_TABLERO='0.32',
                       P_SPM=os.environ.get('P_SPM', '0.45'), P_FUSIBLE=os.environ.get('P_FUSIBLE', '0.35'),
                       P_POLO=os.environ.get('P_POLO', '0.25'), P_DENSA=os.environ.get('P_DENSA', '0.28'),
                       P_BARRA=os.environ.get('P_BARRA', '0.10'), P_TRAFO=os.environ.get('P_TRAFO', '0.15'),
                       P_CELDAS=os.environ.get('P_CELDAS', '0.18'), P_COMPUESTO='0',
                       PESO_TOMAS=os.environ.get('PESO_TOMAS', '6'), CLAUDIO_DS=ds7)
            subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), 'build_all.py')], env=env, check=True)
        DS = ds7
        a.model = 'yolo11n.pt'
        a.epochs = int(os.environ.get('L_EPOCHS', '200'))
        a.hours = 0.0
        print('[train] plan L: desde %s | dataset %s | epocas %d' % (a.model, ds7, a.epochs))
    if a.name in ('J_ft', 'K_full'):
        # 17/09, criterio final de Tomas:
        #   - el fusible y el ojo de buey son componentes SEPARADOS (nunca una sola caja)
        #   - entran los simbolos reales que paso: diferencial, 3 fusibles, ojo de buey, etc.
        #   - los GT ya son oficiales con la caja ajustada al dibujo (sin el texto del atributo)
        # J: fine tuning corto desde G.   K: entrenamiento largo desde cero.
        import subprocess
        ds6 = os.path.join(WORK, 'ds6')
        if not os.path.exists(os.path.join(ds6, 'data.yaml')):
            env = dict(os.environ, NP=os.environ.get('J_NP', '12000'), NN=os.environ.get('J_NN', '3500'),
                       NR=os.environ.get('J_NR', '900'), P_TABLERO='0.32',
                       P_SPM=os.environ.get('P_SPM', '0.45'), P_FUSIBLE=os.environ.get('P_FUSIBLE', '0.35'),
                       P_POLO=os.environ.get('P_POLO', '0.25'), P_DENSA=os.environ.get('P_DENSA', '0.28'),
                       P_BARRA=os.environ.get('P_BARRA', '0.10'), P_TRAFO=os.environ.get('P_TRAFO', '0.15'),
                       P_CELDAS=os.environ.get('P_CELDAS', '0.18'), P_COMPUESTO='0',
                       PESO_TOMAS=os.environ.get('PESO_TOMAS', '6'), CLAUDIO_DS=ds6)
            subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), 'build_all.py')], env=env, check=True)
        DS = ds6
        if a.name == 'J_ft':
            wp = os.path.join(WORK, 'runs', 'G_scratch', 'weights', 'best.pt')
            if not os.path.exists(wp): raise SystemExit('falta el modelo de G: %s' % wp)
            a.model = wp
            a.epochs = int(os.environ.get('J_EPOCHS', '15'))
        else:
            a.model = 'yolo11n.pt'
            a.epochs = int(os.environ.get('K_EPOCHS', '200'))
        a.hours = 0.0
        print('[train] plan %s: desde %s | dataset %s | epocas %d' % (a.name, a.model, ds6, a.epochs))
    if a.name in ('H2_ft', 'I_ft'):
        # Segunda vuelta del fine tuning (17/09), con lo aprendido de H:
        #   - el ojo de buey y el fusible tambien se generan SOLOS (H perdia el ojo suelto)
        #   - mas negativos (H bajo de 3500 a 600 y los FP subieron ~30%)
        #   - menos epocas, para corregir sin desaprender
        # H2 parte de H (ya tiene el testigo resuelto); I parte de G (limpio).
        import subprocess
        ds5 = os.path.join(WORK, 'ds5')
        if not os.path.exists(os.path.join(ds5, 'data.yaml')):
            env = dict(os.environ, NP=os.environ.get('H2_NP', '6000'), NN=os.environ.get('H2_NN', '2000'),
                       NR=os.environ.get('H2_NR', '500'), P_TABLERO='0.32',
                       P_SPM=os.environ.get('P_SPM', '0.50'), P_FUSIBLE=os.environ.get('P_FUSIBLE', '0.35'),
                       P_POLO=os.environ.get('P_POLO', '0.25'), P_DENSA=os.environ.get('P_DENSA', '0.28'),
                       P_BARRA=os.environ.get('P_BARRA', '0.10'), P_TRAFO=os.environ.get('P_TRAFO', '0.15'),
                       P_CELDAS=os.environ.get('P_CELDAS', '0.18'), P_COMPUESTO='0',
                       PESO_TOMAS=os.environ.get('PESO_TOMAS', '5'), CLAUDIO_DS=ds5)
            subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), 'build_all.py')], env=env, check=True)
        DS = ds5
        base_run = 'H_ft' if a.name == 'H2_ft' else 'G_scratch'
        wp = os.path.join(WORK, 'runs', base_run, 'weights', 'best.pt')
        if not os.path.exists(wp):
            raise SystemExit('falta el modelo base %s' % wp)
        a.model = wp
        a.epochs, a.hours = int(os.environ.get('H2_EPOCHS', '12')), 0.0
        print('[train] plan %s: desde %s | dataset %s | epocas %d' % (a.name, wp, ds5, a.epochs))
    if a.name == 'H_ft':
        # Fine tuning corto sobre G (17/09). Dataset CHICO y enfocado en lo que falla en los
        # planos reales, con los simbolos que paso Tomas (data/diferencial.dxf):
        #   - testigo de tension = fusible + ojo de buey en UNA caja  (hoy lo parte al medio)
        #   - fusibles con mas peso                                   (el simbolo peor detectado)
        #   - polos sueltos como NEGATIVO                             (hoy los marca como componente)
        # Pocas epocas y pocos tiles: el objetivo es corregir, no reaprender.
        import subprocess
        ds4 = os.path.join(WORK, 'ds4')
        if not os.path.exists(os.path.join(ds4, 'data.yaml')):
            env = dict(os.environ, NP=os.environ.get('H_NP', '3000'), NN=os.environ.get('H_NN', '600'),
                       NR=os.environ.get('H_NR', '250'), P_TABLERO='0.30',
                       P_SPM=os.environ.get('P_SPM', '0.55'), P_FUSIBLE=os.environ.get('P_FUSIBLE', '0.45'),
                       P_POLO=os.environ.get('P_POLO', '0.30'), P_DENSA=os.environ.get('P_DENSA', '0.20'),
                       P_BARRA=os.environ.get('P_BARRA', '0.08'), P_TRAFO=os.environ.get('P_TRAFO', '0.10'),
                       P_CELDAS=os.environ.get('P_CELDAS', '0.12'), P_COMPUESTO='0',
                       PESO_TOMAS=os.environ.get('PESO_TOMAS', '8'), CLAUDIO_DS=ds4)
            subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), 'build_all.py')], env=env, check=True)
        DS = ds4
        wg = os.path.join(WORK, 'runs', 'G_scratch', 'weights', 'best.pt')   # parte de G
        if not os.path.exists(wg):
            raise SystemExit('no esta el modelo de G en %s' % wg)
        a.model = wg
        a.epochs, a.hours = int(os.environ.get('H_EPOCHS', '25')), 0.0
        print('[train] plan H (fine tuning): desde', a.model, 'dataset', ds4, '| epocas', a.epochs)
    if a.name == 'G_scratch':
        # Modelo NUEVO (no parte de D): backbone yolo11n preentrenado + ds3, el dataset que ya
        # incluye todos los casos que fallaban en los planos reales del cliente (SPM, filas
        # densas, barras, trafo vs par de interruptores, celdas de tabla, sin PAT).
        # Corta por epocas y por paciencia, sin tope de reloj (pedido de Tomas 17/09).
        import subprocess
        ds3 = os.path.join(WORK, 'ds3')
        if not os.path.exists(os.path.join(ds3, 'data.yaml')):
            env = dict(os.environ, NP=os.environ.get('G_NP', '12000'), NN=os.environ.get('G_NN', '3500'),
                       NR=os.environ.get('G_NR', '900'), P_TABLERO='0.35', P_SPM=os.environ.get('P_SPM', '0.30'),
                       P_DENSA=os.environ.get('P_DENSA', '0.30'), P_BARRA=os.environ.get('P_BARRA', '0.12'),
                       P_TRAFO=os.environ.get('P_TRAFO', '0.18'), P_CELDAS=os.environ.get('P_CELDAS', '0.22'),
                       CLAUDIO_DS=ds3)
            subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), 'build_all.py')], env=env, check=True)
        DS = ds3
        a.model = 'yolo11n.pt'
        a.epochs, a.hours = int(os.environ.get('G_EPOCHS', '200')), 0.0
        print('[train] plan G (desde cero): backbone', a.model, 'dataset', ds3, '| epocas', a.epochs,
              '| patience', os.environ.get('PATIENCE', '40'))
    if a.name == 'F_spm':
        # Ajuste fino sobre los fallos vistos en planos reales (16/09). Parte del mejor de D y
        # entrena pocas epocas sobre un dataset nuevo (ds3) que es como ds2 pero con:
        #   P_SPM   portafusible SPM inclinado + lampara piloto  -> 14 de los 15 FN de D
        #   P_DENSA filas de simbolos pegados, caja por simbolo   -> 24 cajas fusionadas en EZE4077
        #   P_BARRA barras colectoras gruesas como negativo       -> FP en cadena en los IE-UNI
        # Se genera el dataset COMPLETO (no solo los casos nuevos) para no olvidar lo ya aprendido.
        import subprocess
        ds3 = os.path.join(WORK, 'ds3')
        if not os.path.exists(os.path.join(ds3, 'data.yaml')):
            env = dict(os.environ, NP=os.environ.get('F_NP', '12000'), NN=os.environ.get('F_NN', '3500'),
                       NR=os.environ.get('F_NR', '900'), P_TABLERO='0.35', P_SPM=os.environ.get('P_SPM', '0.30'),
                       P_DENSA=os.environ.get('P_DENSA', '0.30'), P_BARRA=os.environ.get('P_BARRA', '0.12'),
                       P_TRAFO=os.environ.get('P_TRAFO', '0.18'), P_CELDAS=os.environ.get('P_CELDAS', '0.22'),
                       CLAUDIO_DS=ds3)
            subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), 'build_all.py')], env=env, check=True)
        DS = ds3
        wd = os.path.join(WORK, 'runs', 'B_s_largo', 'weights', 'best.pt')
        a.model = wd
        a.epochs, a.hours = int(os.environ.get('F_EPOCHS', '18')), float(os.environ.get('F_HOURS', '0'))
        print('[train] plan F (ajuste fino SPM/densa/barra): desde', a.model, 'dataset', ds3)
    if a.name == 'B_s_largo':
        # CAMBIO de plan (pedido de Tomas): se saltea el small. En su lugar: dataset v2b (borneras/selectoras/pulsadores
        # sobre borde punteado + Vyre etiquetado) y nano largo partiendo de los pesos de A.
        import subprocess, json as _j
        ds2 = os.path.join(WORK, 'ds2')
        if not os.path.exists(os.path.join(ds2, 'data.yaml')):
            env = dict(os.environ, NP='24000', NN='3000', NR='1200', P_TABLERO='0.35', CLAUDIO_DS=ds2)
            subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), 'build_all.py')], env=env, check=True)
        DS = ds2
        wa = os.path.join(WORK, 'runs', 'A_n_largo', 'weights', 'best.pt')
        if not os.path.exists(wa): wa = os.path.join(WORK, 'runs', 'A_n_largo', 'weights', 'best.pt')
        a.model = wa if os.path.exists(wa) else 'yolo11n.pt'
        a.epochs, a.hours = 150, float(os.environ.get('D_HOURS', '10'))
        print('[train] plan D: nano desde', a.model, 'dataset', ds2)
    for cand in (os.path.join(BASE, a.model), os.path.join(BASE, 'train-maker', a.model)):
        if os.path.exists(cand): a.model = cand; break
    dev = 0 if torch.cuda.is_available() else 'cpu'
    print('[train]', a, 'device', dev, torch.cuda.get_device_name(0) if dev == 0 else '')
    kw = dict(data=os.path.join(DS, 'data.yaml'), imgsz=a.imgsz, epochs=a.epochs, batch=a.batch, device=dev, workers=6,
              project=os.path.join(WORK, 'runs'), name=a.name, exist_ok=True, patience=int(os.environ.get('PATIENCE', '40')), cos_lr=True,
              hsv_h=0, hsv_s=0, hsv_v=.2, degrees=0, translate=.1, scale=.2, fliplr=.2, flipud=0,
              mosaic=1.0, close_mosaic=10, plots=True, amp=True)
    # 22/09: lr y mosaic configurables por entorno. Un fine-tuning parte de un modelo que ya
    # sabe y solo tiene que aprender un simbolo mas: con la lr de un entrenamiento desde cero
    # (0,01) le pisa lo aprendido, que es justo lo que hay que evitar.
    for var, clave, conv in (('LR0', 'lr0', float), ('LRF', 'lrf', float),
                             ('WARMUP', 'warmup_epochs', float), ('MOSAIC', 'mosaic', float),
                             ('CLOSE_MOSAIC', 'close_mosaic', int)):
        if os.environ.get(var):
            kw[clave] = conv(os.environ[var])
    if a.hours > 0: kw['time'] = a.hours
    modelo = YOLO(a.model)
    if cargar:
        for cand in (os.path.join(BASE, cargar), os.path.join(os.path.dirname(os.path.dirname(__file__)), cargar)):
            if os.path.exists(cand): cargar = cand; break
        modelo.load(cargar)
    modelo.train(**kw)
    best = os.path.join(WORK, 'runs', a.name, 'weights', 'best.pt')
    out = os.path.join(WORK, 'runs', a.name, 'best_limpio.pt'); clean(best, out)
    print('[train] pesos limpios ->', out)
