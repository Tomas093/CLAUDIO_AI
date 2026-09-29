import sys, json, cv2, ezdxf, numpy as np
sys.path.insert(0, '/home/claude/pack/claudio_v2/src')
from render import render_doc, cad2px
from postproc import darken
out=[]
for base in sys.argv[2:]:
    doc = ezdxf.readfile(base + '.dxf'); B = json.load(open(base + '.json'))
    img, meta = render_doc(doc, 2.0); img = darken(img)
    v = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for b in B:
        p0 = cad2px(meta, b[0], b[3]); p1 = cad2px(meta, b[2], b[1])
        cv2.rectangle(v, (int(p0[0]), int(p0[1])), (int(p1[0]), int(p1[1])), (0, 0, 255), 2)
    h = 1100; v = cv2.resize(v, (int(v.shape[1]*h/v.shape[0]), h)); out.append(v)
W = sum(x.shape[1] for x in out); canvas = np.hstack(out)
cv2.imwrite(sys.argv[1], canvas)
