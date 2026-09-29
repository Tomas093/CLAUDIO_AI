import sys,json,cv2
sys.path.insert(0,'/home/claude/pack/claudio_v2/src'); from numgrid import numbered_grid
n,bj,tag=sys.argv[1:4]; img=cv2.imread(f'/home/claude/lab/{n}.png',0); D=json.load(open(bj))
B=[it['bbox_px'] if isinstance(it,dict) else it for it in (D['boxes'] if isinstance(D,dict) else D)]
B.sort(key=lambda b:((b[1]+b[3])//400,b[0]))
x0=int(min(b[0] for b in B)-60);y0=int(min(b[1] for b in B)-60);x1=int(max(b[2] for b in B)+60);y1=int(max(b[3] for b in B)+60)
v=cv2.cvtColor(img,cv2.COLOR_GRAY2BGR);crops=[]
for i,b in enumerate(B):
    a,bb,c,dd=map(int,b);cv2.rectangle(v,(a,bb),(c,dd),(0,0,220),2);cv2.putText(v,str(i+1),(a,max(10,bb-3)),0,.5,(200,0,0),1,cv2.LINE_AA)
    pw,ph=int((c-a)*.4)+8,int((dd-bb)*.4)+8
    cc=cv2.cvtColor(img[max(0,bb-ph):dd+ph,max(0,a-pw):c+pw],cv2.COLOR_GRAY2BGR).copy()
    cv2.rectangle(cc,(min(pw,a),min(ph,bb)),(min(pw,a)+c-a,min(ph,bb)+dd-bb),(0,0,220),1);crops.append(cc)
O='/mnt/user-data/outputs/revision_planos/'
groups=[B] if (y1-y0)<5000 else [[b for b in B if b[1]<5000],[b for b in B if b[1]>=5000]]
for k,g in enumerate(groups):
    gx0=int(min(b[0] for b in g)-60);gy0=int(min(b[1] for b in g)-60);gx1=int(max(b[2] for b in g)+60);gy1=int(max(b[3] for b in g)+60)
    w=v[max(0,gy0):gy1,max(0,gx0):gx1]; s=min(1,7000/max(w.shape[:2]))
    cv2.imwrite(O+f'{tag}_A_plano_numerado{"" if len(groups)==1 else "_"+str(k+1)}.jpg',cv2.resize(w,None,fx=s,fy=s,interpolation=cv2.INTER_AREA),[cv2.IMWRITE_JPEG_QUALITY,90])
G=numbered_grid(crops,'',cell=150,cols=15,strip=16,colors=[(0,0,220)]*len(crops),start=0,digits=3,title=tag)
cv2.imwrite(O+f'{tag}_B_grilla.jpg',G,[cv2.IMWRITE_JPEG_QUALITY,90]);print(tag,len(B))
