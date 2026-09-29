import sys,json,cv2
# ctx.py plan boxesjson id1,id2.. out  [pad]
n,bj,ids,out=sys.argv[1:5]; pad=int(sys.argv[5]) if len(sys.argv)>5 else 120
img=cv2.imread(f'/home/claude/lab/{n}.png',0); B=json.load(open(bj))
B=B['cajas'] if isinstance(B,dict) else B
get=lambda it: it['bbox_px']
tiles=[]
for i in map(int,ids.split(',')):
    x0,y0,x1,y1=map(int,get(B[i-1])); X0,Y0=max(0,x0-pad),max(0,y0-pad); X1,Y1=x1+pad,y1+pad
    c=cv2.cvtColor(img[Y0:Y1,X0:X1],cv2.COLOR_GRAY2BGR)
    for j,it in enumerate(B):
        a=list(map(int,get(it)))
        if a[2]>X0 and a[0]<X1 and a[3]>Y0 and a[1]<Y1:
            col=(0,0,255) if j==i-1 else (0,170,0)
            cv2.rectangle(c,(a[0]-X0,a[1]-Y0),(a[2]-X0,a[3]-Y0),col,1)
    s=3 if max(c.shape[:2])<300 else 2
    c=cv2.resize(c,None,fx=s,fy=s,interpolation=cv2.INTER_NEAREST)
    for gx in range((X0//20+1)*20,X1,20):
        cv2.line(c,((gx-X0)*s,0),((gx-X0)*s,6),(255,0,0),1)
        if gx%100==0: cv2.putText(c,str(gx),((gx-X0)*s+2,16),0,.4,(255,0,0),1)
    for gy in range((Y0//20+1)*20,Y1,20):
        cv2.line(c,(0,(gy-Y0)*s),(6,(gy-Y0)*s),(255,0,0),1)
        if gy%100==0: cv2.putText(c,str(gy),(8,(gy-Y0)*s+4),0,.4,(255,0,0),1)
    cv2.putText(c,f'#{i}',(10,c.shape[0]-8),0,.8,(0,0,255),2)
    cv2.imwrite(out.replace('.jpg',f'_{i}.jpg'),c)
