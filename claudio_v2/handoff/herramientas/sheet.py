import sys,json,cv2,numpy as np
n,bj,ids,out=sys.argv[1:5]; pad=int(sys.argv[5]) if len(sys.argv)>5 else 80
img=cv2.imread(f'/home/claude/lab/{n}.png',0); B=json.load(open(bj)); B=B['cajas'] if isinstance(B,dict) else B
T=[]
for i in map(int,ids.split(',')):
    x0,y0,x1,y1=map(int,B[i-1]['bbox_px']); X0,Y0=max(0,x0-pad),max(0,y0-pad); X1,Y1=min(img.shape[1],x1+pad),min(img.shape[0],y1+pad)
    c=cv2.cvtColor(img[Y0:Y1,X0:X1],cv2.COLOR_GRAY2BGR)
    for j,it in enumerate(B):
        a=list(map(int,it['bbox_px']))
        if a[2]>X0 and a[0]<X1 and a[3]>Y0 and a[1]<Y1:
            cv2.rectangle(c,(a[0]-X0,a[1]-Y0),(a[2]-X0,a[3]-Y0),(0,0,255) if j==i-1 else (0,170,0),1)
    s=440/max(c.shape[:2]); c=cv2.resize(c,None,fx=s,fy=s,interpolation=cv2.INTER_AREA)
    C=np.full((480,480,3),255,np.uint8); C[30:30+c.shape[0],30:30+c.shape[1]]=c
    st=10 if (X1-X0)<250 else 50
    for g in range((X0//st+1)*st,X1,st):
        p=30+int((g-X0)*s); cv2.line(C,(p,22),(p,29),(255,0,0),1)
        if g%(st*5)==0: cv2.putText(C,str(g),(p-12,14),0,.35,(255,0,0),1)
    for g in range((Y0//st+1)*st,Y1,st):
        p=30+int((g-Y0)*s); cv2.line(C,(22,p),(29,p),(255,0,0),1)
        if g%(st*5)==0: cv2.putText(C,str(g),(0,p-3),0,.3,(255,0,0),1)
    cv2.putText(C,f'#{i}',(400,470),0,.7,(0,0,255),2); cv2.rectangle(C,(0,0),(479,479),(0,0,0),1); T.append(C)
while len(T)%3: T.append(np.full((480,480,3),255,np.uint8))
cv2.imwrite(out,np.vstack([np.hstack(T[k:k+3]) for k in range(0,len(T),3)]))
