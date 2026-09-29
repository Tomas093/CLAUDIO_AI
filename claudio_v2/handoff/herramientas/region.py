import sys,json,cv2
n,bj,x0,y0,x1,y1,out=sys.argv[1:8]; x0,y0,x1,y1=map(int,(x0,y0,x1,y1)); s=float(sys.argv[8]) if len(sys.argv)>8 else 2
img=cv2.imread(f'/home/claude/lab/{n}.png',0); B=json.load(open(bj)); B=B['cajas'] if isinstance(B,dict) else B
c=cv2.cvtColor(img[y0:y1,x0:x1],cv2.COLOR_GRAY2BGR); c=cv2.resize(c,None,fx=s,fy=s,interpolation=cv2.INTER_AREA if s<1 else cv2.INTER_LINEAR)
P=lambda x,y:(int((x-x0)*s),int((y-y0)*s))
for g in range((x0//50+1)*50,x1,50):
    cv2.line(c,P(g,y0),(P(g,y0)[0],8 if g%100 else 16),(255,120,0),1)
    if g%100==0: cv2.putText(c,str(g),(P(g,y0)[0]+2,26),0,.4,(255,120,0),1)
for g in range((y0//50+1)*50,y1,50):
    cv2.line(c,P(x0,g),(8 if g%100 else 16,P(x0,g)[1]),(255,120,0),1)
    if g%100==0: cv2.putText(c,str(g),(18,P(x0,g)[1]+4),0,.4,(255,120,0),1)
for j,it in enumerate(B):
    a=it['bbox_px']
    if a[2]>x0 and a[0]<x1 and a[3]>y0 and a[1]<y1:
        cv2.rectangle(c,P(a[0],a[1]),P(a[2],a[3]),(0,0,255),1); cv2.putText(c,str(it.get('id',j+1)),P(a[0],a[1]-2),0,.45,(0,120,0),1)
cv2.imwrite(out,c)
