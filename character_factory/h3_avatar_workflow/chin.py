"""Frozen accepted chin geometry and refined mask, independent of experiment paths."""
import cv2,numpy as np
from musetalk.utils.blending import prepare_image_blending_plan,get_image_blending_with_plan,get_image_blending
JAW=[234,93,132,58,172,136,150,149,176,148,152,377,400,378,379,365,397,288,361,323,454]
LIPS=[61,146,91,181,84,17,314,405,321,375,291,409,270,269,267,0,37,39,40,185]
GRID=np.linspace(-1.2,1.2,161)

PARAMS={'standard':None,'neck_cap':(-.10,.10),'chin_preserve':(-.27,-.05),'tracked_lip_preserve':(-.27,-.05)}

def smooth(v):
 v=np.clip(v,0,1);return v*v*(3-2*v)

def axes(p):
 l=p[[33,133]].mean(0);r=p[[263,362]].mean(0);span=np.linalg.norm(r-l)
 horizontal=(r-l)/span;down=np.array([-horizontal[1],horizontal[0]])
 return (l+r)/2,horizontal,down,span

def cap_mask(mask,cb,p,kind,g=None):
 if kind=='standard':return mask
 center,horizontal,down,span=axes(p)
 yy,xx=np.indices(mask.shape,dtype=np.float32);xx+=cb[0]-center[0];yy+=cb[1]-center[1]
 px=(xx*horizontal[0]+yy*horizontal[1])/span;py=(xx*down[0]+yy*down[1])/span
 jaw=p[JAW]-center; jx=jaw@horizontal/span;jy=jaw@down/span;order=np.argsort(jx)
 boundary=np.interp(px,jx[order],jy[order])
 start,end=PARAMS[kind]
 cap=1-smooth((py-boundary-start)/(end-start))
 # Only the bottom central face changes; taper to the original alpha toward the sides.
 lateral=1-smooth((np.abs(px)-.55)/.40)
 weight=1-lateral+lateral*cap
 # Keep a conservative source-derived mouth envelope untouched. It can be cached;
 # generated landmarks are used only later to validate the actual replacement lips.
 lip=np.zeros(mask.shape,np.uint8)
 for points in ([p,g] if kind=='tracked_lip_preserve' else [p]):
  assert points is not None
  hull=cv2.convexHull(np.rint(points[LIPS]-np.asarray(cb[:2])).astype(np.int32))
  cv2.fillConvexPoly(lip,hull,255)
 distance=cv2.distanceTransform(255-lip,cv2.DIST_L2,5)
 if kind=='tracked_lip_preserve':
  # Offline bound: measured output lip hull + same dilation as the pixel check.
  radius=max(4,int(round(.06*span)))
  expanded=cv2.dilate(lip,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*radius+1,2*radius+1)))
  distance=cv2.distanceTransform(255-expanded,cv2.DIST_L2,5)
  guard=1-smooth(distance/(.04*span))
 else:guard=1-smooth((distance-.25*span)/(.06*span))
 weight=np.maximum(weight,guard)
 return np.rint(mask.astype(np.float32)*weight).astype(np.uint8)

def curves(p,g):
 center,horizontal,down,span=axes(p)
 vals=[]
 for points in (p,g):
  j=points[JAW]-center;x=j@horizontal/span;y=j@down/span;o=np.argsort(x)
  vals.append(np.interp(GRID,x[o],y[o]))
 return vals

def prepare(d):
 delta=[]
 for p,g in zip(d['p'],d['g']):
  s,t=curves(p,g);delta.append(s-t)
 delta=np.asarray(delta)
 padded=np.pad(delta,((1,1),(0,0)),mode='edge')
 d['chin_delta']=np.clip(.25*padded[:-2]+.5*padded[1:-1]+.25*padded[2:],0,.18)
 return d

def aligned(preserve,d,i,strength):
 p=d['p'][i];g=d['g'][i];center,horizontal,down,span=axes(p)
 yy,xx=np.indices(preserve.shape[:2],np.float32)
 rx=xx-center[0];ry=yy-center[1]
 u=(rx*horizontal[0]+ry*horizontal[1])/span;v=(rx*down[0]+ry*down[1])/span
 source_curve,_=curves(p,g)
 move=np.interp(u,GRID,d['chin_delta'][i])*strength
 target=np.interp(u,GRID,source_curve)-move
 lip_y=np.max((g[LIPS]-center)@down/span)+.07
 upper=smooth((v-lip_y)/np.maximum(target-lip_y,.12))
 lower=1-smooth((v-target)/.65)
 lateral=1-smooth((np.abs(u)-.60)/.45)
 displacement=move*span*upper*lower*lateral
 mx=(xx+displacement*down[0]).astype(np.float32);my=(yy+displacement*down[1]).astype(np.float32)
 out=cv2.remap(preserve,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101)
 # Interpolation is exact at zero displacement; the whole lip guard remains stationary.
 lip=np.zeros(preserve.shape[:2],np.uint8)
 cv2.fillConvexPoly(lip,cv2.convexHull(np.rint(g[LIPS]).astype(np.int32)),255)
 radius=max(4,int(round(.06*span)))
 lip=cv2.dilate(lip,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*radius+1,2*radius+1)))>0
 diff=np.abs(out.astype(np.int16)-preserve.astype(np.int16))[lip]
 dxdy,dxdx=np.gradient(mx);dydy,dydx=np.gradient(my);jac=dxdx*dydy-dxdy*dydx
 meta={'max_lip_pixel_change':int(diff.max()),'max_shift_px':float(displacement.max()),'min_map_jacobian':float(jac.min()),'source_to_target_chin_shift_px':float(np.interp(0,GRID,d['chin_delta'][i])*span*strength)}
 assert meta['max_lip_pixel_change']==0,meta
 assert meta['min_map_jacobian']>.25,meta
 return out,meta

def warp_roi(preserve, d, i, strength):
    """Same coordinate math, evaluated only over its nonzero support."""
    p, g = d['p'][i], d['g'][i]
    center, horizontal, down, span = axes(p)
    source_curve, _ = curves(p, g)
    lip_y = np.max((g[LIPS] - center) @ down / span) + .07
    bottom = np.max(source_curve) + .65
    corners = np.array([center + span * (u * horizontal + v * down)
                        for u in (-1.05, 1.05) for v in (lip_y, bottom)])
    height, width = preserve.shape[:2]
    x0, y0 = np.maximum(np.floor(corners.min(axis=0)).astype(int) - 2, 0)
    x1, y1 = np.minimum(np.ceil(corners.max(axis=0)).astype(int) + 3, [width, height])
    out = preserve.copy()
    if x1 <= x0 or y1 <= y0:
        return out
    yy, xx = np.indices((y1-y0, x1-x0), np.float32)
    xx += x0
    yy += y0
    rx, ry = xx-center[0], yy-center[1]
    u = (rx*horizontal[0]+ry*horizontal[1])/span
    v = (rx*down[0]+ry*down[1])/span
    move = np.interp(u, GRID, d['chin_delta'][i])*strength
    target = np.interp(u, GRID, source_curve)-move
    upper = smooth((v-lip_y)/np.maximum(target-lip_y, .12))
    lower = 1-smooth((v-target)/.65)
    lateral = 1-smooth((np.abs(u)-.60)/.45)
    displacement = move*span*upper*lower*lateral
    mx = (xx+displacement*down[0]).astype(np.float32)
    my = (yy+displacement*down[1]).astype(np.float32)
    out[y0:y1, x0:x1] = cv2.remap(preserve, mx, my, cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_REFLECT_101)
    return out

class SourceMask:
    def __init__(self,mask,cb,p):
        self.mask=mask;self.cb=cb
        center,horizontal,down,span=axes(p)
        yy,xx=np.indices(mask.shape,dtype=np.float32)
        xx+=cb[0]-center[0];yy+=cb[1]-center[1]
        px=(xx*horizontal[0]+yy*horizontal[1])/span
        py=(xx*down[0]+yy*down[1])/span
        jaw=p[JAW]-center;jx=jaw@horizontal/span;jy=jaw@down/span;order=np.argsort(jx)
        boundary=np.interp(px,jx[order],jy[order])
        start,end=-.27,-.05
        cap=1-smooth((py-boundary-start)/(end-start))
        lateral=1-smooth((np.abs(px)-.55)/.40)
        self.weight=1-lateral+lateral*cap
        self.lip=np.zeros(mask.shape,np.uint8)
        hull=cv2.convexHull(np.rint(p[LIPS]-np.asarray(cb[:2])).astype(np.int32))
        cv2.fillConvexPoly(self.lip,hull,255)
        radius=max(4,int(round(.06*span)))
        self.kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*radius+1,2*radius+1))
        self.feather=.04*span
        self.radius=radius
        self.source_hull=hull
        self.base=np.rint(mask.astype(np.float32)*self.weight).astype(np.uint8)

    def current(self,g):
        hull=cv2.convexHull(np.rint(g[LIPS]-np.asarray(self.cb[:2])).astype(np.int32))
        points=np.concatenate([self.source_hull.reshape(-1,2),hull.reshape(-1,2)])
        padding=self.radius+int(np.ceil(self.feather*1.1))+4
        x0,y0=np.maximum(points.min(0)-padding,0)
        x1,y1=np.minimum(points.max(0)+padding+1,self.mask.shape[::-1])
        lip=self.lip[y0:y1,x0:x1].copy()
        cv2.fillConvexPoly(lip,hull-np.array([x0,y0],np.int32),255)
        expanded=cv2.dilate(lip,self.kernel)
        distance=cv2.distanceTransform(255-expanded,cv2.DIST_L2,5)
        guard=1-smooth(distance/self.feather)
        weight=np.maximum(self.weight[y0:y1,x0:x1],guard)
        mask=self.base.copy()
        mask[y0:y1,x0:x1]=np.rint(self.mask[y0:y1,x0:x1].astype(np.float32)*weight).astype(np.uint8)
        return mask

def prepare_source(d,frames):
    d['source_masks']=[SourceMask(d['masks'][str(i)],d['cache']['cropboxes'][i],d['p'][i]) for i in range(len(frames))]
    d['plans']=[prepare_image_blending_plan(f.shape,d['cache']['boxes'][i],d['masks'][str(i)],d['cache']['cropboxes'][i]) for i,f in enumerate(frames)]
    d['chin_delta']=np.zeros((len(frames),len(GRID)))
    return d

def standard(f,d,i,face):
    x,y,x1,y1=map(int,d['cache']['boxes'][i])
    return get_image_blending_with_plan(f.copy(),cv2.resize(face,(x1-x,y1-y)),d['plans'][i])

def corrected(f,d,i,face):
    mask=d['source_masks'][i].current(d['g'][i])
    x,y,x1,y1=map(int,d['cache']['boxes'][i])
    plan=d['plans'][i].copy()
    ys,xs=plan['clip_slice'];cx,cy=d['cache']['cropboxes'][i][:2]
    alpha=mask[ys.start-cy:ys.stop-cy,xs.start-cx:xs.stop-cx]
    plan['alpha_u8']=alpha[:,:,None]
    plan['alpha']=(alpha.astype(np.float32)/255.)[:,:,None]
    retained=get_image_blending_with_plan(f.copy(),cv2.resize(face,(x1-x,y1-y)),plan)
    return warp_roi(retained,d,i,1.)

class RefinedMask(SourceMask):
    def __init__(self,mask,cb,p):
        super().__init__(mask,cb,p)
        center,horizontal,down,span=axes(p)
        yy,xx=np.indices(mask.shape,np.float32)
        xx+=cb[0]-center[0];yy+=cb[1]-center[1]
        u=(xx*horizontal[0]+yy*horizontal[1])/span
        v=(xx*down[0]+yy*down[1])/span
        jaw=p[JAW]-center;jx=jaw@horizontal/span;jy=jaw@down/span;o=np.argsort(jx)
        relative=v-np.interp(u,jx[o],jy[o])
        lateral=1-smooth((np.abs(u)-.55)/.40)
        # Delta is clamped to [0,.18]. Outside this region the mask is unchanged.
        active=(relative>-.45-.18)&(lateral>0)&(mask>0)
        yy,xx=np.where(active)
        self.region=(slice(int(yy.min()),int(yy.max())+1),slice(int(xx.min()),int(xx.max())+1))
        self.u=u[self.region];self.relative=relative[self.region];self.lateral=lateral[self.region]
        self.feather=.14*span

    def current(self,g,delta):
        shift=np.interp(self.u,GRID,delta)
        cap=1-smooth((self.relative+shift+.45)/.40)
        weight=1-self.lateral+self.lateral*cap
        hull=cv2.convexHull(np.rint(g[LIPS]-np.asarray(self.cb[:2])).astype(np.int32))
        points=np.concatenate([self.source_hull.reshape(-1,2),hull.reshape(-1,2)])
        padding=self.radius+int(np.ceil(self.feather*1.1))+4
        x0,y0=np.maximum(points.min(0)-padding,0)
        x1,y1=np.minimum(points.max(0)+padding+1,self.mask.shape[::-1])
        lip=self.lip[y0:y1,x0:x1].copy()
        cv2.fillConvexPoly(lip,hull-np.array([x0,y0],np.int32),255)
        expanded=cv2.dilate(lip,self.kernel)
        distance=cv2.distanceTransform(255-expanded,cv2.DIST_L2,5)
        guard=1-smooth(distance/self.feather)
        ys,xs=self.region
        ix0,iy0=max(x0,xs.start),max(y0,ys.start)
        ix1,iy1=min(x1,xs.stop),min(y1,ys.stop)
        if ix1>ix0 and iy1>iy0:
            region=weight[iy0-ys.start:iy1-ys.start,ix0-xs.start:ix1-xs.start]
            g=guard[iy0-y0:iy1-y0,ix0-x0:ix1-x0]
            weight[iy0-ys.start:iy1-ys.start,ix0-xs.start:ix1-xs.start]=region+g*(1-region)
        mask=self.mask.copy()
        mask[self.region]=np.rint(self.mask[self.region].astype(np.float32)*weight).astype(np.uint8)
        return mask

def prepare_refined(d,frames):
    prepare_source(d,frames)
    d['refined_masks']=[RefinedMask(d['masks'][str(i)],d['cache']['cropboxes'][i],d['p'][i]) for i in range(len(frames))]
    return d

def corrected_refined(f,d,i,face):
    mask=d['refined_masks'][i].current(d['g'][i],d['chin_delta'][i])
    x,y,x1,y1=map(int,d['cache']['boxes'][i]);plan=d['plans'][i].copy()
    ys,xs=plan['clip_slice'];cx,cy=d['cache']['cropboxes'][i][:2]
    alpha=mask[ys.start-cy:ys.stop-cy,xs.start-cx:xs.stop-cx]
    plan['alpha_u8']=alpha[:,:,None];plan['alpha']=(alpha.astype(np.float32)/255.)[:,:,None]
    retained=get_image_blending_with_plan(f.copy(),cv2.resize(face,(x1-x,y1-y)),plan)
    return warp_roi(retained,d,i,1.)
