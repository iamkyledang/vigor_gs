"""Finite glass supports, first-hit rays, control poses and Phase I extraction."""
import numpy as np
import torch
import torch.nn.functional as F

from utils.point_utils import camera_rays, depths_to_points


def validate_patches(records, device='cuda'):
    if not records: raise ValueError('No finite glass patches.')
    patches=[]
    for record in records:
        n=torch.as_tensor(record['normal'],device=device,dtype=torch.float32)
        d=torch.as_tensor(record['offset'],device=device,dtype=torch.float32)
        vertices=torch.as_tensor(record['vertices'],device=device,dtype=torch.float32)
        if n.shape!=(3,) or not torch.isfinite(n).all() or not torch.isfinite(d) or n.norm()<1e-8:
            raise ValueError('Invalid plane normal.')
        d=d/n.norm(); n=F.normalize(n,dim=0)
        if vertices.ndim!=2 or vertices.shape[1]!=3 or len(vertices)<3 or not torch.isfinite(vertices).all():
            raise ValueError('A patch needs at least three finite 3D vertices.')
        tolerance=max(float(vertices.std()),1.)*1e-4
        if (vertices@n+d).abs().max()>tolerance: raise ValueError('Patch vertices must lie on their plane.')
        edges=vertices.roll(-1,0)-vertices
        area=torch.cross(vertices-vertices[0],vertices.roll(-1,0)-vertices[0],dim=-1).sum(0).norm()
        if area<1e-10: raise ValueError('Patch polygon has zero area.')
        turns=(torch.cross(edges,edges.roll(-1,0),dim=-1)*n).sum(-1)
        if not ((turns>=-1e-6).all() or (turns<=1e-6).all()):
            raise ValueError('Patch vertices must form an ordered convex polygon.')
        patches.append({'normal':n,'offset':d,'vertices':vertices})
    return patches


def first_hit(view, patches, epsilon=1e-4):
    origin,direction=camera_rays(view)
    direction=F.normalize(direction,dim=-1)
    shape=direction.shape[:-1]
    nearest=torch.full(shape,float('inf'),device=direction.device)
    selected=torch.full(shape,-1,device=direction.device,dtype=torch.int32)
    normals=torch.zeros_like(direction)
    for i,patch in enumerate(patches):
        n,d,vertices=patch['normal'],patch['offset'],patch['vertices']
        denom=direction@n
        safe=torch.where(denom.abs()>1e-8,denom,torch.ones_like(denom))
        distance=-(origin@n+d)/safe
        point=origin+distance[...,None]*direction
        signs=[]
        for a,b in zip(vertices,vertices.roll(-1,0)):
            signs.append((torch.cross((b-a).expand_as(point),point-a,dim=-1)*n).sum(-1))
        signs=torch.stack(signs)
        inside=(signs>=-1e-6).all(0)|(signs<=1e-6).all(0)
        valid=inside & (denom.abs()>1e-8) & (distance>epsilon) & (distance<nearest)
        nearest=torch.where(valid,distance,nearest)
        selected=torch.where(valid,i,selected)
        normals=torch.where(valid[...,None],n,normals)
    visible=selected>=0
    distance=torch.where(visible,nearest,torch.zeros_like(nearest))
    hits=origin+distance[...,None]*direction
    reflected=F.normalize(direction-2*(direction*normals).sum(-1,keepdim=True)*normals,dim=-1)
    origins=hits+epsilon*reflected
    cosine=(normals*direction).sum(-1).abs().clamp(0,1)
    return {'patches':selected,'origins':origins,'directions':reflected,'cosine':cosine,
            'hits':hits,'real_directions':direction,'normals':normals,'visible':visible}


def dynamic_virtual_camera(view, geometry, patch_index):
    """Eq. 9b diagnostic/control pose; rendering always uses per-pixel rays."""
    pixels=(geometry['patches']==patch_index).nonzero()
    if pixels.numel()==0: return None
    y,x=pixels[len(pixels)//2]
    forward=geometry['directions'][y,x]
    up=-view.world_view_transform.T.inverse()[:3,1]
    if torch.abs(up@forward)>0.99:
        axis=torch.argmin(forward.abs())
        up=F.one_hot(axis,3).to(forward)
    right=F.normalize(torch.cross(forward,up,dim=0),dim=0)
    up=torch.cross(right,forward,dim=0)
    return {'center':geometry['origins'][y,x], 'rotation':torch.stack([right,-up,forward],1), 'reference_pixel':(x,y)}


@torch.no_grad()
def extract_patches(cameras, interface, deformation, pipe, args, render_fn, extent):
    """Training-only mask-guided RANSAC. Does not claim glass depth is identifiable."""
    from scipy.spatial import ConvexHull, cKDTree
    sample_points=[]
    selected_views=cameras[::max(1,len(cameras)//12)]
    per_view=max(200,args.plane_max_points//len(selected_views))
    for cam in selected_views:
        out=render_fn(cam,interface,deformation,pipe,surface=True)
        valid=cam.gt_transparent_mask.to('cuda')[0] & cam.valid_mask.to('cuda')[0]
        valid &= (out['surface_alpha'][0]>0.3) & (out['glass_map'][0]>0.1) & (out['surface_depth'][0]>0)
        points=depths_to_points(cam,out['surface_depth'].permute(1,2,0))[valid]
        points=points[torch.isfinite(points).all(-1)]
        if len(points)>per_view: points=points[torch.randperm(len(points),device=points.device)[:per_view]]
        if len(points): sample_points.append(points.cpu().numpy())
    if not sample_points: raise RuntimeError('Phase I produced no glass-supported surface samples. Supply fixed geometry or improve Phase I.')
    points=np.concatenate(sample_points)
    if len(points)>args.plane_max_points:
        points=points[np.random.default_rng(args.seed).choice(len(points),args.plane_max_points,replace=False)]
    threshold=args.plane_threshold*extent
    rng=np.random.default_rng(args.seed)
    remaining=points.copy(); records=[]
    for _ in range(args.max_patches):
        if len(remaining)<args.plane_min_inliers: break
        best=np.zeros(len(remaining),bool)
        for batch in range(8):
            triplets=remaining[rng.integers(len(remaining),size=(64,3))]
            n=np.cross(triplets[:,1]-triplets[:,0],triplets[:,2]-triplets[:,0])
            length=np.linalg.norm(n,axis=1)
            n=n/np.maximum(length[:,None],1e-12)
            d=-(n*triplets[:,0]).sum(-1)
            support=np.abs(remaining@n.T+d)<threshold
            support[:,length<1e-8]=False
            index=support.sum(0).argmax()
            if support[:,index].sum()>best.sum(): best=support[:,index]
        if best.sum()<args.plane_min_inliers: break
        centroid=remaining[best].mean(0)
        _,_,basis=np.linalg.svd(remaining[best]-centroid,full_matrices=False)
        n=basis[-1]; d=-n@centroid
        best=np.abs(remaining@n+d)<threshold
        inliers=remaining[best]
        remaining=remaining[~best]
        uv=(inliers-centroid)@basis[:2].T
        # Split disconnected coplanar supports before computing convex boundaries.
        tree=cKDTree(uv)
        distances=tree.query(uv,k=2)[0][:,1]
        radius=max(np.median(distances)*8,threshold*3)
        pairs=tree.query_pairs(radius)
        parent=np.arange(len(uv))
        def find(i):
            while parent[i]!=i: parent[i]=parent[parent[i]]; i=parent[i]
            return i
        for a,b in pairs: parent[find(a)]=find(b)
        labels=np.array([find(i) for i in range(len(uv))])
        for label in np.unique(labels):
            group=labels==label
            if group.sum()<args.plane_min_inliers: continue
            component=uv[group]
            try: hull=ConvexHull(component)
            except Exception: continue
            vertices=centroid+component[hull.vertices]@basis[:2]
            records.append({'normal':n.tolist(),'offset':float(d),'vertices':vertices.tolist(),
                'inliers':int(group.sum()),'rmse':float(np.sqrt(np.mean((inliers[group]@n+d)**2))),
                'source':'phase_i_mask_guided_surface'})
            if len(records)>=args.max_patches: break
        if len(records)>=args.max_patches: break
    if not records: raise RuntimeError('No supported finite glass plane found; Phase I checkpoint was saved. Supply geometry or improve Phase I.')
    patches=validate_patches(records)
    intersection=predicted=annotated=0
    for cam in selected_views:
        visible=first_hit(cam,patches,pipe.ray_epsilon)['visible']
        valid=cam.valid_mask.to('cuda')[0]
        target=cam.gt_transparent_mask.to('cuda')[0]
        intersection+=int((visible&target&valid).sum())
        predicted+=int((visible&valid).sum()); annotated+=int((target&valid).sum())
    quality={'coverage':intersection/max(annotated,1),'precision':intersection/max(predicted,1),
             'sample_points':len(points),'training_views':len(selected_views)}
    print('Glass geometry quality:',quality)
    return records,quality
