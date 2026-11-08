"""Constrain an observed access apron with local static LiDAR ground evidence."""
import re
import numpy as np
import shapely


def fit(access, points):
    polygon=shapely.Polygon(access['polygon_local_xy_m'])
    center=np.mean(access['mouth_local_xy_m'],axis=0)
    rows=points[shapely.contains(polygon,shapely.points(points[:,:2]))]
    # Estimate ground from the lower repeated heights in each horizontal cell.
    cells=np.floor(rows[:,:2]/.75).astype(int);keys,ids=np.unique(cells,axis=0,return_inverse=True)
    anchors=[]
    for i in range(len(keys)):
        cell=rows[ids==i]
        if len(cell)>=12:anchors.append([*np.median(cell[:,:2],axis=0),np.quantile(cell[:,2],.1)])
    anchors=np.array(anchors)
    low=np.quantile(anchors[:,2],.2)
    anchors=anchors[anchors[:,2]<low+1.0]
    design=np.c_[anchors[:,:2]-center,np.ones(len(anchors))]
    coef=np.linalg.lstsq(design,anchors[:,2],rcond=None)[0]
    for _ in range(10):
        residual=anchors[:,2]-design@coef;w=np.minimum(1.,.06/np.maximum(abs(residual),1e-6))
        coef=np.linalg.lstsq(design*w[:,None],anchors[:,2]*w,rcond=None)[0]
    residual=abs(anchors[:,2]-design@coef)
    if len(anchors)<20 or np.quantile(residual,.9)>.25:
        raise ValueError('Access needs additional ground evidence')
    return {'center_xy_m':center.tolist(),'coefficients':coef.tolist(),'support_cells':len(anchors),
            'abs_residual_m_quantiles':np.quantile(residual,[.5,.9,1]).tolist(),
            'method':'Lower static-LiDAR population in 0.75 m cells, robust plane, observed apron only.'}


def apply(scene, roads, access):
    from scripts.scene.build_assets import plane_heights
    model=access['ground_fit'];polygon=shapely.Polygon(access['polygon_local_xy_m'])
    center=np.array(model['center_xy_m']);coef=np.array(model['coefficients'])
    changes=[]
    def update(vertices, label):
        v=np.array(vertices);distance=shapely.distance(shapely.points(v[:,:2]),polygon)
        weight=np.clip(1-distance/3.,0,1);ids=np.flatnonzero(weight>0)
        desired=np.c_[v[ids,:2]-center,np.ones(len(ids))]@coef
        delta=(desired-v[ids,2])*weight[ids];v[ids,2]+=delta
        if len(ids):changes.append({'surface':label,'vertices':len(ids),'max_abs_delta_m':float(abs(delta).max())})
        return v.tolist()
    source_id=access['source_road'].rsplit(':',1)[-1]
    for mesh in roads['meshes']:
        lane=mesh.get('source_lane_id') or ''
        if re.match(r'sumo:-?'+re.escape(source_id)+r'(?:#|_)',lane):
            # Coarse road triangles can span the entire driveway without a
            # vertex inside it. Split at evidence bounds before fitting height.
            old=np.asarray(mesh['vertices'])[mesh['faces']]
            footprints=shapely.polygons(old[:,:,:2]);outer=polygon.buffer(3.)
            pieces=[];parents=[]
            for clipped in (shapely.intersection(footprints,polygon),
                            shapely.intersection(footprints,outer.difference(polygon)),
                            shapely.difference(footprints,outer)):
                part,parent=shapely.get_parts(shapely.constrained_delaunay_triangles(clipped),return_index=True)
                pieces.extend(part);parents.extend(parent)
            xy=shapely.get_coordinates(pieces).reshape(-1,4,2)[:,:3].reshape(-1,2)
            z=plane_heights(np.repeat(old[parents],3,axis=0),xy)
            vertices=np.c_[xy,z];faces=np.arange(len(vertices)).reshape(-1,3)
            wrong=np.cross(vertices[faces[:,1]]-vertices[faces[:,0]],vertices[faces[:,2]]-vertices[faces[:,0]])[:,2]<0
            faces[wrong]=faces[wrong,::-1]
            mesh['vertices']=vertices.tolist();mesh['faces']=faces.tolist()
            mesh['vertices']=update(mesh['vertices'],mesh['id'])
    scene['terrain']['vertices']=update(scene['terrain']['vertices'],'adjacent terrain')
    return {'access':access['id'],'ground_fit':model,'changes':changes,
            'scope':'Correct locally inferred OSM elevation and terrain. AV2 road faces and topology unchanged; native global access routing remains unvalidated.'}
