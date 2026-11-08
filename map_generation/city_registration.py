"""Apply reviewed log frames and infer only missing boundary elevations."""
from copy import deepcopy
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree


def register_sources(sources, review):
    output=[];reports=[]
    for path,raw in sources:
        log_id=Path(path).parts[-3]
        registration=review.get('registrations',{}).get(log_id)
        if registration is None:output.append((path,raw));continue
        data=deepcopy(raw)
        points=np.asarray([[p[k] for k in ('x','y','z')] for lane in data['lane_segments'].values()
            for side in ('left_lane_boundary','right_lane_boundary') for p in lane[side]],float)
        known=np.unique(points[np.isfinite(points).all(axis=1)],axis=0);tree=cKDTree(known[:,:2])
        basis=np.asarray(registration['rotation_matrix']);translation=np.asarray(registration['translation_city_m'])
        completed=[];unknown_assets=[];cache={}
        reviewed_heights = {}
        for row in registration.get('missing_elevation_completions', []):
            if not row.get('reason') or not row.get('evidence'):
                raise ValueError('Reviewed missing elevation requires reasoning and source evidence')
            reviewed_heights[tuple(row['city_xy'])] = row
        def transform(item,location):
            if isinstance(item,dict):
                if {'x','y','z'} <= item.keys():
                    p=np.array([item[k] for k in ('x','y','z')],float)
                    if not np.isfinite(p[2]) and location.startswith('drivable_areas/'):
                        _,nearest=tree.query(p[:2]);p[2]=known[nearest,2]
                        registered=basis@p+translation
                        item.update(x=float(registered[0]),y=float(registered[1]),z=None)
                        unknown_assets.append(dict(location=location,planar_registration_reference_z_m=float(p[2]),
                            use='XY support footprint only; source elevation remains unknown.'))
                        return
                    if not np.isfinite(p[2]):
                        key=tuple(p[:2])
                        if key not in cache and key in reviewed_heights:
                            row = reviewed_heights[key]
                            cache[key] = float(row['inferred_source_z_m'])
                            completed.append(dict(row, location=location,
                                method='Agent-reviewed source-evidenced completion of an originally missing elevation; measured source coordinates are retained.'))
                        if key not in cache:
                            distance,indices=tree.query(p[:2],k=min(12,len(known)));support=known[indices]
                            design=np.c_[support[:,:2]-p[:2],np.ones(len(support))]
                            weight=1/np.sqrt(1+distance**2)
                            # Adjacent boundaries can lie almost on one line.
                            # A slope prior resolves the unobserved transverse
                            # direction without moving any measured vertex.
                            prior=np.array([[1.,0.,0.],[0.,1.,0.]])
                            def fit(weights):
                                return np.linalg.lstsq(np.vstack([design*weights[:,None],prior]),
                                    np.r_[support[:,2]*weights,0.,0.],rcond=None)[0]
                            coef=fit(weight)
                            for _ in range(6):
                                residual=support[:,2]-design@coef
                                robust=np.minimum(1.,.08/np.maximum(abs(residual),1e-6))*weight
                                coef=fit(robust)
                            residual=abs(support[:,2]-design@coef)
                            if distance[0]>12 or np.linalg.norm(coef[:2])>.15 or np.quantile(residual,.9)>.3:
                                raise ValueError(f'Insufficient same-level support for missing height at {path}:{location}')
                            cache[key]=float(coef[2])
                            completed.append(dict(location=location,city_xy=list(key),inferred_source_z_m=float(coef[2]),
                                nearest_support_m=float(distance[0]),support_radius_m=float(distance[-1]),
                                support_city_xyz=support.tolist(),plane_coefficients=coef.tolist(),
                                transverse_slope_prior='Zero slope, unit quadratic weight; measured support determines supported directions.',
                                residual_p90_m=float(np.quantile(residual,.9)),
                                method='Robust local plane of adjacent finite AV2 lane boundary points; inferred elevation, original missing field retained in the raw source.'))
                        p[2]=cache[key]
                    registered=basis@p+translation
                    item.update(zip(('x','y','z'),registered.tolist()))
                else:
                    for key,value in item.items():transform(value,location+'/'+str(key))
            elif isinstance(item,list):
                for i,value in enumerate(item):transform(value,location+'/'+str(i))
        for category in ('lane_segments','drivable_areas','pedestrian_crossings'):transform(data[category],category)
        reports.append(dict(log_id=log_id,source=path,registration=registration,missing_elevation_completion=completed,
                            unknown_drivable_area_elevations=unknown_assets))
        output.append((path,data))
    return output,reports
