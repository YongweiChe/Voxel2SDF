import os
import numpy as np
import point_cloud_utils as pcu

# Path to the bench category as an example
category_path = "./benches"

# Resolution used to convert shapes to watertight manifolds
# Higher value means better quality and slower
manifold_resolution = 20_000

# Number of points in the volume to sample around the shape
num_vol_pts = 100_000

# Number of points on the surface to sample
num_surf_pts = 100_000

count = 0
for model_path in os.listdir(category_path):
    count += 1
    
    print(f'SDF {count}: {model_path}')
    try:
        v, f = pcu.load_mesh_vf(os.path.join(category_path, model_path, "model.obj"))
        
        # Convert mesh to watertight manifold
        vm, fm = pcu.make_mesh_watertight(v, f, manifold_resolution)
        nm = pcu.estimate_mesh_vertex_normals(vm, fm)  # Compute vertex normals for watertight mesh

        # Generate random points in the volume around the shape
        # NOTE: ShapeNet shapes are normalized within [-0.5, 0.5]^3
        p_grid = (np.random.rand(num_vol_pts, 3) - 0.5) * 1.1
        sdf_grid, _, _  = pcu.signed_distance_to_mesh(p_grid, vm, fm)

        # Sample points on the surface as face ids and barycentric coordinates
        fid_surf, bc_surf = pcu.sample_mesh_random(vm, fm, num_surf_pts)

        # Compute 3D coordinates and normals of surface samples
        p_surf = pcu.interpolate_barycentric_coords(fm, fid_surf, bc_surf, vm)
        sdf_surf = np.zeros(np.shape(sdf_grid))
        # n_surf = pcu.interpolate_barycentric_coords(fm, fid_surf, bc_surf, nm)
        
        noise = np.random.randn(*np.shape(p_surf)) * 0.02
        print(noise)
        # sample points near the surface
        p_near = p_surf + noise
        sdf_near, _, _  = pcu.signed_distance_to_mesh(p_near, vm, fm)
        print('---noised sdf---')
        print(sdf_near)

        # Save volume points + SDF and surface points + normals
        # Load using np.load()
        npz_path = os.path.join(category_path, model_path, "samples.npz")
        np.savez(npz_path, p_grid=p_grid, sdf_grid=sdf_grid, p_near=p_near, sdf_near=sdf_near, p_surf=p_surf, sdf_surf=sdf_surf)

        # Save the watertight mesh
        watertight_mesh_path = os.path.join(category_path, model_path, "model_watertight.obj")
        pcu.save_mesh_vfn(watertight_mesh_path, vm, fm, nm)
    except:
        print(f'error for file: {os.path.join(category_path, model_path, "model.obj")}')