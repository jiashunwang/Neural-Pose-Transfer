import trimesh
import numpy as np

#make sure the order of identity points and gt points are same 
#for original_model, please keep the identity and pose points in different order

ours_mesh = trimesh.load('ours.obj')
ours_vertices=ours_mesh.vertices
ours_bbox= np.array([[np.max(ours_vertices[:,0]), np.max(ours_vertices[:,1]), np.max(ours_vertices[:,2])], \
                        [np.min(ours_vertices[:,0]), np.min(ours_vertices[:,1]), np.min(ours_vertices[:,2])]])

ours_vertices_align=ours_vertices-(ours_bbox[0] + ours_bbox[1]) / 2

gt_mesh=trimesh.load('gt.obj')
gt_vertices=gt_mesh.vertices
gt_bbox= np.array([[np.max(gt_vertices[:,0]), np.max(gt_vertices[:,1]), np.max(gt_vertices[:,2])], \
                        [np.min(gt_vertices[:,0]), np.min(gt_vertices[:,1]), np.min(gt_vertices[:,2])]])
gt_vertices_align=gt_vertices-(gt_bbox[0] + gt_bbox[1]) / 2

print(np.mean((ours_vertices_align-gt_vertices_align)**2))

