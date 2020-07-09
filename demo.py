import torch
from model import NPT
import numpy as np
import pymesh

net_G=NPT()
net_G.cuda()
net_G.load_state_dict(torch.load('original_169.model'))


def face_reverse(faces):
    identity_faces=faces
    face_dict={}
    for i in range(len(random_sample)):
        face_dict[random_sample[i]]=i
    new_f=[]
    for i in range(len(identity_faces)):
        new_f.append([face_dict[identity_faces[i][0]],face_dict[identity_faces[i][1]],face_dict[identity_faces[i][2]]])
    new_face=np.array(new_f)
    return new_face

random_sample = np.random.choice(6890,size=6890,replace=False)
random_sample2 = np.random.choice(6890,size=6890,replace=False)


id_mesh=pymesh.load_mesh('./demo_data/13_643.obj')
pose_mesh=pymesh.load_mesh('./demo_data/14_664.obj')

with torch.no_grad():
    id_mesh_points=id_mesh.vertices[random_sample]
    id_mesh_points=id_mesh_points - (id_mesh.bbox[0] + id_mesh.bbox[1]) / 2
    id_mesh_points = torch.from_numpy(id_mesh_points.astype(np.float32)).cuda()

    pose_mesh_points=pose_mesh.vertices#[random_sample2]
    pose_mesh_points=pose_mesh_points-(pose_mesh.bbox[0] + pose_mesh.bbox[1]) / 2
    pose_mesh_points = torch.from_numpy(pose_mesh_points.astype(np.float32)).cuda()


    pointsReconstructed = net_G(pose_mesh_points.transpose(0,1).unsqueeze(0),id_mesh_points.transpose(0,1).unsqueeze(0))  # forward pass

new_face=face_reverse(id_mesh.faces)

pymesh.save_mesh_raw('./demo_data/13_664.obj', pointsReconstructed.cpu().numpy().squeeze(),new_face)

    
    
    
