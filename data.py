import torch.utils.data as data
import torch
import numpy as np
import pymesh
import random

class SMPL_DATA(data.Dataset):
    def __init__(self, train,  npoints=6890, shuffle_point = False):
        self.train = train
        self.shuffle_point = shuffle_point 
        self.npoints = npoints
        self.path='./smpl_data/'

    def __getitem__(self, index):
        
        identity_mesh_i=np.random.randint(0,16)
        identity_mesh_p=np.random.randint(200,600)

        pose_mesh_i=np.random.randint(0,16)
        pose_mesh_p=np.random.randint(200,600)
        
        
        identity_mesh=pymesh.load_mesh(self.path+str(identity_mesh_i)+'_'+str(identity_mesh_p)+'.obj')
        pose_mesh=pymesh.load_mesh(self.path+str(pose_mesh_i)+'_'+str(pose_mesh_p)+'.obj')
        gt_mesh=pymesh.load_mesh(self.path+str(identity_mesh_i)+'_'+str(pose_mesh_p)+'.obj')

        pose_points = pose_mesh.vertices
        identity_points=identity_mesh.vertices
        identity_faces=identity_mesh.faces
        gt_points = gt_mesh.vertices

        pose_points = pose_points - (pose_mesh.bbox[0] + pose_mesh.bbox[1]) / 2
        pose_points = torch.from_numpy(pose_points.astype(np.float32))
        
        identity_points=identity_points-(identity_mesh.bbox[0]+identity_mesh.bbox[1])/2
        identity_points=torch.from_numpy(identity_points.astype(np.float32))

        gt_points=gt_points-(gt_mesh.bbox[0]+gt_mesh.bbox[1]) / 2
        gt_points = torch.from_numpy(gt_points.astype(np.float32))

        #if self.train:
        #    a = torch.FloatTensor(3)
        #    pose_points = pose_points + (a.uniform_(-1,1) * 0.03).unsqueeze(0).expand(-1, 3)

        random_sample = np.random.choice(self.npoints,size=self.npoints,replace=False)
        random_sample2 = np.random.choice(self.npoints,size=self.npoints,replace=False)

        new_face=identity_faces

        if self.shuffle_point:
            pose_points = pose_points[random_sample2]
            identity_points=identity_points[random_sample]
            gt_points=gt_points[random_sample]
            
            face_dict={}
            for i in range(len(random_sample)):
                face_dict[random_sample[i]]=i
            new_f=[]
            for i in range(len(identity_faces)):
                new_f.append([face_dict[identity_faces[i][0]],face_dict[identity_faces[i][1]],face_dict[identity_faces[i][2]]])
            new_face=np.array(new_f)

        return pose_points, random_sample, gt_points, identity_points, new_face
        

    def __len__(self):
        return 4000
