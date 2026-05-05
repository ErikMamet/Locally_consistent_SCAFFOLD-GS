import glob, os, torch
from depth_anything_3.api import DepthAnything3
import numpy as np
from gsc_tool.colmap_read_model import read_model, qvec2rotmat
########## UTILS ##########
import struct


if __name__ == "__main__":
    Neur3D_path = "/home/erikmam/projects/def-scoulomb/erikmam/Locally_consistent_SCAFFOLD-GS/Neur3D"
    for elem in os.listdir(Neur3D_path):
        base_path = os.path.join(Neur3D_path, elem, "colmap_0")
        image_path=os.path.join(base_path, "images")
        #extrinsics_path=os.path.join(base_path, "sparse/0/images.bin")
        #intrinsics_path=os.path.join(base_path, "sparse/0/cameras.bin")
        sparse_path=os.path.join(base_path, "sparse/0/")
    
        cameras, images, points3D = read_model(sparse_path, ".bin")

        print("num_cameras:", len(cameras))
        print("num_images:", len(images))
        print("num_points3D:", len(points3D))

        print("images1 name", images[1].name)
        #extrinsics, intrinsics = re
        #print(extrinsics.shape)  # (N, 4, 4)
        #print(intrinsics.shape)  # (N, 3, 3)
        IMAGES = []
        N = len(cameras)
        EXTRINSICS = np.zeros((N,4,4))
        INTRINSICS = np.zeros((N,3,3))
        for i in range(len(images)):
            IMAGES.append(os.path.join(image_path,images[i+1].name)) #index starts at 1 
            cam = cameras[images[i+1].camera_id]
            print(qvec2rotmat(images[i+1].qvec))
            EXTRINSICS[i][:3,:3] = qvec2rotmat(images[i+1].qvec)
            EXTRINSICS[i][:3,3] = images[i+1].tvec
            EXTRINSICS[i][3,3] = 1
            if cam.model == "SIMPLE_PINHOLE": 
                print("we've got a simple pinhole camera")       
                INTRINSICS[i][0,0]=cam.params[0]
                INTRINSICS[i][1,1]=cam.params[0]
                INTRINSICS[i][0,2]=cam.params[1]
                INTRINSICS[i][1,2]=cam.params[2]
                INTRINSICS[i][2,2]=1
            elif cam.model == "PINHOLE":
                print("we've got a pinhole camera")        
                INTRINSICS[i][0,0]=cam.params[0]
                INTRINSICS[i][1,1]=cam.params[1]
                INTRINSICS[i][0,2]=cam.params[2]
                INTRINSICS[i][1,2]=cam.params[3]
                INTRINSICS[i][2,2]=1
                print(INTRINSICS[i])
            else : 
                print("ERROR : camera model is not supported ")
                exit(-1)


        #model init
        device = torch.device("cuda")
        model = DepthAnything3.from_pretrained("/home/erikmam/projects/def-scoulomb/erikmam/Locally_consistent_SCAFFOLD-GS/shared/models/DA3NESTED-GIANT-LARGE")
        model = model.to(device=device)
        
        #run
        prediction = model.inference(
            image=IMAGES,
            intrinsics=INTRINSICS,
            extrinsics=EXTRINSICS
        )
        # prediction.processed_images : [N, H, W, 3] uint8   array
        print(prediction.processed_images.shape)
        # prediction.depth            : [N, H, W]    float32 array
        print(prediction.depth.shape)  
        # prediction.conf             : [N, H, W]    float32 array
        print(prediction.conf.shape)  
        # prediction.extrinsics       : [N, 3, 4]    float32 array # opencv w2c or colmap format
        print(prediction.extrinsics.shape)
        # prediction.intrinsics       : [N, 3, 3]    float32 array
        print(prediction.intrinsics.shape)
        np.save(os.path.join(base_path ,"processed_image.npy"), prediction.processed_images)
        np.save(os.path.join(base_path ,"depths.npy"), prediction.depth)
        np.save(os.path.join(base_path ,"conf.npy"), prediction.conf)
        np.save(os.path.join(base_path ,"extrinsics.npy"), prediction.extrinsics)
        np.save(os.path.join(base_path ,"intrinsics.npy"), prediction.intrinsics)
