import argparse
from tqdm import tqdm
import sys
import numpy as np
import os
os.chdir(os.path.realpath(__file__)[:-16])
# print("__-------------_______"+os.path.realpath(__file__)[:-16])
print(os.getcwd())
from . import keypoints_json2npy

from numpy.lib.format import open_memmap

sys.path.extend(['../'])

selected_joints = {
    '27_3': np.concatenate(([70, 75, 72, 76, 73, 77, 74], 
                    [95, 99, 100, 103, 104, 107, 108, 111, 112, 115],[116, 120, 121, 124, 125, 128, 129, 132, 133, 136]), axis=0) #27
}

paris = {
    'sign/27_3': ((5, 6), (5, 7),
                            (6, 8), (8, 10), (7, 9), (9, 11), 
                            (12,13),(12,14),(12,16),(12,18),(12,20),
                            (14,15),(16,17),(18,19),(20,21),
                            (22,23),(22,24),(22,26),(22,28),(22,30),
                            (24,25),(26,27),(28,29),(30,31),
                            (10,12),(11,22)
    ),
}

# pose_keypoints_2d : 70~94 / [70, 75, 72, 76, 73, 77, 74]
# hand_left_keypoints_2d : 95~115 / [95, 99, 100, 103, 104, 107, 108, 111, 112, 115]
# hand_right_keypoints_2d : 116 ~136 / [116, 120, 121, 124, 125, 128, 129, 132, 133, 136]

max_body_true = 1
max_frame = 250
num_channels = 3


def gendata(data_path, out_path, config='27_3'):
    data=[]
    selected = selected_joints[config]
    num_joints = len(selected)
    data.append(os.path.join(data_path + 'infer' + '.npy'))

    fp = np.zeros((len(data), max_frame, num_joints, num_channels, max_body_true), dtype=np.float32)
    for i, data_path in enumerate(data):
        
        skel = np.load(data_path)
        skel = skel[:,selected,:]
       
        if skel.shape[0] < max_frame:
            L = skel.shape[0]
            #print('skel.shape[0] :',L)
            fp[i,:L,:,:,0] = skel
            
            rest = max_frame - L
            num = int(np.ceil(rest / L))
            pad = np.concatenate([skel for _ in range(num)], 0)[:rest]
            fp[i,L:,:,:,0] = pad

        else:
            L = skel.shape[0]
            #print('skel.shape[0] :', L)
            fp[i,:,:,:,0] = skel[:max_frame,:,:]

    fp = np.transpose(fp, [0, 3, 1, 2, 4])

    np.save('{}/infer_data_joint.npy'.format(out_path), fp)




def gen_bone_data(out_path) :

    data = np.load('{}/infer_data_joint.npy'.format(out_path))
    N, C, T, V, M = data.shape
    fp_sp = open_memmap(
        '{}/infer_data_bone.npy'.format(out_path),
        dtype='float32',
        mode='w+',
        shape=(N, 3, T, V, M))

    fp_sp[:, :C, :, :, :] = data
    #print('fp_sp.shape : ', fp_sp.shape)
    for v1, v2 in paris['sign/27_3']:
        v1 -= 5
        v2 -= 5
        fp_sp[:, :, :, v2, :] = data[:, :, :, v2, :] - data[:, :, :, v1, :]



def gen_motion_data(out_path):

    parts = {
    'joint', 'bone'
    }

    for part in parts:
        data = np.load('{}/infer_data_{}.npy'.format(out_path, part))
        N, C, T, V, M = data.shape
        #print(data.shape)
        fp_sp = open_memmap(
            '{}/infer_data_{}_motion.npy'.format(out_path, part),
            dtype='float32',
            mode='w+',
            shape=(N, C, T, V, M))
        for t in range(T - 1):
            fp_sp[:, :, t, :, :] = data[:, :, t + 1, :, :] - data[:, :, t, :, :]
        fp_sp[:, :, T - 1, :, :] = 0


def preprocess(data):
    print("============================================"+data)
    keypoints_json2npy.transform(data)
    
    parser = argparse.ArgumentParser(description='Sign Data Converter.')
    parser.add_argument('--data_path', default= os.getenv('HOME')+'/dev/KDT_SignLanguageTranslator/SLR-frog/datasets/npy/infer_npy/')
    
    parser.add_argument('--out_folder', default=os.getenv('HOME')+'/dev/KDT_SignLanguageTranslator/SLR-frog/data/sign')
    
    parser.add_argument('--points', default='27_3')

    arg = parser.parse_args()

    out_path = os.path.join(arg.out_folder, arg.points)
    
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    gendata(
        arg.data_path,
        out_path,
        config=arg.points)
    gen_bone_data(out_path)
    gen_motion_data(out_path)
    