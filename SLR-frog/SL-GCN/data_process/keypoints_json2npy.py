from re import S
import numpy as np
import json
import os
from natsort import natsorted

def transform(data='scenario1_1'):
    print("_______________________________"+data)
    input_path = os.getenv('HOME')+'/dev/KDT_SignLanguageTranslator/SLR-frog/datasets/infer/{}'.format(data)
    keys = ['face_keypoints_2d', 'pose_keypoints_2d', 'hand_left_keypoints_2d', 'hand_right_keypoints_2d']
    n = 3
    target_name = ''
    for root, dirs, _ in os.walk(input_path):
        for dir in dirs:
            dir_path = os.path.join(root, dir)
            target_name = dir
            for root2, _, fnames in os.walk(dir_path):
                ## fnames -> keypoint dir name
                frame = 0
                data = np.zeros((len(fnames), 137, 3), dtype=np.float32)
                for fname in natsorted(fnames):
                    output_npy = os.getenv('HOME')+'/dev/KDT_SignLanguageTranslator/SLR-frog/datasets/npy/infer_npy/{}.npy'.format('infer')
                    #print('fname : ', fname)
                    result = []
                    path = os.path.join(root2, fname)
                    with open(path, 'r') as f:
                        json_data = json.load(f)
                        for key in keys:
                            result += [json_data['people'][key][i * n:(i + 1) * n] for i in range((len(json_data['people'][key]) // n  ))]
                        
                        data[frame,:,:] = result
                    frame += 1
                np.save(output_npy, data)
    print('transform done')
    # return target_name
    
# 33개 sen+word train(1~13) maximum : 231 / minimum : 38
# 33개 sen+word val(1~13) maximum : 210 / minimum : 89