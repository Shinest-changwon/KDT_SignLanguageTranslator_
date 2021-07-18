from moviepy.editor import VideoFileClip, concatenate_videoclips
import os
import cv2
import json
import imutils

"""
이 코드는 STT로 음성을 한글 문장으로 변환한 후 형태소 분해 리스트를 입력으로 받아 concated video 파일을 생성하고 그 경로를 반환하는 코드.

# 예시
입력: stnc_pos = ['마을버스', '일번']
출력: path = main(stnc_pos)
      path 변수에 concated video 파일 경로가 저장된다.
"""

"""
파일 경로  
root  
  ├morpheme  
  │ └1.Training
  │   ├[라벨링]01_real_word_morpheme  
  │   │ └morpheme 
  │   │   └01  
  │   │    ├NIA_SL_WORD0001_REAL01_D_morpheme.json  
  │   │    ├NIA_SL_WORD0001_REAL01_F_morpheme.json  
  │   │    ├NIA_SL_WORD0001_REAL01_L_morpheme.json  
  │   │    ├NIA_SL_WORD0001_REAL01_R_morpheme.json  
  │   │    ├NIA_SL_WORD0001_REAL01_U_morpheme.json  
  │   │    ├...  
  │   │    └NIA_SL_SEN3000_REAL01_U_morpheme.json # 3,000개 단어  
  │   └[라벨링]01_real_sen_morpheme  
  │      └morpheme 
  │        └01  
  │         ├NIA_SL_SEN0001_REAL01_D_morpheme.json  
  │         ├NIA_SL_SEN0001_REAL01_F_morpheme.json  
  │         ├NIA_SL_SEN0001_REAL01_L_morpheme.json  
  │         ├NIA_SL_SEN0001_REAL01_R_morpheme.json  
  │         ├NIA_SL_SEN0001_REAL01_U_morpheme.json  
  │         ├...  
  │         └NIA_SL_SEN2000_REAL01_U_morpheme.json # 2,000개 문장  
  │  
  └video
    ├word # 0~1500번 영상은 짝수 폴더, 1501~3000번 영상은 홀수 폴더
    │  ├[원천]01_real_word_video
    │  │  └01-1
    │  │    ├NIA_SL_WORD1501_REAL_D.mp4  
    │  │    ├NIA_SL_WORD1501_REAL_F.mp4
    │  │    ├NIA_SL_WORD1501_REAL_L.mp4
    │  │    ├NIA_SL_WORD1501_REAL_R.mp4
    │  │    ├NIA_SL_WORD1501_REAL_U.mp4
    │  │    ├...
    │  │    └NIA_SL_WORD3001_REAL_F.mp4 # '데려가다' 추가(3001번)
    │  ├[원천]02_real_word_video
    │  │  └01
    │  │    ├NIA_SL_WORD0001_REAL_D.mp4  
    │  │    ├NIA_SL_WORD0001_REAL_F.mp4
    │  │    ├NIA_SL_WORD0001_REAL_L.mp4
    │  │    ├NIA_SL_WORD0001_REAL_R.mp4
    │  │    ├NIA_SL_WORD0001_REAL_U.mp4
    │  │    ├...
    │  │    └NIA_SL_WORD1500_REAL_U.mp4
    │  │  
    │  ├...
    │  │
    │  └[원천]32_real_word_video
    │
    └sen # 0~1500번 영상은 홀수 폴더, 1501~3000번 영상은 짝수 폴더
       ├[원천]01_real_sen_video
       │  └01
       │    ├NIA_SL_SEN0001_REAL_D.mp4  
       │    ├NIA_SL_SEN0001_REAL_F.mp4
       │    ├NIA_SL_SEN0001_REAL_L.mp4
       │    ├NIA_SL_SEN0001_REAL_R.mp4
       │    ├NIA_SL_SEN0001_REAL_U.mp4
       │    ├...
       │    └NIA_SL_SEN1500_REAL_U.mp4
       ├[원천]02_real_word_video
       │  └01-1
       │    ├NIA_SL_SEN1501_REAL_D.mp4  
       │    ├NIA_SL_SEN1501_REAL_F.mp4
       │    ├NIA_SL_SEN1501_REAL_L.mp4
       │    ├NIA_SL_SEN1501_REAL_R.mp4
       │    ├NIA_SL_SEN1501_REAL_U.mp4
       │    ├...
       │    └NIA_SL_SEN3000_REAL_U.mp4
       │  
       ├...
       │
       └[원천]32_real_word_video
  
"""
# 문장 -> 형태소 분해 리스트 변환
def morpheme(stnc_pos): # 형태소 분해 리스트
    import pandas as pd
    demo_sen = pd.read_csv('./demo_sen.csv', index_col = 'index') # csv파일 경로 설정
    
    num_list = []
    div_list = []

    if '여기' in stnc_pos:
        stnc_pos = ['여기']
    elif '갈아타' in stnc_pos:
        stnc_pos = ['갈아타']
        
    for i in range(len(stnc_pos)):
        a = int(demo_sen[demo_sen['talk']==stnc_pos[i]]['num'])
        num_list.append(a)
        b = demo_sen[demo_sen['talk']==stnc_pos[i]].iloc[0,2]
        div_list.append(b)
    
    return div_list, num_list


################################################################################################
####################################영상 앞뒤 자르기 및 합치기####################################
################################################################################################

# 영상 시작 프레임과 끝 프레임 지정
def redefine_frame(word_sen, num):
    basic_path = os.getcwd() + '/morpheme/1.Training'
    json_path = basic_path+'/[라벨링]01_real_'+word_sen+'_morpheme/morpheme/01/NIA_SL_'+word_sen.upper()+str(num).zfill(4)+'_REAL01_F_morpheme.json'
    with open(json_path, 'r') as morpheme_json:
        morpheme_data = json.load(morpheme_json)
        # define start frame, end frame
        fps = 30
        if num == 3001:
            fps = 60
        start_frame = int(round(morpheme_data['data'][0]['start']*fps,-1))
        end_frame = int(round(morpheme_data['data'][len(morpheme_data['data'])-1]['end']*fps))
    return start_frame, end_frame

def find_video_path(word_sen, num):
    basic_path = os.getcwd()+'/video/'+word_sen+'/[원천]'
    if word_sen == 'word': 
        if 0 <= num <=1500: # word면서 번호가 0~1500이면 짝수 폴더
            folder_num1 = '02'
            folder_num2 = '01'
        else:
            folder_num1 = '01'
            folder_num2 = '01-1'
    else:
        if 0 <= num <=1500: # sen면서 번호가 0~1500이면 홀수 폴더
            folder_num1 = '01'
            folder_num2 = '01'
        else: 
            folder_num1 = '02'
            folder_num2 = '01-1'
    vid_path = basic_path+folder_num1+'_real_'+word_sen+'_video/'+folder_num2+'/NIA_SL_'+word_sen.upper()+str(num).zfill(4)+'_REAL01_F.mp4'
    return vid_path

def find_width_height_fps(vid_path):
    cap = cv2.VideoCapture(vid_path)
    w_frame, h_frame = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    return w_frame, h_frame, fps

def cut_frame_and_save(vid_path,start_frame,end_frame,idx):

   
   

    cap = cv2.VideoCapture(vid_path)
    # end_frame = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    w_frame, h_frame, fps = find_width_height_fps(vid_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    file_name = 'result_'+str(idx)+'.mp4'
    out = cv2.VideoWriter(file_name, fourcc, fps, (w_frame, h_frame))
    cnt=0
    while(cap.isOpened()):
        ret, frame = cap.read()
        cnt+=1
        # Avoid problems when video finish
        if ret==True:
            # Saving from the desired frames
            # print('start frame type: ', type(start_frame))
            if start_frame <= cnt <= end_frame:
                print('if 실행')
                try:
                    out.write(frame)
                    print('!!write 성공 !!')
                except: print('실패write')
        else:
            break
    cap.release()
    out.release()
            
# 영상 개수만큼 반복문 실행, 각 영상별로 strat, end frame 잘라서 따로 영상 저장하기
def main1():
    path = os.getcwd()
    tmp = []
    for i in os.listdir(path):
        if i.startswith('result') :
            # print("-------------------------------------------------------------------"+path+'/'+i)
            os.remove(path+'/'+i)
    for i in range(len(word_sen_list)):
        # 영상 시작 프레임, 끝 프레임 지정
        start_frame, end_frame = redefine_frame(word_sen_list[i], word_sen_num_list[i])
        print('start frame : {}, end frame: {}'.format(start_frame,end_frame))
        vid_path = find_video_path(word_sen_list[i], word_sen_num_list[i])
        print('vid path: ', vid_path)
        
        cut_frame_and_save(vid_path,start_frame,end_frame,i)

# 자른 영상을 하나로 합치기
"""
root
  ├result_0.mp4
  ├result_1.mp4
  └result_2.mp4
"""
def main2(word_sen_list):
    if len(word_sen_list) >= 1:
        file_list = os.listdir(os.getcwd())
        video_list = [file_name for file_name in file_list if file_name.startswith('result')]

        print('video list: ',video_list)
         
        video_list = sorted(video_list)
        caps = []
        for video in video_list:
            print('video: ',video)
            try:
                cap = VideoFileClip(video)
                caps.append(cap)
            except: print('실패!!')
        # print(caps)
        # 입력된 비디오 모두 concatenate
        final_clip = concatenate_videoclips(caps)
        path = os.getcwd()
        final_clip.write_videofile(path + '/static/videos/final.mp4')

# concat한 비디오 경로 반환
# def return_path():
#     path = os.getcwd() +'/final.mp4'
#     return path

def main(stnc_pos, is_ani=False):
    global word_sen_list
    global word_sen_num_list
    os.chdir("/Users/kyeong/dev/python3/KDT_SignLanguageTranslator/cheong_gaeguri")
    word_sen_list, word_sen_num_list = morpheme(stnc_pos)
    # basic_path = os.getcwd()+'/video/'+word_sen+'/[원천]'

    

    if is_ani:
        
        if 1157 in word_sen_num_list: # 내가 데려다 드릴게요(시나리오1-청인B의 애니메이션 출력)

            path = os.getcwd() +'/static/videos/Ani_Final_S1.mp4'
        elif 148 in word_sen_num_list: # 마을버스 일번이요(시나리오2-청인B의 애니메이션 출력)
            path = os.getcwd() +'/static/videos/Ani_Final_S2.mp4'
            
    else: # 실제 촬영 영상 짜집기 출력
        main1()
        main2(word_sen_list)
        # concat한 비디오 경로 반환
        path = os.getcwd()+'/static/videos/final.mp4'

    return path

# if __name__ == '__main__':
#     stnc_pos =  ['지하철', '갈아타', '곳', '안내'] # ['여기', '1', '선', '있']
#     path = main(stnc_pos, is_ani=False)
#     print(path)