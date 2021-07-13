import pandas as pd

demo_sen = pd.read_csv('./demo_sen.csv', index_col = 'index') # csv파일 경로 설정
demo_sen

stnc_pos = nouns # 이전 변수 넣기

num_list = []
div_list = []
for i in range(len(stnc_pos)):
    for j in range(len(stnc_pos[i])):
        a = int(demo_sen[demo_sen['talk']==stnc_pos[i][j]]['num'])
        num_list.append(a)
        b = demo_sen[demo_sen['talk']==stnc_pos[i][j]].iloc[0,2]
        div_list.append(b)
        
print(num_list)
print(div_list)