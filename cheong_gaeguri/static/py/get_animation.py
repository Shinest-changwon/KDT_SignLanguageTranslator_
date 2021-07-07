# from browser import document
# from konlpy.tag import Mecab
import numpy as np
# def convert(event):#애니메이션 생성 이벤트 리스너
#     ani_gen()

def ani_gen():#애니메이션 생성 
    print(int(document.getElementById("mouth").value) - 5)
    # mecab = Mecab()
    # stnc = getElementById("mouth").value
    
    # # stnc = '건너서 타면 됩니다.'

    # stnc_pos = mecab.pos(stnc)

    # nouns = [n for n, tag in stnc_pos if tag in ["NNG","NNP","VV"] ]

    # print(nouns)


    

# document["play"].bind("click", animation_play)
document["convert"].bind("click", convert)


    


