from browser import document


def convert(event):#애니메이션 생성 이벤트 리스너
    ani_gen()

def ani_gen():#애니메이션 생성 
    print(int(document.getElementById("mouth").value) - 5)


    

# document["play"].bind("click", animation_play)
document["convert"].bind("click", convert)


    


