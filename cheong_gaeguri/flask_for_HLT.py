from flask import Flask, render_template, redirect, url_for, request

import sys
from konlpy.tag import Mecab

import model_load_pipeline as load
import tensorflow as tf

app = Flask(__name__)

@app.route('/', methods = ['GET', 'POST'])
def main():
    return render_template("index.html")

@app.route('/mouth',methods = ['GET', 'POST'])
def mouth(): 

    

    if request.method == 'GET':
        return render_template("mouth.html")



    if request.method == 'POST':
        # sent = str(request.form['mouth'])
        sent = request.form.get("mouth")

        result = load.pipeline(sent)
        print(result)
        # mecab = Mecab()
        # stnc_pos = mecab.pos(sent)

        # nouns = [n for n, tag in stnc_pos if tag in ["NR","NNG","NNP","NP","VV","VV+EC"] ]
        # print(f"this is nouns = {nouns}")
        # ###########################태깅 후 애니메이션 처리된 페이지 반환#######################

        sys.path.append("/home/aiffel-dj16/dev/KDT_SignLanguageTranslator/morpheme_and_video_concat")
        import morpheme_and_video_concat_person as mv

        path = ''
        path = mv.main(result)
        print(path+"-----------------------------------------")
        # tmp = ''
        # for i in range(len(path)):
        #     tmp += path[i]

        #     if path[i] == '/':

        #         tmp = ''
        #         continue
            # elif tmp == 'cheong_gaeguri':
            #     path = '/' + path[i+2:]
            #     print(path)
            #     break
                
        return render_template('mouth.html' ,cont = path, res = "번역 결과 : \n\n" + str(result))
    
        
@app.route('/ear',methods = ['GET', 'POST'])
def ear():
    sys.path.append("/home/aiffel-dj16/dev/KDT_SignLanguageTranslator/SLR-frog/SL-GCN")
    import main as ktw

    if request.method=="GET":
        return render_template("ear.html")

    
    elif request.method == 'POST':
        
        # js_variable = request.form
        value = list(dict(request.form).keys())[0]#영상 이름 추출
        result = ktw.pipeline(value)#keypoints -> words
        print("this res : " + result)
        res = result

        return render_template("ear.html", res=result)
    

if __name__ == "__main__":
    app.run(debug = True,port = 5000)