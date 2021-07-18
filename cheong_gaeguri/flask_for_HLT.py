from flask import Flask, render_template, redirect, url_for, request

from konlpy.tag import Mecab
import morpheme_and_video_concat_fin as mv

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
        mecab = Mecab()
        stnc_pos = mecab.pos(sent)

        nouns = [n for n, tag in stnc_pos if tag in ["NR","NNG","NNP","NP","VV","VV+EC"] ]
        print(nouns)
        ###########################태깅 후 애니메이션 처리된 페이지 반환#######################

        path = mv.main(nouns)
        
        tmp = ''
        for i in range(len(path)):
            tmp += path[i]

            if path[i] == '/':

                tmp = ''
                continue
            elif tmp == 'cheong_gaeguri':
                path = '/' + path[i+2:]
                print(path)
                break
                
        return render_template('mouth.html' ,cont = path)
    
        
@app.route('/ear',methods = ['GET', 'POST'])
def ear():
    return render_template("ear.html")

if __name__ == "__main__":
    app.run(debug = True)