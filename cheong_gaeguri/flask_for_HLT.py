from flask import Flask, render_template, redirect, url_for, request

from konlpy.tag import Mecab

app = Flask(__name__)

@app.route('/', methods = ['GET', 'POST'])
def main():
    return render_template("main.html")

@app.route('/mouth',methods = ['GET', 'POST'])
def mouth(): 
    if request.method == 'GET':
        print("gd")
        return render_template("mouth.html")

    if request.method == 'POST':
        # sent = str(request.form['mouth'])
        sent = request.form.get("mouth")
        mecab = Mecab()
        stnc_pos = mecab.pos(sent)

        nouns = [n for n, tag in stnc_pos if tag in ["NNG","NNP","VV"] ]
        print('ㅋㅋㅎㅎ 성공~')
        print(nouns)
        ###########################태깅 후 애니메이션 처리된 페이지 반환#######################
        return render_template('mouth.html')
        
@app.route('/ear',methods = ['GET', 'POST'])
def ear():
    return render_template("ear.html")

if __name__ == "__main__":
    app.run(debug = True)