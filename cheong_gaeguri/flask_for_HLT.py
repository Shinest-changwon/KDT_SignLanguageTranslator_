from flask import Flask, render_template, redirect, url_for

app = Flask(__name__)

@app.route('/', methods = ['GET', 'POST'])
def main():
    return render_template("main.html")

@app.route('/mouth',methods = ['GET', 'POST'])
def mouth():
    return render_template("mouth.html")

@app.route('/ear',methods = ['GET', 'POST'])
def ear():
    return render_template("ear.html")

if __name__ == "__main__":
    app.run(debug = True)