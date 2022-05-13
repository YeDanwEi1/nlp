from flask import Flask,render_template,request,url_for
import snownlp
import sqlite3
app = Flask(__name__)
@app.route('/',methods=['GET','POST'])#用户通过什么访问即  路由解析
def index():
    if request.method == 'GET':
        return render_template("index.html")
    if request.method == 'POST':
        comment= request.form.get('comment')
        pos=snownlp.SnowNLP(comment).sentiments
        neg=(1-pos)*100
        pos *= 100
        pos=format(pos,'.2f')
        neg=format(neg,'.2f')
        pos=str(pos)+'%'
        neg=str(neg)+'%'
        print(pos,neg)
        print(comment)
        comment=comment.replace(' ',',')
        return render_template("index.html",pos=pos,neg=neg,comment=comment)

@app.route('/index')#用户通过什么访问
def indexx():
    return render_template("index.html")

@app.route('/test',methods=['GET','POST'])#用户通过什么访问
def test():
    if request.method == 'GET':
        return render_template("test.html")
    if request.method == 'POST':
        comment= request.form.get('comment')
        pos=snownlp.SnowNLP(comment).sentiments
        neg=(1-pos)*100
        pos *= 100
        pos=format(pos,'.2f')
        neg=format(neg,'.2f')
        pos=str(pos)+'%'
        neg=str(neg)+'%'
        print(pos,neg)
        print(comment)
        comment=comment.replace(' ',',')
        return render_template("test.html",pos=pos,neg=neg,comment=comment)
@app.route('/wordcloud')#用户通过什么访问
def wordcloud():
    return render_template("wordcloud.html")

@app.route('/LDA')#用户通过什么访问
def LDA():
    return render_template("LDA.html")

@app.route('/team')#用户通过什么访问
def team():
    return render_template("team.html")

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8090)
