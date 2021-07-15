from flask import Flask, request, url_for, redirect, render_template, jsonify
import json
from bs4 import BeautifulSoup
import SistemaRecomen

app = Flask(__name__, template_folder = 'templates')

@app.route("/")
def home():
     return render_template("home.html")

@app.route('/usuario', methods = ['POST'])
def recomen():
    features = [x for x in request.form.values()]
    user = SistemaRecomen.Sistema1.usuario(features[0])
    print(user)
    return render_template("home.html", texto = user)

@app.route('/recomen', methods = ['POST'])
def recomen1():
    features1 = [x for x in request.form.values()]
    user1 = SistemaRecomen.Sistema1.recomendacao(features1[0])
    
    return render_template("home.html", boasvindas = "Olá " + SistemaRecomen.Sistema1.user +  
                                "! Sugerimos a leitura do artigo: ", texto = user1[0], pred = user1[1])

@app.route('/avali', methods = ['POST'])
def avali():
    features1 = [x for x in request.form.values()]
    user2 = SistemaRecomen.Sistema1.avaliacao((features1[0]), SistemaRecomen.Sistema1.indice[1],  SistemaRecomen.Sistema1.indice[0])
    
    return render_template("home.html", ava = user2)


@app.route('/tabela')
def tabela():

      tabela = SistemaRecomen.Sistema1.tabelaRecomendação.to_html()
      return tabela  

@app.route('/classifi')
def classifi():
     return render_template("home1.html")

@app.route('/predict', methods = ['POST'])
def predict():
    features = [x for x in request.form.values()]
    pred = SistemaRecomen.classifi(features[0])
    print(pred)
    return render_template("home1.html", texto = features[0], pred =  "Classificação: {}".format(pred))

@app.route('/model_health')
def model_health():
    with open('metricas/metricas_0.json') as f:
        model_metrics = json.load(f)    
        return model_metrics

if __name__ == "__main__":
    app.run()
