import numpy as np
import pandas as pd
import json
import re
import os
import emoji
import spacy
import string
import requests
from bs4 import BeautifulSoup
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from joblib import load

def preprocess_text(text, remove_stop = True, stem_words = False, remove_mentions_hashtags = True):
    """
    eg:
    input: preprocess_text("@water #dream hi hello where are you going be there tomorrow happening happen happens",  
    stem_words = True) 
    output: ['tomorrow', 'happen', 'go', 'hello']
    """

    # Remove emojis
    emoji_pattern = re.compile("[" "\U0001F1E0-\U0001F6FF" "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r"", text)
    text = "".join([x for x in text if x not in emoji.UNICODE_EMOJI])

    if remove_mentions_hashtags:
        text = re.sub(r"@(\w+)", " ", text)
        text = re.sub(r"#(\w+)", " ", text)

    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    regex = re.compile('[' + re.escape(string.punctuation) + '0-9\\r\\t\\n]') # remove punctuation and numbers
    nopunct = regex.sub(" ", text.lower())
    words = (''.join(nopunct)).split()

    if(remove_stop):
        words = [w for w in words if w not in ENGLISH_STOP_WORDS]
        words = [w for w in words if len(w) > 2]  # remove a,an,of etc.

    if(stem_words):
        stemmer = PorterStemmer()
        words = [stemmer.stem(w) for w in words]

    return list(words)


## Recomendação

## Aqui os códigos estão um pouco diferente dos códigos do notebook, adptei eles para usar juntamente com o HTML.


class sistemaAvaliacaoTextos():

    def __init__(self, data, coluna, coluna_titulo, categorias, modelo):

        self.data = data
        self.coluna = coluna
        self.categorias = categorias
        self.modelo = modelo
        self.titulo = coluna_titulo
        df = np.empty([1, data.shape[0]])
        df[:] = np.nan
        self.tabelaRecomendação = pd.DataFrame(df)
        self.tabelaRecomendação.insert(0, 'Usuário', 'A')
        
    
    def usuario(self, user):
        self.user = user
        contador = 0
        for n in range(self.tabelaRecomendação.shape[0]):
            if self.tabelaRecomendação['Usuário'].iloc[n].lower() == self.user.lower():
                contador = 1
                self.user = self.tabelaRecomendação['Usuário'].iloc[n]
        if contador == 0:   
            df1 = np.empty([1, self.data.shape[0]])
            df1[:] = np.nan
            dff = pd.DataFrame(df1)
            dff.insert(0, 'Usuário', 0)
            self.tabelaRecomendação = pd.concat([self.tabelaRecomendação,dff ],
                                                axis = 0).reset_index().drop('index', axis = 1)
            self.tabelaRecomendação['Usuário'].iloc[-1]= self.user
            return 'Usuário criado com sucesso'
        if contador == 1:
            return 'Usuário encontrado no sistema'
    
    def recomendacao(self, tema):
        self.listaCategorias = []
        self.indice = []
        for n in range(len(self.categorias)):
            X = self.data[self.modelo.labels_ == n].reset_index()
            self.listaCategorias.append(X) 
        for n1 in range(len(self.categorias)):
            if tema.lower() == self.categorias[n1]:
                self.indice.append(n1)
                i = np.random.randint(len(self.categorias[n1]))
                self.indice.append(i)
                return self.listaCategorias[n1][self.titulo][i], self.listaCategorias[n1][self.coluna][i]
                
    def avaliacao(self, ava, i, categ):
        self.lugarNota = []
        for n in range(len(self.data[self.coluna])):
            if self.data[self.coluna][n] == self.listaCategorias[categ][self.coluna][i]:
                self.lugarNota.append(n)
        
        while (float(ava) <0) or (float(ava) >5):
            return 'A nota precisa tem que ser um valor entre 0 e 5'
            
        self.tabelaRecomendação.loc[self.tabelaRecomendação['Usuário'] == self.user, self.lugarNota[0]]  = float(ava)
        self.tabelaRecomendação.to_csv('datasets/tabelaRecomendacao.csv')
        return 'Agradecemos o uso do sistema e até breve!'


model1 = load('models/kmeans.joblib')
df = pd.read_csv('datasets/DatasetTextos.csv')
Sistema1 = sistemaAvaliacaoTextos(df, 'news_article', 'news_headline', ['sports', 'politics', 'technology'], model1)



## Classificação


nlp = spacy.load('en_core_web_md')
model = load('models/ICA_n2.joblib')

def vec(s):
    return nlp.vocab[s].vector
def dist(x,y):
    d = np.sqrt((x[0]-y[0])**2 + (x[1]-y[1])**2)
    return d

def classifi(texto):
    centros = model1.cluster_centers_
    textoProcessado = preprocess_text(texto)
    matrix = np.empty([len(textoProcessado), 300])
    for idx, word in enumerate(textoProcessado):
         matrix[idx,:] = vec(word)
    final_feature_matrix = np.empty([1, 300])
    final_feature_matrix = matrix.mean(axis = 0).reshape(1,-1)
    principalComponents = model.transform(final_feature_matrix)
    d1 = dist(principalComponents[0], centros[0])
    d2 = dist(principalComponents[0], centros[1])
    d3 = dist(principalComponents[0], centros[2])
    lista = [d1,d2,d3]
    cluster = np.argmin(lista)
    if cluster == 0:
        return "\nO seu texto fala sobre Esportes"
    if cluster == 1:
        return "\nO seu texto fala sobre Politica"
    if cluster == 2:
        return "\nO seu texto fala sobre Tecnologia"

## README