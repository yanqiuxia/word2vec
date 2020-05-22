# _*_ coding: utf-8 _*_
# @Time : 2020/5/21 上午10:25 
# @Author : yanqiuxia
# @Version：V 0.1
# @File : app.py

import json
from flask import Flask, request
from gensim.models import KeyedVectors
from flask import jsonify
import argparse
import sys
import socket
import time
import logging


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s;%(levelname)s: %(message)s",
                              "%Y-%m-%d %H:%M:%S")
console = logging.StreamHandler()
console.setLevel(logging.DEBUG)
console.setFormatter(formatter)
logger.addHandler(console)


app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False


def isNoneWords(model, word):
    if word is None or len(word)==0 or word not in model.vocab:
        return True
    else:
        return False


# @app.route("/vec", methods=['GET'])
# def vec_route():
#     word = request.args.get("word")
#     if isNoneWords(word):
#         return jsonify("word is null or not in model!")
#     else:
#         return jsonify({'word':word,'vector': model.word_vec(word).tolist()})

def get_vec(model, word):
    vector = []
    if not isNoneWords(model, word):
        vector = model.word_vec(word).tolist()
    return vector


def get_similarity(model, word1, word2):
    similarity = 0
    if not isNoneWords(model, word1) and not isNoneWords(model, word2):
        similarity = float(model.similarity(word1, word2))
    return similarity


def get_nearest(model, word):
    nearest = []
    if not isNoneWords(model, word):
        nearest = model.similar_by_word(word, topn=20, restrict_vocab=None)
    return nearest


def main():

    model_file = './data/vec.txt'
    # global model
    start_time =time.time()
    model = KeyedVectors.load_word2vec_format(model_file, binary=False)
    end_time = time.time()
    run_time = end_time - start_time
    print('load model time:%d s'%run_time)
    word = '户口'
    nearest = get_nearest(model, word)
    print(nearest)

    word1 = '贵州'
    word2 = '贵阳'
    similarity = get_similarity(model, word1, word2)
    print(similarity)


if __name__ == "__main__":
    main()