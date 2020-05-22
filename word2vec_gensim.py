# _*_ coding: utf-8 _*_
# @Time : 2020/5/18 下午5:49 
# @Author : yanqiuxia
# @Version：V 0.1
# @File : word2vec_gensim.py
import time
from gensim.models.word2vec import Word2Vec
from gensim.models.word2vec import LineSentence

def train_word2vec(file_in,model_save_path):
    sentences = LineSentence(file_in)
    model = Word2Vec(sentences, min_count=1, sg=1, negative=20)
    model.save(model_save_path)

if __name__ == '__main__':
    file_in = './data/sor/gzzf.txt'
    model_save_path = './data/w2v.model'
    start_time = time.time()
    # train_word2vec(file_in,model_save_path)
    end_time = time.time()
    run_time = end_time - start_time
    print('程序运行时间：%d s' %run_time)

    model = Word2Vec.load(model_save_path)
    y2 = model.wv.similarity(u"省国土资源厅", u"贵州省国土资源厅")
    print(y2)

    for i in model.wv.most_similar(u"入户"):
        print(i[0], i[1])