# _*_ coding: utf-8 _*_
# from jpype import *
import os
os.environ['JAVA_HOME'] = '/home/ai/mnt/yanqiuxia/software/jdk1.8.0_201'
import re
import requests
import traceback

import jieba
import jieba.posseg as pseg

# from pyhanlp import HanLP
# HanLP = JClass("com.hankcs.hanlp.HanLP")


# def seg_byhanlp(text, stop_words=None):
#
#     terms = HanLP.segment(text)
#     seg_words = []
#     for term in terms:
#         word = term.word
#         if stop_words:
#             if word not in stop_words:
#                  seg_words.append(word)
#         else:
#             seg_words.append(word)
#
#     return seg_words


def seg_byjieba(sentence, use_paddle=False, stop_words=None):

    seg_words = []

    # words = pseg.cut(sentence, use_paddle=use_paddle)
    words = list(jieba.cut(sentence, use_paddle=use_paddle))

    if words and stop_words:
        for word in words:
            if word not in stop_words:
                seg_words.append(word)
    return seg_words


def seg_bytrs(text, url=None, stop_words=None):

    words = []
    tags = []
    try:
        seg_text = requests.post(url, json={"text": text}).text
        pos_data = seg_text.split()
        index_text = 0

        while index_text < len(pos_data):
            data = pos_data[index_text]
            word = ''

            if ('/' not in data
                    and (index_text + 1) < len(pos_data)
                    and not pos_data[index_text + 1].split('/')[0]):
                word = data
                index_text += 1
                try:
                    tag = pos_data[index_text + 1].split('/')[1]
                except IndexError as e:
                    traceback.print_exc()
                    tag = 'O'
                    # print(text)
            else:
                if '/' in data:
                    splits = re.split('/+', data)
                    tag = splits[len(splits) - 1]
                    word_len = len(data) - len(tag) - 1
                    word = data[:word_len]

                else:
                    word = data
                    tag = 'O'
            index_text += 1
            if word:
                if stop_words:
                    if word not in stop_words:
                        words.append(word)
                        tags.append(tag)

    except Exception as e:
        traceback.print_exc()

    return words, tags

def cut_sent(para):
    para = re.sub(r'([；。！？\?])([^”’])', r"\1\n\2", para)  # 单字符断句符
    para = re.sub(r'(\.{6})([^”’])', r"\1\n\2", para)  # 英文省略号
    para = re.sub(r'(\…{2})([^”’])', r"\1\n\2", para)  # 中文省略号
    para = re.sub(r'([；。！？\?][”’])([；。！？\?])', r'\1\n\2', para)
    # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
    para = para.rstrip()  # 段尾如果有多余的\n就去掉它
    # 很多规则中会考虑分号;，但是这里我把它忽略不计，破折号、英文双引号等同样忽略，需要的再做些简单调整即可。
    return para.split("\n")

if __name__ == '__main__':
    '''
    '''
    jieba.enable_paddle()# 启动paddle模式。 0.40版之后开始支持，早期版本不支持
    text = "我来到北京清华大学。"
    words = list(jieba.cut(text, use_paddle=True))
    print(' '.join(words))