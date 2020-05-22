# _*_ coding: utf-8 _*_
# @Time : 2020/1/8 下午2:40
# @Author : yanqiuxia
# @Version：V 0.1
import re
import json
from utils import load_stopwords
from segment import cut_sent, seg_bytrs


def reader_query(file_in,file_out):

    fp = open(file_in, 'r', encoding='utf-8')
    op = open(file_out, 'w', encoding='utf-8')
    json_data = json.load(fp)
    hits = json_data['hits']
    hits_10000 = hits['hits']

    for hit in hits_10000:
        query_object = {}
        try:
            searchWord = hit.get('_source').get('parameter').get('page').get('searchWord')
            searhTerms = hit.get('_source').get('parameter').get('page').get('searhTerms')
            # seg_hanlp_words = seg_byhanlp(searchWord)
            query_object = {
                'searchWord': searchWord,
                'searhTerms': searhTerms,
                # 'seg_hanlp_words': seg_hanlp_words,
            }
        except TypeError as e1:
            print(e1)
        if query_object:
            json.dump(query_object, op,ensure_ascii=False)
            op.write('\n')
    fp.close()
    op.close()


def segment(file_in, file_out, stopwords_file,url=None):
    fp = open(file_in, 'r', encoding='utf-8')
    op = open(file_out, 'a', encoding='utf-8')
    lines = fp.readlines()
    stopwords = load_stopwords(stopwords_file)
    count = 0
    for line in lines:
        if count %3000 == 0:
            print(count)
        data = json.loads(line)
        title = data['title']
        content = data['content']
        sentences = []
        if title:
            sentences.append(title.strip())
        if content:
            ansentences = cut_sent(content)
            sentences.extend(ansentences)
        if sentences:
            seg_words, _ = seg_bytrs(title, url,stopwords)
            op.write(' '.join(seg_words))
            op.write('\n')
        count += 1
    fp.close()
    op.close()


def reader_doc(file_in,file_out):
    '''

    :param file_in:
    :param file_out:
    :return:
    '''
    fp = open(file_in, 'r', encoding='utf-8')
    op = open(file_out, 'w', encoding='utf-8')
    json_data = json.load(fp)
    hits = json_data['hits']
    hits_10000 = hits['hits']

    for hit in hits_10000:
        doc_object = {}
        try:
            content = hit.get('_source').get("INTERACTIONCONTENT")

            title = hit.get('_source').get('INTERACTIONTITLE')

            doc_object = {
                'title': title,
                'content': content,
            }
        except TypeError as e1:
            print(e1)
        if doc_object:
            json.dump(doc_object, op, ensure_ascii=False)
            op.write('\n')
    fp.close()
    op.close()


def deduplication(file_in, file_out):
    fp = open(file_in, 'r', encoding='utf-8')
    op = open(file_out, 'w', encoding='utf-8')
    lines = fp.readlines()
    texts = set()
    count = 0
    for line in lines:
        data = json.loads(line)
        searchWord = data['searchWord'].strip()

        # min_len = len(content)
        if not texts.__contains__(searchWord):
            count += 1
            texts.add(searchWord)
            json.dump(data, op, ensure_ascii=False)
            op.write('\n')
    print('数据总数%d'%count)
    fp.close()
    op.close()

if __name__ == '__main__':
    '''
    '''
    # file_in = 'data/sor/互动交流.json'
    # file_out = 'data/sor/互动交流.re.json'
    # reader_doc(file_in, file_out)

    # file_in = './data/sor/query.result.json'
    # file_out = './data/sor/query.du.json'
    # deduplication(file_in,file_out)

    # file_in = './data/sor/gzzf.json'
    # file_out = './data/sor/gzzf.txt'
    # stopwords_file = './data/stopwords.txt'
    # url = 'http://0.0.0.0:7777/cws'
    # segment(file_in, file_out, stopwords_file,url)



