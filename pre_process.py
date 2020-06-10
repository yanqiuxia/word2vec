# _*_ coding: utf-8 _*_
# @Time : 2020/1/8 下午2:40
# @Author : yanqiuxia
# @Version：V 0.1
import re
import json
import traceback
import ijson
from utils import load_stopwords
from segment import cut_sent, seg_bytrs, seg_byjieba

http_pattern = re.compile(r'http://[a-zA-Z0-9.?/&=:]*', re.S)
www_pattern = re.compile(r'www.[a-zA-Z0-9.?/&=:]*', re.S)
html_pattern = re.compile(r'<[^>]+>', re.S)

digit_pattern = re.compile('[0-9]+')


def contain_digits(s):
    match = digit_pattern.match(s)
    if match:
        return True
    else:
        return False


def remove_html_url(text):
    if text:
        text = html_pattern.sub("", text)
        text = http_pattern.sub("", text)
        text = www_pattern.sub("", text)

        text = re.sub('&nbsp', '', text)
        text = text.strip()
        # text = re.sub('\r\n', '', text)
        # text = re.sub('\n', '', text)

    return text


def cut_sent(para):
    para = re.sub(r'([；。！？\?])([^”’])', r"\1\n\2", para)  # 单字符断句符
    para = re.sub(r'(\.{6})([^”’])', r"\1\n\2", para)  # 英文省略号
    para = re.sub(r'(\…{2})([^”’])', r"\1\n\2", para)  # 中文省略号
    para = re.sub(r'([；。！？\?][”’])([；。！？\?])', r'\1\n\2', para)
    # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
    para = para.rstrip()  # 段尾如果有多余的\n就去掉它
    # 很多规则中会考虑分号;，但是这里我把它忽略不计，破折号、英文双引号等同样忽略，需要的再做些简单调整即可。
    return para.split("\n")


def read_query(file_in, file_out):
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
            json.dump(query_object, op, ensure_ascii=False)
            op.write('\n')
    fp.close()
    op.close()


def segment(file_in, file_out, stopwords_file, url=None):
    fp = open(file_in, 'r', encoding='utf-8')
    op = open(file_out, 'a', encoding='utf-8')
    lines = fp.readlines()
    stopwords = load_stopwords(stopwords_file)
    count = 0
    for line in lines:
        if count % 3000 == 0:
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
            seg_words, _ = seg_bytrs(title, url, stopwords)
            op.write(' '.join(seg_words))
            op.write('\n')
        count += 1
    fp.close()
    op.close()


def segment_file(file_in, file_out, stopwords_file, url=None):
    fp = open(file_in, 'r', encoding='utf-8')
    op = open(file_out, 'w', encoding='utf-8')
    stopwords = load_stopwords(stopwords_file)
    count = 0
    while True:
        count += 1
        if count % 50000 == 0:
            print(count)
        # if count <= 1039705:
        #     continue
        line = ''
        try:
            line = fp.readline()
        except Exception as e:
            print(line)
            traceback.print_stack()
        else:
            if line:
                # seg_words, _ = seg_bytrs(line, url, stopwords)
                words = seg_byjieba(line, True, stopwords)
                op.write(' '.join(words))
                # op.write('\n')
        if not line:
            break

    fp.close()
    op.close()


def read_doc(file_in, file_out):
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


def read_json2(file_in, file_out):
    content_pattern = r'"h_DOCCONTENT":'
    title_pattern = r'"TITLE":'
    count = 0
    with open(file_in, 'r', encoding='utf-8') as fp:
        with open(file_out, 'w', encoding='utf-8') as op:
            while True:
                try:
                    line = fp.readline()
                except Exception as e:
                    print(line)
                    traceback.print_stack()
                else:
                    if re.search(content_pattern, line):
                        line = re.sub(content_pattern, '', line)
                        content = remove_html_url(line)
                        sentences = cut_sent(content)
                        if sentences:
                            for sentence in sentences:
                                # sentence = re.sub('\r\n', '', sentence)
                                # sentence = re.sub('\n', '', sentence)
                                op.write(sentence)
                                op.write('\n')

                    if re.search(title_pattern, line):
                        line = re.sub(title_pattern, '', line)
                        op.write(line)

                    if not line:
                        break

                    count += 1
                    if count % 100000 == 0:
                        print(count)


def read_json(file_in, file_out):
    with open(file_in, 'r', encoding='utf-8') as fp:
        with open(file_out, 'w', encoding='utf-8') as op:
            json_data = json.load(fp)['RECORDS']
            print('数据总数：%d' % len(json_data))
            for data in json_data:
                title = data["name"]
                op.write(title)
                op.write('\n')

                # content = data["h_DOCCONTENT"]
                # content = remove_html_url(content)
                # sentences = cut_sent(content)
                # if sentences:
                #     for sentence in sentences:
                #         op.write(sentence)
                #         op.write('\n')


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
    print('数据总数%d' % count)
    fp.close()
    op.close()


def merger_file(file_in, file_out):
    fp = open(file_in, 'r', encoding='utf-8')
    op = open(file_out, 'a', encoding='utf-8')
    op.write('\n')
    lines = fp.read()

    op.write(lines)
    fp.close()
    op.close()


def remove_digit(file_in, file_out):
    fp = open(file_in, 'r', encoding='utf-8')
    op = open(file_out, 'w', encoding='utf-8')

    count = 0

    not_contains_word = ['-','—','·','"','\u3000','\u2000','\xa0','\\','n','t','r','&#','&','ldquo','rdquo','/',':',',','.','*','〔','〕','mdash','+','{','}'
                         ,'．','…','n1','align','n2','middot','margin','t0','#','n3','～','%','’','‘','!','--','℃','@','≤','×']
    while True:
        try:
            line = fp.readline()
        except Exception as e:
            print(line)
            traceback.print_stack()
        else:
            words = line.split(' ')
            if words:
                for i, word in enumerate(words):
                    word = re.sub('\n', '', word)
                    if (
                            word
                            and not word.isdigit()
                            and not word.isnumeric()
                            and not contain_digits(word)
                            and not word.encode('utf-8').isalpha()
                            and not word.encode('utf-8').isalnum()
                            and word not in not_contains_word
                    ):
                        op.write(word)
                        op.write(' ')
                op.write('\n')

            if not line:
                break

            count += 1
            if count % 100000 == 0:
                print(count)
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

    # file_in = './data/gzzf/gzzf_bssx2.txt'
    # file_out = './data/gzzf/gzzf_seg2.txt'
    # stopwords_file = './data/stopwords.txt'
    # url = 'http://0.0.0.0:7777/cws'
    # print(file_in)
    # segment_file(file_in, file_out, stopwords_file, url)

    # file_in = './data/gzzf/gzzf_zcfg2.txt'
    # file_out = './data/gzzf/gzzf_zcfg_seg2.txt'
    # stopwords_file = './data/stopwords.txt'
    # url = 'http://0.0.0.0:7777/cws'
    # print(file_in)
    # segment_file(file_in, file_out, stopwords_file, url)

    # file_in = './data/gzzf/gzzf_news2.txt'
    # file_out = './data/gzzf/gzzf_seg2.txt'
    # stopwords_file = './data/stopwords.txt'
    # url = 'http://0.0.0.0:7777/cws'
    # print(file_in)
    # segment_file(file_in, file_out, stopwords_file, url)

    # file_in = './data/gzzf/gzzf_xxgk2.txt'
    # file_out = './data/gzzf/gzzf_xxgk_seg2.txt'
    # stopwords_file = './data/stopwords.txt'
    # url = 'http://0.0.0.0:7777/cws'
    # print(file_in)
    # segment_file(file_in, file_out, stopwords_file, url)

    # file_in = './data/gzzf/tys_xxgk_govopendatanews.json'
    # file_out = './data/gzzf/gzzf_xxgk.txt'
    # read_json2(file_in, file_out)

    # file_in = './data/gzzf/tys_xxgk_govopendatanews.json'
    # file_out = './data/gzzf/gzzf_xxgk.txt'
    # read_json(file_in, file_out)

    # file_in = './data/gzzf/gzzf_zcfg_seg2.txt'
    # file_out = './data/gzzf/gzzf_zcfg_seg2.txt'
    # merger_file(file_in, file_out)

    # file_in = './data/v0.0.1/gzzf_remove_digit.txt'
    # file_out = './data/v0.0.1/gzzf_remove_digit2.txt'
    # remove_digit(file_in, file_out)
    #
    # word = '年月'
    # print(word.encode('utf-8').isalpha())
    #
    # word = '年月'
    # print(word.encode('utf-8').isalnum())


