import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.pylab import style #自定义图表风格
style.use('ggplot')
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
plt.rcParams['font.sans-serif'] = ['Simhei'] # 解决中文乱码问题
import imageio
import re
import jieba.posseg as psg
import jieba
import itertools
#conda install -c anaconda gensim
from gensim import corpora,models #主题挖掘，提取关键信息

# pip install wordcloud
from wordcloud import WordCloud,ImageColorGenerator
from collections import Counter

from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

import pyLDAvis.gensim
import graphviz

#读入
reviews=pd.read_csv('./reviews.csv')

#去重
reviews=reviews.drop_duplicates()

#清洗，删除数字、字母
content=reviews['content']
#Series和DataFrame是Pandas(数据分析)的基本数据结构，直接从表格中选一列数据，类型是series(选择多列时数据类型是DataFrame)。
info=re.compile('[0-9a-zA-Z]')
#lambda类似匿名函数：x为参数，info.sub('',x)为函数体
#apply:自动遍历整个 Series，按照相对应的函数进行运算。axis默认值为0--遍历列。
content=content.apply(lambda x: info.sub('',x))  #遍历到的元素作为x，对x进行替换

#分词，
jieba.add_word("差评",freq=10,tag='n')
jieba.add_word("牛逼",freq=10,tag='a')
jieba.add_word("智商税",freq=10,tag='n')
seg_content=content.apply( lambda s:  [(x.word,x.flag) for x in psg.cut(s)] )#使用jieba的psg分词，得到词组和词性
print(seg_content)#由元组（单词，词性）组成的list  外面再套了一个series

#统计每条评论的词数
n_word=seg_content.apply(lambda s: len(s))
print(n_word)

#得到各分词在第几条评论#[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5...
n_content=[ [x+1]*y for x,y in zip(list(seg_content.index),list(n_word))] #[x+1]*y,表示复制y份，由list组成的list
index_content_long=sum(n_content,[]) #sum表示去掉[]，拉平,返回list

#分词及词性去掉[]，拉平,返回list。[('东西', 'ns'), ('收到', 'v'), ('这么久', 'r'),...]
seg_content_long=sum(seg_content,[])

#得到加长版的词组、词性['东西', '收到',...]和['ns', 'v'...]
word_long=[x[0] for x in seg_content_long]
nature_long=[x[1] for x in seg_content_long]

#content_type拉长
n_content_type=[ [x]*y for x,y in zip(list(reviews['content_type']),list(n_word))] #[x]*y,表示复制y份
content_type_long=sum(n_content_type,[]) #表示去掉[]，拉平

#形成表格数据类型dataframe【词组所属评论，词组，词性，词组所属评论是pos还是neg】
#dataframe由多个Series数据列组成
review_long=pd.DataFrame({'index_content':index_content_long,
                        'word':word_long,
                        'nature':nature_long,
                        'content_type':content_type_long})

#去除标点符号：词性为x的所有词
review_clean=review_long[review_long['nature']!='x'] #x表标点符号

#导入停用词
stop_path=open('./stoplist.txt','r',encoding='UTF-8')
stop_words=stop_path.readlines()
stop_words=[word.strip('\n') for word in stop_words]#去换行符

#得到不含停用词的分词表
word_long_clean=list(set(word_long)-set(stop_words))

#保留非停用词 最后还剩26645个词组
review_clean=review_clean[review_clean['word'].isin(word_long_clean)]

#dataframe中增加一列：每个词组在各自评论中的index
n_word=review_clean.groupby('index_content').count()['word']#词数(1, 40) (2, 57)...
index_word=[ list(np.arange(1,x+1)) for x in list(n_word)]#得到1到x的排列 [[1, 2...40], [1... 57],
index_word_long=sum(index_word,[]) #去[]
review_clean['index_word']=index_word_long
review_clean.to_csv('./1_review_clean.csv')#存储（清洗后的表格）

#提取名词
n_review_clean=review_clean[[ 'n' in nat for nat in review_clean.nature]]

#词云图
font=r"C:\Windows\Fonts\msyh.ttc"
background_image=imageio.imread('static/image/pl.jpg')
wordcloud = WordCloud(font_path=font, max_words = 100, background_color='white',mask=background_image,color_func=lambda *args, **kwargs: "green") #width=1600,height=1200, mode='RGBA'
wordcloud.generate_from_frequencies(Counter(review_clean.word.values))#计算频率
wordcloud.to_file('static/image/1_分词后的词云图.png')
plt.figure(figsize=(20,10))
plt.imshow(wordcloud)
plt.axis('off')
plt.show()
#名词词云图
background_image=plt.imread('static/image/pl.jpg')
wordcloud = WordCloud(font_path=font, max_words = 100, mode='RGBA' ,background_color='white',mask=background_image,color_func=lambda *args, **kwargs: "green") #width=1600,height=1200
wordcloud.generate_from_frequencies(Counter(n_review_clean.word.values))
wordcloud.to_file('static/image/1_分词后的词云图(名词).png')
plt.figure(figsize=(20,10))
plt.imshow(wordcloud)
plt.axis('off')
plt.show()

#来自知网发布的情感分析用词语集
pos_comment=pd.read_csv('./正面评价词语（中文）.txt',header=None,encoding='utf-8')
neg_comment=pd.read_csv('./负面评价词语（中文）.txt',header=None,encoding='utf-8')
pos_emotion=pd.read_csv('./正面情感词语（中文）.txt',header=None,encoding='utf-8')
neg_emotion=pd.read_csv('./负面情感词语（中文）.txt',header=None,encoding='utf-8')
#合并-axis=0纵向合并
pos=pd.concat([pos_comment,pos_emotion],axis=0)
neg=pd.concat([neg_comment,neg_emotion],axis=0)

#增加新词 得到positive negative
new_pos=pd.Series(['点赞','棒棒','不错','流畅','好看','舒服','赞','买'])
new_neg=pd.Series(['歇菜','下架','垃圾','恶心'])
positive=pd.concat([pos,new_pos],axis=0)
negative=pd.concat([neg,new_neg],axis=0)

#情感词打标签 pos为1，neg为-1
positive.columns=['review']
positive['weight']=pd.Series([1]*len(positive))
negative.columns=['review']
negative['weight']=pd.Series([-1]*len(negative))
pos_neg=pd.concat([positive,negative],axis=0)

#表联接 weight=0/1/-1
data=review_clean.copy()
review_dic=pd.merge(data,pos_neg,how='left',left_on='word',right_on='review')#left：以左边的表格为基准，left_on和right_on用于连接的列
#删review，Nan替换为0（也就是非情感词的weight设为0）
review_dic=review_dic.drop(['review'],axis=1)#删除指定的列
review_dic=review_dic.replace(np.nan,0)

#读入否定词
notdict=pd.read_csv('./not.csv')
notdict['freq']=[1]*len(notdict)

#amend_weight初始为1，id为索引，用于修改amend_weight值
review_dic['amend_weight']=review_dic['weight']
review_dic['id']=np.arange(0,review_dic.shape[0])

#只保留有情感值的行到sa_review_dic中，待会遍历
sa_review_dic=review_dic[review_dic['weight']!=0]
sa_review_dic.index=np.arange(0,sa_review_dic.shape[0]) #索引重置

#计算每个词组的amend_weight，带有否定词则改为-weight
index = sa_review_dic['id']
for i in range(0, sa_review_dic.shape[0]):
    review_i = review_dic[review_dic['index_content'] == sa_review_dic['index_content'][i]]  # 第i个情感词所处评论的 所有分词
    review_i.index = np.arange(0, review_i.shape[0])  # 重置索引后，索引值等价于index_word
    word_ind = sa_review_dic['index_word'][i]  # 第i个情感值在该条评论的位置
    # 第一种，在句首。则不用判断
    # 第二种，在评论的第2个为位置
    if word_ind == 2:
        ne = sum([review_i['word'][word_ind - 1] in notdict['term']])#只看前一个词是否在not词典中
        if ne == 1:#单重否定改；双重否定不改
            review_dic['amend_weight'][index[i]] = -(review_dic['weight'][index[i]])
    # 第三种，在评论的第2个位置以后
    elif word_ind > 2:
        ne = sum([word in notdict['term'] for word in#看前两个词是否在not词典中
                  review_i['word'][[word_ind - 1, word_ind - 2]]])
        if ne == 1:
            review_dic['amend_weight'][index[i]] = - (review_dic['weight'][index[i]])
review_dic.tail()

#合并，计算每条评论的情感词的总分
emotion_value=review_dic.groupby('index_content',as_index=False)['amend_weight'].sum()


#只取！=0部分 >0标记为pos反之neg，
content_emotion_value=emotion_value.copy()
content_emotion_value=content_emotion_value[content_emotion_value['amend_weight']!=0]#只取！=0部分
content_emotion_value['dic_type']=''
content_emotion_value['dic_type'][content_emotion_value['amend_weight']>0]='pos'
content_emotion_value['dic_type'][content_emotion_value['amend_weight']<0]='neg'

#合并到大表中 根据index_content即评论下标来合并
content_emotion_value=content_emotion_value.drop(['amend_weight'],axis=1)#删除指定列
review_dic=pd.merge(review_dic,content_emotion_value,how='left',left_on='index_content',right_on='index_content')
review_dic=review_dic.drop(['id'],axis=1)
review_dic.to_csv('./1_review_dic.csv',index=True,header=True)#最终的表格

#分析基于情感词典的精确度--89%
cate=['index_content','content_type','dic_type']
data_type=review_dic[cate].drop_duplicates()
data=data_type[['content_type','dic_type']]
data=data.dropna(axis=0)
print("基于情感词典的精确度为:")
print( classification_report(data['content_type'],data['dic_type']) )

#只看情感词的词云图
data=review_dic.copy()
data=data[data['amend_weight']!=0]
word_data_pos=data[data['dic_type']=='pos']
word_data_neg=data[data['dic_type']=='neg']
#pos词云图
wordcloud = WordCloud(font_path=font, max_words = 100, mode='RGBA' ,background_color='white',mask=background_image,color_func=lambda *args, **kwargs: "red") #width=1600,height=1200
wordcloud.generate_from_frequencies(Counter(word_data_pos.word.values))
wordcloud.to_file('static/image/2_词云图(只看pos情感词).png')
plt.figure(figsize=(15,7))
plt.imshow(wordcloud)
plt.axis('off')
plt.show()
#neg词云图
wordcloud = WordCloud(font_path=font, max_words = 100, mode='RGBA' ,background_color='white',mask=background_image,color_func=lambda *args, **kwargs: "black") #width=1600,height=1200
wordcloud.generate_from_frequencies(Counter(word_data_neg.word.values))
wordcloud.to_file('static/image/2_词云图(只看neg情感词).png')
plt.figure(figsize=(15,7))
plt.imshow(wordcloud)
plt.axis('off')
plt.show()

#三、基于LDA的模型的主题分析
#获取pos和neg词组
data=review_dic.copy()
print("data",data)
word_data_pos=data[data['dic_type']=='pos']
word_data_neg=data[data['dic_type']=='neg']

#建立词典，去重
pos_dict=corpora.Dictionary([ [i] for i in word_data_pos.word])
neg_dict=corpora.Dictionary([ [i] for i in word_data_neg.word])

#建立语料库；元组（x,y),x是在词典中的位置，y是1表示存在。
pos_corpus=[ pos_dict.doc2bow(j) for j in [ [i] for i in word_data_pos.word] ] #shape=(n,(2,1))
neg_corpus=[ neg_dict.doc2bow(j) for j in [ [i] for i in word_data_neg.word] ]

#余玄相似度函数
def cos(vector1,vector2):
    dot_product=0.0
    normA=0.0
    normB=0.0
    for a,b in zip(vector1,vector2):
        dot_product +=a*b
        normA +=a**2
        normB +=b**2
    if normA==0.0 or normB==0.0:
        return None
    else:
        return ( dot_product/((normA*normB)**0.5) )

# 计算每个主题数下的平均余玄相似度（2-10），值越小越好
def LDA_k(x_corpus, x_dict):
    # 初始化平均余玄相似度
    mean_similarity = []
    mean_similarity.append(1)#划分为1个主题 相似度肯定是1。

    # 循环生成主题并计算主题间相似度（主题数范围：2-10）
    for i in np.arange(2, 11):
        lda = models.LdaModel(x_corpus, num_topics=i, id2word=x_dict)  # LDA模型训练

        for j in np.arange(i):
            term = lda.show_topics(num_words=50)

        # 提取各主题词
        top_word = []  # shape=(i,50)
        for k in np.arange(i):
            top_word.append([''.join(re.findall('"(.*)"', i)) for i in term[k][1].split('+')])  # 列出所有词

        # 构造词频向量
        word = sum(top_word, [])  # 列出所有词
        unique_word = set(word)  # 去重

        # 构造主题词列表，行表示主题号，列表示各主题词
        mat = []  # shape=(i,len(unique_word))
        for j in np.arange(i):
            top_w = top_word[j]
            mat.append(tuple([top_w.count(k) for k in unique_word]))  # 统计list中元素的频次，返回元组

        # 两两组合。方法一
        p = list(itertools.permutations(list(np.arange(i)), 2))  # 返回可迭代对象的所有数学全排列方式。
        y = len(p)  # y=i*(i-1)
        top_similarity = [0]
        for w in np.arange(y):
            vector1 = mat[p[w][0]]
            vector2 = mat[p[w][1]]
            top_similarity.append(cos(vector1, vector2))

        # 计算每个主题数下的平均余玄相似度（主题数2~10）
        mean_similarity.append(sum(top_similarity) / y)
    return mean_similarity

#调用函数计算每个主题数下的平均余玄相似度，越小越好
pos_k=LDA_k(pos_corpus,pos_dict)
neg_k=LDA_k(neg_corpus,neg_dict)

#主题数由该图确定
#由图可知主题数为2和3相似度最低，往后没有意义
pd.Series(pos_k,index=range(1,11)).plot()
plt.title('正面评论LDA主题数寻优')
plt.show()
pd.Series(neg_k,index=range(1,11)).plot()
plt.title('负面评论LDA主题数寻优')
plt.show()

pos_lda=models.LdaModel(pos_corpus,num_topics=2,id2word=pos_dict)
neg_lda=models.LdaModel(neg_corpus,num_topics=3,id2word=neg_dict)
print('------')
print(pos_lda.print_topics(num_topics=2))
print(neg_lda.print_topics(num_topics=4))
#这种。他只会返回规定主题数下的多少词
# ，每个词后面的小数可认为是这个词属于这个主题的概率，主题下所有词的概率和为1；
# 而这个主题应该是什么，就要靠人工后面分析来定义了。
#那也许能通过概率衡量出这个词和这个主题的关系，但分析不同主题之间的关系 以及一个词和其他主题的关系有点困难。这时就要引出LDA可视化分析工具了。

#LDA可视化：lda: 计算好的话题模型；corpus: 文档词频矩阵；dictionary: 词语空间

#pos
lda = models.LdaModel(pos_corpus, num_topics=2, id2word=pos_dict)  # LDA模型训练
d=pyLDAvis.gensim.prepare(lda, pos_corpus, pos_dict)
pyLDAvis.save_html(d, 'templates/lda_pos.html')# 将结果保存为该html文件


# neg
lda = models.LdaModel(neg_corpus, num_topics=4, id2word=neg_dict)  # LDA模型训练-传入文档词频矩阵 主题数 词典
d=pyLDAvis.gensim.prepare(lda, neg_corpus, neg_dict)#调用建模进行数据可视化
pyLDAvis.save_html(d, 'templates/lda_neg.html')# 将结果保存为该html文件
