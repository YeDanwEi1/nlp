#改3个地方 ：1和3+2个neg

import requests
import pandas as pd
import sqlite3
import json
import time
import csv
import re
import xlwt
#requests请求地址
url = 'https://club.jd.com/comment/productPageComments.action?'
#      https://club.jd.com/comment/productPageComments.action?  利用开发工具获取真实网址（当源码中搜不到时才需要 即动态网页；若静态网页则用css选择器方便）
# callback=fetchJSON_comment98&productId=10036040275246&score=0&sortType=5&page=0&pageSize=10&isShadowSku=0&fold=1
#文件头 user-agent必加
header = {
	'User-Agent':'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/76.0.3809.100 Safari/537.36'
}
#多页评论
def main():
#1、neg
	datalist = []
	savepath = "neg_comment.xls"
	for i in range(80):
		# 请求参数
		data = {
				"productId": 100006584727,
				"score": 1,#3好评；1差评
				"sortType": 5,
				"page": i,
				"pageSize": 10,
				"isShadowSku": 0,
				"rid": 0,
				"fold": 1}
		response = requests.get(url, params=data, headers=header)#发出get请求
		#print(type(response.text))json格式的字符串，要将其转化为字典
		js_data=json.loads(response.text)#转化为字典
		#print(type(js_data))
		for j in range(len(js_data['comments'])):
			data1 = [(js_data['comments'][j]['content'])]
			datalist.append(data1)
		time.sleep(5)
		print("page " + str(1 + i) + " has done")
	attitude="neg"
	saveData_excel(datalist,savepath,attitude)
#2、pos
	datalist = []
	savepath = "pos_comment.xls"
	for i in range(100):
		# 请求参数
		data = {
			"productId": 100006584727,
			"score": 3,  # 3好评；1差评
			"sortType": 5,
			"page": i,
			"pageSize": 10,
			"isShadowSku": 0,
			"rid": 0,
			"fold": 1}
		response = requests.get(url, params=data, headers=header)  # 发出get请求
		# print(type(response.text))json格式的字符串，要将其转化为字典
		js_data = json.loads(response.text)  # 转化为字典
		# print(type(js_data))
		for j in range(len(js_data['comments'])):
			data1 = [(js_data['comments'][j]['content'])]
			datalist.append(data1)
		time.sleep(5)
		print("page " + str(1 + i) + " has done")
	attitude = "pos"
	saveData_excel(datalist, savepath, attitude)

	# excel转csv
	exl1 = pd.read_excel('pos_comment.xls', 'comment', index_col=0)
	exl1.to_csv('pos.csv', encoding='utf-8')

	exl2 = pd.read_excel('neg_comment.xls', 'comment', index_col=0)
	exl2.to_csv('neg.csv', encoding='utf-8')
	# 合并csv
	df1 = pd.read_csv("pos.csv")
	df2 = pd.read_csv("neg.csv")
	df = pd.concat([df1, df2])
	# 保存合并后的文件
	df.to_csv('reviews.csv', index=False, encoding='utf-8')  # 不要下标
#保存数据到excel
def saveData_excel(datalist,savepath,attitude):
    book=xlwt.Workbook(encoding="utf-8")#创建word对象
    sheet=book.add_sheet('comment',cell_overwrite_ok=True)#创建sheet表
    col = ("content","content_type")
    for i in range(0, 2):
        sheet.write(0, i, col[i])
    for i in range(0, len(datalist)):
        data1 = datalist[i]
        for j in range(0, 2):
            if(j==1):sheet.write(i + 1, j, attitude)
            else:sheet.write(i + 1, j, data1)
    book.save(savepath)
if __name__=="__main__":
    main()