# -*- coding: utf-8 -*-
# @Time    : 18-8-2 下午3:16
# @Author  : Marvin
# @File    : demo.py
# @Notes   : 
import requests
import re
import json

result = []
r = requests.get(url='https://www.trackico.io/ajax/ico-statistics/ubcoin/')
data = re.findall(re.compile(r'makeCofig\({(.*?);Highcharts'), r.text)
for i in data:
    dic = {}
    name = re.findall(re.compile(r'text:\"(.*)\"'), i)[0]
    print(name)
    dic['soicalName'] = name
    d = re.findall(re.compile(r'data:\[(.*)}]}'), i)[0]
    d = re.sub(re.compile(r'Date\.UTC'), '', d)
    print(d)
    data_list = re.findall(re.compile(r'\[(.*?)\]'), d)
    dic['item'] = []
    for d_list in data_list:
        # 时间
        date = re.findall(re.compile(r'\((.*?)\)'), d_list)[0]
        # 数值
        num = re.findall(re.compile(r'\),(.*)'), d_list)[0]
        dic['item'].append({'date': date, 'value': num})
    result.append(dic)

print(json.loads(json.dumps(result)))