#encoding:utf-8   
import urllib2
import requests   
import bs4
import re

for page in range(18,40):    
    html = urllib2.urlopen('http://job.bupt.edu.cn/fair-datas/25/' + str(page) + '/1.html')      
    txt = html.read().replace(":","").replace(",","").replace("\"\"","\"").replace("{","").replace("}","").replace("[","").replace("]","")
    file = open('CareerTalk1.csv','ab')  
    txt_list = txt.split("\"")
    row = ""
    timeline = ""
    num = 0
    pattern = re.compile(r'\b\d{2}/\d{2}/\d{4}\b|\b\d{1}:\d{2}\b|\b\d{2}:\d{2}\b')#定义匹配模式
    while num < len(txt_list):
        if txt_list[num] == "id":
            file.write(row + timeline.encode('utf-8') + "\r\n")
            row = ""
            timeline = ""
            detail = requests.get('http://job.bupt.edu.cn/fair.1.'+txt_list[num+1]+'.html')                                                     
            soup = bs4.BeautifulSoup(detail.text, 'html.parser') 
            if len(soup.select('div[class="row-show"]')):
                content = soup.select('div[class="row-show"]')[1].select('div[class="area-show"]')
                #print str(content[0])
                if not str(content[0]) is None:
                    if len(re.findall(pattern,str(content[0]))):
                        time_list = re.findall(pattern,str(content[0]))
                        for time in time_list:
                            timeline = timeline + time + ","
            num = num + 2
        elif txt_list[num] == "title":
            row = row + txt_list[num+1] + ","
            num = num + 2
        elif txt_list[num] == "startTime":
            row = row + txt_list[num+1][0:11] + ","
            num = num + 2
        elif txt_list[num] == "address":
            row = row + txt_list[num+1] + ","
            num = num + 2
        else:
            num = num + 1
    file.close()