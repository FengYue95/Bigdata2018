import sys
reload(sys)
sys.setdefaultencoding('utf-8')
from bs4 import BeautifulSoup  
import requests  
import csv  
import bs4  
  
  
#检查url地址  
def check_link(url):  
    try:  
          
        r = requests.get(url)  
        r.raise_for_status()  
        r.encoding = r.apparent_encoding  
        return r.text  
    except:  
        print('无法链接服务器！！！')  
  
  
#爬取资源  
def get_contents(ulist,rurl):  
    soup = BeautifulSoup(rurl,'lxml')  
    trs = soup.find_all('tr')  
    for tr in trs:  
        ui = []  
        for td in tr:  
            ui.append(td.string)  
        ulist.append(ui)  
      
#保存资源  
def save_contents(urlist,num):  
    file = "C:/Users/moonshine/Documents/Jupyter/2016-"+str(num)+".csv"
    with open(file,'w') as f:  
        writer = csv.writer(f)  
        writer.writerow([urlist[0][0],urlist[0][1],urlist[0][3],urlist[0][4],urlist[0][5],urlist[0][6],urlist[0][7],urlist[0][8],urlist[0][9],urlist[0][10]])  
        for i in range(len(urlist)-1):  
            if i%2 == 0:
                writer.writerow([urlist[i+1][0],urlist[i+1][1],urlist[i+1][3],urlist[i+1][4],urlist[i+1][5],urlist[i+1][6],urlist[i+1][7],urlist[i+1][8],urlist[i+1][9],urlist[i+1][10]])   
            else:
                writer.writerow([urlist[i][0],urlist[i+1][0],urlist[i+1][2],urlist[i+1][3],urlist[i+1][4],urlist[i+1][5],urlist[i+1][6],urlist[i+1][7],urlist[i+1][8],urlist[i+1][9]])
  
def main():    
    for num in range(1,13):
        urli = []
        url = "https://tianqi.911cha.com/beijing/2016-"+str(num)+".html"
        rs = check_link(url)  
        get_contents(urli,rs)  
        save_contents(urli,num) 
main()  