#encoding:utf-8  
import urllib2 
import requests   
import bs4
import re
  
#里面写你的cookie    
HEADERS = {"cookie": '_ga=GA1.3.605766813.1525780564; Hm_lvt_41e71a1bb3180ffdb5c83f253d23d0c0=1523851406,1525931380,1525931510,1526040246; JSESSIONID=0000pENNDVWc9T77IAlWk583DMO:19fuml00i'}  
#url = 'http://my.bupt.edu.cn/index.portal'
file = open('AcademicLecture.csv','ab')  
for page in range(4,25):          
    url = 'http://my.bupt.edu.cn/detach.portal?.p=Znxjb20ud2lzY29tLnBvcnRhbC5jb250YWluZXIuY29yZS5pbXBsLlBvcnRsZXRFbnRpdHlXaW5kb3d8cGU3MjJ8dmlld3xub3JtYWx8YWN0aW9uPWxlY3R1cmVCcm93c2Vy&pageIndex='+str(page) 
    req = urllib2.Request(url, headers=HEADERS)  
    text = urllib2.urlopen(req).read()  
    page_soup = bs4.BeautifulSoup(text, 'html.parser') 
    #print page_soup
    content = page_soup.select('div[class=" pull-right"]')
    li_list =  content[0].select('li') 
    for li in li_list:
        row = ""
        li_str = li.select('a')[0].string.replace("\t","").replace("\r\n","")
        li_href = li.select('a')[0]['href']
        detail_req = urllib2.Request('http://my.bupt.edu.cn/'+ li_href, headers=HEADERS)  
        detail_text = urllib2.urlopen(detail_req).read()   
        detail_soup = bs4.BeautifulSoup(detail_text, 'html.parser') 
        detail_content = detail_soup.select('div[class="singleinfo"]')
        p_list = detail_content[0].select('p')
        row = row + li_str[0:10] + "," + li_str[17:] + "," + p_list[0].text[5:] + "," + p_list[1].text[17:] + "," + p_list[2].text[17:] + "," + p_list[3].text[4:]
        file.write(row.encode('utf-8') + "\r\n")
        print row
file.close()
