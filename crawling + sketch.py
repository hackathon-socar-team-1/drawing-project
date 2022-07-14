# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 09:25:51 2022

@author: kiseok2
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 13:22:31 2022

@author: kiseok2
"""



import time
from urllib.request import (urlopen, urlparse, urlunparse, urlretrieve)

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
import urllib.request
import os
import pandas as pd

from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
import cv2



path = "./"
file_list = os.listdir(path)

chrome_path =r'C:\Users\kiseok2\Desktop\python\chromedriver.exe'
base_url = "https://www.google.co.kr/imghp"

chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument("lang=ko_KR") # 한국어
chrome_options.add_argument('window-size=1920x1080')


def selenium_scroll_option():
  SCROLL_PAUSE_SEC = 3
  
  # 스크롤 높이 가져옴
  last_height = driver.execute_script("return document.body.scrollHeight")
  
  while True:
    # 끝까지 스크롤 다운
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

    # 1초 대기
    time.sleep(SCROLL_PAUSE_SEC)

    # 스크롤 다운 후 스크롤 높이 다시 가져옴
    new_height = driver.execute_script("return document.body.scrollHeight")
  
    if new_height == last_height:
        break
    last_height = new_height
    
    


our_keyword_list = ["mountain",'beach','buliding']


# 키워드 검색하기

a=input("검색할 키워드를 입력 : ")
image_name = input("저장할 이미지 이름 : ")
our_keyword_list.append(image_name)
#b=int(input("몇 개 저장할래? : "))
driver = webdriver.Chrome(chrome_path,chrome_options=chrome_options)
driver.get('http://www.google.co.kr/imghp?hl=ko')
browser = driver.find_element(By.NAME, "q") #구글 검색창 선택
browser.send_keys(a) # 검색어를 넣어줌
browser.send_keys(Keys.RETURN)  #검색어를 넣고 엔터를치는 것




# 클래스를 찾고 해당 클래스의 src 리스트를 만들자

selenium_scroll_option() # 스크롤하여 이미지를 많이 확보
driver.find_elements(By.XPATH,'//*[@id="islmp"]/div/div/div/div[1]/div[2]/div[2]/input')[0].click() # 이미지 더보기 클릭
selenium_scroll_option()



'''이미지 src요소를 리스트업해서 이미지 url 저장'''

images = driver.find_elements(By.CSS_SELECTOR,".rg_i.Q4LuWd") #  클래스 네임에서 공백은 .을 찍어줌

images_url = []
for i in images: 
   
   if i.get_attribute('src')!= None :
        images_url.append(i.get_attribute('src'))
   else :
       images_url.append(i.get_attribute('data-src'))






if image_name in file_list:
    for t, url in enumerate(images_url, 0):        
        urlretrieve(url,'./'+image_name+'/' + 'a_' + str(t) + '.png')
    driver.close()  

else:
    os.makedirs(path+image_name, exist_ok=True)
    for t, url in enumerate(images_url, 0):        
        urlretrieve(url,'./'+image_name+'/' + 'a_' + str(t) + '.png')
    driver.close() 
    


## picture > sketch 변환



image_path = path+image_name
sketch_path =path+image_name+"_sketch" 

os.makedirs(sketch_path, exist_ok=True)


def img2sketch(photo, k_size, name):
    #Read Image
    img=cv2.imread(photo)
    
    # Convert to Grey Image
    grey_img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Invert Image
    invert_img=cv2.bitwise_not(grey_img)
    #invert_img=255-grey_img

    # Blur image
    blur_img=cv2.GaussianBlur(invert_img, (k_size,k_size),0)

    # Invert Blurred Image
    invblur_img=cv2.bitwise_not(blur_img)
    #invblur_img=255-blur_img

    # Sketch Image
    sketch_img=cv2.divide(grey_img,invblur_img, scale=256.0)

    # Save Sketch 
    
    cv2.imwrite(f'./{sketch_path}/{name}.png', sketch_img)

    # Display sketch
    # cv2.imshow(f'./korea/pic_to_sketch/{name}.png',sketch_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
#Function call

print(len(os.listdir(path+image_name)))
for i in range(len(os.listdir(path+image_name))):
    img2sketch(photo=f'./{image_path}/a_{i}.png', k_size=7, name=f'a_{i}')
    




