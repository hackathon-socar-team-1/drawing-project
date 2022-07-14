# drawing

사용자가 꿈꾸는 여행지 추천

<h1 style = "text-align: center">
  
✨쏘카 실전 데이터로 배우는 AI 엔지니어 육성 부트캠프 2기✨ <br><br>
사용자가 꿈꾸는 여행지 추천[1팀-찾아 DREAM]
</h1>




## 🌟 drawing-project : 사용자가 꿈꾸는 여행지 추천

일반적인 키워드 검색이나 SNS 검색이 아닌, 그림을 통해 사용자가 원하는 여행지를 추천하는 서비스 입니다<br>
<br>
이러한 분들을 위해 저희 서비스를 추천합니다<br>
1) 여행을 어디가야 할지 막막한 사람 <br>
2) 신선한 방식으로 여행을 가고 싶은 사람.<br>
3) 여행을 왔는데, 찾아본 여행지가 맘에 안들어 새로운 여행지를 원하는 사람<br>


## ✈순서 및 로직

1. 사용자가 가고 싶은 여행지를 간단하게 스케치 합니다<br>
  -잘 그릴 필요 없습니다! 아주 간단한 스케치면 충분합니다! 
2. 사용자의 스캐치는 학습된 pix2pix 모델에 의해 하나의 풍경화로 바뀌게 됩니다.
3. 사용자로 부터 키워드를 입력받아 네이버 지도에서 크롤링을 하여 "가볼만한 곳"의 이미지와 장소의 정보를 크롤링 합니다.
4. 크롤링한 이미지들과 사용자의 풍경화를 vgg16모델을 fine-tuning한 feature-extractor 모델을 사용하여 유사도를 비교합니다
5. 가장 유사한 장소의 이미지와 정보를 사용자에게 보여줍니다


## 👇🏻 서비스 링크

아직 local host

## 📷 스크린샷

![image](https://user-images.githubusercontent.com/75923078/178495453-3cd92ce8-6e64-47d4-97d6-1df931e3dabf.png)

## ⚙️ 아키텍처

![image](https://user-images.githubusercontent.com/75923078/178495791-5202456c-aafa-4c84-9acd-78c2d7714c55.png)


## 📌 팀원 정보

👩‍🦰 김명희 [@huista][https://github.com/huista]

😆 류호원 [@howon-ryu][https://github.com/howon-ryu]

👨 김효석 [@gytjr8422][https://github.com/gytjr8422]

🤵 김기석 [@kiseseok][https://github.com/kiseseok]



## 🛠 사용 기술

- ### **프론트엔드** - html, css

- ### **백엔드** - python, flask

- ### **크롤링** - python

- ### **사모델** - vgg16, feature0-extractor, unet
