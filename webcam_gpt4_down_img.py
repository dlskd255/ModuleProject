api_key = "AIzaSyC1Ax9tlpuKI8qC7hh0RVnUNkJYrAOJe10"

import cv2
import mediapipe as mp
import requests
import numpy as np
from multiprocessing import Process, Manager
import multiprocessing
import math
import time
import traceback
import asyncio
import os
import sys
import urllib.request
import json
import speech_recognition as sr
from copy import deepcopy
import pickle
import re
import pygame

def play_mp3(mp3_file):
    pygame.mixer.init()
    pygame.mixer.music.load(mp3_file)
    pygame.mixer.music.play()

mp3_file = "./son.mp3"

"""
# 관측 시야(Field Of View) - 최대 120 기본값 90
fov = "120"
# 방향 - 범위 0 ~ 360 (0 or 360::북, 180: 남)
heading = "-45"
# 카메라 상하 방향 설정 - 범위 -90 ~ 90 기본값 0
pitch = "30"
"""
def distance_with_cv2(a,b):
    return cv2.norm((a.x,a.y,a.z),(b.x,b.y,b.z))

def center_point_distance(a,b,c,d):
    return cv2.norm(((a.x+c.x)/2,(a.y+c.y)/2,(a.z+c.z)/2),((b.x+d.x)/2,(b.y+d.y)/2,(b.z+d.z)/2))

def calculate_angle(a, b, c):
    """세 점 간의 각도를 계산하는 함수"""
    radians = math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0])
    angle = abs(math.degrees(radians))
    return angle

def extract_and_sum_numbers(text):
    # 숫자만 추출하여 리스트로 저장
    numbers = text.split(".")
    
    # 숫자를 문자열에서 추출하고 합치기
    number = numbers[0]+numbers[1]

    return number


def head_nod_algorithm(left_eye,right_eye,poseLandmarks,steps,avg_hori,avg_vert):#(left_shoulder_coords, right_shoulder_coords, head_coords):
    """고개를 상하좌우로 흔드는 경우를 판단하는 알고리즘"""
    
    # 고개 좌우로 움직이는 경우에 대한 계산
    head_horizontal_movement = (left_eye.z - right_eye.z) # 머리의 z축 방향 이동량 계산 
    horiDiff = distance_with_cv2(left_eye,right_eye) # 눈 사이 거리 계산
    head_horizontal_movement /= horiDiff # 이동량을 눈 사이 거리로 나눠 tan 값을 계산.
    
    calib_hori = 0
    calib_vert = 0
        
    ListReturn = [0,0,head_horizontal_movement,0,"",""]
    if steps > 300:
        calib_hori = avg_hori
        calib_vert = avg_vert
        head_horizontal_movement -= calib_hori
        if head_horizontal_movement > 0.35:  # 예제의 임계값
            ListReturn[0] = 1
            ListReturn[4] = "right"
        elif head_horizontal_movement < -0.35:
            ListReturn[0] = -1    
            ListReturn[4] = "left"
    
    return ListReturn

def speech_recognator(firstLocation,condition):
    api_key = "AIzaSyC1Ax9tlpuKI8qC7hh0RVnUNkJYrAOJe10"
    # 1) 음성 인식기
    condition = True
    while condition == True:
        try:
            
            print("record process is on")
            r = sr.Recognizer() # 음성 인식을 위한 객체 생성            
            mic = sr.Microphone(device_index = 1)
            # 마이크 객체 선언, 인덱스는 각 노트북의 마이크 번호를 의미합니다. 만약 인식이 안되시면 바꿔보시면서 테스트 해보시면 될 듯 합니다.
            with mic as source:
                audio = r.listen(source,timeout=5, phrase_time_limit = 5) # 마이크에서 5초 동안 음성을 듣고 audio 변수에 저장합니다.

            result = r.recognize_google(audio, language = "ko-KR",show_all=True) # 인식한 음성을 텍스트로 변환

            confidence = float(result['alternative'][0]['confidence'])
            if confidence > 0.85:
                result1 = result['alternative'][0]['transcript']
                if result1 != "보정":
                    print(f"입력 받았습니다. {result1}(으)로 이동합니다.")
                    values = geocode_address(result1,api_key)
                    if values[0] == True:
                        location2 = [result1]+[round(values[1],6)]+[round(values[2],6)]
                        print(f"경도와 위도를 반환합니다 : {location2}")
                        firstLocation[:] = location2
                        prevLoc = location2

        except:
            try:
                firstLocation[:] = prevLoc

            except:   
                pass
            time.sleep(0.1)
        

def geocode_address(address, api_key):
    base_url = "https://maps.googleapis.com/maps/api/geocode/json"

    # Geocoding API에 요청 보내기
    response = requests.get(
        base_url,
        params={
            "address": address,
            "key": api_key,
        }
    )

    #print(response.json())
    # 응답 확인
    if response.status_code == 200:
        # JSON 응답 파싱
        data = response.json()

        # 결과 확인
        if data["status"] == "OK":
            #print(data)
            # 첫 번째 결과의 위도와 경도 추출
            location1 = data["results"][0]["geometry"]["location"]
            latitude = location1["lat"]
            longitude = location1["lng"]

            return [True,latitude, longitude]
        else:
            print(f"Geocoding API error: {data['status']} - {data.get('error_message')}")
    else:
        print(f"Geocoding API request failed with status code: {response.status_code}")

    return [False]

def get_street_view_image( loc, ph):
    api_key = "AIzaSyC1Ax9tlpuKI8qC7hh0RVnUNkJYrAOJe10"
    base_url = "https://maps.googleapis.com/maps/api/streetview"
    fov = 90
    #print("fov",fov)
    #print("loc",loc)
    #print("ph",ph)
    params = {
        "size": "960x720",
        "location": f"{loc[0]},{loc[1]}",
        "heading": ph[1],
        "pitch": ph[0],
        "fov": fov,
        "key": api_key
    }
    response = requests.get(base_url, params=params)

    if response.status_code == 200:
        image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
        street_view_image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        return street_view_image
    else:
        print(f"Error {response.status_code}: {response.text}")
        return None



def make_street_view_pickle(location):
    doCycle = True
    mapFolder = "./maps/"
    os.makedirs(mapFolder,exist_ok=True)
    
    while doCycle == True:
        try:
            
            street_path = mapFolder+f"{extract_and_sum_numbers(str(location[1]))}_{extract_and_sum_numbers(str(location[2]))}.pkl"
            #print(get_street_view_image(location[1:3],[0,0]))
            if geocode_address(location[0],api_key) != None:
                street_view_image_list = [] 
                if not os.path.exists(street_path):
                    print("down process is on")
                    svil = []
                    for i in range(36):
                        svil.append(get_street_view_image(location[1:3],[0,10*i]))
                    street_view_image_list.append(location[1:3])
                    street_view_image_list.append(svil)
                
                    with open(street_path,"wb") as file:
                        pickle.dump(street_view_image_list,file)
                    print("download is done")
                time.sleep(0.02)
            else: 
                time.sleep(0.02)
        except:
            #print(traceback.format_exc())
            time.sleep(0.2)

def update_street_view(location, pitchheading,Picture):
    firstTime = True
    pathFolder = "./picture/"
    mapFolder = "./maps/"
    os.makedirs(mapFolder,exist_ok=True)
    os.makedirs(pathFolder,exist_ok=True)
    pictureIndex = 0
    street_view_image_list = [0]

    while firstTime == True:        
        try:
            street_path = mapFolder+f"{extract_and_sum_numbers(str(location[1]))}_{extract_and_sum_numbers(str(location[2]))}.pkl"
            if os.path.exists(street_path):
                
                if street_view_image_list[0] != location[1:3]:
                    with open(street_path,'rb') as file:
                        street_view_image_list = pickle.load(file)
                street_view_image = street_view_image_list[1][int(round(pitchheading[1]/10))]
                        
            else:
                street_view_image = get_street_view_image(location[1:3], pitchheading)  
                
            cv2.imshow('Street View', street_view_image)
            #print("after :",len(street_view_image_list),pitchheading)
            if Picture.value > 0:
                #path
                print("snapshot")
                cv2.imwrite(pathFolder+f"{extract_and_sum_numbers(str(location[1]))}_{extract_and_sum_numbers(str(location[2]))}_{pictureIndex}.jpg",street_view_image)
                pictureIndex += 1
                Picture.value = 0
            #print("after 2 :",len(street_view_image_list),pitchheading)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
        except:
            #print(traceback.format_exc())
            time.sleep(0.01)
    

def webcam_pose_estimation(PitchHeading,sharedPicture):
    cap = cv2.VideoCapture(0)
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)  # Initialize once
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands
    phLocal = [0,0]
    HeadDelay = 0
    steps = 0
    avghori = 0
    avgvert = 0
    photo_Delay = 0
    pathSubFolder = "./picture/webcam/"
    os.makedirs(pathSubFolder,exist_ok=True)
    listsub = os.listdir(pathSubFolder)
    mp_selfie_segmentation = mp.solutions.selfie_segmentation
    selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)
    
    with mp_hands.Hands(min_detection_confidence=0.6, min_tracking_confidence=0.6) as hands:
        last_capture_time = time.time()  # 마지막 촬영 시간 초기화
        capture_interval = 6  # 촬영 간격 (초)
        

        while cap.isOpened():
            HeadDelay += 1
            try:
                ret, frame = cap.read()

                if not ret:
                    continue
                for_picture_frame = cv2.cvtColor(cv2.flip(frame,1),cv2.COLOR_BGR2RGB)
                flipped_frame = cv2.flip(frame, 1)
                # BGR을 RGB로 변환
                rgb_frame = cv2.cvtColor(flipped_frame, cv2.COLOR_BGR2RGB)

                
                results = pose.process(rgb_frame)
                rpl = results.pose_landmarks
                if rpl:
                    poseLandmarks = rpl.landmark
                    # 어깨(landmark 12, 11)
                    left_shoulder = poseLandmarks[12]
                    right_shoulder = poseLandmarks[11]
                    
                    left_eye = poseLandmarks[2]
                    right_eye = poseLandmarks[5]


                    shoulder_distance1 = cv2.norm((left_shoulder.x,left_shoulder.y,left_shoulder.z),(right_shoulder.x,right_shoulder.y,right_shoulder.z))
                    left_thumb1 = poseLandmarks[21]
                    right_index1 = poseLandmarks[20]
                    finger_distance1 = cv2.norm((left_thumb1.x,left_thumb1.y,left_thumb1.z),(right_index1.x,right_index1.y,right_index1.z))
                    finger_distance1/=shoulder_distance1
                
                    steps += 1
                    
                    hna = head_nod_algorithm(left_eye,right_eye,poseLandmarks,steps,avghori,avgvert)                    
                    cv2.putText(flipped_frame, f"LR  diff : {hna[2]:.2f}   {hna[4]}", (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                    if steps < 300:
                        cv2.putText(flipped_frame, f"In calibration : {steps:d}/300", (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                        avghori = avghori*(steps-1)/steps +hna[2]/steps
                        avgvert = avgvert*(steps-1)/steps +hna[3]/steps
                    elif steps>= 300 and steps <420:
                        cv2.putText(flipped_frame, f"Calibration is done !!", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                        cv2.putText(flipped_frame, f"x axis : {avghori:.2f}",(50, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    if steps % 100 == 0 and steps < 420:
                        print("hori avg :",avghori)
                        print("vert avg :",avgvert)

                    ph = [int(10*hna[1]),int(10*hna[0])]
                    if HeadDelay > 0:
                        for i2,p2 in enumerate(ph):
                            phLocal[i2] += p2
                            if i2 == 1:
                                phLocal[i2] = phLocal[i2] % 360
                            else:
                                if phLocal[i2] >= 60:
                                    phLocal[i2] = 60
                                elif phLocal[i2] <= -60:
                                    phLocal[i2] = -60
                        HeadDelay =0
                            
                        
                    PitchHeading[:] = phLocal
                    
                        
                # 손 감지 수행
                resultsHand = hands.process(rgb_frame)
                # 손 키포인트를 그리기 위한 코드
                if resultsHand.multi_hand_landmarks:
                    for landmarks in resultsHand.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(flipped_frame, landmarks, mp_hands.HAND_CONNECTIONS)

                        # 검지와 엄지 각도 계산
                        indexfinger_tip = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                        thumb_tip = landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                        indexMCP = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
                        wrist = landmarks.landmark[mp_hands.HandLandmark.WRIST]
                        dist_indexTIP_thumbTIP = distance_with_cv2(indexfinger_tip,thumb_tip)
                        dist_indeMCP_wrist = distance_with_cv2(indexMCP,wrist)
                        
                        #print(distance2)
                        if dist_indexTIP_thumbTIP/dist_indeMCP_wrist > 1.5:  # 이 값은 실험을 통해 조절할 수 있습니다.
                            photo_Delay+=1
                            if photo_Delay >30 and dist_indexTIP_thumbTIP/dist_indeMCP_wrist > 1.5:
                                photo_Delay = 0
                                current_time = time.time()
                                if current_time - last_capture_time >= capture_interval:
                                    
                                    sharedPicture.value += 1
                                    cv2.imwrite('webcam.jpg', for_picture_frame)
                                    play_mp3(mp3_file)
                                    # 화면 어둡게 만들기 (가중치 조절 가능)
                                    dark_frame = np.zeros_like(flipped_frame)
                                    alpha = 0.1
                                    cv2.addWeighted(flipped_frame, alpha, dark_frame, 1 - alpha, 0, flipped_frame)
                                    # 내가 원하는 이미지와 함께 촬영
                                    
                                    last_capture_time = current_time
                                
                        #print(Picture)
                
              
                cv2.imshow('Webcam', flipped_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            except:
                time.sleep(0.5)
                print(traceback.format_exc())

        cap.release()
        cv2.destroyAllWindows()



# In the main function
if __name__ == '__main__':
    with Manager() as manager:
        PitchHeading= manager.list()
        Location = manager.list()
        sharedPicture = manager.Value('i', 0)
        calibStart = manager.Value('i',0)
        
        pose_process = Process(target=webcam_pose_estimation, args=(PitchHeading,sharedPicture))
        sound_process = Process(target=speech_recognator,args=(Location,True))
        make_pickle_process = Process(target=make_street_view_pickle,args=(Location,))
        street_view_process = Process(target=update_street_view, args=(Location,PitchHeading,sharedPicture))

        
        pose_process.start()
        sound_process.start()
        make_pickle_process.start()
        street_view_process.start()

        pose_process.join()
        sound_process.join()
        make_pickle_process.join()
        street_view_process.join()
        