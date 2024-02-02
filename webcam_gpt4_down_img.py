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

def speech_recognator(firstLocation,condition,paintStart,voicePicture):
    api_key = "AIzaSyC1Ax9tlpuKI8qC7hh0RVnUNkJYrAOJe10"
    # 1) 음성 인식기
    condition = True
    while condition == True:
        try:
            print(voicePicture.value)
            listOrder = ['보정','그림 시작','그림 종료','지워 줘','사진']
            alter = ['드림 시작','드림 종료']
            print("record process is on | 명령어 목록 : 보정, 그림 시작, 그림 종료, 지워 줘, 사진 ")
            r = sr.Recognizer() # 음성 인식을 위한 객체 생성            
            mic = sr.Microphone(device_index = 1)
            # 마이크 객체 선언, 인덱스는 각 노트북의 마이크 번호를 의미합니다. 만약 인식이 안되시면 바꿔보시면서 테스트 해보시면 될 듯 합니다.
            with mic as source:
                audio = r.listen(source,timeout=5, phrase_time_limit = 5) # 마이크에서 5초 동안 음성을 듣고 audio 변수에 저장합니다.

            result = r.recognize_google(audio, language = "ko-KR",show_all=True) # 인식한 음성을 텍스트로 변환

            confidence = float(result['alternative'][0]['confidence'])
            if confidence > 0.85:
                result1 = result['alternative'][0]['transcript']
                if not result1 in listOrder :
                    print(f"입력 받았습니다. {result1}(으)로 이동합니다.")
                    values = geocode_address(result1,api_key)
                    if values[0] == True:
                        location2 = [result1]+[round(values[1],6)]+[round(values[2],6)]
                        print(f"경도와 위도를 반환합니다 : {location2}")
                        firstLocation[:] = location2
                        prevLoc = location2
                elif result1 == listOrder[0]:
                    pass
                elif result1 == listOrder[1] or result1 == alter[0]:
                    print(f"입력 받았습니다. 그림을 시작합니다")
                    paintStart.value = 1
                elif result1 == listOrder[2] or result1 == alter[1]:
                    print(f"입력 받았습니다. 그림을 종료합니다")
                    paintStart.value = 0
                elif result1 == listOrder[3]:
                    print(f"입력 받았습니다. 그림을 지웁니다")
                    paintStart.value = -1
                elif result1 == listOrder[4]:
                    print(f"사진 촬영 시작합니다")
                    voicePicture.value = 1
                    

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
    PictureFolder = "./picture/pic/"
    SubFolder = "./picture/bgSub/"
    mapFolder = "./maps/"
    pathSubFolder = "./picture/webcam/"
    os.makedirs(mapFolder,exist_ok=True)
    os.makedirs(SubFolder,exist_ok=True)
    
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
                Name = f"{extract_and_sum_numbers(str(location[1]))}_{extract_and_sum_numbers(str(location[2]))}"
                #path
                print("snapshot")
                cv2.imwrite(SubFolder+Name+f"_{pictureIndex}.png",street_view_image)
                
                picFoldPath = PictureFolder+Name+"/"
                os.makedirs(picFoldPath,exist_ok=True)
                ldp = len(os.listdir(picFoldPath))+1
                

                # 전경 이미지(사람) 및 배경 이미지 로드
                foreground = cv2.imread(pathSubFolder+'/webcam.png', cv2.IMREAD_UNCHANGED)  # 알파 채널 포함하여 로드
                background = cv2.imread(SubFolder+Name+f"_{pictureIndex}.png")

                # 전경 이미지의 해상도를 50%로 줄임
                scale_percent = 50  # 전경 이미지의 해상도를 50%로 줄이기 위한 비율
                width = int(foreground.shape[1] * scale_percent / 100)
                height = int(foreground.shape[0] * scale_percent / 100)
                dim = (width, height)

                # 리사이즈된 전경 이미지
                resized_foreground = cv2.resize(foreground, dim, interpolation=cv2.INTER_AREA)

                # 배경 및 전경 이미지의 크기 확인
                bg_height, bg_width = background.shape[:2]
                fg_height, fg_width = resized_foreground.shape[:2]

                # 전경 이미지를 배경 이미지 맨 아래쪽에 배치하기 위한 시작점 계산
                start_y = bg_height - fg_height
                start_x = (bg_width - fg_width) // 2

                # 전경 이미지의 알파 채널을 사용하여 배경과 전경 합성
                fore_alpha = resized_foreground[:, :, 3] / 255.0
                fore_rgb = resized_foreground[:, :, :3]

                background_rgb = background[start_y:start_y+fg_height, start_x:start_x+fg_width, :3]
                back_alpha = 1.0 - fore_alpha

                # 전경 및 배경 이미지 합성
                composite_rgb = fore_rgb * fore_alpha[:, :, np.newaxis] + background_rgb * back_alpha[:, :, np.newaxis]

                # 합성된 영역을 원본 배경 이미지에 복사
                background[start_y:start_y+fg_height, start_x:start_x+fg_width, :3] = composite_rgb

                newPicPath = picFoldPath+f"/{ldp}.png"
                # 결과 이미지 저장
                cv2.imwrite(newPicPath, background)

                # 결과 확인을 위해 이미지 표시
                cv2.imshow('Composite Image', background)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                pictureIndex += 1
                Picture.value = 0
                
                
            #print("after 2 :",len(street_view_image_list),pitchheading)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
        except:
            #print(traceback.format_exc())
            time.sleep(0.01)
    

def webcam_pose_estimation(PitchHeading,sharedPicture,paintStart,voicePicture):
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
    photo_Switch = False
    pathSubFolder = "./picture/webcam/"
    os.makedirs(pathSubFolder,exist_ok=True)
    canvas = None

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
                #for_picture_frame = cv2.cvtColor(cv2.flip(frame,1),cv2.COLOR_BGR2RGB)
                flipped_frame = cv2.flip(frame, 1)
                # BGR을 RGB로 변환
                rgb_frame = cv2.cvtColor(flipped_frame, cv2.COLOR_BGR2RGB)
                results1 = selfie_segmentation.process(rgb_frame)


                condition = np.stack((results1.segmentation_mask,) * 3, axis=-1) > 0.2
                bg_color = np.ones(rgb_frame.shape, dtype=np.uint8) * 192  # 배경을 회색으로 설정
                processed_frame = np.where(condition, rgb_frame, bg_color)

                # 처리된 프레임을 BGR 형식으로 변환 후 표시
                processed_frame_bgr = cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR)
                
                flipped_frame = processed_frame_bgr
                if canvas is None:
                    # canvas 초기화 (처음 한 번만 실행)
                    canvas = np.zeros_like(frame)

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
                    """x, y = int(indexfinger_tip.x * frame.shape[1]), int(indexfinger_tip.y * frame.shape[0])
                    cv2.circle(canvas, (x, y), 10, (162, 180, 255), -1)"""
                    if paintStart.value>0:

                        curr_x, curr_y = int(indexfinger_tip.x * frame.shape[1]), int(indexfinger_tip.y * frame.shape[0])

                        # 이전 좌표와 현재 좌표 사이에 선 그리기
                        if prev_x != 0 and prev_y != 0:
                            cv2.line(canvas, (prev_x, prev_y), (curr_x, curr_y), (162, 180, 255), 5)

                        prev_x, prev_y = curr_x, curr_y  # 현재 좌표를 이전 좌표로 업데이트
                    elif paintStart.value <0:
                        canvas = np.zeros_like(flipped_frame)
                else:
                    # 손이 감지되지 않으면 이전 좌표 초기화
                    prev_x, prev_y = 0, 0      
                    dist_indexTIP_thumbTIP = 1 
                    dist_indeMCP_wrist = 1
                current_time = time.time()
                if current_time - last_capture_time >= capture_interval:
                    if sharedPicture.value < 1 or voicePicture.value > 0:
                        
                        if dist_indexTIP_thumbTIP/dist_indeMCP_wrist > 1.5 or voicePicture.value > 0 or photo_Switch == True:  # 이 값은 실험을 통해 조절할 수 있습니다.
                            photo_Delay+=1
                            photo_Switch = True

                            cv2.putText(flipped_frame, f"take a picture in : {50-photo_Delay}",(200, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                            
                            if photo_Delay > 50:
                                if dist_indexTIP_thumbTIP/dist_indeMCP_wrist > 1.5 or voicePicture.value or photo_Switch == True> 0:
                                    photo_Switch = False
                                    photo_Delay = 0
                                    
                                    voicePicture.value = 0
                                
                                    alpha_channel = np.where(results1.segmentation_mask > 0.2, 255, 0).astype(np.uint8)

                                    # RGB 프레임으로부터 BGR 프레임 생성
                                    bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
                                    # BGR 프레임과 알파 채널을 결합하여 RGBA 이미지 생성
                                    rgba_frame = cv2.merge((bgr_frame, alpha_channel))
                                    
                                    cv2.imwrite(pathSubFolder+f'webcam.png', rgba_frame)
                                    play_mp3(mp3_file)
                                    # 화면 어둡게 만들기 (가중치 조절 가능)
                                    dark_frame = np.zeros_like(flipped_frame)
                                    alpha2 = 0.1
                                    cv2.addWeighted(flipped_frame, alpha2, dark_frame, 1 - alpha2, 0, flipped_frame)
                                    # 내가 원하는 이미지와 함께 촬영
                                    
                                    last_capture_time = current_time
                                    sharedPicture.value = 1
                              
                
                #print(Picture)
                alpha = 1  # canvas의 투명도
                flipped_frame = cv2.addWeighted(flipped_frame, 1, canvas, alpha, 0)
                
                #cv2.imshow('Processed Frame', processed_frame_bgr)
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
        voicePicture = manager.Value('i',0)
        calibStart = manager.Value('i',0)
        paintStart = manager.Value('i',0)

        pose_process = Process(target=webcam_pose_estimation, args=(PitchHeading,sharedPicture,paintStart,voicePicture))
        sound_process = Process(target=speech_recognator,args=(Location,True,paintStart,voicePicture))
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
        