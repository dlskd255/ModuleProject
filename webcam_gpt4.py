api_key = "AIzaSyC1Ax9tlpuKI8qC7hh0RVnUNkJYrAOJe10"

import cv2
import mediapipe as mp
import requests
import numpy as np
from multiprocessing import Process, Manager
import multiprocessing
import math
import time

import os
import sys
import urllib.request
import json
import speech_recognition as sr
from copy import deepcopy

import pygame

def play_mp3(mp3_file):
    pygame.mixer.init()
    pygame.mixer.music.load(mp3_file)
    pygame.mixer.music.play()

mp3_file = "./son.mp3"
"""mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose"""


def calculate_angle(a, b, c):
    """세 점 간의 각도를 계산하는 함수"""
    radians = math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0])
    angle = abs(math.degrees(radians))
    return angle

# Pose 모델 로드
#pose = mp_pose.Pose()

def head_nod_algorithm(left_shoulder_coords, right_shoulder_coords, head_coords):
    """고개를 좌우로 흔드는 경우를 판단하는 알고리즘"""
    shoulder_center_x = (left_shoulder_coords[0] + right_shoulder_coords[0]) // 2

    # 머리의 좌우 이동량 계산
    head_horizontal_movement = head_coords[0] - shoulder_center_x
    print(head_horizontal_movement)
    if abs(head_horizontal_movement) > 50:  # 예제의 임계값
        return True
    else:
        return False


def speech_recognator(firstLocation,condition):
    api_key = "AIzaSyC1Ax9tlpuKI8qC7hh0RVnUNkJYrAOJe10"
    # 1) 음성 인식기
    condition = True
    while condition == True:
        try:
            time.sleep(1)
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
                print(f"입력 받았습니다. {result1}로 이동합니다.")
                values = geocode_address(result1,api_key)
                if values[0] == True:
                    location2 = values[1:]
                    print(f"경도와 위도를 반환합니다 : {location2}")
                    firstLocation[:] = location2
                    prevLoc = location2
                    #print(firstLocation)
        except:
            try:
                firstLocation[:] = prevLoc
                #print(prevLoc)
            except:   
                pass
        time.sleep(1)

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

def update_street_view(location, pitchheading,Picture):
    firstTime = True
    pathFolder = "./picture/"
    os.makedirs(pathFolder,exist_ok=True)
    pictureIndex = 0
    street_view_image_prev = None
    
    
    while firstTime == True:        
        try:            
            #print(location,pitchheading)
            street_view_image = get_street_view_image(location, pitchheading)#(location, pitchheading)
            if street_view_image is not None:
                street_view_image_prev = street_view_image
                print("good?")

            #print(street_view_image[5])
            if street_view_image is not None:
                cv2.imshow('Street View', street_view_image) 
                
            if street_view_image_prev is not None:
                print("recived :",Picture)
                if Picture.value > 0:
                    print("go")
                    cv2.imwrite(pathFolder+f"{pictureIndex}.jpg",street_view_image_prev)
                    pictureIndex += 1
                    Picture.value = 0
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        except:
            time.sleep(0.5)
    

def webcam_pose_estimation(PitchHeading,sharedPicture):
    cap = cv2.VideoCapture(0)
    
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()  # Initialize once
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands
    with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        last_capture_time = time.time()  # 마지막 촬영 시간 초기화
        capture_interval = 4  # 촬영 간격 (초)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                continue
            # BGR을 RGB로 변환
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 손 감지 수행
            resultsHand = hands.process(rgb_frame)
            # 손 키포인트를 그리기 위한 코드
            if resultsHand.multi_hand_landmarks:
                for landmarks in resultsHand.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

                    # 검지와 엄지 각도 계산
                    index_finger = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    thumb = landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                    indexMCP = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
                    distance = cv2.norm(
                        (index_finger.x, index_finger.y,index_finger.z),
                        (thumb.x, thumb.y,thumb.z)
                    )
                    distance1 =cv2.norm(
                        (index_finger.x, index_finger.y,index_finger.z),
                        (indexMCP.x,indexMCP.y,indexMCP.z)
                    )
                    
                    if distance/distance1 > 1.4:  # 이 값은 실험을 통해 조절할 수 있습니다.
                        current_time = time.time()
                        
                        if current_time - last_capture_time >= capture_interval:
                            sharedPicture.value = 1
                            play_mp3(mp3_file)
                            # 내가 원하는 이미지와 함께 촬영
                            #cv2.imwrite('captured_desired_image.jpg', desired_image)
                            last_capture_time = current_time
                            
                            

                    #print(Picture)
                            
                    

            results = pose.process(frame)

            if results.pose_landmarks:
                # 어깨(landmark 12, 11)와 머리(landmark 30)의 좌표 획득
                left_shoulder = results.pose_landmarks.landmark[12]
                right_shoulder = results.pose_landmarks.landmark[11]
                head = results.pose_landmarks.landmark[30]

                # 각 랜드마크의 좌표를 화면 좌표로 변환
                left_shoulder_coords = (int(left_shoulder.x * frame.shape[1]), int(left_shoulder.y * frame.shape[0]))
                right_shoulder_coords = (int(right_shoulder.x * frame.shape[1]), int(right_shoulder.y * frame.shape[0]))
                head_coords = (int(head.x * frame.shape[1]), int(head.y * frame.shape[0]))

                # 어깨 사이의 각도 계산
                shoulder_angle = calculate_angle(left_shoulder_coords, head_coords, right_shoulder_coords)

                # 각도에 따라 고개의 상태 표시
                if shoulder_angle > 20:  # 예제의 임계값
                    cv2.putText(frame, "Head Down", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                else:
                    cv2.putText(frame, "Head Up", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # 각도를 화면에 표시
                cv2.putText(frame, f"Angle: {shoulder_angle:.2f} degrees", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                ph = [0,0]
                PitchHeading[:] = ph
                
            

            cv2.imshow('Webcam', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()



# In the main function
if __name__ == '__main__':
    with Manager() as manager:
        PitchHeading= manager.list()
        Location = manager.list()
        sharedPicture = manager.Value('i', 0)
        
        pose_process = Process(target=webcam_pose_estimation, args=(PitchHeading,sharedPicture))
        sound_process = Process(target=speech_recognator,args=(Location,True))
        street_view_process = Process(target=update_street_view, args=(Location,PitchHeading,sharedPicture))
        
        pose_process.start()
        sound_process.start()
        street_view_process.start()

        pose_process.join()
        sound_process.join()
        street_view_process.join()
        