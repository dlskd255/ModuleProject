import subprocess
import sys

# requirements.txt 파일을 열고 각 라인을 읽음
with open('requirements.txt', 'r') as file:
    for line in file:
        package = line.strip()
        try:
            # pip install 명령을 실행
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
        except subprocess.CalledProcessError:
            # 설치 실패 시 메시지 출력
            print(f"Failed to install {package}")

# 스크립트 완료 메시지
print("Installation process completed")