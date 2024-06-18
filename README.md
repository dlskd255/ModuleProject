
코드 실행을 위한 팁
----

pip install -r requirements.txt : 설치 명령어. python 3.8 environment 먼저 만들고 그 env를 activate 한다음에 그 안에서 타이핑하면됩니다.
** for /f %%i in (requirements.txt) do pip install %%i || echo Failed to install %%i ** : 도중에 에러있을경우 이렇게 실행.  


## 지의 버전 : visual studio 터미널에서 하는 방법.
- (mp) C:\Users\bluecom010>conda deactivate : 이거 해주면 base로 나가게 되고 
- (base) C:\Users\bluecom010>conda create -n python38 python=3.8 : 이거 해주면 python38 안에 python 3.8깔아짐 
- (base) C:\Users\bluecom010>conda activate python38 : 이거 해주면
- (python38) C:\Users\bluecom010> 여기로 들어가지게 됨 
- shift_ctrl+p해서 인터럽트를 python38(python3.8)로 바꿔주고 안깔린 모듈들 pip install로 깔아주면 됨 



pip list --format=freeze > requirements.txt : 가상환경 requirements.txt로 저장
----
