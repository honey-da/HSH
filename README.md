<img width="2083" height="482" alt="image" src="https://github.com/user-attachments/assets/22e9c6f8-d34d-4ecd-9caa-d1c0f0515a41" /># HSH
##YOLO 

cuda 12.2.0:
widows, x86_64 11 exe(local)

파이썬 아나콘다로 conda create -n wake python=3.10 -y
conda activate wake
cd C:\Users\jyh10\source\repos\wake_alarm

pytorch 2.5.1버전

numpy 1.26.4
pip install numpy==1.26.4

pip install onnxruntime-gpu
pip install opencv-python numpy

conda install -c conda-forge cudnn=9.1


##web-UI

현재까지 진행사항 : HTML자체에서 구현가능한 버튼들은 구현 완료.. <예시. 알람 설정, 화면 세팅 등등>
해야할것 : IOT 연결하는 통신 부분 코딩 필요.

실행하는법 : 그냥 HTML 파일 클릭 하면 UI 열림
수정하는법 : 메모장 또는 VS_CODE로 .html 로 저장하면 수정 끝
향후추가됐으면 좋겠는 부분 : 지금은 웹으로 실행. 향후에는 앱으로 만들어서 실행하면은 좋을듯?!


(yolo폴더하고 메인폴더 source->repos에 저장해둠)
