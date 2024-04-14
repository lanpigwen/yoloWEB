from flask import Flask, render_template, request, Response,send_file
import base64
import numpy as np
import cv2
from newFFmpegTest import predict,YOLO,judge_shoot
import ffmpeg
from werkzeug.utils import secure_filename
import subprocess as sp
import os

model = YOLO('pts/best-ball-rim-4.engine')
from collections import deque
i=0
preBallStack=[]
rim_t_ls=[]
wait_frame=0
# out=None
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 编解码器
# 在 Flask 应用初始化时创建一个固定长度的队列
MAX_QUEUE_LENGTH = 3  # 假设队列长度为5
frame_queue = deque(maxlen=MAX_QUEUE_LENGTH)
all_frame=[]

# 读取错误图片并转换为OpenCV格式
error_image = cv2.imread('test.png')

# 将错误图片转换为 Base64 编码
_, buffer = cv2.imencode('.png', error_image)
error_image_base64 = base64.b64encode(buffer).decode('utf-8')


app = Flask(__name__)
@app.route('/')
def index():
    model.predict("test.png")
    return render_template('index.html')

@app.route('/process_frame', methods=['POST'])
def process_frame():
    # 从请求中获取图像数据
    image_data = request.json.get('image_data')
    jumpORnot=request.json.get('jumpORnot')
    flip=(request.json.get('imgFlip')=='true')
    try:
        decoded_data = base64.b64decode(image_data.split(',')[1])
        np_data = np.frombuffer(decoded_data, np.uint8)
        frame = cv2.imdecode(np_data, cv2.IMREAD_COLOR)
    except Exception as e:
        print("Error processing frame:", e)
        # 返回错误图片的 Base64 编码数据
        return Response(response=error_image_base64, status=500, mimetype='text/plain')
    
    if flip:
        frame = cv2.flip(frame, 1)

    # 将处理后的图像加入队列
    frame_queue.append(frame)
    newest_frame = frame_queue[-1]


    global i,preBallStack,rim_t_ls,wait_frame
    # 仅处理队列中的最新帧
    if jumpORnot:
        newest_frame,preBallStack,rim_t_ls=predict(model,newest_frame,preBallStack)
    
    score,newest_frame,wait_frame=judge_shoot(preBallStack,newest_frame,preBallStack,rim_t_ls,wait_frame)

    all_frame.append(newest_frame)
    _, buffer = cv2.imencode('.jpg', newest_frame)
    processed_data = base64.b64encode(buffer).decode('utf-8')

    # 返回处理后的图像数据
    return Response(response=processed_data, status=200, mimetype='text/plain')



@app.route('/upload', methods=['POST'])
def upload():
    width=int(request.form.get('imgWidth'))
    height=int(request.form.get('imgHeight'))
    print(width,height)
    isFromCamera=request.form.get('isFromCamera')
    if os.path.exists('output.mp4'):
        os.remove('output.mp4')
        
    # Compile frames into a video
    out = cv2.VideoWriter('temp.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))

    for frame in all_frame:
        out.write(frame)

    out.release()
    
    print("帧数为：",len(all_frame))
    all_frame.clear()

    if isFromCamera=='true' and 'audio' in request.files:
        audio_file = request.files['audio']
        audio_file.save('temp.wav')
        sp.run(['ffmpeg', '-i', 'temp.mp4', '-i', 'temp.wav', '-c:v', 'copy', '-c:a', 'aac', 'output.mp4'])
        os.remove('temp.wav')
    elif isFromCamera=='false':
        sp.run(['ffmpeg', '-i', 'temp.mp4', '-i', 'audiotemp.mp4', '-map', '0:v','-map', '1:a','-c:v', 'copy', '-c:a', 'copy', 'output.mp4'])
        # os.remove('audiotemp.mp4')
    else:
        return 'No file part', 400
    # os.remove('temp.mp4')
    return send_file('output.mp4', as_attachment=True)


@app.route('/upload_mp4', methods=['POST'])
def uploadmp4():
    if 'mp4File' not in request.files:
        return 'No file part', 400
    audio_file = request.files['mp4File']
    audio_file.save('audiotemp.mp4')
    return 'File uploaded successfully.', 200  # 返回成功响应

if __name__ == '__main__':
    app.run(debug=True)  # 在调试模式下运行 Flask 应用
