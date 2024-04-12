from flask import Flask, render_template, request, Response,send_file
import base64
import numpy as np
import cv2
from newFFmpegTest import predict,YOLO
import ffmpeg
from werkzeug.utils import secure_filename
import subprocess as sp
import os

model = YOLO('pts/best-ball-rim-4.engine')
from collections import deque
i=0
preBallStack=[]
rim_t_ls=[]
# out=None
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 编解码器
# 在 Flask 应用初始化时创建一个固定长度的队列
MAX_QUEUE_LENGTH = 3  # 假设队列长度为5
frame_queue = deque(maxlen=MAX_QUEUE_LENGTH)
all_frame=[]
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

    # 将 base64 编码的图像数据解码成图像
    decoded_data = base64.b64decode(image_data.split(',')[1])
    np_data = np.frombuffer(decoded_data, np.uint8)
    frame = cv2.imdecode(np_data, cv2.IMREAD_COLOR)
    frame = cv2.flip(frame, 1)

    # 将处理后的图像加入队列
    frame_queue.append(frame)
    newest_frame = frame_queue[-1]


    global i,preBallStack,rim_t_ls,out
    # 仅处理队列中的最新帧
    if jumpORnot:
        newest_frame,preBallStack,rim_t_ls=predict(model,newest_frame,preBallStack)
    all_frame.append(newest_frame)

    _, buffer = cv2.imencode('.jpg', newest_frame)
    processed_data = base64.b64encode(buffer).decode('utf-8')

    # 返回处理后的图像数据
    return Response(response=processed_data, status=200, mimetype='text/plain')


@app.route('/upload', methods=['POST'])
def upload():

    if os.path.exists('output.mp4'):
        os.remove('output.mp4')
        
    # Compile frames into a video
    out = cv2.VideoWriter('temp.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (640, 480))

    for frame in all_frame:
        out.write(frame)

    out.release()
    all_frame.clear()

    if 'audio' not in request.files:
        return 'No file part', 400
    audio_file = request.files['audio']
    audio_file.save('temp.wav')
    # 使用FFmpeg将音频转换成MP3格式
    # sp.run(['ffmpeg', '-i', 'temp.wav', 'output.mp3'])
    # 执行 FFmpeg 命令
    sp.run(['ffmpeg', '-i', 'temp.mp4', '-i', 'temp.wav', '-c:v', 'copy', '-c:a', 'aac', 'output.mp4'])

    # 检查命令是否成功执行

        # 删除临时音频文件和视频文件
    
    os.remove('temp.wav')
    os.remove('temp.mp4')

    return send_file('output.mp4', as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)  # 在调试模式下运行 Flask 应用
