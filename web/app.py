from flask import Flask, render_template, request, Response
import base64
import numpy as np
import cv2

model = YOLO('pts/best-ball-rim-4.engine')
preBallStack=[]
rim_t_ls=[]
app = Flask(__name__)
@app.route('/')
def index():
    return render_template('camera.html')

@app.route('/process_frame', methods=['POST'])
def process_frame():
    # 从请求中获取图像数据
    image_data = request.json.get('image_data')

    # 将 base64 编码的图像数据解码成图像
    decoded_data = base64.b64decode(image_data.split(',')[1])
    np_data = np.frombuffer(decoded_data, np.uint8)
    frame = cv2.imdecode(np_data, cv2.IMREAD_COLOR)

    # 在这里进行对图像的处理
    # 这里只是简单地将图像水平翻转，您可以根据需要进行更复杂的处理
    frame,preBallStack,rim_t_ls=predict(model,frame,preBallStack)
    processed_frame = cv2.flip(frame, 1)

    # 将处理后的图像编码为 base64 字符串
    _, buffer = cv2.imencode('.jpg', processed_frame)
    processed_data = base64.b64encode(buffer).decode('utf-8')

    # 返回处理后的图像数据
    return Response(response=processed_data, status=200, mimetype='text/plain')

if __name__ == '__main__':
    app.run(debug=True)  # 在调试模式下运行 Flask 应用
