from flask import Flask, render_template, request, Response,send_file,jsonify,redirect,url_for,json
import base64
import numpy as np
import cv2
from newFFmpeg import predict,YOLO,yolo_process,judge_shoot,manage_shoot_score
import ffmpeg
from werkzeug.utils import secure_filename
import subprocess as sp
import os
from PIL import Image
from io import BytesIO
from randomtest import generate_random_timestamps

model = YOLO('pts/ball-rim-pose.engine')
from collections import deque
allDataList=dict()
allShootingInfo=dict()
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 编解码器
# 在 Flask 应用初始化时创建一个固定长度的队列
MAX_QUEUE_LENGTH = 3  # 假设队列长度为5
frame_queue = deque(maxlen=MAX_QUEUE_LENGTH)
all_frame=[]
shootingInfo=[]
# 读取错误图片并转换为OpenCV格式
error_image = cv2.imread('test.png')

# 将错误图片转换为 Base64 编码
_, buffer = cv2.imencode('.png', error_image)
error_image_base64 = base64.b64encode(buffer).decode('utf-8')


app = Flask(__name__)
@app.route('/')
def index():
    # model.predict("test.png")
    return render_template('index.html')

@app.route('/shootingPractice')
def shootingPractice():
    model.predict("test.png")
    return render_template('shooting.html')

@app.route('/dribblePractice')
def dribblePractice():
    model.predict("test.png")
    return render_template('drible.html')

@app.route('/dribbleCountPractice')
def dribbleCountPractice():
    model.predict("test.png")
    return render_template('counter.html')

@app.route('/dribbleReactPractice')
def dribbleReactPractice():
    model.predict("test.png")
    return render_template('handsReact.html')

@app.route('/dataView')
def dataView():
    # model.predict("test.png")
    return render_template('view.html')

@app.route('/afterShooting')
def afterShooting():
    return render_template('afterShooting.html')


@app.route('/getShootingInfo',methods=['POST'])
def getShootingInfo():
    uuid=request.json.get('uuid')
    shootData=allShootingInfo.get(uuid,{
                                        'trainStartTime':0,
                                        'trainEndTime':0,
                                        'Shoots':[]
                                    })
    data={
        'trainStartTime':shootData['trainStartTime'],
        'trainEndTime':shootData['trainEndTime'],
        'shootingInfo' : shootData['Shoots']
    }
    # print(allShootingInfo)
    # print('data',data)
    return jsonify(data)


@app.route('/getAllShootingInfo',methods=['POST'])
def getAllShootingInfo():
    userID=request.json.get('userID')
    start = request.json.get('start')
    end = request.json.get('end')
    count=int(request.json.get('count'))

    user_data = generate_random_timestamps(start, end,count)
    return jsonify(user_data)

@app.route('/process_frame', methods=['POST'])
def process_frame():
    # 从请求中获取图像数据
    image_data = request.json.get('image_data')
    jumpORnot=request.json.get('jumpORnot')
    flip=(request.json.get('imgFlip')=='true')
    uuid=request.json.get('uuid')


    try:
        decoded_data = base64.b64decode(image_data.split(',')[1])
        np_data = np.frombuffer(decoded_data, np.uint8)
        frame = cv2.imdecode(np_data, cv2.IMREAD_COLOR)
    except Exception as e:
        print("Error processing frame---------------------------------------------:", e)
        # 返回错误图片的 Base64 编码数据
        return Response(response=error_image_base64, status=500, mimetype='text/plain')
    if uuid not in allDataList.keys():
        allDataList[uuid] = allDataList.setdefault(uuid, {
            'preBallStack':[],
            'shooting_balls_line':[],
            'ball_cxy':[],
            'ball_thickness':[],
            'jump':0,
            'wait_frame':0,
            'frame_idx':0,
            'shooting_count':0,
            'score_count':0,
            'ball_state':'normal',
            'transparent_layer':np.zeros_like(frame, dtype=np.uint8),
            'score_layer':np.zeros_like(frame, dtype=np.uint8),
            'frame_queue':deque(maxlen=MAX_QUEUE_LENGTH),
            'all_frame':[],
            'b_ball':[],
            'keypoints':[],
            'player':[]
        })
    data=allDataList[uuid]

    if flip:
        frame = cv2.flip(frame, 1)

    # 将处理后的图像加入队列
    data['frame_queue'].append(frame)
    newest_frame = data['frame_queue'][-1]
    newest_frame=yolo_process(model,data,newest_frame,jumpORnot,conf=0.75)

    data['all_frame'].append(newest_frame)
    _, buffer = cv2.imencode('.jpg', newest_frame)
    processed_data = base64.b64encode(buffer).decode('utf-8')
    extra_data = {
    'ball': data['b_ball'],
    'coordinates': data['keypoints'],
    'player':data['player'],
    'ball_state':data['ball_state'],
    'shooting_count':data['shooting_count'],
    'score_count':data['score_count']
    }
    response_data={
        'image_data': processed_data,  # 之前处理的图像数据
        'extra_data': extra_data  # 添加的额外信息
    }
    # 返回处理后的图像数据
    return jsonify(response_data)




@app.route('/upload', methods=['POST'])
def upload():
    width=int(request.form.get('imgWidth'))
    height=int(request.form.get('imgHeight'))
    isFromCamera=request.form.get('isFromCamera')
    uuid=request.form.get('uuid')
    fps=int(request.form.get('fps'))
    shooting_info_data = request.form.getlist('shootingInfo')
    allShootInfo = [json.loads(item) for item in shooting_info_data] 
    trainStartTime=request.form.get('trainStartTime')
    trainEndTime=request.form.get('trainEndTime')
    allShootInfo={
        'trainStartTime':trainStartTime,
        'trainEndTime':trainEndTime,
        'Shoots':allShootInfo
    }

    # print("upload",50*uuid)
    if os.path.exists('output.mp4'):
        os.remove('output.mp4')
    if uuid in allDataList:
        data=allDataList[uuid]
        # print('uuid',uuid)
        allShootingInfo[uuid]=allShootInfo
        # print('添加后',allShootingInfo)
        # Compile frames into a video
        out = cv2.VideoWriter('temp.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        for frame in data['all_frame']:
            out.write(frame)
        out.release()
        # print("帧数为：",len(data['all_frame']))

        if isFromCamera=='true' and 'audio' in request.files:
            # print("aaaaa")
            audio_file = request.files['audio']
            audio_file.save('temp.wav')
            sp.run(['ffmpeg', '-i', 'temp.mp4', '-i', 'temp.wav', '-c:v', 'copy', '-c:a', 'aac', 'output.mp4'])
            os.remove('temp.wav')
        elif isFromCamera=='false':
            sp.run(['ffmpeg', '-i', 'temp.mp4', '-i', 'audiotemp.mp4', '-map', '0:v','-map', '1:a','-c:v', 'copy', '-c:a', 'copy', 'output.mp4'])
            # os.remove('audiotemp.mp4')
        else:
            return 'No file part', 400
        del allDataList[uuid]
        
        # os.remove('temp.mp4')
        # return send_file('output.mp4', as_attachment=True)
        # return redirect(url_for('afterShooting'))
        # print(allShootInfo)
        return 'ok', 200

    else:
        print(uuid)
        return 'No UUID', 400
    


@app.route('/upload_mp4', methods=['POST'])
def uploadmp4():
    if 'mp4File' not in request.files:
        return 'No file part', 400
    audio_file = request.files['mp4File']
    audio_file.save('audiotemp.mp4')
    return 'File uploaded successfully.', 200  # 返回成功响应

@app.route('/save_image', methods=['POST'])
def save_image():
    try:
        # 获取前端发送的数据
        data = request.json
        image_data = data['image_data']

        # 解码base64编码的图像数据
        image_bytes = base64.b64decode(image_data.split(',')[1])
        
        # 将字节流转换为图像
        image = Image.open(BytesIO(image_bytes))

        # 保存图像到文件
        image_path = 'static/saved_image.jpg'  # 图像保存路径
        image.save(image_path)

        return jsonify({'message': 'Image saved successfully', 'image_path': image_path}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/calculate_perspective_matrix', methods=['POST'])
def calculate_perspective_matrix():
    # 获取 POST 请求中的数据
    data = request.get_json()

    # 解析数据并转换为 NumPy 数组
    src_points = np.array([[point['x'], point['y']] for point in data['src_points']], dtype=np.float32)
    dst_points = np.array([[point['x'], point['y']] for point in data['dst_points']], dtype=np.float32)
    court_img='saved_image.jpg'
    input_image = cv2.imread('static/'+court_img)
    # 定义输出图像的大小
    output_size = (input_image.shape[1], input_image.shape[0])  # 使用输入图像的大小
    try:
        # 计算透视变换矩阵
        perspective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)

        inverse_perspective_matrix=cv2.getPerspectiveTransform(dst_points,src_points)
        inverse_perspective_matrix_list = inverse_perspective_matrix.tolist()

        output_image = cv2.warpPerspective(input_image, perspective_matrix, output_size,borderMode=cv2.BORDER_CONSTANT,borderValue=(0, 0, 0))


        retval, buffer = cv2.imencode('.jpg', output_image)
        output_image_base64 = base64.b64encode(buffer).decode('utf-8')

        # 将图像的 base64 编码字符串添加到 JSON 对象中
        response_data = {
            'perspective_matrix': inverse_perspective_matrix_list,
            'output_image_base64': output_image_base64
        }

        # 将 JSON 对象发送到前端
        return jsonify(response_data), 200
    except Exception as e:
        print(e)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
     app.run(debug=True)  # 在调试模式下运行 Flask 应用
