import torch
from ultralytics import YOLO
import os
import cv2
import time
from joblib import Parallel, delayed
import numpy as np
from numpy.polynomial import Polynomial
from scipy.spatial import distance
import math
from multiprocessing import Process,Queue
import subprocess
from colorama import Fore, init
init()


def b2b_distance(a, b):
    ax, ay = (a[0] + a[2]) / 2, (a[1] + a[3]) / 2
    bx, by = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    aconf, bconf = a[4], b[4]
    o_dst = ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5
    # dst = (2.5 - aconf - bconf) * o_dst
    dst=o_dst
    return dst

def only_show_one_ball(last2ballxy, dataLS, clsNames,all_N=100,rear_N=3):
    outList = []
    ballList = []
    dstRace = [0 for i in range(len(dataLS))]
    if len(dataLS) < 1:
        return dataLS, last2ballxy

    if len(last2ballxy) < 1:
        for d in dataLS:
            x1, y1, x2, y2, conf, label_i = [int(i) for i in d.tolist()]
            if clsNames[label_i] == 'ball':
                ballList.append(d)
            else:
                outList.append(d)
        confLS = [c[4] for c in ballList]
        if len(confLS)>0:
            outList.append(ballList[confLS.index(max(confLS))])

        return outList, outList

    for d_i, d in enumerate(iterable=dataLS):
        x1, y1, x2, y2, conf, label_i = [int(i) for i in d.tolist()]
        if clsNames[label_i] != 'ball':
            outList.append(d)
            continue
        else:
            ballList.append(d)
        for preBall in last2ballxy[::-1][:min(rear_N,len(last2ballxy))]:
            dst = b2b_distance(d.tolist(), preBall)
            dstRace[d_i] += dst
    if len([i for i in dstRace if i>0])>0:
        one_ball_i = dstRace.index(min([i for i in dstRace if i>0]))
        outList.append(dataLS[one_ball_i])
        last2ballxy.append(dataLS[one_ball_i])
    if len(last2ballxy)>all_N:
        last2ballxy.pop(0)
    return outList, last2ballxy

def predict(model,video,size=640,confidence=0.10):
    cap = cv2.VideoCapture(video)
    preBallStack = []
    while cap.isOpened():
        success, frame = cap.read()
        if success:
            results = model(frame,conf=confidence)  # predict on an image
            dataLS = results[0].boxes.data
            findCLS = results[0].boxes.cls
            clsNames = results[0].names
            # 只显示一个球
            # if len(findCLS)>len(set(findCLS.tolist())):
            #     now_one_ball, preBallStack = only_show_one_ball(preBallStack, dataLS, clsNames)
            #     dataLS = now_one_ball

            for d in dataLS:

                x1, y1, x2, y2, conf, label_i = d.tolist()
                x1, y1, x2, y2, conf, label_i=int(x1), int(y1), int(x2), int(y2),round(conf,2), int(label_i)
                label = clsNames[label_i]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 8)  # 绿色，8个像素宽度
                cv2.putText(frame, label+' '+str(conf), (x1, max(2, y1 - 15)), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 0), 8)
            # draw_ball_line(frame,preBallStack,rear_N=10)

            try:
                # try语句写在这不会卡顿
                        # try语句写在这不会卡顿
                # frame=cv2.resize(frame,640)
                # size = 640
                # 获取原始图像宽高。
                height, width = frame.shape[0], frame.shape[1]
                # 等比例缩放尺度。
                scale = height/size
                # 获得相应等比例的图像宽度。
                width_size = int(width/scale)
                # resize
                frame = cv2.resize(frame, (width_size, size))
                cv2.imshow('img', frame)
                pass

            except:
                pass
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break

def draw_ball_line(frame,last2ballxy,rear_N=10):
    ball_xy_tensor=last2ballxy[::-1][:min(rear_N,len(last2ballxy))][::-1]
    if len(ball_xy_tensor)>2:
        ball_xy_list=[]
        for d in ball_xy_tensor:
            x1, y1, x2, y2, conf, label_i = [int(i) for i in d.tolist()]
            ball_xy_list.append([(x1+x2)/2.0,(y1+y2)/2.0])
        pts=np.array(ball_xy_list,dtype=np.int32)
        cv2.polylines(frame,[pts],color=(0,0,255),thickness=8,isClosed=False)

def draw_rim(frame,rim,clsNames):
    for d_i,d in enumerate([i for i in rim if i is not None]):
        x1, y1, x2, y2, conf, label_i = d.tolist()
        x1, y1, x2, y2, conf, label_i=int(x1), int(y1), int(x2), int(y2),round(conf,2), int(label_i)
        label = clsNames[label_i]
        textPUT=f'{label} {conf}'
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 100, 255), 3)  
        cv2.putText(frame, textPUT, (x1, max(2, y1 - 15)), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)
    return frame

def draw_ball(frame,ball,clsNames,tsize1=2,tsize2=2,recsize=3,rec_color=(0,100,255),text_color=(255,255,255),show_rec=True,show_text=True):
    if ball is None:
        return frame
    for d_i,d in enumerate([i for i in ball if i is not None]):
        x1, y1, x2, y2, conf, label_i = d.tolist()
        x1, y1, x2, y2, conf, label_i=int(x1), int(y1), int(x2), int(y2),round(conf,2), int(label_i)
        label = clsNames[label_i]
        textPUT=f'{label} {conf}'
        if show_rec:
            cv2.rectangle(frame, (x1, y1), (x2, y2), rec_color, recsize)  
        if show_text:
            cv2.putText(frame, textPUT, (x1, max(2, y1 - 15)), cv2.FONT_HERSHEY_SIMPLEX, tsize1,text_color, tsize2)
    return frame

def get_cls_idx_tensors(results,cls_idx=0):
    dataLS = results[0].boxes.data
    findCLS = results[0].boxes.cls
    clsNames = results[0].names
    cls_idx_tensors=[i for i in dataLS if i[-1].item()==cls_idx]
    return cls_idx_tensors

def b2b_distance_v2(ball_tensor,pre_n_frames,pre_balls_in_n_frames):

    #前15frame内所有的 dst*pre_conf累加
    dst=0
    divN=(len(pre_balls_in_n_frames)+len(pre_n_frames))/2.0
    offset=1 if len(pre_balls_in_n_frames)>1 else divN/2.0
    for d_i,d in enumerate(pre_n_frames):
        if d is None:
            continue
        else:
            dst+=((d_i+1+offset)/divN)*b2b_distance(ball_tensor,d)
        

    return dst

def get_balls_dst(balls_tensors,pre_n_frames):
    balls_dst=[0 for i in range(len(balls_tensors))]
    pre_balls_in_n_frames=[i for i in pre_n_frames if i is not None]

    for b_i,ball_tensor in enumerate(balls_tensors):
        balls_dst[b_i]=b2b_distance_v2(ball_tensor,pre_n_frames,pre_balls_in_n_frames)
    return balls_dst

def fit_curve_dst(pre_n_frames):
    points=[[((i[0]+i[2])/2.0).item(),((i[1]+i[3])/2.0).item()] for i in pre_n_frames if i is not None]
    if len(points)<2:
        return None
    points = np.array(points)

    x = points[:, 0]
    y = points[:, 1]
    
    # 多项式拟合，这里使用三次多项式
    # p = Polynomial.fit(x, y, 3)
    p = Polynomial.fit(y, x, 3)
    return p

def draw_fit(frame,balls_tensors,p):
    for ball in balls_tensors:
        cx=int(((ball[0]+ball[2])/2.0).item())
        cy=int(((ball[1]+ball[3])/2.0).item())
        fit_y=min(frame.shape[0],max(int(p(cx)),0))
        print(fit_y)
        cv2.circle(frame, [cx,fit_y], 20, (0,0,255), 10)

def tensor_round(tensorData,n=2):
    tensorData=tensorData.item() if torch.is_tensor(tensorData) else tensorData
    return round(tensorData,n)

def b2b_dis_circle(pre_n_frames,balls_tensors):
    dst=[0 for i in range(len(balls_tensors))]
    for ball in balls_tensors:
        one_ball_dst=[]
        for d_i,d in enumerate(pre_n_frames):
            if d is None:
                one_ball_dst.append(None)
                continue
            one_ball_dst.append(b2b_distance(d,ball))
        dst.append(one_ball_dst)
    return dst
              
def possible_balls(pre_n_frames,balls_tensors,r_scales,cxcy):
    min_r_dst_ids=-1
    p_balls=[]
    b2b_circle=[]
    if not all(i is None for i in r_scales ):
        min_r_dst_ids=0
        max_r_dst_ids=0
        for i,r in enumerate(r_scales[::-1]):
            if r is None:
                continue
            else:
                min_r_dst_ids=len(r_scales)-i-1
                break
        for i,r in enumerate(r_scales):
            if r is None:
                continue
            else:
                max_r_dst_ids=i
                break
        for b_i,b in enumerate(balls_tensors):
            if b[-2]<0.3:
                b2b=b2b_distance(pre_n_frames[min_r_dst_ids],b)
                if b2b<=r_scales[min_r_dst_ids]:
                    p_balls.append(b)
                    b2b_circle.append(b2b)
            elif b[-2]<0.7:
                b2b=b2b_distance(pre_n_frames[max_r_dst_ids],b)
                if b2b<=r_scales[max_r_dst_ids]:
                    p_balls.append(b)
                    b2b_circle.append(b2b)
            elif b[-2]>=0.7:
                if pre_n_frames[-1] is None:
                    p_balls.append(b)
                    b2b_circle.append(-b[-2])
                else:
                    b2b=b2b_distance(pre_n_frames[-1],b)
                    if b2b<=r_scales[max_r_dst_ids]:
                        p_balls.append(b)
                        b2b_circle.append(b2b)
        return p_balls,b2b_circle
    else:
        # >0.8
        for b_i,b in enumerate(balls_tensors):
            if b[-2]>=0.8:
                p_balls.append(b)
                b2b_circle.append(-b[-2])
        return p_balls,b2b_circle

def get_best_p_ball(p_balls,b2b_circle):
    if len(b2b_circle)<1:
        return None
    else:
        idx=b2b_circle.index(min(b2b_circle,default=0))
        return p_balls[idx]

def update_pre_n_frames(pre_n_frames,balls_tensors,frame,clsNames,pre_n=3):

    r_scales=[]
    cxcy=[]
    r_scales_big_p=[]
    min_r_dst=-1
    if len(pre_n_frames)>=pre_n:
        pre_n_frames.pop(0)
    if len(pre_n_frames)!=0:

        r_scales=[]
        cxcy=[]
        r_scales_big_p=[]
        for d_i,d in enumerate(pre_n_frames):
            if d is None:
                r_scales.append(None)
                r_scales_big_p.append(None)
                cxcy.append(None)
                continue
            x1, y1, x2, y2, conf, label_i = d.tolist()
            x1, y1, x2, y2, conf, label_i=int(x1), int(y1), int(x2), int(y2),round(conf,2), int(label_i)
            label = clsNames[label_i]

            cx,cy,r_2=int((x1+x2)/2),int((y1+y2)/2),abs(x1-x2)
            r_2=int(math.log(max(2,len(pre_n_frames)-d_i),1.6)*r_2)
            r_scales.append(r_2)
            cxcy.append((cx,cy))


    if len(pre_n_frames)==0 or len(balls_tensors)==0:
        pre_n_frames.append(max(balls_tensors,key=lambda _:_[-2],default=None))
        return pre_n_frames,balls_tensors,None
    else:
        p_balls,b2b_circle=possible_balls(pre_n_frames,balls_tensors,r_scales,cxcy)
        b_ball=get_best_p_ball(p_balls,b2b_circle)
        
        if b_ball is not None:            
            pre_n_frames.append(b_ball)
        else:
            pre_n_frames.append(None)

        return pre_n_frames,balls_tensors,b_ball

def frame_risize(frame,size=640):
    height, width = frame.shape[0], frame.shape[1]
    # 等比例缩放尺度。
    scale = height/size
    # 获得相应等比例的图像宽度。
    width_size = int(width/scale)
    # resize
    frame = cv2.resize(frame, (width_size, size))
    return frame

def write_frames(video_path, frame_queue, video_fps, video_size):
    videoWriter = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), video_fps, video_size)
    while True:
        frame = frame_queue.get()
        if frame is None:
            break
        videoWriter.write(frame)
        print('还在写')
    videoWriter.release()

def merge_audio_with_video(video_with_audio_path, video_without_audio_path, output_path):
    # FFmpeg命令
    command = [
        'ffmpeg',
        '-i', video_with_audio_path,  # 视频文件1，带有音频
        '-i', video_without_audio_path,  # 视频文件2，没有音频
        '-c:v', 'copy',  # 复制视频流
        '-c:a', 'aac',  # 重新编码音频为AAC格式
        '-strict', 'experimental',
        '-map', '1:v:0',  # 选择视频文件1的视频流
        '-map', '0:a:0',  # 选择视频文件0的音频流
        output_path  # 输出文件路径
    ]

    # 执行FFmpeg命令
    subprocess.run(command)

def judge_shoot(preBallStack,frame,ball,rim,wait_frame):
    try:
        p=fit_curve_dst(preBallStack)
    except:
        return False,frame,max(0,wait_frame)

    # draw_ball(frame,ball,)
    balls=[i for i in ball if i is not None]
    balls_in_pre=[i for i in preBallStack if i is not None]
    rims=[i for i in rim if i is not None]
    if len(balls)<1 or len(rims)<1 or p is None:
        wait_frame-=1
        return False,frame,max(0,wait_frame)
    else:
        ball=balls[-1]
        rim=rims[-1]
        c_x,c_y=((rim[0]+rim[2])/2.0).item(),((rim[1]+rim[3])/2.0).item()
        fit_x=p(c_y)
        # try:
        #     cv2.circle(frame,(int(fit_x),int(c_y)),4,(0,0,255),4)
        # except:
        #     print(fit_x)
        #     pass
        if wait_frame==0 and balls_in_pre[-1][1]>rim[1] and fit_x>rim[0] and fit_x<rim[2] and b2b_distance(rim,balls_in_pre[0])<4*abs(balls_in_pre[0][0]-balls_in_pre[0][2]) and balls_in_pre[0][1]<=balls_in_pre[-1][1]:
            print('!!!!!!!!!!!进球！！！！！！！！！')
            cv2.putText(frame, 'score', (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 2)
            cv2.circle(frame,(int(fit_x),int(c_y)),4,(0,0,255),4)
            wait_frame=30
            return True,frame,wait_frame
        else:
            wait_frame-=1
            if(wait_frame>5):
                cv2.putText(frame, 'score', (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 2)
            return False,frame,max(0,wait_frame)

def testfun(model,video,size=640,confidence=0.10):
    cap = cv2.VideoCapture(video)
    fourcc=cv2.VideoWriter_fourcc(*'mp4v')
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    print(video_fps)
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_size=(video_width,video_height)
    print(video_fps)
    print(video_size)


    videoWriter = cv2.VideoWriter('./result.mp4', fourcc, video_fps, video_size)
    preBallStack = []

    # frame_queue = Queue()
    # p = Process(target=write_frames, args=('./result.mp4', frame_queue, video_fps, video_size))
    # p.start()
    
    while cap.isOpened():
        success, frame = cap.read()
        if success:
            results = model(frame,conf=confidence)  # predict on an image
            clsNames = results[0].names
            ball_t_ls=get_cls_idx_tensors(results,cls_idx=0)
            rim_t_ls=get_cls_idx_tensors(results,cls_idx=1)
            preBallStack,ball_t_ls,b_ball=update_pre_n_frames(preBallStack,ball_t_ls,frame,clsNames,pre_n=8)
            frame=draw_rim(frame,rim_t_ls,clsNames)
            frame=draw_ball(frame,[b_ball],clsNames)
            try:
                # try语句写在这不会卡顿
                height, width = frame.shape[0], frame.shape[1]
                scale = height/size
                width_size = int(width/scale)
                # frame_queue.put(frame)
                # # resize
                # p=Process(target=writerV,args=(videoWriter,frame))
                # p.start() # 开启了两个进程
                # [p.join() for p in process]   
                videoWriter.write(frame)

                frame = cv2.resize(frame, (width_size, size))
                cv2.imshow('img', frame)
                pass

            except:
                pass
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break


    
    # # Signal the writer process to stop
    # frame_queue.put(None)
    # p.join()

    # Release resources
    cap.release()
    # videoWriter.release()
    cv2.destroyAllWindows()
    merge_audio_with_video(video,'./result.mp4','./result_v_a.mp4')


# # video = r"D:\NBA-DATASETS\tiktok-shoot\tiktok-shoot-16.mp4"
# video=r"D:\NBA-DATASETS\tiktok-shoot\tiktok-shoot-3.mp4"
# imgs=r"D:\NBA-DATASETS\篮球图片"
# imgspath=[os.path.join(imgs,i ) for i in os.listdir(imgs)]

# for i in imgspath:
#     predict(model,i)
video=r"https://xzbonlinepull.pq8.co/live/hd-zh-2-3736296.m3u8?txSecret=7533569dc9cf6841f299eaab9cecd748&txTime=1708904184"
# video=r"D:\NBA-DATASETS\tiktok-shoot\tiktok-shoot-16.mp4"
# video=r"D:\NBA-DATASETS\tiktok-shoot\tiktok-shoot-1.mp4"
# # video=r"D:\NBA-DATASETS\videos\Harden-shoot-2.mp4"
# video=
video=r"D:\NBA-DATASETS\videos\NBA-replay-27-4.mp4"
# video=0
# video=r"https://v3-web.douyinvod.com/c2554c35f60ae6a869726b33c49b6cea/65d9e9ef/video/tos/cn/tos-cn-ve-15/okM8LANDC7EQmaeDABgEInfhRQYrXyZKz5ALBY/?a=6383&ch=11&cr=3&dr=0&lr=all&cd=0%7C0%7C0%7C3&cv=1&br=1492&bt=1492&cs=0&ds=3&ft=bvTKJbQQqU-mfJ40Do0OqY8hFgpiW4isejKJChUkoG0P3-I&mime_type=video_mp4&qs=1&rc=ZTtmaGg1Ozs1Ozg7ZzMzOkBpM2VsM2U6ZjNwcTMzNGkzM0BfNjZeY2M0XmMxXmMxNGJhYSNnY2xycjRnYGBgLS1kLS9zcw%3D%3D&btag=e00028000&dy_q=1708776282&feature_id=46a7bb47b4fd1280f3d3825bf2b29388&l=202402242004415EDD12B277F8E89CB207"
model = YOLO('pts/best-ball-rim-4.pt')


if __name__ =='__main__':
    t1=time.time()
    testfun(model,video)
    t2=time.time()
    print(f'耗时{t2-t1}s')