import cv2
import subprocess as sp
import numpy as np
import time
import ffmpeg
from ultralytics import YOLO
import os
from utils import get_cls_idx_tensors, update_pre_n_frames, draw_rim, draw_ball,judge_shoot
from threading import Thread
from queue import Queue
import re

def extract_video_devices(output):
    pattern = r'"(.+?)" \(video\)'
    matches = re.findall(pattern, output)
    return matches

def predict(model,frame,preBallStack,confidence=0.1):
    results = model(frame,conf=confidence,imgsz=640,int8=True,task='detect')  # predict on an image
    clsNames = results[0].names
    ball_t_ls=get_cls_idx_tensors(results,cls_idx=0)
    rim_t_ls=get_cls_idx_tensors(results,cls_idx=1)
    preBallStack,ball_t_ls,b_ball=update_pre_n_frames(preBallStack,ball_t_ls,frame,clsNames,pre_n=8)
    frame=draw_rim(frame,rim_t_ls,clsNames)
    frame=draw_ball(frame,preBallStack,clsNames,tsize1=1,tsize2=1,recsize=1,rec_color=(255,0,0),show_text=False)
    frame=draw_ball(frame,[b_ball],clsNames)
    return frame,preBallStack,rim_t_ls

def read_video(show_queue,save_queue,video_file,model,preset,model_predict,video_size,jump_frame,isCamera,isFront):
    # 设置FFmpeg命令行
    if isCamera:
        command = ['ffmpeg', '-list_devices', 'true', '-f', 'dshow', '-i', 'dummy']
        result = sp.run(command, capture_output=True, text=True)
        output = str(result)
        video_devices = extract_video_devices(output)
        command1 = ['ffmpeg',
                    '-f', 'dshow',   # 使用DirectShow设备
                    '-i', f'video={video_devices[0]}',  # 使用摄像头作为输入
                    '-vf', f'scale={video_size[0]}:{video_size[1]}',
                    '-f', 'image2pipe',
                    '-pix_fmt', 'bgr24',
                    '-preset', preset,
                    '-c:v', 'rawvideo',
                    '-']
    else:
        command1 = ['ffmpeg',
                # '-re', # 按原始帧率发送
                '-i', video_file,
                '-vf', f'scale={video_size[0]}:{video_size[1]}',
                '-f', 'image2pipe',
                '-pix_fmt', 'bgr24',
                '-preset', preset,
                '-c:v', 'rawvideo',
                '-']
    



    pipe1 = sp.Popen(command1, stdout=sp.PIPE)

    preBallStack=[]
    jump=0
    wait_frame=0
    # 使用OpenCV读取并显示视频帧
    while True:
        # 从管道中读取帧数据
        raw_image = pipe1.stdout.read(video_size[0] * video_size[1] * 3)

        if len(raw_image) != video_size[0] * video_size[1] * 3:
            break

        # 将原始数据转换为NumPy数组
        frame = np.frombuffer(raw_image, dtype='uint8')
        frame = frame.reshape((video_size[1], video_size[0] , 3))
        # predict frame
        if not jump_frame:jump=0
        if model_predict and jump==0:
            frame = cv2.resize(frame, (video_size[0],video_size[1]))
            #镜像
            if isFront:
                frame = cv2.flip(frame, 1)

            frame,preBallStack,rim_t_ls=predict(model,frame,preBallStack)

            score,frame,wait_frame=judge_shoot(preBallStack,frame,preBallStack,rim_t_ls,wait_frame)
            if score:
                print(preBallStack[-1])
                print(rim_t_ls[-1])


        jump=(jump+1)%(jump_frame+1)
        save_queue.put(frame)
        show_queue.put(frame)

    show_queue.put(None)
    save_queue.put(None)

def show_video(show_queue,show_height,origin_fps,show_fps,fit_show_fps):
    time_start=time.time()
    last_frame=time.time()
    showed_fps=0
    frame_ms=[]
    while True:

        frame=show_queue.get()
        if frame is None:
            break

        size = show_height
        height, width = frame.shape[0], frame.shape[1]
        scale = height / size
        width_size = int(width / scale)
        showed_fps+=1

        if show_fps:
            frame_ms.append(time.time()-last_frame)
            if showed_fps==1:
                fps=int(1.0/(time.time()-last_frame))
            if len(frame_ms)==10:
                ms_10=sum(frame_ms)
                fps=int(10.0/ms_10)
                frame_ms.clear()
            last_frame=time.time()
            frame=cv2.resize(frame,(frame.shape[1],frame.shape[0]))
            rec_color=(0, 0, 0)
            text_color=(255, 255, 255)

            text = f'fps:{fps}'
            # 获取文本的尺寸
            text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)
            text_width, text_height = text_size
            cv2.rectangle(frame, (0, 0),(int(text_width), int(text_height)), rec_color, -1)  
            cv2.putText(frame, text, (0, int(text_height)), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 1)

        if fit_show_fps:
            while int(showed_fps/(time.time()-time_start))>origin_fps:    
                time.sleep(0.001)

        # # 显示视频帧
        cv2.namedWindow('video',cv2.WINDOW_NORMAL)

        cv2.imshow('video', frame)
        cv2.resizeWindow('video',(width_size, size))

        key = cv2.waitKey(1) & 0xff
        if key == ord(" "):
            cv2.waitKey(0)
        if key == ord("q"):
            break

def save_video(save_queue,video_file,out_video_file,encoder,preset,video_size,fps,get_bitrate,bitrate,isCamera):


    if isCamera:
        out_video_file='camera'+'-predict.'+'mp4'
        out_video_file=os.path.join('./results',out_video_file)
        audio_device_name="麦克风阵列 (英特尔® 智音技术)"
        command2 = [
            'ffmpeg',
            '-y',  # 允许覆盖输出文件
            '-f', 'rawvideo',
            '-s', f'{video_size[0]}x{video_size[1]}',
            '-pix_fmt', 'bgr24',
            '-r', f'{fps}',
            '-i', '-',  # 从标准输入读取视频帧
            '-f', 'dshow',
            '-i', f'audio={audio_device_name}',  # 捕获摄像头的声音
            '-map', '0:v',  # 明确指定第一个输入的视频流
            '-map', '1:a',  # 明确指定第二个输入的音频流
            '-c:v', encoder,  # 使用指定编码器编码视频
            '-c:a', 'aac',  # 复制原始音频流
        ]
    else:
        command2 = [
                'ffmpeg',
                # '-re', # 按原始帧率发送
                '-y',
                '-f', 'rawvideo',
                '-s', f'{video_size[0]}x{video_size[1]}',
                '-pix_fmt', 'bgr24',
                '-r', f'{fps}',
                '-i', '-',
                '-i', video_file,
                # '-i', 'audio=麦克风阵列 (英特尔® 智音技术)',
                '-map', '0:v',  # 明确指定第一个输入的视频流
                '-map', '1:a', 
                '-c:v', encoder,  # 使用 libx264 编码器
                '-c:a', 'copy',  # 复制原始音频流
            ]
        if get_bitrate:
            command2.extend(['-b:v', f'{bitrate}k'])
        command2.extend(['-preset', preset, out_video_file])  
    pipe2 = sp.Popen(command2, stdin=sp.PIPE)

    while True:
        frame=save_queue.get()
        if frame is None:
            break
        pipe2.stdin.write(frame.tobytes())

def mainThreads(video_file,model,out_video_file,show_height,out_height,bitrate_k,encoder,preset,model_predict,jump_frame=0,show_fps=True,fit_show_fps=True,isCamera=False,isFront=False):
    get_bitrate=True
    bitrate='20M'
    if isCamera:
        get_bitrate=False
        fps=31
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Unable to open camera.")
            exit()
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video_size = (width, height)
        cap.release()
    else:
        # 打印摄像头尺寸
        probe = ffmpeg.probe(video_file)
        video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
        fps = eval(video_stream['r_frame_rate'])
        width = video_stream['width']
        height = video_stream['height']
        
        try:
            bitrate = int(video_stream['bit_rate'])/1024
            bitrate=int(bitrate*bitrate_k)
        except:
            get_bitrate=False

    if height>out_height:
        scale=height/out_height
        width=int(width/scale)
        height=out_height
    # 偶数x偶数才能正常播放
    width=width-width%2
    height=height-height%2

    video_size=(width,height)
    show_queue=Queue()
    save_queue=Queue()

    read_thread = Thread(target=read_video, args=(show_queue,save_queue,video_file,model,preset,model_predict,video_size,jump_frame,isCamera,isFront))
    show_thread = Thread(target=show_video, args=(show_queue,show_height,fps,show_fps,fit_show_fps))
    save_thread = Thread(target=save_video,args=(save_queue,video_file,out_video_file,encoder,preset,video_size,fps,get_bitrate,bitrate,isCamera))

    read_thread.start()
    show_thread.start()
    save_thread.start()

    read_thread.join()
    show_thread.join()
    save_thread.join()

if __name__ == '__main__':
    t0 = time.time()
    video_file = r"C:\NBA-DATASETS\videos\NBA-replay-27.mp4"
    # video_file=r"https://play.aomeila.cn/live/sd-318q6nt6kk7zmo9.m3u8"
    # video_file=r"D:\NBA-DATASETS\NBA-Replay\NBA-replay-20.mp4"
    # video_file=r"D:\NBA-DATASETS\tiktok-shoot\tiktok-shoot-2.mp4"
    # video_file="https://play2nm.hnyongshun.cn/live/hd-en-4wyrn1to9720q86.m3u8"

    video_file=r"C:\NBA-DATASETS\tiktok-shoot\shoot.mp4"
    # video_file=r"D:\NBA-DATASETS\tiktok-shoot\tiktok-shoot-2.mp4"
    # video_file=r"D:\NBA-DATASETS\CunBA-replay\CunBA-replay-8.mp4"
    # video_file="https://v3-web.douyinvod.com/ad247f653d54a50c77583451b8972e3c/661699cf/video/tos/cn/tos-cn-ve-15c001-alinc2/og9FSAsMAK5oQgZjwnDeDqWvQBFbMZc8FgnfAs/?a=6383&ch=11&cr=3&dr=0&lr=all&cd=0%7C0%7C0%7C3&cv=1&br=830&bt=830&cs=0&ds=6&ft=LjhJEL998xztuo0mo0P5fQhlpPiXEkUWxVJEUA-jpbPD-Ipz&mime_type=video_mp4&qs=0&rc=aTRkMzw4PGllZmU3PDM0OkBpMztyMzg6ZmU6aDMzNGkzM0A1Ml8wY2M1X2AxLWJeLTM0YSNgamhfcjQwaXNgLS1kLTBzcw%3D%3D&btag=e00028000&cquery=101n_100B_100x_100z_100o&dy_q=1712753472&feature_id=f0150a16a324336cda5d6dd0b69ed299&l=2024041020511257482ACBCA1CCB0C5FE9"
    model = YOLO('pts/best-ball-rim-4.engine')
    # model = YOLO('pts/best-ball-rim-n-600s.engine')
    outfile=''.join(os.path.basename(video_file).split('.')[:-1])+'-predict.'+'mp4'
    outfile=os.path.join('./results',outfile)
    # outfile="result.mp4"
    # print(outfile)
    mainThreads(video_file,model,out_video_file=outfile,
                show_height=640,
                out_height=640,
                bitrate_k=2,
                encoder='mpeg4',
                preset='ultrafast',
                model_predict=True,
                jump_frame=1,
                show_fps=True,
                fit_show_fps=False,
                isCamera=False,
                isFront=False
                )
    t2 = time.time()
    # print(f'耗时{round(t2 - t0,3)}s')
    print(outfile)