import cv2
video_file=r"D:\BALL-RIM-PROJECT\results\hd-en-4wyrn1to9720q86-predict.mp4"
vc = cv2.VideoCapture(video_file
)

if vc.isOpened():
    ret, frame = vc.read()
while ret:
    ret, frame = vc.read()
    cv2.imshow('img',frame)
    key = cv2.waitKey(0) & 0xff
    if key == ord(" "):
        cv2.waitKey(0)
    if key == ord("q"):
        break