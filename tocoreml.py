# # from ultralytics import YOLO

# # # Load the YOLOv8 model
# # model = YOLO(r'D:\BALL-RIM-PROJECT\pts\best-ball-rim-4.pt')

# # # Export the model to CoreML format
# # model.export(format='coreml')  # creates '.mlpackage'

# # # Load the exported CoreML model
# # coreml_model = YOLO(r'D:\BALL-RIM-PROJECT\pts\best-ball-rim-4.mlpackage')

# # # Run inference
# # results = coreml_model(r"D:\NBA-DATASETS\videos\frames\DL-shoot-1_000168.jpg")



# from ultralytics import YOLO

# # Load the YOLOv8 model
# model = YOLO('yolov8n.pt')

# # Export the model to NCNN format
# model.export(format='onnx',simplify=True,opset=12) # creates '/yolov8n_ncnn_model'

# # Load the exported NCNN model
# # ncnn_model = YOLO('./yolov8n_onnx_model')

# # Run inference
# # results = ncnn_model('bus.jpg')


from ultralytics import YOLO

# # Load a model
# model = YOLO('best-ball-rim-4.pt')  # load an official model
# # model = YOLO('yolov8n.pt')  # load a custom trained

# # Use the model
# success=model.export(format='onnx',simplify=True)

# Load the exported ONNX model
onnx_model = YOLO('pts\yolov8n-pose.engine')
# onnx_model = YOLO('best-ball-rim-4.onnx',task='detect')


# Run inference

video_file=r"C:\NBA-DATASETS\tiktok-shoot\练完核心后的各种离谱投篮，甚至可以干拔三分.mp4"
results = onnx_model(video_file,show=True,save=True)