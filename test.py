import subprocess as sp
import re

def extract_video_devices(output):
    pattern = r'"(.+?)" \(video\)'
    matches = re.findall(pattern, output)
    return matches

def extract_audio_devices(output):
    pattern = r'"(.+?)" \(audio\)'
    matches = re.findall(pattern, output)
    return matches

command = ['ffmpeg', '-list_devices', 'true', '-f', 'dshow', '-i', 'dummy']
result = sp.run(command, capture_output=True, text=True)
# print(result)
output = str(result)
video_devices = extract_video_devices(output)
# audio_devices=extract_audio_devices(output)
print(video_devices[0])
# print(audio_devices[0])




# 麦克风阵列 (英特尔® 智音技术)