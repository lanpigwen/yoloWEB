self.onmessage = function(event) {
    // 接收来自主线程的视频帧数据
    const videoFrame = event.data;
    
    // 创建 OffscreenCanvas 对象
    const offscreenCanvas = new OffscreenCanvas(videoFrame.width, videoFrame.height);
    const offscreenContext = offscreenCanvas.getContext('2d');
    
    // 将视频帧绘制到 OffscreenCanvas 上
    offscreenContext.drawImage(videoFrame, 0, 0);
    
    // 将 OffscreenCanvas 的图像数据转换为 Data URL
    const frameDataURL = offscreenCanvas.toDataURL('image/jpeg');
    
    // 将处理后的图像数据发送回主线程
    postMessage(frameDataURL);
};