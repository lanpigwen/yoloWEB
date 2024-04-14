// worker.js

self.onmessage = function(event) {
    const imageData = event.data.imageData;
    const resultCanvasWidth = event.data.resultCanvasWidth;
    const resultCanvasHeight = event.data.resultCanvasHeight;

    createImageBitmap(base64ToBlob(imageData)).then(function(bitmap) {
        const offscreenCanvas = new OffscreenCanvas(resultCanvasWidth, resultCanvasHeight);
        const offscreenContext = offscreenCanvas.getContext("2d");

        // 清除之前的渲染
        offscreenContext.clearRect(0, 0, resultCanvasWidth, resultCanvasHeight);
        // 将图像渲染到离屏 canvas 上
        offscreenContext.drawImage(bitmap, 0, 0, resultCanvasWidth, resultCanvasHeight);

        // 将渲染后的图像数据发送回主线程
        self.postMessage(offscreenCanvas.transferToImageBitmap());
    });
};

function base64ToBlob(base64) {
    const byteString = atob(base64.split(',')[1]);
    const mimeString = base64.split(',')[0].split(':')[1].split(';')[0];
    const ab = new ArrayBuffer(byteString.length);
    const ia = new Uint8Array(ab);

    for (let i = 0; i < byteString.length; i++) {
        ia[i] = byteString.charCodeAt(i);
    }

    return new Blob([ab], { type: mimeString });
}
