// Web Worker 中
self.onmessage = function(event) {
    var width = event.data.width;
    var height = event.data.height;
    var imageData = event.data.imageData;

    // 将像素数据编码为 base64 字符串
    var canvas = new OffscreenCanvas(width, height);
    var context = canvas.getContext('2d');
    var imageDataArray = new Uint8ClampedArray(imageData);
    var imageDataObject = new ImageData(imageDataArray, width, height);
    context.putImageData(imageDataObject, 0, 0);
    var base64ImageData = canvas.toDataURL("image/jpeg").split(',')[1];

    // 发送编码后的数据给主线程
    self.postMessage({ base64ImageData: base64ImageData });
};