// var canvas = document.getElementById('myCanvas');
// var ctx = canvas.getContext('2d');

function distanceBetweenPoints(x1, y1, x2, y2) {
    var dx = x2 - x1;
    var dy = y2 - y1;
    return Math.sqrt(dx * dx + dy * dy);
  }
  
  function toPercentage(num1, num2) {
      if (num2 === 0) {
          // 避免除以零的错误
          return "0.0%";
      }
      // 计算百分比
      var percentage = (num1 / num2) * 100;
      // 格式化为两位小数的百分比字符串
      return percentage.toFixed(1) + "%";
  }
  
  function drawZones(zones,canvas,ctx) {
      zones.forEach(function(zone) {
          let ax,ay,num;
          ax=0;ay=0;num=0;
          let min_x,max_x,min_y,max_y;
          min_x=canvas.width;max_x=0;min_y=canvas.height;max_y=0;
          
          ctx.fillStyle = zone.color;
          ctx.beginPath();
          for (var x = 0; x < canvas.width; x++) {
              for (var y = 0; y < canvas.height; y++) {
              if (zone.area(x, y)) {
                  ctx.fillRect(x, y, 1, 1);
                  ax+=x;ay+=y;num++;
                  min_x=Math.min(min_x,x);
                  max_x=Math.max(max_x,x);
                  min_y=Math.min(min_y,y);
                  max_y=Math.max(max_y,y);
              }
              }
          }
          ctx.fill();
          let fontSize=20;
          ctx.font = fontSize+"px Arial";
          ctx.fillStyle = "white"; // 文字颜色
          ctx.textAlign = "center"; 
          ax/=num;ay/=num;
          if(zone.idx==0){
              ay+=0.2*(max_y-min_y);
          }
          if(zone.idx==1){
              ax-=0.1*(max_x-min_x);
          }
          if(zone.idx==2){
              ay-=0.2*(max_y-min_y);
          }
          if(zone.idx==3){
              ax+=0.1*(max_x-min_x);
          }
          if(zone.idx==6){
              ay-=0.19*(max_y-min_y);
          }
  
          if(zone.attampt>0){
              let text_percent=toPercentage(zone.make,zone.attampt);
              let t1Width = ctx.measureText(text_percent).width;
              while(t1Width>0.65*(max_x-min_x)){
                  fontSize--;
                  ctx.font = fontSize+"px Arial";
                  t1Width = ctx.measureText(text_percent).width;
              }
              ctx.fillText(text_percent,ax,ay);
  
              let text_make_attampt=zone.make+'/'+zone.attampt;
              let t2Width=ctx.measureText(text_make_attampt).width;
              while(t2Width>0.65*(max_x-min_x)){
                  fontSize--;
                  ctx.font = fontSize+"px Arial";
                  t2Width=ctx.measureText(text_make_attampt).width;
              }
              ctx.fillText(text_make_attampt,ax,ay+fontSize+2);
          }
  
  
      });
      }
  
  function nxny2Zones(zones,canvas,ctx,nx,ny,score,
                  null_color='rgba(180, 180, 180, 1)',
                  lower_color='rgba(89, 164, 222, 1)',
                  average_color='rgba(247, 214, 73,1)',
                  high_color='rgba(220, 117, 58, 1)',
                  lower_percent=0.3,
                  high_percent=0.5){
  
  
      let cx,cy;
      cx=nx*canvas.width;
      cy=ny*canvas.height;
      zones.forEach(function(zone) {
          if(zone.area(cx,cy)){
              zone.attampt++;
              if(score){
                  zone.make++;
              }
  
              let color;
              if(zone.attampt==0){
                  color=null_color;
              }
              else{
                  let percent=zone.make/zone.attampt;
                  if(percent<lower_percent){
                      color=lower_color;
                  }
                  else if(percent<high_percent){
                      color=average_color;
                  }
                  else{
                      color=high_color;
                  }
              }
              zone.color=color;
          }
      })
  }
  
  function drawZonesLine(canvas,ctx,zonesInfo){
      let r1r2_1,r2r1_1;
      let min_r1=1;let min_r1_2=1;
  
      let r2r3_1,r2r3_2,r3r2_1,r3r2_2;
      let min_r2=1;let min_r2_2=1;let min_2_r2=1;let min_2_r2_2=1;
  
      let r3r4_1,r4r3_1;
      let min_r3=1;let min_r3_1=1;
  
  
          for (var y = 0; y < canvas.height; y++) {
              for (var x = 0; x < canvas.width; x++) {
                  let tempAngle=angleRim_XY(x,y,zonesInfo.rimX,zonesInfo.rimY);
                  //中距离左分割
                  if(Math.abs(distanceBetweenPoints(x,y,zonesInfo.rimX,zonesInfo.rimY)-zonesInfo.r1)<2 &&
                  Math.abs(tempAngle-zonesInfo.r1_angle1)<min_r1 && x<zonesInfo.rimX){
                      r1r2_1={x:x,y:y};
                      min_r1=Math.abs(tempAngle-zonesInfo.r1_angle1);
                  }
                  if(Math.abs(distanceBetweenPoints(x,y,zonesInfo.rimX,zonesInfo.rimY)-zonesInfo.r2)<2 &&
                  Math.abs(tempAngle-zonesInfo.r1_angle1)<min_r1_2 && x<zonesInfo.rimX){
                      r2r1_1={x:x,y:y};
                      min_r1_2=Math.abs(tempAngle-zonesInfo.r1_angle1);
                  }
                  //长两分 左1
                  if(Math.abs(distanceBetweenPoints(x,y,zonesInfo.rimX,zonesInfo.rimY)-zonesInfo.r2)<2 &&
                  Math.abs(tempAngle-zonesInfo.r2_angle1)<min_r2 && x<zonesInfo.rimX){
                      r2r3_1={x:x,y:y};
                      min_r2= Math.abs(tempAngle-zonesInfo.r2_angle1);
                  }
                  if(Math.abs(distanceBetweenPoints(x,y,zonesInfo.rimX,zonesInfo.rimY)-zonesInfo.r3)<2 &&
                  Math.abs(tempAngle-zonesInfo.r2_angle1)<min_r2_2 && x<zonesInfo.rimX){
                      r3r2_1={x:x,y:y};
                      min_r2_2=Math.abs(tempAngle-zonesInfo.r2_angle1);
                  }
                  //长两分 左2
                  if(Math.abs(distanceBetweenPoints(x,y,zonesInfo.rimX,zonesInfo.rimY)-zonesInfo.r2)<2 &&
                  Math.abs(tempAngle-zonesInfo.r2_angle2)<min_2_r2 && x<zonesInfo.rimX){
                      r2r3_2={x:x,y:y};
                      min_2_r2= Math.abs(tempAngle-zonesInfo.r2_angle2);
                  }
                  if(Math.abs(distanceBetweenPoints(x,y,zonesInfo.rimX,zonesInfo.rimY)-zonesInfo.r3)<2 &&
                  Math.abs(tempAngle-zonesInfo.r2_angle2)<min_2_r2_2 && x<zonesInfo.rimX){
                      r3r2_2={x:x,y:y};
                      min_2_r2_2=Math.abs(tempAngle-zonesInfo.r2_angle2);
                  }
  
                  if(Math.abs(distanceBetweenPoints(x,y,zonesInfo.rimX,zonesInfo.rimY)-zonesInfo.r3)<2 &&
                  Math.abs(tempAngle-zonesInfo.r3_angle1)<min_r3 && x<zonesInfo.rimX){
                      r3r4_1={x:x,y:y};
                      min_r3= Math.abs(tempAngle-zonesInfo.r3_angle1);
                  }
                  if(y==zonesInfo.centerLineY-1 &&
                  Math.abs(tempAngle-zonesInfo.r3_angle1)<min_r3_1 && x<zonesInfo.rimX){
                      r4r3_1={x:x,y:y};
                      min_r3_1= Math.abs(tempAngle-zonesInfo.r3_angle1);
                  }
  
  
              }
          }
  
          //圆圈 划分长两分 短两分
          ctx.strokeStyle = 'white';
          ctx.lineWidth=Math.max(1,zonesInfo.borderXY*0.4);
          ctx.beginPath();
          ctx.arc(zonesInfo.rimX,zonesInfo.rimY,zonesInfo.r1,0,2*Math.PI);
          ctx.stroke();
          ctx.beginPath();
          ctx.arc(zonesInfo.rimX,zonesInfo.rimY,zonesInfo.r2,0,2*Math.PI);
          ctx.stroke();
  
          ctx.beginPath();
          ctx.moveTo(r1r2_1.x,r1r2_1.y);
          ctx.lineTo(r2r1_1.x,r2r1_1.y);
          ctx.stroke();
          ctx.beginPath();
          ctx.moveTo(r1r2_1.x+2*(zonesInfo.rimX-r1r2_1.x),r1r2_1.y);
          ctx.lineTo(r2r1_1.x+2*(zonesInfo.rimX-r2r1_1.x),r2r1_1.y);
          ctx.stroke();
  
          ctx.beginPath();
          ctx.moveTo(r2r3_1.x,r2r3_1.y);
          ctx.lineTo(r3r2_1.x,r3r2_1.y);
          ctx.stroke();
          ctx.beginPath();
          ctx.moveTo(r2r3_1.x+2*(zonesInfo.rimX-r2r3_1.x),r2r3_1.y);
          ctx.lineTo(r3r2_1.x+2*(zonesInfo.rimX-r3r2_1.x),r3r2_1.y);
          ctx.stroke();
  
          ctx.beginPath();
          ctx.moveTo(r2r3_2.x,r2r3_2.y);
          ctx.lineTo(r3r2_2.x,r3r2_2.y);
          ctx.stroke();
          ctx.beginPath();
          ctx.moveTo(r2r3_2.x+2*(zonesInfo.rimX-r2r3_2.x),r2r3_2.y);
          ctx.lineTo(r3r2_2.x+2*(zonesInfo.rimX-r3r2_2.x),r3r2_2.y);
          ctx.stroke();
  
          ctx.beginPath();
          ctx.moveTo(r3r4_1.x,r3r4_1.y);
          ctx.lineTo(r4r3_1.x,r4r3_1.y);
          ctx.stroke();
          ctx.beginPath();
          ctx.moveTo(r3r4_1.x+2*(zonesInfo.rimX-r3r4_1.x),r3r4_1.y);
          ctx.lineTo(r4r3_1.x+2*(zonesInfo.rimX-r4r3_1.x),r4r3_1.y);
          ctx.stroke();
  
          ctx.beginPath()
          ctx.moveTo(0,2.99*zonesInfo.scale+zonesInfo.borderXY);
          ctx.lineTo(zonesInfo.borderXY+0.9*zonesInfo.scale+zonesInfo.borderXY/2,2.99*zonesInfo.scale+zonesInfo.borderXY);
          ctx.stroke();
  
          ctx.beginPath()
          ctx.moveTo(0+2*zonesInfo.rimX,2.99*zonesInfo.scale+zonesInfo.borderXY);
          ctx.lineTo(zonesInfo.borderXY+0.9*zonesInfo.scale+zonesInfo.borderXY/2+2*(zonesInfo.rimX-(zonesInfo.borderXY+0.9*zonesInfo.scale+zonesInfo.borderXY/2)),2.99*zonesInfo.scale+zonesInfo.borderXY);
          ctx.stroke();
  
  
  
  
  }
  
  function angleRim_XY(x,y,rimX,rimY){
      let dh=y-rimY;
      let dw=x-rimX;
      let tan=Math.abs(dh/dw);
      let sign=0;
      if(y<rimY){
          sign=1;
      }
      return Math.pow(-1,sign)*tan;
  }
      // 当图片加载完成后，绘制图片
  function drawCourt(canvas,ctx,shootingInfo,courtWidth=800,borderXY=10,drawMainLineColor='white',
                  null_color='rgba(180, 180, 180, 1)',
                  lower_color='rgba(89, 164, 222, 1)',
                  average_color='rgba(247, 214, 73,1)',
                  high_color='rgba(220, 117, 58, 1)',
                  lower_percent=0.3,
                  high_percent=0.5) {
  
      let courtHeight;
      courtHeight=(courtWidth-2*borderXY)*(13.95/15)+2*borderXY;
      canvas.width=courtWidth;
      canvas.height=courtHeight;
      ctx.lineWidth=borderXY;
      let scale=(canvas.width-2*borderXY)/15;
  
  
      let x=borderXY+7.5*scale;
      let y=borderXY+1.575*scale;
      let zonesInfo={
          rimX:x,rimY:y,
          r1:2.45*scale-borderXY/2,
          r2:Math.sqrt(Math.pow(3.9*scale+3*0.05*scale,2)+Math.pow(2.45*scale,2)),
          r3:6.75*scale-borderXY/2,
          r1_angle1:4.6/2.45,
          r2_angle1:0.5,
          r2_angle2:4.6/2.45+0.3,
          r3_angle0:2.99*scale+borderXY,
          r3_angle1:4.6/2.45+0.8,
          centerLineY:canvas.height,
          scale:scale,
          borderXY:borderXY
      };
  
      let radius=6.75*scale-borderXY/2;
  
  
      // 定义区域和命中率
      let zones = [
          {//禁区
              color: null_color,
              area: function (x, y) { return distanceBetweenPoints(x, y, zonesInfo.rimX, zonesInfo.rimY) < zonesInfo.r1; },
              make:0,
              attampt:0,
              idx:0
  
          },
          {//左中距离
              color: null_color,
              area: function (x, y) {
                  return ((distanceBetweenPoints(x, y, zonesInfo.rimX, zonesInfo.rimY) < zonesInfo.r2) &&
                      (distanceBetweenPoints(x, y, zonesInfo.rimX, zonesInfo.rimY) > zonesInfo.r1)) &&
                      angleRim_XY(x,y,zonesInfo.rimX,zonesInfo.rimY)<zonesInfo.r1_angle1 &&
                      x<=zonesInfo.rimX;
              },
              make:0,
              attampt:0,
              idx:1
          },
  
          {//中中距离
              color: null_color,
              area: function (x, y) {
                  return ((distanceBetweenPoints(x, y, zonesInfo.rimX, zonesInfo.rimY) < zonesInfo.r2) &&
                      (distanceBetweenPoints(x, y, zonesInfo.rimX, zonesInfo.rimY) > zonesInfo.r1)) &&
                      angleRim_XY(x,y,zonesInfo.rimX,zonesInfo.rimY)>=zonesInfo.r1_angle1;
              },
              make:0,
              attampt:0,
              idx:2
          },
          {//右中距离
              color: null_color,
              area: function (x, y) {
                  return ((distanceBetweenPoints(x, y, zonesInfo.rimX, zonesInfo.rimY) < zonesInfo.r2) &&
                      (distanceBetweenPoints(x, y, zonesInfo.rimX, zonesInfo.rimY) > zonesInfo.r1) )&&
                      angleRim_XY(x,y,zonesInfo.rimX,zonesInfo.rimY)<zonesInfo.r1_angle1 &&
                      x>=zonesInfo.rimX;
              },
  
              make:0,
              attampt:0,
              idx:3
          },
  
          {//左低长两分
              color: null_color,
              area: function (x, y) {
                  return ((distanceBetweenPoints(x, y, zonesInfo.rimX, zonesInfo.rimY) < zonesInfo.r3) &&
                      (distanceBetweenPoints(x, y, zonesInfo.rimX, zonesInfo.rimY) > zonesInfo.r2) &&
                      (x > borderXY + 0.9 * scale + borderXY / 2) &&
                      (x < borderXY + 14.1 * scale - borderXY / 2) ||
                      ((x > borderXY + 0.9 * scale + borderXY / 2) &&
                      y<borderXY + 2.99 * scale + borderXY / 2 &&
                      distanceBetweenPoints(x, y, zonesInfo.rimX, zonesInfo.rimY) > zonesInfo.r2))&&
                      angleRim_XY(x,y,zonesInfo.rimX,zonesInfo.rimY)<zonesInfo.r2_angle1 &&
                      x<=zonesInfo.rimX;
              },
              make:0,
              attampt:0,
              idx:4
          },
          {//左45长两分
              color: null_color,
              area: function (x, y) {
                  return ((distanceBetweenPoints(x, y, zonesInfo.rimX, zonesInfo.rimY) < zonesInfo.r3) &&
                      (distanceBetweenPoints(x, y, zonesInfo.rimX, zonesInfo.rimY) > zonesInfo.r2) &&
                      (x > borderXY + 0.9 * scale + borderXY / 2) &&
                      (x < borderXY + 14.1 * scale - borderXY / 2)) &&
                      angleRim_XY(x,y,zonesInfo.rimX,zonesInfo.rimY)>zonesInfo.r2_angle1 && angleRim_XY(x,y,zonesInfo.rimX,zonesInfo.rimY)<zonesInfo.r2_angle2 &&
                      x<=zonesInfo.rimX;
              },
              make:0,
              attampt:0,
              idx:5
          },
          {//中长两分
              color: null_color,
              area: function (x, y) {
                  return ((distanceBetweenPoints(x, y, zonesInfo.rimX, zonesInfo.rimY) < zonesInfo.r3) &&
                      (distanceBetweenPoints(x, y, zonesInfo.rimX, zonesInfo.rimY) > zonesInfo.r2) &&
                      (x > borderXY + 0.9 * scale + borderXY / 2) &&
                      (x < borderXY + 14.1 * scale - borderXY / 2)) &&
                      angleRim_XY(x,y,zonesInfo.rimX,zonesInfo.rimY)>=zonesInfo.r2_angle2;
              },
              make:0,
              attampt:0,
              idx:6
          },
          {//右45长两分
              color: null_color,
              area: function (x, y) {
                  return ((distanceBetweenPoints(x, y, zonesInfo.rimX, zonesInfo.rimY) < zonesInfo.r3) &&
                      (distanceBetweenPoints(x, y, zonesInfo.rimX, zonesInfo.rimY) > zonesInfo.r2) &&
                      (x > borderXY + 0.9 * scale + borderXY / 2) &&
                      (x < borderXY + 14.1 * scale - borderXY / 2)) &&
                      angleRim_XY(x,y,zonesInfo.rimX,zonesInfo.rimY)>zonesInfo.r2_angle1 && angleRim_XY(x,y,zonesInfo.rimX,zonesInfo.rimY)<zonesInfo.r2_angle2 &&
                      x>=zonesInfo.rimX;
              },
              make:0,
              attampt:0,
              idx:7
          },
          {//右低长两分
              color: null_color,
              area: function (x, y) {
                  return ((distanceBetweenPoints(x, y, zonesInfo.rimX, zonesInfo.rimY) < zonesInfo.r3) &&
                      (distanceBetweenPoints(x, y, zonesInfo.rimX, zonesInfo.rimY) > zonesInfo.r2) &&
                      (x > borderXY + 0.9 * scale + borderXY / 2) &&
                      (x < borderXY + 14.1 * scale - borderXY / 2) ||
                      ((x < borderXY + 14.1 * scale - borderXY / 2) &&
                      y<borderXY + 2.99 * scale + borderXY / 2 &&
                      distanceBetweenPoints(x, y, zonesInfo.rimX, zonesInfo.rimY) > zonesInfo.r2))&&
                      angleRim_XY(x,y,zonesInfo.rimX,zonesInfo.rimY)<zonesInfo.r2_angle1 &&
                      x>=zonesInfo.rimX;
              },
              make:0,
              attampt:0,
              idx:8
          },
          {//左底角三分
              color: null_color,
              area: function (x, y) {
                  return ((distanceBetweenPoints(x, y, zonesInfo.rimX, zonesInfo.rimY) > zonesInfo.r3) && (y > 2.99 * scale + borderXY) ||
                      (x < borderXY + 0.9 * scale + borderXY / 2) ||
                      (x > borderXY + 14.1 * scale - borderXY / 2)) &&
                      y<=zonesInfo.r3_angle0 &&
                      x<zonesInfo.rimX;
              },
              make:0,
              attampt:0,
              idx:9
          },
          {//左45三分
              color: null_color,
              area: function (x, y) {
                  return ((distanceBetweenPoints(x, y, zonesInfo.rimX, zonesInfo.rimY) > zonesInfo.r3) && (y > 2.99 * scale + borderXY) ||
                      (x < borderXY + 0.9 * scale + borderXY / 2) ||
                      (x > borderXY + 14.1 * scale - borderXY / 2)) &&
                      y>zonesInfo.r3_angle0 &&
                      angleRim_XY(x,y,zonesInfo.rimX,zonesInfo.rimY)<zonesInfo.r3_angle1 &&
                      x<=zonesInfo.rimX;
              },
              make:0,
              attampt:0,
              idx:10
          },
          {//弧顶三分
              color: null_color,
              area: function (x, y) {
                  return ((distanceBetweenPoints(x, y, zonesInfo.rimX, zonesInfo.rimY) > zonesInfo.r3) && (y > 2.99 * scale + borderXY) ||
                      (x < borderXY + 0.9 * scale + borderXY / 2) ||
                      (x > borderXY + 14.1 * scale - borderXY / 2)) &&
                      y>zonesInfo.r3_angle0 &&
                      angleRim_XY(x,y,zonesInfo.rimX,zonesInfo.rimY)>=zonesInfo.r3_angle1;
              },
              make:0,
              attampt:0,
              idx:11
          },            
          {//右45三分
              color: null_color,
              area: function (x, y) {
                  return ((distanceBetweenPoints(x, y, zonesInfo.rimX, zonesInfo.rimY) > zonesInfo.r3) && (y > 2.99 * scale + borderXY) ||
                      (x < borderXY + 0.9 * scale + borderXY / 2) ||
                      (x > borderXY + 14.1 * scale - borderXY / 2)) &&
                      y>zonesInfo.r3_angle0 &&
                      angleRim_XY(x,y,zonesInfo.rimX,zonesInfo.rimY)<zonesInfo.r3_angle1 &&
                      x>=zonesInfo.rimX;
              },
              make:0,
              attampt:0,
              idx:12
          },
          {//右底角三分
              color: null_color,
              area: function (x, y) {
                  return ((distanceBetweenPoints(x, y, zonesInfo.rimX, zonesInfo.rimY) > zonesInfo.r3) && (y > 2.99 * scale + borderXY) ||
                      (x < borderXY + 0.9 * scale + borderXY / 2) ||
                      (x > borderXY + 14.1 * scale - borderXY / 2)) &&
                      y<=zonesInfo.r3_angle0 &&
                      x>zonesInfo.rimX;
              },
              make:0,
              attampt:0,
              idx:13
          },
      ];
  
      for(let i=0;i<shootingInfo.length;i++){
          nxny2Zones(zones,canvas,ctx,shootingInfo[i].nx,shootingInfo[i].ny,shootingInfo[i].score,
                  null_color,
                  lower_color,
                  average_color,
                  high_color,
                  lower_percent,
                  high_percent);
      }
      
      drawZones(zones,canvas,ctx);
  
      drawZonesLine(canvas,ctx,zonesInfo);  
  
      function drawMainLine(color='white'){
          ctx.strokeStyle=color;
          ctx.lineWidth=borderXY;
          //边框
          ctx.beginPath();
          let x1,y1,x2,y2,x3,y3,x4,y4;
          x1=0+borderXY/2;
          y1=0;
          x2=x1;
          y2=borderXY+13.95*scale+borderXY;
          ctx.moveTo(x1,y1);
          ctx.lineTo(x2,y2);
  
          y2-=borderXY/2;
          x3=borderXY+15*scale+borderXY;
          y3=y2;
          ctx.moveTo(x2,y2);
          ctx.lineTo(x3,y3);
  
          x3-=borderXY/2;
          x4=x3;
          y4=0;
          ctx.moveTo(x3,y3);
          ctx.lineTo(x4,y4);
  
          y4+=borderXY/2;
          y1=y4;
          ctx.moveTo(x4,y4);
          ctx.lineTo(x1,y1);
          ctx.stroke();
  
          //三分线
          //除去底角三分不画圆
          let tanValue = 1.425 / 6.6;
          let radians = Math.atan(tanValue);
  
          ctx.beginPath();
          ctx.arc(x,y,radius,radians,Math.PI-radians);
  
          //底角三分的竖直线
          //左
          x1=borderXY+0.9*scale+borderXY/2;
          y1=0;
          x2=x1;
          y2=2.99*scale+borderXY;
          ctx.moveTo(x1, y1);
          ctx.lineTo(x2, y2);
          //右
          x1=borderXY+14.1*scale-borderXY/2;
          x2=x1;
          ctx.moveTo(x1, y1);
          ctx.lineTo(x2, y2);
          ctx.stroke();
  
          //左罚球线
          x1=borderXY+5.05*scale+borderXY/2;
          y1=0;
          x2=x1;
          // y2=borderXY+5.65*scale+3*borderXY-borderXY/2;
          y2=borderXY+5.65*scale+3*0.05*scale-borderXY/2;
  
          ctx.moveTo(x1,y1);
          ctx.lineTo(x2,y2);
          //中罚球线
          x1=borderXY+9.95*scale-borderXY/2;
          y1=y2;
          ctx.lineTo(x1,y1);
          let freeX=(x1+x2)/2;
          let freeY=y1;
          let freeR=1.8*scale-borderXY/2;
  
          //右罚球线
          x2=x1;
          y2=0;
          y1+=borderXY/2;
          ctx.moveTo(x1,y1);
          ctx.lineTo(x2,y2);
          ctx.stroke();
          
          //罚球线靠后半圆
          ctx.beginPath();
          ctx.arc(freeX,freeY,freeR,2*Math.PI,Math.PI);
          ctx.stroke();
          // //油漆区半圆
          // ctx.beginPath();
          // let sm_radius=1.25*scale+borderXY/2;
          // ctx.arc(x,y,sm_radius,2*Math.PI,Math.PI);
          // ctx.stroke();
          //球筐
          ctx.beginPath();
          let smm_radius=0.2*scale+borderXY/2;
          ctx.arc(x,y,smm_radius,0,2*Math.PI);
          ctx.stroke();
          //篮板
          ctx.beginPath();
          let basketY=borderXY+1.2*scale-borderXY/2;
          ctx.moveTo(x-0.9*scale,basketY);
          ctx.lineTo(x+0.9*scale,basketY);
          ctx.stroke();
          //中场圆
          ctx.beginPath();
          let c_radius=1.75*scale+borderXY/2;
          ctx.arc(x,y3,c_radius,Math.PI,2*Math.PI);
          ctx.stroke();
          //球筐连接篮板
          ctx.beginPath();
          ctx.lineWidth=10;
          ctx.moveTo(x,basketY);
          ctx.lineTo(x,basketY+0.2*scale);
          ctx.stroke();
      }
      
      drawMainLine(drawMainLineColor); 
  
    
  };
  
  // let shootingInfo=[
  //     {nx:0.5,ny:0.5,score:true},
  //     {nx:0.5,ny:0.5,score:true},
  //     {nx:0.2,ny:0.5,score:false},
  //     {nx:0.5,ny:0.2,score:true},
  //     {nx:0.1,ny:0.2,score:true},
  //     {nx:0.1,ny:0.23,score:true},
  //     {nx:0.7,ny:0.2,score:true},
  //     {nx:0.65,ny:0.32,score:false},
  //     {nx:0.45,ny:0.78,score:true},
  //     {nx:0.57,ny:0.68,score:false},
  //     {nx:0.5,ny:0.88,score:false}
  
  // ];
  
  // drawCourt(canvas,ctx,shootingInfo,600,10,'white',
  //                 null_color='rgba(180, 180, 180, 1)',
  //                 lower_color='rgba(89, 164, 222, 1)',
  //                 average_color='rgba(247, 214, 73,1)',
  //                 high_color='rgba(220, 117, 58, 1)',
  //                 lower_percent=0.3,
  //                 high_percent=0.5);