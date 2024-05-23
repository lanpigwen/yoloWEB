function drawLineGraph(data, svgId='LineSvg') {
    // 计算最大的命中数和不命中数
    const maxHits = d3.max(data, d => d.hits);
    const maxMisses = d3.max(data, d => d.misses);

    // 使用较大的那个值作为Y轴的最大值，并向上取十的倍数
    const maxY = Math.ceil(Math.max(maxHits, maxMisses) / 10) * 10;
    const svgDoc = document.getElementById(svgId);
    const margin = {top: 60, right: 20, bottom: 50, left: 50},
        width = svgDoc.getAttribute("width") - margin.left - margin.right,
        height = svgDoc.getAttribute("height") - margin.top - margin.bottom;

    const svg = d3.select("#"+svgId)
        .attr("width", width + margin.left + margin.right)
        .attr("height", height + margin.top + margin.bottom)
        .append("g")
        .attr("transform", `translate(${margin.left},${margin.top})`);

    const x = d3.scaleLinear()
        .domain([1, data.length])
        .range([0, width]);

    const y = d3.scaleLinear()
        .domain([0, maxY])  // 将Y轴的域设置为从0到最大值
        .range([height, 0]);

    const xAxis = d3.axisBottom(x)
        .tickValues(d3.range(1, data.length+1, 1+(data.length>12?1:0)))
        .tickFormat(d3.format('d'));

    const yAxis = d3.axisLeft(y).ticks(5).tickSizeInner(-width);

    // 绘制X轴和Y轴
    svg.append("g")
        .attr("class", "axis")
        .attr("transform", `translate(0,${height})`)
        .call(xAxis)
        .selectAll("text")  // 选择所有文本元素
        .style("font-size", "16px")
        .style("font-weight", "bold");  // 设置字体大小;

    svg.append("g")
        .attr("class", "axis")
        .call(yAxis)
        .selectAll("text")  // 选择所有文本元素
        .style("font-size", "16px")


    // 添加Y轴的标题
    svg.append("text")
    .attr("transform", "rotate(-90)")  // 将文字旋转90度
    .attr("y", 0 - margin.left)  // 定位到Y轴的左侧
    .attr("x", 0 - (height / 2))  // 定位到Y轴的中心位置
    .attr("dy", "1em")  // 微调位置
    .style("text-anchor", "middle")  // 文本对齐方式为中间对齐
    .text("投篮数");  // Y轴标题内容


    // 折线函数
    const line = d3.line()
        .x(d => x(d.day))
        .y(d => y(d.hits))
        // .curve(d3.curveCatmullRom.alpha(0.01));

    const line2 = d3.line()
        .x(d => x(d.day))
        .y(d => y(d.misses))
        // .curve(d3.curveCatmullRom.alpha(0.1));

    // 绘制折线
    svg.append("path")
        .datum(data)
        .attr("class", "line make")
        .attr("d", line);

    svg.append("path")
        .datum(data)
        .attr("class", "line miss")
        .attr("d", line2);

    // // 添加标题
    // svg.append("text")
    //     .attr("x", width / 2)
    //     .attr("y", 0 - (margin.top / 2))
    //     .attr("text-anchor", "middle")
    //     .style("font-size", "24px")
    //     .text("投篮记录");

    // 添加图例
    const el_make = document.querySelectorAll('.make')[0];
    const style1 = window.getComputedStyle(el_make);
    const el_miss = document.querySelectorAll('.miss')[0];
    const style2 = window.getComputedStyle(el_miss);
    const make_color = style1.stroke || style1.getPropertyValue('stroke');
    const miss_color = style2.stroke || style2.getPropertyValue('stroke');

    const colors = { '投丢数': miss_color, '命中数': make_color, };
    
    const legend = svg.selectAll(".legend")
        .data(Object.entries(colors))
        .enter().append("g")
        .attr("class", "legend")
        .attr("transform", (d, i) => `translate(${-i * 80},-${20})`);  // 将图例定位到图表的顶部右侧

    legend.append("rect")
        .attr("x", width - 80)
        .attr("width", 18)
        .attr("height", 18)
        .style("fill", d => d[1]);

    legend.append("text")
        .attr("x", width - 9)
        .attr("y", 9)
        .attr("dy", ".35em")
        .style("text-anchor", "end")
        .text(d => d[0]);

}

