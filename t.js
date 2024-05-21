var zones = [
    {
        color: 'rgba(255, 0, 0, 0.9)',
        area: function (x, y) { return distanceBetweenPoints(x, y, rimX, rimY) < r1; },
        efficiency: 40
    },
    {
        color: 'rgba(0, 255, 0, 0.9)',
        area: function (x, y) {
            return (distanceBetweenPoints(x, y, rimX, rimY) < r2) &&
                (distanceBetweenPoints(x, y, rimX, rimY) > r1);
        },
        efficiency: 60
    },
    {
        color: 'rgba(0, 0, 255, 0.9)',
        area: function (x, y) {
            return (distanceBetweenPoints(x, y, rimX, rimY) < r3) &&
                (distanceBetweenPoints(x, y, rimX, rimY) > r2) &&
                (x > borderXY + 0.9 * scale + borderXY / 2) &&
                (x < borderXY + 14.1 * scale - borderXY / 2);
        },
        efficiency: 60
    },
    {
        color: 'rgba(255, 255, 0, 0.9)',
        area: function (x, y) {
            return (distanceBetweenPoints(x, y, rimX, rimY) > r3) && (y > 2.99 * scale + borderXY) ||
                (x < borderXY + 0.9 * scale + borderXY / 2) ||
                (x > borderXY + 14.1 * scale - borderXY / 2);
        },
        efficiency: 40
    },
];