// Function to create a random progress circle
function createRandomProgressCircle(index) {
    const progressCircle = document.createElement('div');
    progressCircle.classList.add('progress-circle');
    progressCircle.style.left = `${Math.random() * (window.innerWidth - 100)}px`;
    progressCircle.style.top = `${Math.random() * (window.innerHeight - 100)}px`;
    progressCircle.style.display = 'block'; // Ensure it's displayed
    let fontSize=50;
    progressCircle.innerHTML = `
    <svg>
        <circle stroke="var(--inactive-color)" />
        <circle stroke="var(--color)" class="progress-value" />
        <circle class="background-circle" />
        <text x="-50%" y="-50%" dominant-baseline="middle" text-anchor="middle" font-size="${fontSize}px" transform="rotate(-90) scale(1, -1)">${index}</text>
    </svg>
    `;

    document.body.appendChild(progressCircle);
    animateProgress(progressCircle);
    }

    // Function to animate progress
    function animateProgress(progressCircle) {
    let percent = 100;
    const progressBar = progressCircle.querySelector('.progress-value');

    const interval = setInterval(() => {
        percent -= 1;
        progressBar.style.strokeDasharray = `calc(2 * 3.1415 * var(--r) * ${percent} / 100), 1000`;
        if (percent <= 0) {
        clearInterval(interval);
        progressCircle.remove();
        }
    }, 40);
    }
    for (let i = 1; i <= 4; i++) {
    createRandomProgressCircle(i);
    }
    // Generate 4 random progress circles every 5 seconds
    setInterval(() => {
    for (let i = 1; i <= 4; i++) {
        createRandomProgressCircle(i);
    }
    }, 5000);