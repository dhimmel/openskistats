// Snowflake generator
const snowContainer = document.getElementById('snow-container');
let snowflakeInterval;

function createSnowflake() {
  const snowflake = document.createElement('div');
  snowflake.classList.add('snowflake');
  snowflake.textContent = '❄';
  snowflake.style.left = `${Math.random() * 100}vw`;
  const baseSize = Math.random() * 0.7;
  const fontSize = baseSize * Math.sqrt(window.innerWidth) + 1
  snowflake.style.fontSize = `${fontSize}px`;
  snowflake.style.animationDuration = `${Math.random() * 30 + 3}s`;
  // snowflake.style.animationDelay = `${Math.random() * 2}s`;
  snowContainer.appendChild(snowflake);

  // Remove snowflake after animation
  setTimeout(() => snowflake.remove(), 10000);
}

// Generate snowflakes continuously
function adjustSnowflakeFrequency() {
  const width = window.innerWidth;

  if (snowflakeInterval) {
    clearInterval(snowflakeInterval);
  }

  // Set the interval based on width
  let intervalTime;
  if (width > 800) {
    intervalTime = 400; // Frequent for larger screens
  } else {
    intervalTime = 600; // Less frequent for smaller screens
  }

  snowflakeInterval = setInterval(createSnowflake, intervalTime);
}

// Initial adjustment
adjustSnowflakeFrequency();

// Adjust frequency on window resize
window.addEventListener('resize', adjustSnowflakeFrequency);
