// MNIST Digit Recognizer — frontend logic
// ----------------------------------------
// Provides: drawing canvas, file upload, real-time prediction, probability bars.

const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const clearBtn = document.getElementById("clearBtn");
const predictBtn = document.getElementById("predictBtn");
const fileInput = document.getElementById("fileInput");
const realtimeToggle = document.getElementById("realtimeToggle");
const digitEl = document.getElementById("digit");
const confidenceEl = document.getElementById("confidence");
const barsEl = document.getElementById("bars");

// --- Init canvas with a white background ----------------------------------
function clearCanvas() {
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  digitEl.textContent = "–";
  confidenceEl.textContent = "—";
  renderBars(new Array(10).fill(0));
}

// Brush setup
ctx.lineCap = "round";
ctx.lineJoin = "round";
ctx.lineWidth = 18;
ctx.strokeStyle = "#000000";
clearCanvas();

// --- Drawing handlers (mouse + touch) -------------------------------------
let drawing = false;
let lastX = 0;
let lastY = 0;
let predictTimer = null;

function getPos(evt) {
  const rect = canvas.getBoundingClientRect();
  const scaleX = canvas.width / rect.width;
  const scaleY = canvas.height / rect.height;
  const point = evt.touches ? evt.touches[0] : evt;
  return {
    x: (point.clientX - rect.left) * scaleX,
    y: (point.clientY - rect.top) * scaleY,
  };
}

function startDraw(evt) {
  evt.preventDefault();
  drawing = true;
  const { x, y } = getPos(evt);
  lastX = x;
  lastY = y;
  ctx.beginPath();
  ctx.arc(x, y, ctx.lineWidth / 2, 0, Math.PI * 2);
  ctx.fillStyle = "#000000";
  ctx.fill();
}

function moveDraw(evt) {
  if (!drawing) return;
  evt.preventDefault();
  const { x, y } = getPos(evt);
  ctx.beginPath();
  ctx.moveTo(lastX, lastY);
  ctx.lineTo(x, y);
  ctx.stroke();
  lastX = x;
  lastY = y;
}

function endDraw(evt) {
  if (!drawing) return;
  evt.preventDefault();
  drawing = false;
  if (realtimeToggle.checked) schedulePredict();
}

canvas.addEventListener("mousedown", startDraw);
canvas.addEventListener("mousemove", (e) => {
  moveDraw(e);
  if (drawing && realtimeToggle.checked) schedulePredict(150);
});
canvas.addEventListener("mouseup", endDraw);
canvas.addEventListener("mouseleave", endDraw);
canvas.addEventListener("touchstart", startDraw, { passive: false });
canvas.addEventListener("touchmove", (e) => {
  moveDraw(e);
  if (realtimeToggle.checked) schedulePredict(150);
}, { passive: false });
canvas.addEventListener("touchend", endDraw);

// --- Buttons ---------------------------------------------------------------
clearBtn.addEventListener("click", clearCanvas);
predictBtn.addEventListener("click", () => predict());

// --- File upload -----------------------------------------------------------
fileInput.addEventListener("change", async (e) => {
  const file = e.target.files[0];
  if (!file) return;
  const url = URL.createObjectURL(file);
  const img = new Image();
  img.onload = () => {
    // Draw uploaded image into the canvas, fitting it.
    clearCanvas();
    const scale = Math.min(canvas.width / img.width, canvas.height / img.height);
    const w = img.width * scale;
    const h = img.height * scale;
    ctx.drawImage(img, (canvas.width - w) / 2, (canvas.height - h) / 2, w, h);
    URL.revokeObjectURL(url);
    predict();
  };
  img.src = url;
});

// --- Prediction ------------------------------------------------------------
function schedulePredict(delay = 250) {
  clearTimeout(predictTimer);
  predictTimer = setTimeout(() => predict(), delay);
}

async function predict() {
  const dataUrl = canvas.toDataURL("image/png");
  try {
    const res = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image: dataUrl }),
    });
    const data = await res.json();
    if (data.error) {
      digitEl.textContent = "!";
      confidenceEl.textContent = data.error;
      return;
    }
    digitEl.textContent = data.digit;
    confidenceEl.textContent = `${(data.confidence * 100).toFixed(1)}%`;
    renderBars(data.probabilities, data.digit);
  } catch (err) {
    digitEl.textContent = "!";
    confidenceEl.textContent = "Network error";
  }
}

// --- Probability bars ------------------------------------------------------
function renderBars(probs, topDigit = -1) {
  barsEl.innerHTML = "";
  for (let i = 0; i < 10; i++) {
    const row = document.createElement("div");
    row.className = "bar-row" + (i === topDigit ? " top" : "");
    row.innerHTML = `
      <span class="label">${i}</span>
      <div class="bar-track"><div class="bar-fill" style="width:${(probs[i] * 100).toFixed(1)}%"></div></div>
      <span class="value">${(probs[i] * 100).toFixed(1)}%</span>
    `;
    barsEl.appendChild(row);
  }
}

renderBars(new Array(10).fill(0));
