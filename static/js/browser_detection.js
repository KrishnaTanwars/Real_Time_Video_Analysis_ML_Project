(function () {
  const root = document.querySelector("[data-detector-mode]");
  if (!root) return;

  const mode = root.dataset.detectorMode;
  const video = root.querySelector('[data-role="video"]');
  const canvas = root.querySelector('[data-role="canvas"]');
  const ctx = canvas.getContext("2d");

  const startButton = root.querySelector('[data-action="start"]');
  const stopButton = root.querySelector('[data-action="stop"]');
  const cameraSelect = root.querySelector('[data-role="camera"]');
  const fpsInput = root.querySelector('[data-role="fps"]');
  const fpsValue = root.querySelector('[data-role="fps-value"]');
  const profileSelect = root.querySelector('[data-role="profile"]');

  const statusEl = root.querySelector('[data-role="status"]');
  const latencyEl = root.querySelector('[data-role="latency"]');
  const metaEl = root.querySelector('[data-role="meta"]');

  const captureCanvas = document.createElement("canvas");
  const captureCtx = captureCanvas.getContext("2d");

  // 🔥 FIXED SMALL RESOLUTION
  captureCanvas.width = 416;
  captureCanvas.height = 416;

  const outputImage = new Image();
  outputImage.onload = function () {
    if (!outputImage.width || !outputImage.height) return;

    if (canvas.width !== outputImage.width || canvas.height !== outputImage.height) {
      canvas.width = outputImage.width;
      canvas.height = outputImage.height;
    }

    ctx.drawImage(outputImage, 0, 0, canvas.width, canvas.height);
  };

  const clientId = localStorage.getItem("detector_client_id") ||
    `client_${Math.random().toString(36).slice(2)}_${Date.now()}`;

  localStorage.setItem("detector_client_id", clientId);

  let stream = null;
  let timerId = null;
  let inFlight = false;
  let currentFps = Math.min(8, Number(fpsInput.value || 4));

  function setStatus(text) {
    statusEl.textContent = text;
  }

  function applyButtonState(running) {
    startButton.disabled = running;
    stopButton.disabled = !running;
    cameraSelect.disabled = running;
    if (profileSelect) profileSelect.disabled = running;
  }

  function buildConstraints() {
    return {
      audio: false,
      video: {
        facingMode: "user",
        width: { ideal: 640 },
        height: { ideal: 480 },
      },
    };
  }

  function stopLoop() {
    if (timerId) {
      clearInterval(timerId);
      timerId = null;
    }
  }

  function startLoop() {
    stopLoop();
    const intervalMs = Math.max(1000 / currentFps, 120);
    timerId = setInterval(processFrame, intervalMs);
  }

  function stopCamera() {
    stopLoop();
    if (stream) {
      stream.getTracks().forEach((track) => track.stop());
      stream = null;
    }
    video.srcObject = null;
    applyButtonState(false);
    setStatus("Camera stopped");
  }

  async function startCamera() {
    try {
      stopCamera();
      setStatus("Starting camera...");

      stream = await navigator.mediaDevices.getUserMedia(buildConstraints());
      video.srcObject = stream;
      await video.play();

      applyButtonState(true);
      setStatus("Camera running");
      startLoop();
    } catch (error) {
      setStatus(`Camera error: ${error.message}`);
    }
  }

  async function processFrame() {
    if (!stream || inFlight) return;
    if (!video.videoWidth) return;

    inFlight = true;

    try {
      captureCtx.drawImage(video, 0, 0, 416, 416);

      // 🔥 LOWER JPEG QUALITY
      const frameDataUrl = captureCanvas.toDataURL("image/jpeg", 0.55);

      const startTime = performance.now();

      const response = await fetch(`/api/detect/${mode}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          client_id: clientId,
          profile: profileSelect ? profileSelect.value : "balanced",
          frame: frameDataUrl,
        }),
      });

      const latency = Math.round(performance.now() - startTime);
      latencyEl.textContent = `Latency: ${latency} ms`;

      const payload = await response.json();
      if (!response.ok || !payload.ok) {
        throw new Error(payload.error || `HTTP ${response.status}`);
      }

      outputImage.src = payload.image;
      metaEl.textContent = Object.entries(payload.meta || {})
        .map(([k, v]) => `${k}: ${v}`)
        .join(" | ");

      setStatus("Processing live frames");
    } catch (error) {
      setStatus(`Detection error: ${error.message}`);
    } finally {
      inFlight = false;
    }
  }

  function updateFps() {
    currentFps = Math.min(8, Math.max(2, Number(fpsInput.value || 4)));
    fpsValue.textContent = `${currentFps} fps`;
    if (stream) startLoop();
  }

  startButton.addEventListener("click", startCamera);
  stopButton.addEventListener("click", stopCamera);
  fpsInput.addEventListener("input", updateFps);

  updateFps();
})();
