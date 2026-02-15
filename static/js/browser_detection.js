(function () {
  const root = document.querySelector("[data-detector-mode]");
  if (!root) {
    return;
  }

  const mode = root.dataset.detectorMode;
  const video = root.querySelector('[data-role="video"]');
  const canvas = root.querySelector('[data-role="canvas"]');
  const ctx = canvas.getContext("2d");

  const startButton = root.querySelector('[data-action="start"]');
  const stopButton = root.querySelector('[data-action="stop"]');
  const cameraSelect = root.querySelector('[data-role="camera"]');
  const fpsInput = root.querySelector('[data-role="fps"]');
  const fpsValue = root.querySelector('[data-role="fps-value"]');

  const statusEl = root.querySelector('[data-role="status"]');
  const latencyEl = root.querySelector('[data-role="latency"]');
  const metaEl = root.querySelector('[data-role="meta"]');

  const captureCanvas = document.createElement("canvas");
  const captureCtx = captureCanvas.getContext("2d");

  const outputImage = new Image();
  outputImage.onload = function () {
    if (!outputImage.width || !outputImage.height) {
      return;
    }

    if (canvas.width !== outputImage.width || canvas.height !== outputImage.height) {
      canvas.width = outputImage.width;
      canvas.height = outputImage.height;
    }

    ctx.drawImage(outputImage, 0, 0, canvas.width, canvas.height);
  };

  const clientId = ensureClientId();

  let stream = null;
  let timerId = null;
  let inFlight = false;
  let currentFps = Number(fpsInput.value || 6);
  let consecutiveFailures = 0;

  function ensureClientId() {
    const key = "detector_client_id";
    const existing = window.localStorage.getItem(key);
    if (existing) {
      return existing;
    }

    const generated = `client_${Math.random().toString(36).slice(2)}_${Date.now()}`;
    window.localStorage.setItem(key, generated);
    return generated;
  }

  function setStatus(text) {
    statusEl.textContent = text;
  }

  function updateMeta(meta) {
    if (!meta || typeof meta !== "object") {
      metaEl.textContent = "";
      return;
    }

    const entries = Object.entries(meta).filter(([_, value]) => value !== null && value !== undefined);
    metaEl.textContent = entries.map(([k, v]) => `${k}: ${v}`).join(" | ");
  }

  function applyButtonState(running) {
    startButton.disabled = running;
    stopButton.disabled = !running;
    cameraSelect.disabled = running;
  }

  async function listCameras() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) {
      return;
    }

    const devices = await navigator.mediaDevices.enumerateDevices();
    const cameras = devices.filter((device) => device.kind === "videoinput");

    cameraSelect.innerHTML = "";

    cameras.forEach((camera, index) => {
      const option = document.createElement("option");
      option.value = camera.deviceId;
      option.textContent = camera.label || `Camera ${index + 1}`;
      cameraSelect.appendChild(option);
    });

    if (!cameras.length) {
      const option = document.createElement("option");
      option.value = "";
      option.textContent = "No camera found";
      cameraSelect.appendChild(option);
      cameraSelect.disabled = true;
    }
  }

  function buildConstraints() {
    const selected = cameraSelect.value;
    if (selected) {
      return {
        audio: false,
        video: {
          deviceId: { exact: selected },
          width: { ideal: 960 },
          height: { ideal: 540 },
        },
      };
    }

    return {
      audio: false,
      video: {
        facingMode: "user",
        width: { ideal: 960 },
        height: { ideal: 540 },
      },
    };
  }

  function stopLoop() {
    if (timerId) {
      window.clearInterval(timerId);
      timerId = null;
    }
  }

  function startLoop() {
    stopLoop();

    const intervalMs = Math.max(1000 / currentFps, 80);
    timerId = window.setInterval(processFrame, intervalMs);
  }

  function stopCamera() {
    stopLoop();

    if (stream) {
      stream.getTracks().forEach((track) => track.stop());
      stream = null;
    }

    video.srcObject = null;
    inFlight = false;
    applyButtonState(false);
    setStatus("Camera stopped");
  }

  async function startCamera() {
    try {
      stopCamera();
      setStatus("Requesting camera access...");

      stream = await navigator.mediaDevices.getUserMedia(buildConstraints());
      video.srcObject = stream;
      await video.play();

      captureCanvas.width = video.videoWidth || 960;
      captureCanvas.height = video.videoHeight || 540;

      applyButtonState(true);
      setStatus("Camera running");
      consecutiveFailures = 0;
      startLoop();

      await listCameras();
    } catch (error) {
      setStatus(`Camera error: ${error.message || error}`);
      stopCamera();
    }
  }

  async function processFrame() {
    if (!stream || inFlight) {
      return;
    }

    if (!video.videoWidth || !video.videoHeight) {
      return;
    }

    inFlight = true;

    try {
      captureCtx.drawImage(video, 0, 0, captureCanvas.width, captureCanvas.height);
      const frameDataUrl = captureCanvas.toDataURL("image/jpeg", 0.72);

      const startedAt = performance.now();
      const response = await fetch(`/api/detect/${mode}`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          client_id: clientId,
          frame: frameDataUrl,
        }),
      });

      const elapsed = Math.round(performance.now() - startedAt);
      latencyEl.textContent = `Latency: ${elapsed} ms`;

      const payload = await response.json();
      if (!response.ok || !payload.ok) {
        throw new Error(payload.error || `HTTP ${response.status}`);
      }

      outputImage.src = payload.image;
      updateMeta(payload.meta);
      setStatus("Processing live frames");
      consecutiveFailures = 0;
    } catch (error) {
      consecutiveFailures += 1;
      setStatus(`Detection error: ${error.message || error}`);

      if (consecutiveFailures >= 5) {
        stopCamera();
      }
    } finally {
      inFlight = false;
    }
  }

  function updateFps() {
    const nextFps = Number(fpsInput.value || 6);
    currentFps = Math.min(15, Math.max(2, nextFps));
    fpsValue.textContent = `${currentFps} fps`;

    if (stream) {
      startLoop();
    }
  }

  startButton.addEventListener("click", startCamera);
  stopButton.addEventListener("click", stopCamera);

  cameraSelect.addEventListener("change", function () {
    if (stream) {
      startCamera();
    }
  });

  fpsInput.addEventListener("input", updateFps);

  window.addEventListener("beforeunload", stopCamera);

  updateFps();
  listCameras().then(startCamera).catch((error) => setStatus(`Init error: ${error.message || error}`));
})();
