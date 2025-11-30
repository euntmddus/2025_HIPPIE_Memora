// js/viewer2d.js
// NIfTI 로딩, 2D MPR 렌더링, 슬라이스/밝기 조작

let mriOwnerPatientIndex = null;

const AXIAL_ZOOM = 1.15;
const SAGITTAL_ZOOM = 1.15;
const CORONAL_ZOOM = 1.15;

let axialTransform = {
  dx: 0, dy: 0, scale: 1,
  minX: 0, minY: 0, srcW: 1, srcH: 1
};

function renderViewer() {
  const currentMri = mri[idx];
  const hasImage = currentMri && (currentMri.src || currentMri.file);

  const isOwner =
    mriOwnerPatientIndex !== null &&
    typeof currentPatientIndex !== 'undefined' &&
    mriOwnerPatientIndex === currentPatientIndex;

  if (defaultImageTextEl) defaultImageTextEl.style.display = 'none';
  if (viewerImgEl) viewerImgEl.style.display = 'none';
  if (mprContainerEl) mprContainerEl.style.display = 'none';
  if (viewer3DContainerEl) viewer3DContainerEl.style.display = 'none';

  const needOwnerCheck = !!imageData;

  if (!hasImage || (needOwnerCheck && !isOwner)) {
    if (defaultImageTextEl) {
      defaultImageTextEl.textContent = "No Image";
      defaultImageTextEl.style.display = 'flex';
    }
    return;
  }

  if (currentViewMode === '2D') {
    if (mprContainerEl) mprContainerEl.style.display = 'block';

    if (currentMri.file) {
      if (!imageData || !dims) {
        loadNiftiFile(currentMri.file).catch(err => {
          console.error('로컬 NIfTI 로딩 오류:', err);
        });
      } else {
        renderAll();
      }
    } else {
      if (imageData && dims) {
        renderAll();
      }
      else {
        const name = currentMri.title || currentMri.src || '';
        if (
          name.endsWith('.nii') || name.endsWith('.nii.gz') ||
          currentMri.src.endsWith('.nii') || currentMri.src.endsWith('.nii.gz')
        ) {
          loadNiftiFromURL(currentMri.src).catch(err => {
            console.error('원격 NIfTI 로딩 오류:', err);
          });
        } else {
          ['axialCanvas', 'sagittalCanvas', 'coronalCanvas'].forEach(id => {
            const c = document.getElementById(id);
            if (!c) return;
            const ctx = c.getContext('2d', { willReadFrequently: true });
            ctx.fillStyle = '#000';
            ctx.fillRect(0, 0, c.width || 256, c.height || 256);
          });
        }
      }
    }
  }
}

async function loadNiftiFile(file) {
  mriOwnerPatientIndex = currentPatientIndex;

  const url = URL.createObjectURL(file);
  await loadNiftiFromURL(url);
}

async function loadNiftiFromURL(url) {
  mriOwnerPatientIndex = currentPatientIndex;

  try {
    const res = await fetch(url);
    const raw = new Uint8Array(await res.arrayBuffer());

    let buf = raw;
    if (raw[0] === 0x1f && raw[1] === 0x8b && typeof pako !== 'undefined') {
      buf = pako.inflate(raw);
    }

    const header = parseNiftiHeader(buf);
    dims = [header.dim[1], header.dim[2], header.dim[3]];
    const offset = Math.round(header.vox_offset || 352);
    imageData = extractImageData(buf, offset, header);

    currentSlice = {
      axial: Math.floor(dims[2] / 2),
      sagittal: Math.floor(dims[0] / 2),
      coronal: Math.floor(dims[1] / 2)
    };

    renderAll();
    console.log('NIfTI 로드 완료:', dims);
  } catch (err) {
    console.error(err);
  }
}

function setupInteraction() {
  let isDragging = false;
  const panels = [
    { id: 'axialCanvas', view: 'axial' },
    { id: 'sagittalCanvas', view: 'sagittal' },
    { id: 'coronalCanvas', view: 'coronal' }
  ];

  panels.forEach(p => {
    const canvas = document.getElementById(p.id);
    if (!canvas) return;

    const updateSlice = (e) => {
      if (!dims) return;
      const rect = canvas.getBoundingClientRect();
      const xPix = (e.clientX - rect.left) * (canvas.width / rect.width);
      const yPix = (e.clientY - rect.top) * (canvas.height / rect.height);

      let x = 0;
      let y = 0;

      if (p.view === 'axial') {
        const { dx, dy, scale, minX, minY, srcW, srcH } = axialTransform;
        const lx = (xPix - dx) / scale;
        const ly = (yPix - dy) / scale;

        // 크롭 영역 밖을 클릭하면 무시
        if (lx < 0 || lx >= srcW || ly < 0 || ly >= srcH) return;

        x = Math.floor(minX + lx);
        y = Math.floor(minY + ly);
      } else {
        // sagittal / coronal 기존 로직 유지
        x = Math.floor(xPix);
        y = Math.floor(yPix);
      }

      if (x < 0 || x >= dims[0] || y < 0 || y >= dims[1]) return;

      if (p.view === 'axial') {
        currentSlice.sagittal = x;
        currentSlice.coronal = dims[1] - 1 - y;
      } else if (p.view === 'sagittal') {
        currentSlice.coronal = dims[1] - 1 - x;
        currentSlice.axial = dims[2] - 1 - y;
      } else if (p.view === 'coronal') {
        currentSlice.sagittal = dims[0] - 1 - x;
        currentSlice.axial = dims[2] - 1 - y;
      }
      renderAll();
    };


    canvas.addEventListener('mousedown', e => { isDragging = true; updateSlice(e); });
    canvas.addEventListener('mousemove', e => { if (isDragging) updateSlice(e); });
    canvas.addEventListener('mouseup', () => { isDragging = false; });
    canvas.addEventListener('mouseleave', () => { isDragging = false; });
  });
}

async function loadMaskFromBase64(base64String) {
  try {
    const binaryString = window.atob(base64String);
    const len = binaryString.length;
    const bytes = new Uint8Array(len);
    for (let i = 0; i < len; i++) bytes[i] = binaryString.charCodeAt(i);

    let niftiData = bytes;
    if (bytes[0] === 0x1f && bytes[1] === 0x8b && typeof pako !== 'undefined') {
      niftiData = pako.inflate(bytes);
    }

    const header = parseNiftiHeader(niftiData);
    const offset = Math.round(header.vox_offset || 352);

    const maskVoxels = header.dim[1] * header.dim[2] * header.dim[3];
    const imageVoxels = dims ? dims[0] * dims[1] * dims[2] : null;
    console.log('mask header dim=', header.dim, 'mask voxels=', maskVoxels, 'image dims=', dims, 'image voxels=', imageVoxels);

    if (!dims) {
      log('원본 MRI가 로드되어 있지 않습니다. 먼저 MRI를 로드하세요.', true);
      return;
    }

    if (maskVoxels !== imageVoxels) {
      log(`마스크 크기 불일치: mask=${maskVoxels}, image=${imageVoxels}. 오버레이를 적용할 수 없습니다.`, true);
      console.warn('마스크/이미지 크기 불일치. 서버에서 반환한 mask가 다른 스페이스일 수 있습니다.');
      return;
    }

    maskData = extractImageData(niftiData, offset, header, true);
    log('마스크 오버레이 적용됨');
    renderAll();
  } catch (err) {
    console.error(err);
    log("마스크 처리 중 오류 발생: " + err.message, true);
  }
}

function parseNiftiHeader(buf) {
  const v = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  return {
    dim: [
      v.getInt16(40, true),
      v.getInt16(42, true),
      v.getInt16(44, true),
      v.getInt16(46, true)
    ],
    datatype: v.getInt16(70, true),
    bitpix: v.getInt16(72, true),
    vox_offset: v.getFloat32(108, true)
  };
}

function extractImageData(buf, offset, header, isMask = false) {
  const v = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  const n = header.dim[1] * header.dim[2] * header.dim[3];
  const data = new Float32Array(n);
  let min = Infinity, max = -Infinity;

  for (let i = 0; i < n; i++) {
    let val = 0;
    const bytePos = offset + i * (header.bitpix / 8);
    if (bytePos >= v.byteLength) break;

    try {
      if (header.bitpix === 8) val = v.getUint8(bytePos);
      else if (header.bitpix === 16) val = v.getInt16(bytePos, true);
      else if (header.bitpix === 32) val = v.getFloat32(bytePos, true);
    } catch (e) {
      val = 0;
    }

    data[i] = val;
    if (val < min) min = val;
    if (val > max) max = val;
  }

  if (!isMask) {
    const range = max - min;
    if (range > 0 && isFinite(range)) {
      for (let i = 0; i < n; i++) {
        data[i] = ((data[i] - min) / range) * 255;
      }
    }
  } else {
    console.log(`마스크 데이터 범위: Min=${min}, Max=${max}`);
  }

  return data;
}

function renderAll() {
  renderAxial();
  renderSagittal();
  renderCoronal();
}

function renderAxial() {
  if (!dims || !imageData) return;
  const c = document.getElementById('axialCanvas');
  if (!c) return;

  const baseW = dims[0];  // X
  const baseH = dims[1];  // Y

  // 오프스크린 캔버스에 원본 slice 그림
  const off = document.createElement('canvas');
  off.width = baseW;
  off.height = baseH;
  const offCtx = off.getContext('2d', { willReadFrequently: true });

  const img = offCtx.createImageData(baseW, baseH);
  const z = currentSlice.axial;

  for (let y = 0; y < baseH; y++) {
    const flippedY = baseH - 1 - y;
    for (let x = 0; x < baseW; x++) {
      const idx = x + flippedY * baseW + z * baseW * baseH;
      if (idx < 0 || idx >= imageData.length) continue;

      const v = Math.min(255, imageData[idx] * brightness);
      const p = (y * baseW + x) * 4;

      img.data[p] = v;
      img.data[p + 1] = v;
      img.data[p + 2] = v;
      img.data[p + 3] = 255;

      if (maskData && idx < maskData.length) {
        const maskVal = Math.round(maskData[idx]);
        if (maskVal > 0) {
          if (maskVal === 1) {
            img.data[p] = 0;
            img.data[p + 1] = 255;
            img.data[p + 2] = 0;
            img.data[p + 3] = 150;
          } else {
            img.data[p] = 255;
            img.data[p + 1] = 0;
            img.data[p + 2] = 0;
            img.data[p + 3] = 150;
          }
        }
      }
    }
  }

  offCtx.putImageData(img, 0, 0);

  // 메인 캔버스를 패널 크기로 맞추기
  const panel = c.parentElement;
  let cw = baseW, ch = baseH;
  if (panel) {
    const rect = panel.getBoundingClientRect();
    cw = rect.width;
    ch = rect.height;
  }
  c.width = cw;
  c.height = ch;

  const ctx = c.getContext('2d', { willReadFrequently: true });
  ctx.clearRect(0, 0, cw, ch);
  ctx.imageSmoothingEnabled = true;

  // 패널에 맞게 + 살짝 확대
  const fitScale = Math.min(cw / baseW, ch / baseH);
  const scale = fitScale * AXIAL_ZOOM;

  const scaledW = baseW * scale;
  const scaledH = baseH * scale;
  const dx = (cw - scaledW) / 2;
  const dy = (ch - scaledH) / 2;

  ctx.drawImage(off, 0, 0, baseW, baseH, dx, dy, scaledW, scaledH);

  // 크로스헤어 좌표
  const cxVoxel = currentSlice.sagittal;              // X
  const cyVoxel = baseH - 1 - currentSlice.coronal;  // Y 뒤집힘
  const cxCanvas = dx + cxVoxel * scale;
  const cyCanvas = dy + cyVoxel * scale;

  drawCrosshair(ctx, cw, ch, cxCanvas, cyCanvas);

  const infoEl = document.getElementById('axialInfo');
  if (infoEl) infoEl.textContent = `${z + 1} / ${dims[2]}`;
}

function renderSagittal() {
  if (!dims || !imageData) return;
  const c = document.getElementById('sagittalCanvas');
  if (!c) return;

  const baseW = dims[1];  // 가로: Y
  const baseH = dims[2];  // 세로: Z

  // 오프스크린에 원본 slice 그림
  const off = document.createElement('canvas');
  off.width = baseW;
  off.height = baseH;
  const offCtx = off.getContext('2d', { willReadFrequently: true });

  const img = offCtx.createImageData(baseW, baseH);
  const x = currentSlice.sagittal;

  for (let z = 0; z < dims[2]; z++) {
    const flippedZ = dims[2] - 1 - z;
    for (let y = 0; y < dims[1]; y++) {
      const flippedY = dims[1] - 1 - y;
      const idx = x + flippedY * dims[0] + flippedZ * dims[0] * dims[1];
      if (idx < 0 || idx >= imageData.length) continue;

      const v = Math.min(255, imageData[idx] * brightness);
      const p = (z * baseW + y) * 4;

      img.data[p] = v;
      img.data[p + 1] = v;
      img.data[p + 2] = v;
      img.data[p + 3] = 255;

      if (maskData && idx < maskData.length) {
        const maskVal = Math.round(maskData[idx]);
        if (maskVal > 0) {
          if (maskVal === 1) {
            img.data[p] = 0;
            img.data[p + 1] = 255;
            img.data[p + 2] = 0;
            img.data[p + 3] = 150;
          } else {
            img.data[p] = 255;
            img.data[p + 1] = 0;
            img.data[p + 2] = 0;
            img.data[p + 3] = 150;
          }
        }
      }
    }
  }
  offCtx.putImageData(img, 0, 0);

  // 메인 캔버스를 패널 크기로 맞춤
  const panel = c.parentElement;
  let cw = baseW, ch = baseH;
  if (panel) {
    const rect = panel.getBoundingClientRect();
    cw = rect.width;
    ch = rect.height;
  }
  c.width = cw;
  c.height = ch;

  const ctx = c.getContext('2d', { willReadFrequently: true });
  ctx.clearRect(0, 0, cw, ch);
  ctx.imageSmoothingEnabled = true;

  // 패널에 맞게 + 약간 확대
  const fitScale = Math.min(cw / baseW, ch / baseH);
  const scale = fitScale * SAGITTAL_ZOOM;

  const scaledW = baseW * scale;
  const scaledH = baseH * scale;
  const dx = (cw - scaledW) / 2;
  const dy = (ch - scaledH) / 2;

  ctx.drawImage(off, 0, 0, baseW, baseH, dx, dy, scaledW, scaledH);

  // 크로스헤어 좌표 변환
  const cxVoxel = dims[1] - 1 - currentSlice.coronal; // Y
  const cyVoxel = dims[2] - 1 - currentSlice.axial;   // Z
  const cxCanvas = dx + cxVoxel * scale;
  const cyCanvas = dy + cyVoxel * scale;

  drawCrosshair(ctx, cw, ch, cxCanvas, cyCanvas);

  const infoEl = document.getElementById('sagittalInfo');
  if (infoEl) infoEl.textContent = `${x + 1} / ${dims[0]}`;
}


function renderCoronal() {
  if (!dims || !imageData) return;
  const c = document.getElementById('coronalCanvas');
  if (!c) return;

  const baseW = dims[0];  // 가로: X
  const baseH = dims[2];  // 세로: Z

  const off = document.createElement('canvas');
  off.width = baseW;
  off.height = baseH;
  const offCtx = off.getContext('2d', { willReadFrequently: true });

  const img = offCtx.createImageData(baseW, baseH);
  const y = currentSlice.coronal;

  for (let z = 0; z < dims[2]; z++) {
    const flippedZ = dims[2] - 1 - z;
    for (let x = 0; x < dims[0]; x++) {
      const flippedX = dims[0] - 1 - x;
      const idx = flippedX + y * dims[0] + flippedZ * dims[0] * dims[1];
      if (idx < 0 || idx >= imageData.length) continue;

      const v = Math.min(255, imageData[idx] * brightness);
      const p = (z * baseW + x) * 4;

      img.data[p] = v;
      img.data[p + 1] = v;
      img.data[p + 2] = v;
      img.data[p + 3] = 255;

      if (maskData && idx < maskData.length) {
        const maskVal = Math.round(maskData[idx]);
        if (maskVal > 0) {
          if (maskVal === 1) {
            img.data[p] = 0;
            img.data[p + 1] = 255;
            img.data[p + 2] = 0;
            img.data[p + 3] = 150;
          } else {
            img.data[p] = 255;
            img.data[p + 1] = 0;
            img.data[p + 2] = 0;
            img.data[p + 3] = 150;
          }
        }
      }
    }
  }
  offCtx.putImageData(img, 0, 0);

  const panel = c.parentElement;
  let cw = baseW, ch = baseH;
  if (panel) {
    const rect = panel.getBoundingClientRect();
    cw = rect.width;
    ch = rect.height;
  }
  c.width = cw;
  c.height = ch;

  const ctx = c.getContext('2d', { willReadFrequently: true });
  ctx.clearRect(0, 0, cw, ch);
  ctx.imageSmoothingEnabled = true;

  const fitScale = Math.min(cw / baseW, ch / baseH);
  const scale = fitScale * CORONAL_ZOOM;

  const scaledW = baseW * scale;
  const scaledH = baseH * scale;
  const dx = (cw - scaledW) / 2;
  const dy = (ch - scaledH) / 2;

  ctx.drawImage(off, 0, 0, baseW, baseH, dx, dy, scaledW, scaledH);

  const cxVoxel = dims[0] - 1 - currentSlice.sagittal; // X
  const cyVoxel = dims[2] - 1 - currentSlice.axial;    // Z
  const cxCanvas = dx + cxVoxel * scale;
  const cyCanvas = dy + cyVoxel * scale;

  drawCrosshair(ctx, cw, ch, cxCanvas, cyCanvas);

  const infoEl = document.getElementById('coronalInfo');
  if (infoEl) infoEl.textContent = `${y + 1} / ${dims[1]}`;
}


function drawCrosshair(ctx, width, height, x, y) {
  ctx.beginPath();
  ctx.setLineDash([8, 3]);
  ctx.strokeStyle = "#9ab3fd";
  ctx.lineWidth = 0.5;
  ctx.translate(0.5, 0.5);
  ctx.moveTo(x, 0);
  ctx.lineTo(x, height - 1);
  ctx.moveTo(0, y);
  ctx.lineTo(width - 1, y);
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.resetTransform();
}

// 밝기 슬라이더
if (brightnessEl) {
  brightnessEl.addEventListener('input', (e) => {
    brightness = parseFloat(e.target.value);
    const bv = document.getElementById('brightnessValue');
    if (bv) bv.textContent = brightness.toFixed(1);
    renderAll();
  });
}

// 휠 슬라이스 이동
if (axialPanel) {
  axialPanel.addEventListener('wheel', (e) => {
    if (!imageData) return;
    e.preventDefault();
    const delta = e.deltaY > 0 ? 1 : -1;
    currentSlice.axial = Math.max(0, Math.min(dims[2] - 1, currentSlice.axial + delta));
    renderAxial();
  });
}

if (sagittalPanel) {
  sagittalPanel.addEventListener('wheel', (e) => {
    if (!imageData) return;
    e.preventDefault();
    const delta = e.deltaY > 0 ? 1 : -1;
    currentSlice.sagittal = Math.max(0, Math.min(dims[0] - 1, currentSlice.sagittal + delta));
    renderSagittal();
  });
}

if (coronalPanel) {
  coronalPanel.addEventListener('wheel', (e) => {
    if (!imageData) return;
    e.preventDefault();
    const delta = e.deltaY > 0 ? 1 : -1;
    currentSlice.coronal = Math.max(0, Math.min(dims[1] - 1, currentSlice.coronal + delta));
    renderCoronal();
  });
}

setupInteraction();
