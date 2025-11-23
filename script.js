// js/script.js

// ---------------------------------------------------------
// 1) 전역 상태 및 DOM 요소 참조
// ---------------------------------------------------------

let currentViewMode = '2D';
let patients = []; // DB에서 불러온 환자 목록
let mri = [
  { src: 'assets/mri-axial.png', title: 'Axial', type: "PNG" },
  { src: 'assets/mri-coronal.png', title: 'Coronal', type: "PNG" }
];
let idx = 0;
let currentPatientIndex = 0;

let maskData = null; // 2D
let currentMaskBase64 = null; // 3D

// DOM 요소
const list = document.getElementById('patientList');
const ptNameEl = document.getElementById('ptName');
const ptSexAgeEl = document.getElementById('ptSexAge');
const viewerRoot = document.getElementById('mainViewer');
const issuedAtEl = document.getElementById('issuedAt');
const summaryEl = document.getElementById('summaryBox');
const logList = document.getElementById('logList');
const featureListEl = document.getElementById('featureList');

const btnView2D = document.getElementById('btnView2D');
const btnView3D = document.getElementById('btnView3D');
const btnUpload = document.getElementById('btnUpload');
const fileInput = document.getElementById('fileInput');

const defaultImageTextEl = document.getElementById('defaultImageText');
const viewerImgEl = document.getElementById('viewerImg');
const mprContainerEl = document.getElementById('mprContainer');
const viewer3DContainerEl = document.getElementById('viewer3DContainer');
const axialPanel = document.getElementById('axialPanel');
const sagittalPanel = document.getElementById('sagittalPanel');
const coronalPanel = document.getElementById('coronalPanel');
const brightnessEl = document.getElementById('brightness');


// ---------------------------------------------------------
// 2) 전역 NIfTI 상태
// ---------------------------------------------------------

let imageData = null;
let dims = null;
let brightness = 1.0;
let currentSlice = { axial: 0, sagittal: 0, coronal: 0 };


// ---------------------------------------------------------
// 3) 유틸 함수
// ---------------------------------------------------------

function todayISO() {
  const t = new Date();
  return `${t.getFullYear()}-${String(t.getMonth() + 1).padStart(2, '0')}-${String(t.getDate()).padStart(2, '0')}`;
}

function log(msg, isError = false) {
  const row = document.createElement('div');
  row.className = 'row';
  if (isError) row.style.color = 'red';
  const now = new Date().toLocaleString();
  row.innerHTML = `<span class="mono">${now}</span><span>${msg}</span><span class="mono">sys</span>`;
  if (logList) logList.prepend(row);
}


// ---------------------------------------------------------
// 4) 환자 데이터 로직 (로드/렌더/선택/이력 호출)
// ---------------------------------------------------------

// 서버에서 환자 목록을 가져와 내부 구조로 변환
async function loadPatients() {
  try {
    const res = await fetch('http://127.0.0.1:8000/api/patients');
    if (!res.ok) throw new Error('환자 목록 로드 실패');
    const data = await res.json();

    patients = data.map(row => {
      let vitalsStr = '';
      const parts = [];
      if (row.height_cm) parts.push(`키 ${row.height_cm}cm`);
      if (row.weight_kg) parts.push(`몸무게 ${row.weight_kg}kg`);
      vitalsStr = parts.join(' | ');

      return {
        id: row.patient_id,
        name: row.name,
        sex: row.sex,
        age: row.age,
        vitals: vitalsStr,
        icv: row.icv,
        apoe4: row.apoe4,
        features: null,
        summary: ''
      };
    });

    renderPatients();
  } catch (err) {
    console.error(err);
    log('환자 목록 로드 실패: ' + err.message, true);
  }
}

// 환자 목록 DOM 렌더링 및 기본 선택
function renderPatients() {
  if (!list) return;
  list.innerHTML = '';
  patients.forEach((p, i) => {
    const li = document.createElement('li');
    li.innerHTML = `<span>•</span><span>${p.name}</span><span class="mono">${p.sex}/${p.age}</span>`;
    li.onclick = () => selectPatient(i, li);
    list.appendChild(li);
    if (i === 0) selectPatient(0, li);
  });
}

// 환자 선택 시 화면 업데이트
function selectPatient(i, li) {
  document.querySelectorAll('.patient-list li').forEach(n => n.classList.remove('active'));
  if (li) li.classList.add('active');

  currentPatientIndex = i;
  const p = patients[i];
  if (ptNameEl) ptNameEl.textContent = p.name;
  if (ptSexAgeEl) ptSexAgeEl.textContent = `${p.sex} / ${p.age}세`;
  renderVitals(p.vitals);
  renderFeatures(p.features);
  if (summaryEl) summaryEl.textContent = p.summary || '';
  loadPatientHistory(p.id);
  renderViewer();
}

// 환자 검사 이력 로드
async function loadPatientHistory(patientId) {
  const historyList = document.getElementById('historyList');
  if (!historyList) return;
  historyList.innerHTML = '<li style="justify-content: center; color: #888;">로딩 중...</li>';

  try {
    const res = await fetch(`http://127.0.0.1:8000/api/patients/${patientId}/history`);
    if (!res.ok) throw new Error('History fetch failed');
    const historyData = await res.json();
    renderHistory(historyData);
  } catch (err) {
    console.error(err);
    historyList.innerHTML = '<li style="justify-content: center; color: red;">이력 로드 실패</li>';
  }
}

// 이력 DOM 렌더링
function renderHistory(historyData) {
  const historyList = document.getElementById('historyList');
  if (!historyList) return;
  historyList.innerHTML = '';

  if (historyData.length === 0) {
    historyList.innerHTML = '<li style="justify-content: center; color: #888;">이전 검사 기록이 없습니다.</li>';
    return;
  }

  historyData.forEach(item => {
    const li = document.createElement('li');
    li.style.display = 'flex';
    li.style.flexDirection = 'column';
    li.style.alignItems = 'flex-start';
    li.style.gap = '2px';
    li.style.borderBottom = '1px solid #eee';
    li.style.padding = '8px';
    const dateStr = item.exam_datetime.substring(0, 16);
    const labelColor = item.label === 'CN' ? '#167c3a' : '#c43d35';
    li.innerHTML = `
            <div style="display: flex; justify-content: space-between; width: 100%;">
                <span style="font-weight: bold; font-size: 12px;">${dateStr}</span>
                <span style="color: ${labelColor}; font-weight: bold; font-size: 11px; background: #fff; border: 1px solid ${labelColor}; padding: 1px 4px; border-radius: 3px;">${item.label}</span>
            </div>
            <div style="font-size: 11px; color: #555;">
                총 해마 부피: ${item.total_hipp_vol.toLocaleString()} mm³
            </div>
        `;
    historyList.appendChild(li);
  });
}

// 신체 정보(키/몸무게) render
function renderVitals(text) {
  const root = document.getElementById('ptVitals');
  if (!root) return;
  const items = (text || '').split('|').map(s => s.trim()).filter(Boolean);
  root.innerHTML = '';
  items.forEach(item => {
    const [key, ...rest] = item.split(/\s+/);
    const value = rest.join(' ');
    const li = document.createElement('li');
    li.innerHTML = `<span class="vkey">${key}</span><span class="vval">${value}</span>`;
    root.appendChild(li);
  });
}

// 분석 특성 목록 render
function renderFeatures(features) {
  if (!featureListEl) return;
  featureListEl.innerHTML = '';
  if (!features) return;

  const rows = [
    ['ICV', features.icv ? `${features.icv.toLocaleString()} mm³` : '—'],
    ['좌 해마 부피', features.left_hipp_vol_mm3 ? `${features.left_hipp_vol_mm3} mm³` : '—'],
    ['우 해마 부피', features.right_hipp_vol_mm3 ? `${features.right_hipp_vol_mm3} mm³` : '—'],
    ['총 해마 부피', features.total_hipp_vol_mm3 ? `${features.total_hipp_vol_mm3} mm³` : '—'],
    ['ICV 보정 해마 지수 (좌/우/총)',
      (features.left_hipp_vol_icv_norm != null &&
        features.right_hipp_vol_icv_norm != null &&
        features.total_hipp_vol_icv_norm != null)
        ? `${features.left_hipp_vol_icv_norm} / ${features.right_hipp_vol_icv_norm} / ${features.total_hipp_vol_icv_norm}`
        : '—'
    ],
    ['APOE4 유전자형', (features.APOE4 != null ? features.APOE4 : (features.apoe4 != null ? features.apoe4 : '정보 없음'))]
  ];

  rows.forEach(([k, v]) => {
    const li = document.createElement('li');
    li.innerHTML = `<span class="vkey">${k}</span><span class="vval">${v}</span>`;
    featureListEl.appendChild(li);
  });
}


// ---------------------------------------------------------
// 5) 뷰어 렌더링 / NIfTI 로드
// ---------------------------------------------------------

// 2D/3D 버튼에 따라 뷰어 전환
function renderViewer() {
  const currentMri = mri[idx];
  const hasImage = currentMri && currentMri.src;

  if (defaultImageTextEl) defaultImageTextEl.style.display = 'none';
  if (viewerImgEl) viewerImgEl.style.display = 'none';
  if (mprContainerEl) mprContainerEl.style.display = 'none';
  if (viewer3DContainerEl) viewer3DContainerEl.style.display = 'none';

  if (!hasImage) {
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
      const name = currentMri.title || currentMri.src || '';
      if (name.endsWith('.nii') || name.endsWith('.nii.gz') || currentMri.src.endsWith('.nii') || currentMri.src.endsWith('.nii.gz')) {
        loadNiftiFromURL(currentMri.src).catch(err => {
          console.error('원격 NIfTI 로딩 오류:', err);
        });
      } else {
        ['axialCanvas', 'sagittalCanvas', 'coronalCanvas'].forEach(id => {
          const c = document.getElementById(id);
          if (c) {
            const ctx = c.getContext('2d', { willReadFrequently: true });
            ctx.fillStyle = '#000';
            ctx.fillRect(0, 0, c.width || 256, c.height || 256);
          }
        });
      }
    }
  } 
}

// File 객체를 Blob URL로 변환하여 로드
async function loadNiftiFile(file) {
  const url = URL.createObjectURL(file);
  await loadNiftiFromURL(url);
}

// URL에서 NIfTI 데이터를 불러와 파싱
async function loadNiftiFromURL(url) {
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


// ---------------------------------------------------------
// 6) 상호작용: 패널 드래그/휠 등
// ---------------------------------------------------------

// 캔버스 클릭 드래그로 슬라이스 변경
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
      const x = Math.floor((e.clientX - rect.left) * (canvas.width / rect.width));
      const y = Math.floor((e.clientY - rect.top) * (canvas.height / rect.height));
      if (x < 0 || x >= canvas.width || y < 0 || y >= canvas.height) return;

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
    canvas.addEventListener('mouseup', () => isDragging = false);
    canvas.addEventListener('mouseleave', () => isDragging = false);
  });
}


// ---------------------------------------------------------
// 7) 3D 뷰어
// ---------------------------------------------------------

// 서버에 base64 마스크를 보내 Plotly JSON을 받아 렌더링
async function loadAndRenderPlotly(maskBase64) {
  const container = document.getElementById('viewer3DContainer');
  if(container) container.innerHTML = '';
  const mprContainer = document.getElementById('mprContainer');
  if (!maskBase64) {
    alert("분석된 데이터가 없습니다.");
    return;
  }
  mprContainer.style.display = 'none';
  container.style.display = 'block';

  container.innerHTML = '';
  const loading = document.createElement('div');
  loading.textContent = "3D 모델링 중...";
  loading.style.color = 'white';
  loading.style.position = 'absolute';
  loading.style.top = '50%';
  loading.style.left = '50%';
  container.appendChild(loading);

  try {
    const res = await fetch('http://127.0.0.1:8000/api/get_plotly_3d', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ mask_base64: maskBase64 })
    });

    if (!res.ok) throw new Error("서버 오류");
    const figData = await res.json();
    console.log("PLOTLY 데이터 확인:", figData);
    if (figData.status === 'error') throw new Error(figData.message);

    container.innerHTML = '';
    const config = { responsive: true, displayModeBar: true, displaylogo: false };
    if (!figData.layout.scene) figData.layout.scene = {};
    await Plotly.newPlot('viewer3DContainer', figData.data, figData.layout, config);

    setTimeout(() => {
      const container = document.getElementById('viewer3DContainer');
      Plotly.Plots.resize(container);
      console.log("3D 뷰어 리사이징 완료");
    }, 100);

    window.onresize = function () {
      Plotly.Plots.resize('viewer3DContainer');
    };

  } catch (err) {
    console.error(err);
    alert("3D 로드 실패: " + err.message);
    mprContainer.style.display = 'block';
    container.style.display = 'none';
  } finally {
    if (loading) loading.style.display = 'none';
  }
}


// ---------------------------------------------------------
// 8) 이벤트 리스너 (업로드/뷰 전환 등)
// ---------------------------------------------------------

if (btnView2D) btnView2D.onclick = () => {
  currentViewMode = '2D';
  if (mprContainerEl) mprContainerEl.style.display = 'block';
  if (viewer3DContainerEl) viewer3DContainerEl.style.display = 'none';
  renderViewer();
};

if (btnView3D) btnView3D.onclick = () => {
  currentViewMode = '3D';
  let dataToRender = window.latest_mask_base64 || currentMaskBase64;
  if (!dataToRender) {
    alert('분석된 데이터가 아직 없습니다.');
    return;
  }
  loadAndRenderPlotly(dataToRender);
};

if (btnUpload) btnUpload.onclick = () => {
  if (fileInput) {
    fileInput.value = '';
    fileInput.click();
  }
};

if (fileInput) {
  fileInput.onchange = async e => {
    const f = e.target.files[0];
    if (!f) return;

    const url = URL.createObjectURL(f);
    mri.push({ src: url, title: f.name, file: f });
    idx = mri.length - 1;

    maskData = null;
    currentMaskBase64 = null;
    window.latest_mask_base64 = null;

    if (f.name.endsWith('.nii') || f.name.endsWith('.nii.gz')) {
      currentViewMode = '2D';
      await loadNiftiFile(f);
    } else {
      renderViewer();
    }

    log(`MRI 업로드: ${f.name}`);

    const p = patients[currentPatientIndex];

    try {
      if (btnUpload) btnUpload.disabled = true;

      const fd = new FormData();
      fd.append('file', f);
      fd.append('patient_id', p?.id || '');
      if (p?.age != null) fd.append('age', String(p.age));
      if (p?.apoe4 != null) fd.append('apoe4', String(p.apoe4));
      if (p?.sex) fd.append('sex', p.sex);
      if (p?.icv) fd.append('icv', String(p.icv));

      const now = new Date();
      const examDT = new Date(now.getTime() - (now.getTimezoneOffset() * 60000)).toISOString().slice(0, 19).replace('T', ' ');
      fd.append('exam_datetime', examDT);

      const res = await fetch('http://127.0.0.1:8000/api/process_mri', {
        method: 'POST',
        body: fd
      });

      if (!res.ok) {
        const errorText = await res.text();
        throw new Error(`Server Error: ${res.status} - ${errorText}`);
      }

      const result = await res.json();
      applyServerResult(result);

      if (result.exam_datetime) {
        log(`[검사 완료] 저장된 검사일시: ${result.exam_datetime}`);
      }

      if (p && p.id) {
        loadPatientHistory(p.id);
      }

      if (result.mask_base64) {
        console.log("마스크 데이터 수신 성공 (길이):", result.mask_base64.length);
        window.latest_mask_base64 = result.mask_base64;
        currentMaskBase64 = result.mask_base64;
        await loadMaskFromBase64(result.mask_base64);
        log('마스크 데이터 수신 완료. 3D 보기가 가능합니다.');
      } else {
        console.warn("서버 응답에 mask_base64가 없습니다.");
        log('결과에 마스크 데이터가 없습니다.');
      }

      log('분석 완료: ' + (result.label || 'N/A'));

    } catch (err) {
      console.error(err);
      log(`분석 실패: ${err.message}`, true);
      alert("분석 중 오류가 발생했습니다.");
    } finally {
      if (btnUpload) btnUpload.disabled = false;
    }
  };
}


// ---------------------------------------------------------
// 9) Base64 마스크 처리 및 NIfTI 파싱/렌더링
// ---------------------------------------------------------

// 서버에서 받은 base64 마스크를 파싱해 maskData에 적용
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
    const mask = extractImageData(niftiData, offset, header);

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

// 서버 결과를 UI에 적용
function applyServerResult(result) {
  const { probs, label, summary, features } = result;
  if (summaryEl && summary) {
      summaryEl.textContent = summary;
    if (patients[currentPatientIndex]) {
      patients[currentPatientIndex].summary = summary;
    }
  } else if (summaryEl && probs && label) {
    const CN = Math.round(probs.CN || 0);
    const AD = Math.round(probs.AD || 0);
    const text = `모델 예측: ${label}\n\n확률 분포: CN ${CN}% · AD ${AD}%`;
    summaryEl.textContent = text;
    if (patients[currentPatientIndex]) {
      patients[currentPatientIndex].summary = text;
    }
  }

  if (features) {
    patients[currentPatientIndex].features = features;
    renderFeatures(features);
  }
}


// ---------------------------------------------------------
// 10) NIfTI 파싱 및 렌더링 함수
// ---------------------------------------------------------

function parseNiftiHeader(buf) {
  const v = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  return {
    dim: [v.getInt16(40, true), v.getInt16(42, true), v.getInt16(44, true), v.getInt16(46, true)],
    datatype: v.getInt16(70, true),
    bitpix: v.getInt16(72, true),
    vox_offset: v.getFloat32(108, true)
  };
}

// 원시 버퍼에서 이미지 볼륨을 추출하여 Float32Array로 반환
function extractImageData(buf, offset, header, isMask = false) {
  const v = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  const n = header.dim[1] * header.dim[2] * header.dim[3];
  const data = new Float32Array(n);
  let min = Infinity, max = -Infinity;

  for (let i = 0; i < n; i++) {
    let val = 0;
    if (offset + i * (header.bitpix / 8) >= v.byteLength) break;
    try {
      if (header.bitpix === 8) val = v.getUint8(offset + i);
      else if (header.bitpix === 16) val = v.getInt16(offset + i * 2, true);
      else if (header.bitpix === 32) val = v.getFloat32(offset + i * 4, true);
    } catch (e) { val = 0; }

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

// 세 방향 다 render
function renderAll() {
  renderAxial();
  renderSagittal();
  renderCoronal();
}

// Axial
function renderAxial() {
  if (!dims || !imageData) return;
  const c = document.getElementById('axialCanvas');
  if (!c) return;
  const ctx = c.getContext('2d', { willReadFrequently: true });
  c.width = dims[0];
  c.height = dims[1];
  const img = ctx.createImageData(dims[0], dims[1]);
  const z = currentSlice.axial;

  for (let y = 0; y < dims[1]; y++) {
    const flippedY = dims[1] - 1 - y;
    for (let x = 0; x < dims[0]; x++) {
      const idx = x + flippedY * dims[0] + z * dims[0] * dims[1];
      if (idx < 0 || idx >= imageData.length) continue;
      const v = Math.min(255, imageData[idx] * brightness);
      const p = (y * dims[0] + x) * 4;
      img.data[p] = v; img.data[p + 1] = v; img.data[p + 2] = v; img.data[p + 3] = 255;

      if (maskData && idx < maskData.length) {
        const maskVal = Math.round(maskData[idx]);
        if (maskVal > 0) {
          if (maskVal === 1) { img.data[p] = 0; img.data[p + 1] = 255; img.data[p + 2] = 0; img.data[p + 3] = 150; }
          else { img.data[p] = 255; img.data[p + 1] = 0; img.data[p + 2] = 0; img.data[p + 3] = 150; }
        }
      }
    }
  }

  ctx.putImageData(img, 0, 0);
  const cx = currentSlice.sagittal;
  const cy = dims[1] - 1 - currentSlice.coronal;
  drawCrosshair(ctx, c.width, c.height, cx, cy);

  const infoEl = document.getElementById('axialInfo');
  if (infoEl) infoEl.textContent = `${z + 1} / ${dims[2]}`;
}

// Sagittal
function renderSagittal() {
  if (!dims || !imageData) return;
  const c = document.getElementById('sagittalCanvas');
  if (!c) return;
  const ctx = c.getContext('2d', { willReadFrequently: true });
  c.width = dims[1];
  c.height = dims[2];
  const img = ctx.createImageData(dims[1], dims[2]);
  const x = currentSlice.sagittal;

  for (let z = 0; z < dims[2]; z++) {
    const flippedZ = dims[2] - 1 - z;
    for (let y = 0; y < dims[1]; y++) {
      const flippedY = dims[1] - 1 - y;
      const idx = x + flippedY * dims[0] + flippedZ * dims[0] * dims[1];
      if (idx < 0 || idx >= imageData.length) continue;
      const v = Math.min(255, imageData[idx] * brightness);
      const p = (z * dims[1] + y) * 4;
      img.data[p] = v; img.data[p + 1] = v; img.data[p + 2] = v; img.data[p + 3] = 255;

      if (maskData && idx < maskData.length) {
        const maskVal = Math.round(maskData[idx]);
        if (maskVal > 0) {
          if (maskVal === 1) { img.data[p] = 0; img.data[p + 1] = 255; img.data[p + 2] = 0; img.data[p + 3] = 150; }
          else { img.data[p] = 255; img.data[p + 1] = 0; img.data[p + 2] = 0; img.data[p + 3] = 150; }
        }
      }
    }
  }
  ctx.putImageData(img, 0, 0);
  const cx = dims[1] - 1 - currentSlice.coronal;
  const cy = dims[2] - 1 - currentSlice.axial;
  drawCrosshair(ctx, c.width, c.height, cx, cy);

  const infoEl = document.getElementById('sagittalInfo');
  if (infoEl) infoEl.textContent = `${x + 1} / ${dims[0]}`;
}

// Coronal
function renderCoronal() {
  if (!dims || !imageData) return;
  const c = document.getElementById('coronalCanvas');
  if (!c) return;
  const ctx = c.getContext('2d', { willReadFrequently: true });
  c.width = dims[0];
  c.height = dims[2];
  const img = ctx.createImageData(dims[0], dims[2]);
  const y = currentSlice.coronal;

  for (let z = 0; z < dims[2]; z++) {
    const flippedZ = dims[2] - 1 - z;
    for (let x = 0; x < dims[0]; x++) {
      const flippedX = dims[0] - 1 - x;
      const idx = flippedX + y * dims[0] + flippedZ * dims[0] * dims[1];
      if (idx < 0 || idx >= imageData.length) continue;
      const v = Math.min(255, imageData[idx] * brightness);
      const p = (z * dims[0] + x) * 4;
      img.data[p] = v; img.data[p + 1] = v; img.data[p + 2] = v; img.data[p + 3] = 255;

      if (maskData && idx < maskData.length) {
        const maskVal = Math.round(maskData[idx]);
        if (maskVal > 0) {
          if (maskVal === 1) { img.data[p] = 0; img.data[p + 1] = 255; img.data[p + 2] = 0; img.data[p + 3] = 150; }
          else { img.data[p] = 255; img.data[p + 1] = 0; img.data[p + 2] = 0; img.data[p + 3] = 150; }
        }
      }
    }
  }
  ctx.putImageData(img, 0, 0);
  const cx = dims[0] - 1 - currentSlice.sagittal;
  const cy = dims[2] - 1 - currentSlice.axial;
  drawCrosshair(ctx, c.width, c.height, cx, cy);

  const infoEl = document.getElementById('coronalInfo');
  if (infoEl) infoEl.textContent = `${y + 1} / ${dims[1]}`;
}

// 십자선
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

if (brightnessEl) {
  brightnessEl.addEventListener('input', (e) => {
    brightness = parseFloat(e.target.value);
    const bv = document.getElementById('brightnessValue');
    if (bv) bv.textContent = brightness.toFixed(1);
    renderAll();
  });
}

// 휠 조작
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


// ---------------------------------------------------------
// 초기화
// ---------------------------------------------------------

async function init() {
  if (issuedAtEl) issuedAtEl.textContent = todayISO();
  await loadPatients();
  setupInteraction();
}
init();