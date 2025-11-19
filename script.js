let currentViewMode = '2D';

// DB에서 불러올 환자 배열
let patients = [];

// 기본 MRI 이미지
let mri = [
  { src: 'assets/mri-axial.png', title: 'Axial', type: "PNG" },
  { src: 'assets/mri-coronal.png', title: 'Coronal', type: "PNG" }
];

let idx = 0;
let currentPatientIndex = 0;

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

// 유틸
function todayISO() {
  const t = new Date();
  return `${t.getFullYear()}-${String(t.getMonth() + 1).padStart(2, '0')}-${String(t.getDate()).padStart(2, '0')}`;
}

function log(msg) {
  const row = document.createElement('div');
  row.className = 'row';
  const now = new Date().toLocaleString();
  row.innerHTML = `<span class="mono">${now}</span><span>${msg}</span><span class="mono">sys</span>`;
  logList.prepend(row);
}

// 환자 데이터 DB에서 가져오기
async function loadPatients() {
  const res = await fetch('http://127.0.0.1:8000/api/patients');
  const data = await res.json();

  patients = data.map(row => ({
    id: row.patient_id,
    name: row.name,
    sex: row.sex,
    age: row.age,
    vitals: '',
    icv: '',
    apoe4: 0,
    features: null
  }));

  renderPatients();
}

// Vitals / Features 렌더링
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
    ['APOE4 유전자형', features.apoe4 ?? '정보 없음']
  ];

  rows.forEach(([k, v]) => {
    const li = document.createElement('li');
    li.innerHTML = `<span class="vkey">${k}</span><span class="vval">${v}</span>`;
    featureListEl.appendChild(li);
  });
}

// 환자 리스트 렌더링
function renderPatients() {
  list.innerHTML = '';
  patients.forEach((p, i) => {
    const li = document.createElement('li');
    li.innerHTML = `<span>•</span><span>${p.name}</span><span class="mono">${p.sex}/${p.age}</span>`;
    li.onclick = () => selectPatient(i, li);
    list.appendChild(li);
    if (i === 0) selectPatient(0, li);
  });
}

function selectPatient(i, li) {
  document.querySelectorAll('.patient-list li').forEach(n => n.classList.remove('active'));
  li.classList.add('active');
  currentPatientIndex = i;
  const p = patients[i];
  ptNameEl.textContent = p.name;
  ptSexAgeEl.textContent = `${p.sex} / ${p.age}세`;
  renderVitals(p.vitals);
  renderFeatures(p.features);
  renderViewer();
}

// 뷰어 렌더링
function renderViewer() {
  document.getElementById('defaultImageText').style.display = 'none';
  document.getElementById('viewerImg').style.display = 'none';
  document.getElementById('mprContainer').style.display = 'none';

  const mprContainer = document.getElementById('mprContainer');
  // 3D
  if (currentViewMode === '3D') {
    // document.getElementById('defaultImageText').style.display = 'none';
    // viewerRoot.innerHTML = '';
    viewerRoot.appendChild(mprContainer);
    mprContainer.style.display = 'block';
    // document.querySelector('.three-panel').style.display = 'grid';

    if (!mri[idx].src.endsWith('.nii') && !mri[idx].src.endsWith('.nii.gz')) {
      ['axialCanvas', 'sagittalCanvas', 'coronalCanvas'].forEach(id => {
        const c = document.getElementById(id);
        if (c) {
          const ctx = c.getContext('2d');
          ctx.fillStyle = '#000';
          ctx.fillRect(0, 0, c.width || 256, c.height || 256);
        }
      });
      return;
    }
    loadNiftiFromURL(mri[idx].src);
  }
  // 2D
  else {
    mprContainer.style.display = 'none';
    viewerRoot.innerHTML = '';
    const img = document.createElement('img');
    img.src = mri[idx].src;
    img.alt = '2D MRI';
    img.style.maxWidth = '100%';
    img.style.maxHeight = '100%';
    viewerRoot.appendChild(img);

    if (mri[idx].src === undefined) {
      viewerRoot.innerText = "No Image";
    }
  }
}

function render3DView(p) {
  // const div = document.createElement('div');
  // div.style.display = 'flex';
  // div.style.alignItems = 'center';
  // div.style.justifyContent = 'center';
  // div.style.height = '100%';
  // //  div.textContent = '3D 해마 세그 뷰어 (추후 연동)';
  // viewerRoot.appendChild(div);
}

async function loadNiftiFromURL(url) {
  const loadingText = document.getElementById('loadingText');
  try {
    loadingText.textContent = 'NIfTI 로딩 중…';

    const res = await fetch(url);
    const buf = new Uint8Array(await res.arrayBuffer());

    const header = parseNiftiHeader(buf);
    dims = [header.dim[1], header.dim[2], header.dim[3]];
    imageData = extractImageData(buf, header.vox_offset, header);

    currentSlice = {
      axial: Math.floor(dims[2] / 2),
      sagittal: Math.floor(dims[0] / 2),
      coronal: Math.floor(dims[1] / 2)
    };

    renderAll();
    loadingText.style.display = 'none';
  } catch (err) {
    console.error(err);
    loadingText.textContent = '로드 실패';
  }
}

// 버튼 이벤트
btnView2D.onclick = () => {
  currentViewMode = '2D';
  renderViewer();
};

btnView3D.onclick = () => {
  currentViewMode = '3D';
  document.querySelector('.three-panel').style.display = 'grid';
  renderViewer();
};

btnUpload.onclick = () => {
  const input = document.getElementById('fileInput');
  if (input) input.click();
};

document.getElementById('fileInput').onchange = async e => {
  const f = e.target.files[0];
  if (!f) return;

  const url = URL.createObjectURL(f);
  mri.push({ src: url, title: f.name });
  idx = mri.length - 1;
  renderViewer();
  log(`MRI 업로드: ${f.name}`);

  const p = patients[currentPatientIndex];

  try {
    log('서버 분석 시작…');

    const fd = new FormData();
    fd.append('file', f);
    fd.append('patient_id', p.id);
    fd.append('age', String(p.age));
    fd.append('apoe4', String(p.apoe4));
    fd.append('sex', p.sex);
    fd.append('icv', String(p.icv || ""));

    const res = await fetch('http://127.0.0.1:8000/api/process_mri', {
      method: 'POST',
      body: fd
    });

    if (!res.ok) {
      log('서버 오류: ' + res.status);
      return;
    }

    const result = await res.json();
    applyServerResult(result);
    log('분석 완료: ' + (result.label || 'N/A'));
  } catch (err) {
    console.error(err);
    log('분석 실패(네트워크 또는 서버 오류)');
  }
};

// 서버 결과 반영
function applyServerResult(result) {
  const { probs, label, summary, features } = result;

  if (summaryEl && summary) {
    summaryEl.value = summary;
  } else if (summaryEl && probs && label) {
    const CN = Math.round(probs.CN || 0);
    const AD = Math.round(probs.AD || 0);
    const header = `모델 예측: ${label}`;
    const dist = `확률 분포: CN ${CN}% · AD ${AD}%`;
    summaryEl.value = `${header}\n\n${dist}`;
  }

  if (features) {
    patients[currentPatientIndex].features = features;
    renderFeatures(features);
  }
}

// 초기 화면 세팅
async function init() {
  issuedAtEl.textContent = todayISO();
  await loadPatients();   // DB에서 환자 불러오고 renderPatients() 실행
  renderViewer();         // 첫 환자 기준 뷰어 렌더
}

// 스크립트가 body 끝에 있다면 바로 호출해도 됨
init();


let imageData = null, dims = null, brightness = 1.0;
let currentSlice = { axial: 0, sagittal: 0, coronal: 0 };

document.getElementById('btnUpload').addEventListener('click', () => {
  if (currentViewMode !== '3D') return;
  document.getElementById('fileInput').click();
});

document.getElementById('fileInput').addEventListener('change', async (e) => {
  const file = e.target.files[0];
  if (!file) return;
  await loadNiftiFile(file);
});

document.getElementById('sliceSlider').addEventListener('input', (e) => {
  const val = parseInt(e.target.value);
  currentSlice.axial = Math.floor((val / 100) * (dims[2] - 1));
  document.getElementById('sliceValue').textContent = val;
  renderAxial();
});

document.getElementById('brightness').addEventListener('input', (e) => {
  brightness = parseFloat(e.target.value);
  document.getElementById('brightnessValue').textContent = brightness.toFixed(1);
  renderAll();
});

// NIfTI 파일 로드
async function loadNiftiFile(file) {
  const loadingText = document.getElementById('loadingText');
  loadingText.textContent = '로딩 중...';
  try {
    const arrayBuffer = await file.arrayBuffer();
    const raw = new Uint8Array(arrayBuffer);
    const niftiBuffer = file.name.endsWith('.gz') ? pako.inflate(raw) : raw;

    const header = parseNiftiHeader(niftiBuffer);
    dims = [header.dim[1], header.dim[2], header.dim[3]];
    imageData = extractImageData(niftiBuffer, header.vox_offset || 352, header);

    // 초기 슬라이스 설정
    currentSlice.axial = Math.floor(dims[2] / 2);
    currentSlice.sagittal = Math.floor(dims[0] / 2);
    currentSlice.coronal = Math.floor(dims[1] / 2);

    renderAll();
    loadingText.style.display = 'none';

    console.log('NIfTI 로드 완료:', dims);
  } catch (err) {
    console.error(err);
    loadingText.textContent = '로드 실패';
  }
}

// 메타데이터 추출
function parseNiftiHeader(buf) {
  const v = new DataView(buf.buffer);
  return {
    dim: [v.getInt16(40, true), v.getInt16(42, true), v.getInt16(44, true), v.getInt16(46, true)],
    datatype: v.getInt16(70, true),
    bitpix: v.getInt16(72, true),
    vox_offset: v.getFloat32(108, true)
  };
}

// 영상 데이터 추출
function extractImageData(buf, offset, header) {
  const v = new DataView(buf.buffer);
  const n = header.dim[1] * header.dim[2] * header.dim[3];
  const data = new Float32Array(n);
  let min = Infinity, max = -Infinity;

  for (let i = 0; i < n; i++) {
    let val;
    if (header.bitpix === 8) val = v.getUint8(offset + i);
    else if (header.bitpix === 16) val = v.getInt16(offset + i * 2, true);
    else if (header.bitpix === 32) val = v.getFloat32(offset + i * 4, true);
    else val = 0;

    data[i] = val;
    if (val < min) min = val;
    if (val > max) max = val;
  }

  const range = max - min;
  for (let i = 0; i < n; i++) {
    data[i] = ((data[i] - min) / range) * 255;
  }

  return data;
}

function renderAll() {
  renderAxial();
  renderSagittal();
  renderCoronal();
}

// Axial 뷰 렌더링
function renderAxial() {
  if (!dims || !imageData) return;
  const c = document.getElementById('axialCanvas');
  const ctx = c.getContext('2d');
  c.width = dims[0];
  c.height = dims[1];
  const img = ctx.createImageData(dims[0], dims[1]);
  const z = currentSlice.axial;

  for (let y = 0; y < dims[1]; y++) {
    const flippedY = dims[1] - 1 - y;
    for (let x = 0; x < dims[0]; x++) {
      const idx = x + flippedY * dims[0] + z * dims[0] * dims[1];
      const v = Math.min(255, imageData[idx] * brightness);
      const p = (y * dims[0] + x) * 4;
      img.data[p] = img.data[p + 1] = img.data[p + 2] = v;
      img.data[p + 3] = 255;
    }
  }

  ctx.putImageData(img, 0, 0);

  // 십자선
  ctx.beginPath();
  ctx.setLineDash([8, 3]);
  ctx.strokeStyle = "#9ab3fd";
  ctx.lineWidth = 0.5;
  ctx.translate(0.5, 0.5);
  const cx = currentSlice.sagittal;
  const cy = currentSlice.coronal;

  ctx.moveTo(cx, 0);
  ctx.lineTo(cx, c.height - 1);
  ctx.moveTo(0, cy);
  ctx.lineTo(c.width - 1, cy);

  ctx.stroke();
  ctx.setLineDash([]);
  ctx.resetTransform();

  document.getElementById('axialInfo').textContent = `${z + 1} / ${dims[2]}`;
}


// Sagittal 뷰 렌더링
function renderSagittal() {
  if (!dims || !imageData) return;
  const c = document.getElementById('sagittalCanvas');
  const ctx = c.getContext('2d');
  c.width = dims[1];
  c.height = dims[2];
  const img = ctx.createImageData(dims[1], dims[2]);
  const x = currentSlice.sagittal;

  for (let z = 0; z < dims[2]; z++) {
    const flippedZ = dims[2] - 1 - z; // 상하 반전
    for (let y = 0; y < dims[1]; y++) {
      const flippedY = dims[1] - 1 - y; // 좌우 반전 추가
      const idx = x + flippedY * dims[0] + flippedZ * dims[0] * dims[1];
      const v = Math.min(255, imageData[idx] * brightness);
      const p = (z * dims[1] + y) * 4;
      img.data[p] = img.data[p + 1] = img.data[p + 2] = v;
      img.data[p + 3] = 255;
    }
  }

  ctx.putImageData(img, 0, 0);

  ctx.beginPath();
  ctx.setLineDash([8, 3]);
  ctx.strokeStyle = "#9ab3fd";
  ctx.lineWidth = 0.5;
  ctx.translate(0.5, 0.5);
  const cx = currentSlice.coronal;
  const cy = currentSlice.axial;
  ctx.moveTo(cx, 0);
  ctx.lineTo(cx, c.height - 1);
  ctx.moveTo(0, cy);
  ctx.lineTo(c.width - 1, cy);
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.resetTransform();

  document.getElementById('sagittalInfo').textContent = `${x + 1} / ${dims[0]}`;
}

// Coronal 뷰 렌더링
function renderCoronal() {
  if (!dims || !imageData) return;
  const c = document.getElementById('coronalCanvas');
  const ctx = c.getContext('2d');
  c.width = dims[0];
  c.height = dims[2];
  const img = ctx.createImageData(dims[0], dims[2]);
  const y = currentSlice.coronal;

  for (let z = 0; z < dims[2]; z++) {
    const flippedZ = dims[2] - 1 - z;
    for (let x = 0; x < dims[0]; x++) {
      const flippedX = dims[0] - 1 - x;
      const idx = flippedX + y * dims[0] + flippedZ * dims[0] * dims[1];
      const v = Math.min(255, imageData[idx] * brightness);
      const p = (z * dims[0] + x) * 4;
      img.data[p] = img.data[p + 1] = img.data[p + 2] = v;
      img.data[p + 3] = 255;
    }
  }

  ctx.putImageData(img, 0, 0);

  ctx.beginPath();
  ctx.setLineDash([8, 3]);
  ctx.strokeStyle = "#9ab3fd";
  ctx.lineWidth = 0.5;
  ctx.translate(0.5, 0.5);
  const cx = currentSlice.sagittal;
  const cy = currentSlice.axial;
  ctx.moveTo(cx, 0);
  ctx.lineTo(cx, c.height - 1);
  ctx.moveTo(0, cy);
  ctx.lineTo(c.width - 1, cy);
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.resetTransform();

  document.getElementById('coronalInfo').textContent = `${y + 1} / ${dims[1]}`;
}

// 마우스 휠 조작
document.getElementById('axialPanel').addEventListener('wheel', (e) => {
  if (!imageData) return;
  e.preventDefault();
  const delta = e.deltaY > 0 ? 1 : -1;
  currentSlice.axial = Math.max(0, Math.min(dims[2] - 1, currentSlice.axial + delta));
  renderAxial();

  const percent = Math.round((currentSlice.axial / (dims[2] - 1)) * 100);
  document.getElementById('sliceSlider').value = percent;
  document.getElementById('sliceValue').textContent = percent;
});

document.getElementById('sagittalPanel').addEventListener('wheel', (e) => {
  if (!imageData) return;
  e.preventDefault();
  const delta = e.deltaY > 0 ? 1 : -1;
  currentSlice.sagittal = Math.max(0, Math.min(dims[0] - 1, currentSlice.sagittal + delta));
  renderSagittal();
});

document.getElementById('coronalPanel').addEventListener('wheel', (e) => {
  if (!imageData) return;
  e.preventDefault();
  const delta = e.deltaY > 0 ? 1 : -1;
  currentSlice.coronal = Math.max(0, Math.min(dims[1] - 1, currentSlice.coronal + delta));
  renderCoronal();
});