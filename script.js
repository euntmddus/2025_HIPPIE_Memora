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

let maskData = null; // 해마 마스크 데이터
let currentMaskBase64 = null;

// ---------------------------------------------------------
// 1. DOM 요소 참조
// ---------------------------------------------------------
const list = document.getElementById('patientList');
const ptNameEl = document.getElementById('ptName');
const ptSexAgeEl = document.getElementById('ptSexAge');
const viewerRoot = document.getElementById('mainViewer');
const issuedAtEl = document.getElementById('issuedAt');
const summaryEl = document.getElementById('summaryBox');
const logList = document.getElementById('logList');
const featureListEl = document.getElementById('featureList');

// 버튼들
const btnView2D = document.getElementById('btnView2D');
const btnView3D = document.getElementById('btnView3D');
const btnUpload = document.getElementById('btnUpload');
const fileInput = document.getElementById('fileInput');

// 뷰어 관련 요소
const defaultImageTextEl = document.getElementById('defaultImageText');
const viewerImgEl = document.getElementById('viewerImg');
const mprContainerEl = document.getElementById('mprContainer');
const viewer3DContainerEl = document.getElementById('viewer3DContainer');
const loadingText = document.getElementById('loadingText');

// 2D 뷰어 패널
const axialPanel = document.getElementById('axialPanel');
const sagittalPanel = document.getElementById('sagittalPanel');
const coronalPanel = document.getElementById('coronalPanel');

// 밝기 조절
const brightnessEl = document.getElementById('brightness');


// ---------------------------------------------------------
// 2. 전역 변수 (NIfTI 데이터용)
// ---------------------------------------------------------
let imageData = null;
let dims = null;
let brightness = 1.0;
let currentSlice = { axial: 0, sagittal: 0, coronal: 0 };


// ---------------------------------------------------------
// 3. 유틸리티 함수
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
// 4. 환자 데이터 로직
// ---------------------------------------------------------
async function loadPatients() {
  try {
    const res = await fetch('http://127.0.0.1:8000/api/patients');
    if (!res.ok) throw new Error('환자 목록 로드 실패');
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

    // [수정 후] renderPatients 함수 호출 추가 (목록 표시용)
    renderPatients();
  } catch (err) {
    console.error(err);
    log('환자 목록 로드 실패: ' + err.message, true);
  }
}

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

function selectPatient(i, li) {
  document.querySelectorAll('.patient-list li').forEach(n => n.classList.remove('active'));
  // li.classList.add('active'); // [수정 전] null 체크 없음
  if (li) li.classList.add('active'); // [수정 후]

  currentPatientIndex = i;
  const p = patients[i];
  if (ptNameEl) ptNameEl.textContent = p.name;
  if (ptSexAgeEl) ptSexAgeEl.textContent = `${p.sex} / ${p.age}세`;
  renderVitals(p.vitals);
  renderFeatures(p.features);
  renderViewer();
}


// ---------------------------------------------------------
// 5. 뷰어 렌더링 로직
// ---------------------------------------------------------
function renderViewer() {
  const currentMri = mri[idx];
  const hasImage = currentMri && currentMri.src;

  if (defaultImageTextEl) defaultImageTextEl.style.display = 'none';
  if (viewerImgEl) viewerImgEl.style.display = 'none';
  if (mprContainerEl) mprContainerEl.style.display = 'none';

  if (!hasImage) {
    if (defaultImageTextEl) {
      defaultImageTextEl.textContent = "No Image";
      defaultImageTextEl.style.display = 'flex';
    }
    return;
  }

  if (currentViewMode === '2D') {
    if (mprContainerEl) mprContainerEl.style.display = 'block';
    // 만약 mri 항목에 file 객체가 있으면 로컬 로더로 처리
    if (currentMri.file) {
      // 이미 로드되어 있지 않다면 로드
      if (!imageData || !dims) {
        loadNiftiFile(currentMri.file).catch(err => {
          console.error('로컬 NIfTI 로딩 오류:', err);
        });
      } else {
        renderAll();
      }
    } else {
      // src가 외부 URL(.nii/.nii.gz)이라면 URL 로더로 처리
      const name = currentMri.title || currentMri.src || '';
      if (name.endsWith('.nii') || name.endsWith('.nii.gz') || currentMri.src.endsWith('.nii') || currentMri.src.endsWith('.nii.gz')) {
        loadNiftiFromURL(currentMri.src).catch(err => {
          console.error('원격 NIfTI 로딩 오류:', err);
        });
      } else {
        // NIfTI가 아니면 캔버스 블랙 처리
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
  } else {
    if (viewerImgEl) {
      viewerImgEl.style.display = 'block';
      viewerImgEl.src = currentMri.src;
      viewerImgEl.alt = currentMri.title || '3D MRI';
    }
  }
}

// 2D 뷰어 - NIfTI 로드 (메인 로직)
async function loadNiftiFromURL(url) {
  try {
    if (loadingText) {
      loadingText.style.display = 'block';
      loadingText.textContent = 'NIfTI 로딩 중…';
    }

    const res = await fetch(url);
    const raw = new Uint8Array(await res.arrayBuffer());

    let buf = raw;
    // gzip 체크 및 압축 해제
    if (raw[0] === 0x1f && raw[1] === 0x8b && typeof pako !== 'undefined') {
      buf = pako.inflate(raw);
    }

    // [수정 전] 
    // const header = parseNiftiHeader(buf);
    // dims = [header.dim[1], header.dim[2], header.dim[3]];
    // imageData = extractImageData(buf, header.vox_offset, header);

    // [수정 후]
    const header = parseNiftiHeader(buf);
    dims = [header.dim[1], header.dim[2], header.dim[3]];
    // vox_offset을 정수로 변환 (중요)
    const offset = Math.round(header.vox_offset || 352);
    imageData = extractImageData(buf, offset, header);

    // 센터 슬라이스로 초기화
    currentSlice = {
      axial: Math.floor(dims[2] / 2),
      sagittal: Math.floor(dims[0] / 2),
      coronal: Math.floor(dims[1] / 2)
    };

    renderAll();
    if (loadingText) loadingText.style.display = 'none';
    console.log('NIfTI 로드 완료:', dims);

  } catch (err) {
    console.error(err);
    if (loadingText) loadingText.textContent = '로드 실패';
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
      // 화면상 클릭 좌표 (0 ~ width, 0 ~ height)
      const x = Math.floor((e.clientX - rect.left) * (canvas.width / rect.width));
      const y = Math.floor((e.clientY - rect.top) * (canvas.height / rect.height));

      if (x < 0 || x >= canvas.width || y < 0 || y >= canvas.height) return;

      // [수정] 렌더링 로직의 반전(Flip)을 고려하여 데이터 인덱스로 변환
      if (p.view === 'axial') {
        // renderAxial: x는 그대로, y는 flippedY (dims[1] - 1 - y)
        currentSlice.sagittal = x;
        currentSlice.coronal = dims[1] - 1 - y; // Y축 반전 보정
      } else if (p.view === 'sagittal') {
        // renderSagittal: x는 flippedY (dims[0] - 1 - x) -> 아니오, sagittal은 dims[1] 사용
        // Sagittal 뷰: 가로축(Y) = Coronal(dims[1]), 세로축(Z) = Axial(dims[2])
        // 렌더링 시: y(가로)는 flippedY(dims[1]-1-y), z(세로)는 flippedZ(dims[2]-1-z)

        // 마우스 X -> 데이터 Coronal (반전됨)
        currentSlice.coronal = dims[1] - 1 - x;
        // 마우스 Y -> 데이터 Axial (반전됨)
        currentSlice.axial = dims[2] - 1 - y;
      } else if (p.view === 'coronal') {
        // Coronal 뷰: 가로축(X) = Sagittal(dims[0]), 세로축(Z) = Axial(dims[2])
        // 렌더링 시: x(가로)는 flippedX(dims[0]-1-x), z(세로)는 flippedZ(dims[2]-1-z)

        // 마우스 X -> 데이터 Sagittal (반전됨)
        currentSlice.sagittal = dims[0] - 1 - x;
        // 마우스 Y -> 데이터 Axial (반전됨)
        currentSlice.axial = dims[2] - 1 - y;
      }
      renderAll();
    };

    // ... (이벤트 리스너 등록 코드는 그대로 유지) ...
    canvas.addEventListener('mousedown', e => { isDragging = true; updateSlice(e); });
    canvas.addEventListener('mousemove', e => { if (isDragging) updateSlice(e); });
    canvas.addEventListener('mouseup', () => isDragging = false);
    canvas.addEventListener('mouseleave', () => isDragging = false);
  });
}


// 3D 뷰어
async function loadAndRenderPlotly(maskBase64) {
  const container = document.getElementById('viewer3DContainer');
  const mprContainer = document.getElementById('mprContainer');

  if (!maskBase64) {
    alert("분석된 데이터가 없습니다.");
    return;
  }

  // UI 전환
  mprContainer.style.display = 'none';
  container.style.display = 'block';

  setTimeout(() => {
    Plotly.Plots.resize(container);
  }, 50);

  container.innerHTML = ''; // 기존 그래프 초기화 (필수)

  // 로딩 표시 (간단히)
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

    // 로딩 제거
    container.innerHTML = '';

    // Plotly 그리기 (핵심)
    // figData는 Python의 fig.to_json() 구조를 그대로 가짐: {data: [], layout: {}}
    const config = {
      responsive: true,
      displayModeBar: true, // 상단 툴바 표시
      displaylogo: false    // Plotly 로고 숨김
    };

    // 배경을 어둡게 강제 조정 (필요시)
    if (!figData.layout.scene) figData.layout.scene = {};
    // figData.layout.paper_bgcolor = '#000'; // 검은 배경 원하면 주석 해제

    await Plotly.newPlot('viewer3DContainer', figData.data, figData.layout, config);

    setTimeout(() => {
      const container = document.getElementById('viewer3DContainer');
      Plotly.Plots.resize(container);
      console.log("3D 뷰어 리사이징 완료");
    }, 100);

    // 3. 창 크기 조절 대응
    window.onresize = function () {
      Plotly.Plots.resize('viewer3DContainer');
    };

  } catch (err) {
    console.error(err);
    alert("3D 로드 실패: " + err.message);
    mprContainer.style.display = 'block';
    container.style.display = 'none';
  }
}

// ---------------------------------------------------------
// 6. 이벤트 리스너
// ---------------------------------------------------------

if (btnView2D) btnView2D.onclick = () => {
  currentViewMode = '2D';
  // UI 강제 전환
  if (mprContainerEl) mprContainerEl.style.display = 'block';
  if (viewer3DContainerEl) viewer3DContainerEl.style.display = 'none';
  renderViewer();
};

// [수정] 3D 버튼: 3D 모드로 설정하고 Plotly 로드
if (btnView3D) btnView3D.onclick = () => {
  currentViewMode = '3D';
  // 3D 함수 호출 (마스크 데이터가 없으면 함수 내부에서 alert 뜸)
  loadAndRenderPlotly(currentMaskBase64);
};

if (btnUpload) btnUpload.onclick = () => { if (fileInput) fileInput.click(); };


if (fileInput) {
  fileInput.onchange = async e => {
    const f = e.target.files[0];
    if (!f) return;

    const url = URL.createObjectURL(f);
    // 파일 객체도 함께 저장
    mri.push({ src: url, title: f.name, file: f });
    idx = mri.length - 1;

    maskData = null; // 초기화
    currentMaskBase64 = null;

    // NIfTI이면 자동 2D 모드로 전환하고 즉시 로컬 파일 로드
    if (f.name.endsWith('.nii') || f.name.endsWith('.nii.gz')) {
      currentViewMode = '2D';
      // ★ [핵심 수정] 없는 함수 loadNiftiFile(f) 대신 존재하는 loadNiftiFromURL(url) 사용
      await loadNiftiFromURL(url);
    } else {
      renderViewer();
    }

    log(`MRI 업로드: ${f.name}`);

    const p = patients[currentPatientIndex];

    try {
      log('서버 분석 시작…');

      const fd = new FormData();
      fd.append('file', f);
      fd.append('patient_id', p?.id || '');
      fd.append('age', String(p?.age || 0));
      fd.append('apoe4', String(p?.apoe4 || 0));
      fd.append('sex', p?.sex || 'M');
      if (p?.icv) fd.append('icv', String(p.icv));

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

      // 마스크가 base64로 오면 처리 (마스크는 MRI와 동일한 크기여야 함)
      if (result.mask_base64) {
        log('마스크 데이터 수신 완료. 처리 중...');
        currentMaskBase64 = result.mask_base64;
        await loadMaskFromBase64(result.mask_base64);
      } else {
        log('결과에 마스크 데이터가 없습니다.');
      }

      log('분석 완료: ' + (result.label || 'N/A'));
      // 업로드 후 renderAll로 덮어쓰기 (마스크까지 있으면 오버레이 적용)
      renderAll();

    } catch (err) {
      console.error(err);
      log(`분석 실패: ${err.message}`, true);
    }
  };
}

// [추가] Base64 문자열을 마스크 데이터로 로드
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

    // dims 일치 여부 확인
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

if (brightnessEl) {
  brightnessEl.addEventListener('input', (e) => {
    brightness = parseFloat(e.target.value);
    const bv = document.getElementById('brightnessValue');
    if (bv) bv.textContent = brightness.toFixed(1);
    renderAll();
  });
}

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
// 7. NIfTI 파싱 및 렌더링 (★ 핵심 수정됨)
// ---------------------------------------------------------

function parseNiftiHeader(buf) {
  // [수정 전] 
  // const v = new DataView(buf.buffer);
  // [수정 후] byteOffset 명시
  const v = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  return {
    dim: [v.getInt16(40, true), v.getInt16(42, true), v.getInt16(44, true), v.getInt16(46, true)],
    datatype: v.getInt16(70, true),
    bitpix: v.getInt16(72, true),
    vox_offset: v.getFloat32(108, true)
  };
}

function extractImageData(buf, offset, header, isMask = false) {
  // [수정 전]
  // const v = new DataView(buf.buffer);
  // [수정 후] byteOffset 명시
  const v = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  const n = header.dim[1] * header.dim[2] * header.dim[3];
  const data = new Float32Array(n);
  let min = Infinity, max = -Infinity;

  for (let i = 0; i < n; i++) {
    let val = 0;
    // [추가] 범위 초과 방지 안전장치
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

  // 0~255 정규화 (안전장치 추가: range가 0이거나 무한대일 때 방지)
  if (!isMask) {
    const range = max - min;
    if (range > 0 && isFinite(range)) {
      for (let i = 0; i < n; i++) {
        data[i] = ((data[i] - min) / range) * 255;
      }
    }
  } else {
    // 마스크인 경우: 디버깅용 로그 출력
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
  const ctx = c.getContext('2d', { willReadFrequently: true });
  c.width = dims[0];
  c.height = dims[1];

  const img = ctx.createImageData(dims[0], dims[1]);
  const z = currentSlice.axial;

  for (let y = 0; y < dims[1]; y++) {
    const flippedY = dims[1] - 1 - y;
    for (let x = 0; x < dims[0]; x++) {
      const idx = x + flippedY * dims[0] + z * dims[0] * dims[1];
      // [추가] 범위 체크
      if (idx < 0 || idx >= imageData.length) continue;

      const v = Math.min(255, imageData[idx] * brightness);
      const p = (y * dims[0] + x) * 4;

      // 기본: 흑백
      img.data[p] = v;
      img.data[p + 1] = v;
      img.data[p + 2] = v;
      img.data[p + 3] = 255;

      if (maskData && idx < maskData.length) {
        const rawVal = maskData[idx];

        // 값이 소수점일 수도 있으니 반올림하여 정수로 처리
        const maskVal = Math.round(rawVal);

        if (maskVal > 0) {
          // HippMapp3r: 보통 1=Left, 2=Right
          // (혹시 반대라면 아래 조건문 숫자를 바꾸세요)

          if (maskVal === 1) {
            // [왼쪽 해마] -> 초록색 (Green)
            img.data[p] = 0;       // R
            img.data[p + 1] = 255; // G (선명한 초록)
            img.data[p + 2] = 0;   // B
            img.data[p + 3] = 150; // 투명도
          } else {
            // [오른쪽 해마] -> 빨간색 (Red)
            img.data[p] = 255;     // R (선명한 빨강)
            img.data[p + 1] = 0;   // G
            img.data[p + 2] = 0;   // B
            img.data[p + 3] = 150; // 투명도
          }
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

      img.data[p] = v;
      img.data[p + 1] = v;
      img.data[p + 2] = v;
      img.data[p + 3] = 255;

      if (maskData && idx < maskData.length) {
        const rawVal = maskData[idx];

        // 값이 소수점일 수도 있으니 반올림하여 정수로 처리
        const maskVal = Math.round(rawVal);

        if (maskVal > 0) {
          // HippMapp3r: 보통 1=Left, 2=Right
          // (혹시 반대라면 아래 조건문 숫자를 바꾸세요)

          if (maskVal === 1) {
            // [왼쪽 해마] -> 초록색 (Green)
            img.data[p] = 0;       // R
            img.data[p + 1] = 255; // G (선명한 초록)
            img.data[p + 2] = 0;   // B
            img.data[p + 3] = 150; // 투명도
          } else {
            // [오른쪽 해마] -> 빨간색 (Red)
            img.data[p] = 255;     // R (선명한 빨강)
            img.data[p + 1] = 0;   // G
            img.data[p + 2] = 0;   // B
            img.data[p + 3] = 150; // 투명도
          }
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

      img.data[p] = v;
      img.data[p + 1] = v;
      img.data[p + 2] = v;
      img.data[p + 3] = 255;

      if (maskData && idx < maskData.length) {
        const rawVal = maskData[idx];

        // 값이 소수점일 수도 있으니 반올림하여 정수로 처리
        const maskVal = Math.round(rawVal);

        if (maskVal > 0) {
          // HippMapp3r: 보통 1=Left, 2=Right
          // (혹시 반대라면 아래 조건문 숫자를 바꾸세요)

          if (maskVal === 1) {
            // [왼쪽 해마] -> 초록색 (Green)
            img.data[p] = 0;       // R
            img.data[p + 1] = 255; // G (선명한 초록)
            img.data[p + 2] = 0;   // B
            img.data[p + 3] = 150; // 투명도
          } else {
            // [오른쪽 해마] -> 빨간색 (Red)
            img.data[p] = 255;     // R (선명한 빨강)
            img.data[p + 1] = 0;   // G
            img.data[p + 2] = 0;   // B
            img.data[p + 3] = 150; // 투명도
          }
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

// 초기화 실행
async function init() {
  if (issuedAtEl) issuedAtEl.textContent = todayISO();
  await loadPatients();
  setupInteraction();
}

init();