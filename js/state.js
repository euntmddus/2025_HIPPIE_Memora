// js/state.js
// 전역 상태, DOM 참조, 공용 유틸

let currentViewMode = '2D';
let patients = [];
let mri = [
  { src: 'assets/mri-axial.png', title: 'Axial', type: "PNG" },
  { src: 'assets/mri-coronal.png', title: 'Coronal', type: "PNG" }
];
let idx = 0;
let currentPatientIndex = 0;

let maskData = null;
let currentMaskBase64 = null;

const list = document.getElementById('patientList');
const ptNameEl = document.getElementById('ptName');
const ptSexAgeEl = document.getElementById('ptSexAge');
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

// NIfTI 상태
let imageData = null;
let dims = null;
let brightness = 1.0;
let currentSlice = { axial: 0, sagittal: 0, coronal: 0 };

// 유틸
function todayISO() {
  const t = new Date();
  return `${t.getFullYear()}-${String(t.getMonth() + 1).padStart(2, '0')}-${String(t.getDate()).padStart(2, '0')}`;
}

function log(msg, isError = false) {
  if (!logList) return;
  const row = document.createElement('div');
  row.className = 'row';
  if (isError) row.style.color = 'red';
  const now = new Date().toLocaleString();
  row.innerHTML = `<span class="mono">${now}</span><span>${msg}</span><span class="mono">sys</span>`;
  logList.prepend(row);
}

async function openExamFromHistory(examId) {
  try {
    console.log('[history] openExamFromHistory called with examId =', examId);

    // 1) 검사 상세 조회 (반드시 절대 URL 사용!)
    const res = await fetch(`http://127.0.0.1:8000/api/exams/${examId}`);
    if (!res.ok) {
      console.error('exam detail fetch failed:', res.status);
      alert('검사 정보를 불러오지 못했습니다.');
      return;
    }
    const json = await res.json();

    if (json.status !== 'success') {
      console.error('exam api error:', json);
      alert('검사 정보를 불러오지 못했습니다.');
      return;
    }

    const data = json.data;
    console.log('[history] exam detail:', data);

    const detailsBox = document.getElementById('detailsBox');
    if (detailsBox) {
      detailsBox.value = data.detailed_opinion || '';
    }

    // 2) NIfTI 파일 다시 받아오기
    const niftiResp = await fetch(data.file_url);
    if (!niftiResp.ok) {
      console.error('nifti fetch failed:', niftiResp.status);
      alert('NIfTI 파일을 불러오지 못했습니다.');
      return;
    }
    const niftiBlob = await niftiResp.blob();

    const file = new File(
      [niftiBlob],
      data.filename || 'history.nii.gz',
      { type: niftiBlob.type || 'application/octet-stream' }
    );

    // 3) 업로드 때와 같은 방식으로 전역 상태 갱신
    const url = URL.createObjectURL(file);
    mri.push({ src: url, title: file.name, file });
    idx = mri.length - 1;

    maskData = null;
    currentMaskBase64 = null;
    window.latest_mask_base64 = null;
    imageData = null;
    dims = null;

    // 4) 2D 뷰어에 렌더링
    currentViewMode = '2D';
    await loadNiftiFile(file);
    log(`[ 검사 이력 호출 ] 파일: ${file.name} | exam_id = ${examId}`);

    // 5) 검사 결과 요약 / 영상 분석 지표 갱신
    const converted = {
      features: data.features,   // 영상 분석 지표
      summary: data.summary,     // 텍스트 요약
      label: data.label,         // AD / CN
      probs: data.probs,         // 확률
      exam_datetime: null,
      mask_base64: data.mask_base64 || null
    };

    applyServerResult(converted);

    if (data.mask_base64) {
      window.latest_mask_base64 = data.mask_base64;
      currentMaskBase64 = data.mask_base64;
      await loadMaskFromBase64(data.mask_base64);
    } else {
      if (typeof clearViewerMask === 'function') {
        clearViewerMask();
        console.log('[history] 마스크 데이터 없음 -> 뷰어 마스크 제거');
      }
    }

  } catch (err) {
    console.error('[history] openExamFromHistory error:', err);
    alert('검사 이력을 불러오는 중 오류가 발생했습니다.');
  }
}

window.openExamFromHistory = openExamFromHistory;
