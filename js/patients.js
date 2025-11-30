// js/patients.js
// 환자 목록 로드, 선택, 이력, vitals/피처 렌더링

async function loadPatients() {
  try {
    const res = await fetch('http://127.0.0.1:8000/api/patients');
    if (!res.ok) throw new Error('환자 목록 로드 실패');
    const data = await res.json();

    patients = data.map(row => {
      const parts = [];
      if (row.height_cm) parts.push(`키 ${row.height_cm}cm`);
      if (row.weight_kg) parts.push(`몸무게 ${row.weight_kg}kg`);

      return {
        id: row.patient_id,
        name: row.name,
        sex: row.sex,
        age: row.age,
        vitals: parts.join(' | '),
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

function renderPatients() {
  if (!list) return;
  list.innerHTML = '';
  patients.forEach((p, i) => {
    const li = document.createElement('li');
    li.innerHTML = `<span>•</span><span>${p.name} (${p.id})</span>`;
    li.onclick = () => selectPatient(i, li);
    list.appendChild(li);
    if (i === 0) selectPatient(0, li);
  });
}

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

function renderHistory(historyData) {
  const historyList = document.getElementById('historyList');
  if (!historyList) return;
  historyList.innerHTML = '';

  if (!historyData.length) {
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
    li.style.cursor = 'pointer';

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

    li.addEventListener('click', () => {
      console.log('[history] clicked exam_id =', item.exam_id);
      if (window.openExamFromHistory) {
        window.currentExamId = item.exam_id;
        window.openExamFromHistory(item.exam_id);
      } else {
        console.error('openExamFromHistory is not defined on window');
      }
    });

    historyList.appendChild(li);
  });
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
    ['총 해마 부피', features.total_hipp_vol_mm3 ? `${features.total_hipp_vol_mm3} mm³` : '—'],
    ['좌 해마 부피', features.left_hipp_vol_mm3 ? `${features.left_hipp_vol_mm3} mm³` : '—'],
    ['우 해마 부피', features.right_hipp_vol_mm3 ? `${features.right_hipp_vol_mm3} mm³` : '—'],

    ['총 해마 Z-score',
      features.total_hipp_vol_zscore != null
        ? `${features.total_hipp_vol_zscore.toFixed(2)}`
        : '—'
    ],
    ['좌 해마 Z-score',
      features.left_hipp_vol_zscore != null
        ? `${features.left_hipp_vol_zscore.toFixed(2)}`
        : '—'
    ],
    ['우 해마 Z-score',
      features.right_hipp_vol_zscore != null
        ? `${features.right_hipp_vol_zscore.toFixed(2)}`
        : '—'
    ],

    [
      'ICV 보정 해마 지수 (좌/우/총)',
      (features.left_hipp_vol_icv_norm != null &&
        features.right_hipp_vol_icv_norm != null &&
        features.total_hipp_vol_icv_norm != null)
        ? `${features.left_hipp_vol_icv_norm} / ${features.right_hipp_vol_icv_norm} / ${features.total_hipp_vol_icv_norm}`
        : '—'
    ],
    [
      'APOE4 유전자형',
      features.APOE4 != null
        ? features.APOE4
        : (features.apoe4 != null ? features.apoe4 : '정보 없음')
    ]
  ];

  rows.forEach(([k, v]) => {
    const li = document.createElement('li');
    li.innerHTML = `<span class="vkey">${k}</span><span class="vval">${v}</span>`;
    featureListEl.appendChild(li);
  });
}

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

const detailsBox = document.getElementById('detailsBox');

let detailSaveTimer = null;

if (detailsBox) {
  detailsBox.addEventListener('input', () => {
    clearTimeout(detailSaveTimer);

    // 타이핑 멈춘 뒤 500ms 후 저장
    detailSaveTimer = setTimeout(() => {
      autoSaveDetail(detailsBox.value);
    }, 500);
  });
}

async function autoSaveDetail(text) {
  if (!window.currentExamId) return;

  const formData = new FormData();
  formData.append("detailed_opinion", text);

  try {
    await fetch(`http://127.0.0.1:8000/api/exams/${window.currentExamId}/opinion`, {
      method: 'POST',
      body: formData   // 서버는 Form(...) 이므로 JSON 말고 FormData!
    });
  } catch (err) {
    console.error('상세소견 자동 저장 실패', err);
  }
}
