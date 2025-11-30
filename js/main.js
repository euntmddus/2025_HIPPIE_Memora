// js/main.js
// 2D/3D 버튼, 업로드 처리, 초기화

if (btnView2D) {
  btnView2D.onclick = () => {
    currentViewMode = '2D';
    if (mprContainerEl) mprContainerEl.style.display = 'block';
    if (viewer3DContainerEl) viewer3DContainerEl.style.display = 'none';
    renderViewer();
  };
}

if (btnView3D) {
  btnView3D.onclick = () => {
    currentViewMode = '3D';
    const dataToRender = window.latest_mask_base64 || currentMaskBase64;
    if (!dataToRender) {
      alert('분석된 데이터가 아직 없습니다.');
      return;
    }
    loadAndRenderPlotly(dataToRender);
  };
}

if (btnUpload) {
  btnUpload.onclick = () => {
    if (fileInput) {
      fileInput.value = '';
      fileInput.click();
    }
  };
}

if (fileInput) {
  fileInput.onchange = async e => {
    const f = e.target.files[0];
    if (!f) return;

    // === 디버깅 로그 추가 ===
    console.log('=== 업로드 시작 ===');
    console.log('currentPatientIndex:', currentPatientIndex);
    console.log('patients 배열:', patients);
    console.log('선택된 환자:', patients[currentPatientIndex]);
    console.log('patient_id:', patients[currentPatientIndex]?.id);
    // =====================


    const url = URL.createObjectURL(f);
    mri.push({ src: url, title: f.name, file: f });
    idx = mri.length - 1;

    maskData = null;
    currentMaskBase64 = null;
    window.latest_mask_base64 = null;
    imageData = null;
    dims = null;

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
      const examDT = new Date(now.getTime() - (now.getTimezoneOffset() * 60000))
        .toISOString().slice(0, 19).replace('T', ' ');
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
        log(`[ 검사 완료 ] 저장된 검사일시: ${result.exam_datetime}`);
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

      log('[ 분석 완료 ] ' + (result.label || 'N/A'));
    } catch (err) {
      console.error(err);
      log(`[ 분석 실패 ] ${err.message}`, true);
      alert("분석 중 오류가 발생했습니다.");
    } finally {
      if (btnUpload) btnUpload.disabled = false;
    }
  };
}

window.openExamFromHistory = async function (examId) {
  try {
    log(`검사 기록 불러오는 중... (exam_id: ${examId})`);

    const res = await fetch(`http://127.0.0.1:8000/api/exams/${examId}`);
    if (!res.ok) throw new Error('검사 기록 로드 실패');

    const result = await res.json();

    if (result.status === 'success' && result.data) {
      const data = result.data;

      if (data.features && patients[currentPatientIndex]) {
        data.features.apoe4 = patients[currentPatientIndex].apoe4;
      }

      applyServerResult({
        label: data.label,
        probs: data.probs,
        summary: data.summary,
        features: data.features,
        total_hipp_vol_zscore: data.total_hipp_vol_zscore,
        left_hipp_vol_zscore: data.left_hipp_vol_zscore,
        right_hipp_vol_zscore: data.right_hipp_vol_zscore,
      });

      const detailsBox = document.getElementById('detailsBox');
      if (detailsBox && data.detailed_opinion) {
        detailsBox.value = data.detailed_opinion;
      }

      if (data.mask_base64) {
        window.latest_mask_base64 = data.mask_base64;
        currentMaskBase64 = data.mask_base64;
      }

      if (data.file_url) {
        await loadNiftiFromURL(data.file_url);

        if (data.mask_base64) {
          await loadMaskFromBase64(data.mask_base64);
        }
      }

      // 3D 모드일 때 자동 렌더링
      if (currentViewMode === '3D' && data.mask_base64) {
        await loadAndRenderPlotly(data.mask_base64);
      }

      if (currentViewMode === '2D') {
        renderViewer();
      }

      log(`검사 기록 불러오기 완료 (exam_id: ${examId})`);


    }
  } catch (err) {
    console.error('검사 기록 로드 오류:', err);
    log('검사 기록 불러오기 실패: ' + err.message, true);
  }
};

async function init() {
  if (issuedAtEl) issuedAtEl.textContent = todayISO();
  await loadPatients();
  setupInteraction();
}

init();
