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

      log('[ 분석 완료 ]' + (result.label || 'N/A'));
    } catch (err) {
      console.error(err);
      log(`[ 분석 실패 ] ${err.message}`, true);
      alert("분석 중 오류가 발생했습니다.");
    } finally {
      if (btnUpload) btnUpload.disabled = false;
    }
  };
}

async function init() {
  if (issuedAtEl) issuedAtEl.textContent = todayISO();
  await loadPatients();
  setupInteraction();
}

init();
