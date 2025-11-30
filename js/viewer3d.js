// js/viewer3d.js
// 해마 마스크 NIfTI(base64) → Plotly 3D Mesh

async function loadAndRenderPlotly(maskBase64) {
  const container = document.getElementById('viewer3DContainer');
  const mprContainer = document.getElementById('mprContainer');

  if (!maskBase64) {
    alert("분석된 데이터가 없습니다.");
    return;
  }
  if (!container) return;

  // 2D 숨기고 3D 보여주기
  if (mprContainer) mprContainer.style.display = 'none';
  container.style.display = 'block';
  container.innerHTML = '';

  // 🔹 부모 패널 기준으로 정사각형 크기 결정
  let size = 600;
  const parent = container.parentElement;
  if (parent) {
    const maxW = parent.clientWidth;
    const maxH = parent.clientHeight;
    size = Math.min(maxW, maxH);
  }
  container.style.width = size + 'px';
  container.style.height = size + 'px';

  // 로딩 표시
  const loading = document.createElement('div');
  loading.textContent = "3D 모델링 중...";
  loading.style.position = 'absolute';
  loading.style.top = '50%';
  loading.style.left = '50%';
  loading.style.transform = 'translate(-50%, -50%)';
  loading.style.padding = '6px 10px';
  loading.style.background = 'rgba(255,255,255,0.9)';
  loading.style.borderRadius = '4px';
  loading.style.fontSize = '12px';
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
    if (!figData.layout) figData.layout = {};
    if (!figData.layout.scene) figData.layout.scene = {};

    figData.layout.width = size;
    figData.layout.height = size;

    await Plotly.newPlot(container, figData.data, figData.layout, config);

    // 처음 한 번 리사이즈
    setTimeout(() => {
      Plotly.Plots.resize(container);
    }, 100);

    // 창 크기 바뀔 때도 정사각형 유지
    window.addEventListener('resize', () => {
      if (container.offsetParent === null) return;

      const parent2 = container.parentElement;
      if (parent2) {
        const maxW2 = parent2.clientWidth;
        const maxH2 = parent2.clientHeight;
        const newSize = Math.min(maxW2, maxH2);
        container.style.width = newSize + 'px';
        container.style.height = newSize + 'px';

        Plotly.Plots.resize(container);
      }
    });

  } catch (err) {
    console.error(err);
    alert("3D 로드 실패: " + err.message);
    if (mprContainer) mprContainer.style.display = 'block';
    container.style.display = 'none';
  } finally {
    if (loading) loading.style.display = 'none';
  }
}
