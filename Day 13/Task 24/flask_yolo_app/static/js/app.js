(() => {
  const dropZone = document.getElementById('dropZone');
  const fileInput = document.getElementById('fileInput');
  const browseBtn = document.getElementById('browseBtn');
  const previewGrid = document.getElementById('previewGrid');
  const controls = document.getElementById('controls');
  const detectBtn = document.getElementById('detectBtn');
  const clearBtn = document.getElementById('clearBtn');
  const confSlider = document.getElementById('confSlider');
  const confValue = document.getElementById('confValue');
  const loadingOverlay = document.getElementById('loadingOverlay');
  const resultsSection = document.getElementById('resultsSection');
  const resultsGrid = document.getElementById('resultsGrid');
  const resultsCount = document.getElementById('resultsCount');

  let selectedFiles = [];

  // ── Confidence slider
  confSlider.addEventListener('input', () => {
    confValue.textContent = parseFloat(confSlider.value).toFixed(2);
  });

  // ── Browse button
  browseBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    fileInput.click();
  });

  // ── Drop zone click (whole area)
  dropZone.addEventListener('click', () => fileInput.click());

  // ── File input change
  fileInput.addEventListener('change', (e) => {
    addFiles(Array.from(e.target.files));
    fileInput.value = ''; // allow re-selecting same file
  });

  // ── Drag & drop
  ['dragenter', 'dragover'].forEach(evt => {
    dropZone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropZone.classList.add('dragover');
    });
  });

  ['dragleave', 'drop'].forEach(evt => {
    dropZone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropZone.classList.remove('dragover');
    });
  });

  dropZone.addEventListener('drop', (e) => {
    const files = Array.from(e.dataTransfer.files);
    addFiles(files);
  });

  // ── Add files
  function addFiles(files) {
    const allowed = ['image/png', 'image/jpeg', 'image/gif', 'image/bmp', 'image/webp'];
    const valid = files.filter(f => allowed.includes(f.type));

    if (valid.length === 0) {
      showToast('No valid image files selected.', 'error');
      return;
    }

    valid.forEach(file => {
      // Avoid duplicates by name+size
      const exists = selectedFiles.find(f => f.name === file.name && f.size === file.size);
      if (!exists) selectedFiles.push(file);
    });

    renderPreviews();
    controls.style.display = 'flex';
  }

  // ── Render previews
  function renderPreviews() {
    previewGrid.innerHTML = '';
    selectedFiles.forEach((file, idx) => {
      const reader = new FileReader();
      reader.onload = (e) => {
        const item = document.createElement('div');
        item.className = 'preview-item';
        item.innerHTML = `
          <img src="${e.target.result}" alt="${file.name}">
          <button class="remove-btn" data-idx="${idx}" title="Remove">✕</button>
          <div class="file-name">${file.name}</div>
        `;
        previewGrid.appendChild(item);
      };
      reader.readAsDataURL(file);
    });

    // Delegate remove buttons
    previewGrid.addEventListener('click', (e) => {
      if (e.target.classList.contains('remove-btn')) {
        const idx = parseInt(e.target.dataset.idx);
        selectedFiles.splice(idx, 1);
        renderPreviews();
        if (selectedFiles.length === 0) {
          controls.style.display = 'none';
        }
      }
    });
  }

  // ── Clear all
  clearBtn.addEventListener('click', () => {
    selectedFiles = [];
    previewGrid.innerHTML = '';
    controls.style.display = 'none';
    resultsSection.style.display = 'none';
    resultsGrid.innerHTML = '';
  });

  // ── Detect
  detectBtn.addEventListener('click', async () => {
    if (selectedFiles.length === 0) {
      showToast('Please select at least one image.', 'error');
      return;
    }

    detectBtn.disabled = true;
    loadingOverlay.style.display = 'flex';
    resultsSection.style.display = 'none';

    const formData = new FormData();
    selectedFiles.forEach(file => formData.append('files', file));
    formData.append('confidence', confSlider.value);

    try {
      const response = await fetch('/predict', {
        method: 'POST',
        body: formData
      });

      const data = await response.json();

      if (!response.ok) {
        showToast(data.error || 'Prediction failed.', 'error');
        return;
      }

      renderResults(data.results);
    } catch (err) {
      showToast('Network error: ' + err.message, 'error');
    } finally {
      detectBtn.disabled = false;
      loadingOverlay.style.display = 'none';
    }
  });

  // ── Render results
  function renderResults(results) {
    resultsGrid.innerHTML = '';
    const total = results.filter(r => !r.error).length;
    const totalObjs = results.reduce((s, r) => s + (r.total_objects || 0), 0);
    resultsCount.textContent = `${total} image${total !== 1 ? 's' : ''} · ${totalObjs} object${totalObjs !== 1 ? 's' : ''} detected`;

    results.forEach(result => {
      if (result.error) {
        const card = document.createElement('div');
        card.className = 'error-card';
        card.innerHTML = `<strong>✕ ${escHtml(result.original_name || result.filename)}</strong><br>${escHtml(result.error)}`;
        resultsGrid.appendChild(card);
        return;
      }

      const card = document.createElement('div');
      card.className = 'result-card';

      const detectionsHtml = result.detections.length > 0
        ? result.detections.map(d => `
            <div class="det-item">
              <span class="det-label">${escHtml(d.label)}</span>
              <div class="conf-bar-wrap">
                <div class="conf-bar">
                  <div class="conf-bar-fill" style="width: ${d.confidence}%"></div>
                </div>
                <span class="conf-pct">${d.confidence}%</span>
              </div>
            </div>
          `).join('')
        : `<div class="no-detection">No objects detected above threshold</div>`;

      // Count unique classes
      const uniqueClasses = [...new Set(result.detections.map(d => d.label))];

      card.innerHTML = `
        <div class="card-images">
          <div class="img-wrap">
            <span class="img-label">Original</span>
            <img src="${result.upload_url}?t=${Date.now()}" alt="Original" onclick="openFullscreen(this.src)">
          </div>
          <div class="divider-v"></div>
          <div class="img-wrap">
            <span class="img-label">Detected</span>
            <img src="${result.result_url}?t=${Date.now()}" alt="Result" onclick="openFullscreen(this.src)">
          </div>
        </div>
        <div class="card-body">
          <div class="card-filename">📄 ${escHtml(result.original_name)}</div>
          <div class="detection-list">${detectionsHtml}</div>
          <div class="card-summary">
            <span>${uniqueClasses.length} class${uniqueClasses.length !== 1 ? 'es' : ''}: ${uniqueClasses.map(escHtml).join(', ') || '—'}</span>
            <span class="count-badge">${result.total_objects} obj</span>
          </div>
        </div>
      `;

      resultsGrid.appendChild(card);
    });

    resultsSection.style.display = 'block';
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  // ── Fullscreen viewer
  window.openFullscreen = function(src) {
    const overlay = document.createElement('div');
    overlay.style.cssText = `
      position:fixed;inset:0;background:rgba(0,0,0,0.92);z-index:9999;
      display:flex;align-items:center;justify-content:center;cursor:zoom-out;
    `;
    const img = document.createElement('img');
    img.src = src;
    img.style.cssText = 'max-width:92vw;max-height:92vh;border-radius:8px;box-shadow:0 0 60px rgba(0,0,0,0.8)';
    overlay.appendChild(img);
    overlay.addEventListener('click', () => document.body.removeChild(overlay));
    document.body.appendChild(overlay);
  };

  // ── Toast
  function showToast(msg, type = 'info') {
    const t = document.createElement('div');
    const color = type === 'error' ? '#ff4566' : '#00ffb4';
    t.style.cssText = `
      position:fixed;bottom:24px;right:24px;z-index:10000;
      background:#0e141b;border:1px solid ${color};color:${color};
      font-family:'Space Mono',monospace;font-size:0.8rem;
      padding:12px 20px;border-radius:8px;
      box-shadow:0 4px 24px rgba(0,0,0,0.4);
      animation:fadeIn 0.3s ease;
    `;
    t.textContent = msg;
    document.body.appendChild(t);
    setTimeout(() => t.remove(), 3500);
  }

  function escHtml(str) {
    const d = document.createElement('div');
    d.textContent = str;
    return d.innerHTML;
  }
})();
