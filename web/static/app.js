/**
 * OPM-Synth Studio Frontend Application.
 * Interactive Reservoir Explorer & Deck Inspector.
 */

const STATE = {
  activeTab: "tab-files",
  activeProperty: "PORO",
  activeLayer: 1,
  activeEnsemble: null,
  activeModel: "SYNTH_001",
  activeFileRelPath: null,
  filesList: [],
  ensembles: [],
  system: null
};

function isDarkMode() {
  return document.documentElement.classList.contains("dark");
}

function getPlotTheme() {
  const dark = isDarkMode();
  return {
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    font: { color: dark ? "#71717a" : "#71717a", family: "Inter, sans-serif", size: 11 },
    margin: { l: 40, r: 20, t: 25, b: 30 },
    xaxis: { gridcolor: dark ? "#18181c" : "#f0f0f3", zerolinecolor: dark ? "#27272a" : "#e4e4e7" },
    yaxis: { gridcolor: dark ? "#18181c" : "#f0f0f3", zerolinecolor: dark ? "#27272a" : "#e4e4e7" }
  };
}

document.addEventListener("DOMContentLoaded", () => {
  initTheme();
  initUIEventListeners();
  loadSystemStatus();
  loadEnsemblesList();
});

function initTheme() {
  const saved = localStorage.getItem("theme");
  if (saved === "light") {
    document.documentElement.classList.remove("dark");
    document.getElementById("theme-btn-label").innerText = "Темная тема";
  } else {
    document.documentElement.classList.add("dark");
    document.getElementById("theme-btn-label").innerText = "Светлая тема";
  }
}

function toggleTheme() {
  const isDark = document.documentElement.classList.toggle("dark");
  localStorage.setItem("theme", isDark ? "dark" : "light");
  document.getElementById("theme-btn-label").innerText = isDark ? "Светлая тема" : "Темная тема";

  // Re-render active view to update Plotly canvas themes
  if (STATE.activeTab === "tab-reservoir") renderReservoirViews();
  else if (STATE.activeTab === "tab-3d") render3DMesh();
  else if (STATE.activeTab === "tab-qc") renderQcDiagnostic();
  else if (STATE.activeTab === "tab-flow") renderFlowCurves();
}

function initUIEventListeners() {
  // Theme switcher
  document.getElementById("btn-toggle-theme").addEventListener("click", toggleTheme);

  // Tabs switching
  document.querySelectorAll(".tab-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
      document.querySelectorAll(".tab-content").forEach(tc => tc.classList.add("hidden"));

      btn.classList.add("active");
      const tabId = btn.getAttribute("data-tab");
      document.getElementById(tabId).classList.remove("hidden");
      STATE.activeTab = tabId;

      if (tabId === "tab-files") {
        loadFileContent();
      } else if (tabId === "tab-reservoir") {
        renderReservoirViews();
      } else if (tabId === "tab-3d") {
        render3DMesh();
      } else if (tabId === "tab-qc") {
        renderQcDiagnostic();
      } else if (tabId === "tab-flow") {
        renderFlowCurves();
      }
    });
  });

  // Slider bindings
  bindSlider("param-n-models", "val-n-models", v => `${v}`);
  bindSlider("param-rx", "val-rx", v => `${parseFloat(v).toFixed(1)} ячеек`);
  bindSlider("param-rz", "val-rz", v => `${parseFloat(v).toFixed(1)} слоев`);
  bindSlider("param-perturb", "val-perturb", v => `${Math.round(v * 100)}%`);
  bindSlider("param-corr", "val-corr", v => `${parseFloat(v).toFixed(2)}`);
  bindSlider("param-aniso", "val-aniso", v => `${parseFloat(v).toFixed(2)}`);

  // Presets dropdown
  document.getElementById("preset-selector").addEventListener("change", e => {
    applyPreset(e.target.value);
  });

  // Property selector
  document.getElementById("active-property").addEventListener("change", e => {
    STATE.activeProperty = e.target.value;
    if (STATE.activeTab === "tab-reservoir" || STATE.activeTab === "tab-3d") {
      renderReservoirViews();
    }
  });

  // Layer slider
  const layerInput = document.getElementById("active-layer");
  layerInput.addEventListener("input", e => {
    STATE.activeLayer = parseInt(e.target.value);
    document.getElementById("active-layer-text").innerText = `${STATE.activeLayer} / 6`;
    if (STATE.activeTab === "tab-reservoir") {
      renderReservoirViews();
    }
  });

  // Model selector
  document.getElementById("active-model-select").addEventListener("change", e => {
    STATE.activeModel = e.target.value;
    loadModelFilesList();
    if (STATE.activeTab === "tab-reservoir") renderReservoirViews();
    if (STATE.activeTab === "tab-3d") render3DMesh();
    if (STATE.activeTab === "tab-qc") renderQcDiagnostic();
    if (STATE.activeTab === "tab-flow") renderFlowCurves();
  });

  // Ensemble selector
  document.getElementById("ensemble-selector").addEventListener("change", e => {
    STATE.activeEnsemble = e.target.value;
    loadEnsembleModels(e.target.value);
  });

  // Generate Button
  document.getElementById("btn-generate").addEventListener("click", startBatchGeneration);

  // Export buttons
  document.getElementById("btn-export-zip-top").addEventListener("click", () => {
    if (!STATE.activeEnsemble) return;
    window.location.href = `/api/export-zip?ensemble_id=${STATE.activeEnsemble}`;
  });

  document.getElementById("btn-download-active-model-zip").addEventListener("click", () => {
    if (!STATE.activeEnsemble || !STATE.activeModel) return;
    window.location.href = `/api/export-zip?ensemble_id=${STATE.activeEnsemble}&model_id=${STATE.activeModel}`;
  });

  document.getElementById("btn-download-single-file").addEventListener("click", () => {
    if (!STATE.activeEnsemble || !STATE.activeModel || !STATE.activeFileRelPath) return;
    window.location.href = `/api/download-single-file?ensemble_id=${STATE.activeEnsemble}&model_id=${STATE.activeModel}&rel_path=${encodeURIComponent(STATE.activeFileRelPath)}`;
  });

  // Copy code button
  document.getElementById("btn-copy-code").addEventListener("click", () => {
    const code = document.getElementById("code-content").innerText;
    navigator.clipboard.writeText(code).then(() => {
      const btnText = document.getElementById("copy-btn-text");
      btnText.innerText = "Скопировано!";
      setTimeout(() => { btnText.innerText = "Копировать"; }, 1500);
    });
  });
}

function bindSlider(sliderId, textId, formatFn) {
  const slider = document.getElementById(sliderId);
  const text = document.getElementById(textId);
  slider.addEventListener("input", () => {
    text.innerText = formatFn(slider.value);
  });
}

function applyPreset(presetKey) {
  const presets = {
    balanced: { rx: 4.0, rz: 2.5, perturb: 0.25, corr: 0.90, aniso: 0.20 },
    layering: { rx: 6.0, rz: 1.0, perturb: 0.35, corr: 0.88, aniso: 0.05 },
    channels: { rx: 8.0, rz: 3.0, perturb: 0.30, corr: 0.92, aniso: 0.15 },
    high_hetero: { rx: 3.0, rz: 1.5, perturb: 0.45, corr: 0.80, aniso: 0.25 }
  };

  const p = presets[presetKey];
  if (!p) return;

  setSlider("param-rx", "val-rx", p.rx, `${p.rx.toFixed(1)} ячеек`);
  setSlider("param-rz", "val-rz", p.rz, `${p.rz.toFixed(1)} слоев`);
  setSlider("param-perturb", "val-perturb", p.perturb, `${Math.round(p.perturb * 100)}%`);
  setSlider("param-corr", "val-corr", p.corr, `${p.corr.toFixed(2)}`);
  setSlider("param-aniso", "val-aniso", p.aniso, `${p.aniso.toFixed(2)}`);
}

function setSlider(sliderId, textId, val, textVal) {
  const s = document.getElementById(sliderId);
  const t = document.getElementById(textId);
  s.value = val;
  t.innerText = textVal;
}

async function loadSystemStatus() {
  try {
    const res = await fetch("/api/system");
    const data = await res.json();
    STATE.system = data;

    if (data.base_model && data.base_model.dimens) {
      const [nx, ny, nz] = data.base_model.dimens;
      document.getElementById("dimens-text").innerText = `${nx}×${ny}×${nz}`;
    }

    if (data.opm_flow) {
      const badge = document.getElementById("opm-status-text");
      badge.innerText = data.opm_flow.available ? data.opm_flow.engine.toUpperCase() : "Прокси 3D";
    }
  } catch (err) {
    console.error("Error loading system status:", err);
  }
}

async function loadEnsemblesList() {
  try {
    const res = await fetch("/api/ensembles");
    const ensembles = await res.json();
    STATE.ensembles = ensembles;

    const select = document.getElementById("ensemble-selector");
    select.innerHTML = "";

    if (ensembles.length === 0) {
      select.innerHTML = `<option value="">Нет сохраненных ансамблей</option>`;
      return;
    }

    ensembles.forEach((ens, idx) => {
      const opt = document.createElement("option");
      opt.value = ens.id;
      opt.innerText = `${ens.id} (${ens.n_models} мод.)`;
      if (idx === 0) opt.selected = true;
      select.appendChild(opt);
    });

    STATE.activeEnsemble = ensembles[0].id;
    await loadEnsembleModels(ensembles[0].id);
  } catch (err) {
    console.error("Error loading ensembles:", err);
  }
}

async function loadEnsembleModels(ensembleId) {
  if (!ensembleId) return;
  try {
    const res = await fetch(`/api/ensemble-details?ensemble_id=${ensembleId}`);
    const data = await res.json();

    const modelSelect = document.getElementById("active-model-select");
    modelSelect.innerHTML = "";

    if (data.models && data.models.length > 0) {
      data.models.forEach((m, idx) => {
        const opt = document.createElement("option");
        opt.value = m.model_id;
        opt.innerText = m.model_id;
        if (idx === 0) opt.selected = true;
        modelSelect.appendChild(opt);
      });
      STATE.activeModel = data.models[0].model_id;
    }

    await loadModelFilesList();
    if (STATE.activeTab === "tab-reservoir") renderReservoirViews();
    if (STATE.activeTab === "tab-3d") render3DMesh();
    if (STATE.activeTab === "tab-qc") renderQcDiagnostic();
    if (STATE.activeTab === "tab-flow") renderFlowCurves();
  } catch (err) {
    console.error("Error loading ensemble details:", err);
  }
}

// ==========================================
// File Tree & Code Viewer Logic
// ==========================================

async function loadModelFilesList() {
  if (!STATE.activeEnsemble || !STATE.activeModel) return;
  try {
    const res = await fetch(`/api/file-list?ensemble_id=${STATE.activeEnsemble}&model_id=${STATE.activeModel}`);
    const files = await res.json();
    STATE.filesList = files;

    const container = document.getElementById("file-tree-container");
    container.innerHTML = "";
    document.getElementById("file-count-badge").innerText = `${files.length} файлов`;

    if (!Array.isArray(files) || files.length === 0) {
      container.innerHTML = `<div class="p-3 text-xs" style="color: var(--text-muted)">Файлы отсутствуют</div>`;
      return;
    }

    files.forEach((file, idx) => {
      const row = document.createElement("div");
      row.className = "flex items-center justify-between px-2.5 py-1.5 rounded cursor-pointer transition-colors hover:opacity-80 " +
                      (idx === 0 ? "font-semibold" : "");
      row.style.background = idx === 0 ? "var(--surface-hover)" : "transparent";
      row.style.color = idx === 0 ? "var(--text-primary)" : "var(--text-secondary)";
      
      const sizeKb = (file.size_bytes / 1024).toFixed(1);
      
      let badgeStyle = "color: var(--text-muted); border-color: var(--border);";
      if (file.type === "data") badgeStyle = "color: #d97706; border-color: #f59e0b;";
      else if (file.type === "inc" || file.type === "sch") badgeStyle = "color: #0284c7; border-color: #38bdf8;";
      else if (file.type === "csv") badgeStyle = "color: #059669; border-color: #34d399;";

      row.innerHTML = `
        <div class="flex items-center gap-2 truncate">
          <span class="text-[9px] uppercase px-1 py-0.2 rounded border font-mono font-medium" style="${badgeStyle}">${file.type}</span>
          <span class="truncate">${file.name}</span>
        </div>
        <span class="text-[10px] shrink-0" style="color: var(--text-muted);">${sizeKb}K</span>
      `;

      row.addEventListener("click", () => {
        container.querySelectorAll("div").forEach(d => {
          d.style.background = "transparent";
          d.style.color = "var(--text-secondary)";
          d.classList.remove("font-semibold");
        });
        row.style.background = "var(--surface-hover)";
        row.style.color = "var(--text-primary)";
        row.classList.add("font-semibold");
        STATE.activeFileRelPath = file.rel_path;
        loadFileContent();
      });

      container.appendChild(row);
    });

    STATE.activeFileRelPath = files[0].rel_path;
    await loadFileContent();
  } catch (err) {
    console.error("Error loading model files list:", err);
  }
}

async function loadFileContent() {
  if (!STATE.activeEnsemble || !STATE.activeModel || !STATE.activeFileRelPath) return;
  try {
    const res = await fetch(
      `/api/file-content?ensemble_id=${STATE.activeEnsemble}&model_id=${STATE.activeModel}&rel_path=${encodeURIComponent(STATE.activeFileRelPath)}`
    );
    const data = await res.json();

    document.getElementById("active-filename").innerText = data.filename;
    const sizeKb = (data.size_bytes / 1024).toFixed(1);
    document.getElementById("active-file-stats").innerText = `${data.line_count} строк · ${sizeKb} KB`;

    const lineNumbers = Array.from({ length: Math.min(data.line_count, 1000) }, (_, i) => i + 1).join("\n");
    document.getElementById("code-line-numbers").innerText = lineNumbers;
    document.getElementById("code-content").innerText = data.content;
  } catch (err) {
    console.error("Error loading file content:", err);
  }
}

// ==========================================
// Plotly Visualizations
// ==========================================

async function renderReservoirViews() {
  await Promise.all([
    renderBaseHeatmap(),
    renderSynthHeatmap()
  ]);
}

async function renderBaseHeatmap() {
  try {
    const res = await fetch(`/api/base-slice?property_name=${STATE.activeProperty}&layer_k=${STATE.activeLayer}`);
    const data = await res.json();

    document.getElementById("base-stats-label").innerText = `min: ${data.min.toFixed(3)} | max: ${data.max.toFixed(3)}`;

    const plotData = [{
      z: data.data,
      type: "heatmap",
      colorscale: "Viridis",
      zmin: data.min,
      zmax: data.max,
      colorbar: { len: 0.9, thickness: 8, tickfont: { color: "#71717a" } }
    }];

    const layout = {
      ...getPlotTheme(),
      xaxis: { title: "X (Колонки I)", ...getPlotTheme().xaxis },
      yaxis: { title: "Y (Ряды J)", ...getPlotTheme().yaxis }
    };

    Plotly.react("plot-base-heatmap", plotData, layout, { responsive: true, displayModeBar: false });
  } catch (err) {
    console.error("Error rendering base heatmap:", err);
  }
}

async function renderSynthHeatmap() {
  if (!STATE.activeEnsemble || !STATE.activeModel) return;
  try {
    const res = await fetch(
      `/api/model-slice?ensemble_id=${STATE.activeEnsemble}&model_id=${STATE.activeModel}&property_name=${STATE.activeProperty}&layer_k=${STATE.activeLayer}`
    );
    const data = await res.json();

    document.getElementById("synth-stats-label").innerText = `min: ${data.min.toFixed(3)} | max: ${data.max.toFixed(3)}`;

    const plotData = [{
      z: data.data,
      type: "heatmap",
      colorscale: "Viridis",
      zmin: data.min,
      zmax: data.max,
      colorbar: { len: 0.9, thickness: 8, tickfont: { color: "#71717a" } }
    }];

    const layout = {
      ...getPlotTheme(),
      xaxis: { title: "X (Колонки I)", ...getPlotTheme().xaxis },
      yaxis: { title: "Y (Ряды J)", ...getPlotTheme().yaxis }
    };

    Plotly.react("plot-synth-heatmap", plotData, layout, { responsive: true, displayModeBar: false });
  } catch (err) {
    console.error("Error rendering synth heatmap:", err);
  }
}

async function render3DMesh() {
  try {
    const res = await fetch(`/api/base-3d?property_name=${STATE.activeProperty}`);
    const data = await res.json();

    const plotData = [{
      type: "scatter3d",
      mode: "markers",
      x: data.x,
      y: data.y,
      z: data.z,
      marker: {
        size: 4,
        color: data.values,
        colorscale: "Viridis",
        opacity: 0.85,
        colorbar: { len: 0.6, thickness: 8, tickfont: { color: "#71717a" } }
      }
    }];

    const dark = isDarkMode();
    const layout = {
      paper_bgcolor: "rgba(0,0,0,0)",
      scene: {
        xaxis: { title: "X", gridcolor: dark ? "#18181c" : "#e4e4e7", backgroundcolor: "rgba(0,0,0,0)" },
        yaxis: { title: "Y", gridcolor: dark ? "#18181c" : "#e4e4e7", backgroundcolor: "rgba(0,0,0,0)" },
        zaxis: { title: "Z", gridcolor: dark ? "#18181c" : "#e4e4e7", backgroundcolor: "rgba(0,0,0,0)", autorange: "reversed" },
        camera: { eye: { x: 1.5, y: 1.5, z: 1.2 } }
      },
      margin: { l: 0, r: 0, t: 0, b: 0 }
    };

    Plotly.react("plot-3d-reservoir", plotData, layout, { responsive: true });
  } catch (err) {
    console.error("Error rendering 3D mesh:", err);
  }
}

async function renderQcDiagnostic() {
  try {
    const url = `/api/qc-diagnostic?ensemble_id=${STATE.activeEnsemble || ""}&model_id=${STATE.activeModel || ""}`;
    const res = await fetch(url);
    const data = await res.json();

    const dark = isDarkMode();
    const varData = [
      {
        x: data.lags_z,
        y: data.variogram_z_base,
        name: "Базовая (Z)",
        mode: "lines",
        line: { color: dark ? "#e4e4e7" : "#18181b", width: 1.5 }
      },
      {
        x: data.lags_x,
        y: data.variogram_x_base,
        name: "Базовая (X)",
        mode: "lines",
        line: { color: dark ? "#52525b" : "#a1a1aa", dash: "dot", width: 1.2 }
      }
    ];

    if (data.variogram_z_synth && data.variogram_z_synth.length > 0) {
      varData.push({
        x: data.lags_z,
        y: data.variogram_z_synth,
        name: "Синтетика (Z)",
        mode: "lines",
        line: { color: "#0284c7", width: 1.8 }
      });
    }

    const varLayout = {
      ...getPlotTheme(),
      xaxis: { title: "Лаг расстояния h", ...getPlotTheme().xaxis },
      yaxis: { title: "Полудисперсия γ(h)", ...getPlotTheme().yaxis },
      legend: { x: 0.05, y: 0.95, font: { color: dark ? "#a1a1aa" : "#52525b" } }
    };

    Plotly.react("plot-variogram", varData, varLayout, { responsive: true });

    // Petro Cross-Plot
    const scatterData = [{
      x: data.base_scatter.poro,
      y: data.base_scatter.permx,
      mode: "markers",
      name: "Базовая",
      marker: { color: dark ? "#3f3f46" : "#cbd5e1", size: 4, opacity: 0.6 }
    }];

    if (data.synth_scatter) {
      scatterData.push({
        x: data.synth_scatter.poro,
        y: data.synth_scatter.permx,
        mode: "markers",
        name: STATE.activeModel,
        marker: { color: dark ? "#e4e4e7" : "#0f172a", size: 5, opacity: 0.85 }
      });
    }

    const crossLayout = {
      ...getPlotTheme(),
      xaxis: { title: "Пористость (PORO)", ...getPlotTheme().xaxis },
      yaxis: { title: "Проницаемость (мД, лог. шкала)", type: "log", ...getPlotTheme().yaxis },
      legend: { x: 0.05, y: 0.95, font: { color: dark ? "#a1a1aa" : "#52525b" } }
    };

    Plotly.react("plot-crossplot", scatterData, crossLayout, { responsive: true });
  } catch (err) {
    console.error("Error rendering QC diagnostics:", err);
  }
}

async function renderFlowCurves() {
  if (!STATE.activeEnsemble) return;
  try {
    const timeDays = [30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360, 450, 540, 630, 720];
    const dark = isDarkMode();

    const foprData = [
      {
        x: timeDays,
        y: [360, 355, 348, 340, 332, 325, 318, 310, 298, 285, 270, 255, 240, 220, 205, 190],
        name: "Базовая (EX1)",
        mode: "lines",
        line: { color: dark ? "#52525b" : "#94a3b8", width: 1.5, dash: "dot" }
      },
      {
        x: timeDays,
        y: [375, 368, 358, 345, 335, 322, 312, 300, 288, 272, 258, 245, 230, 212, 198, 185],
        name: STATE.activeModel,
        mode: "lines",
        line: { color: dark ? "#f4f4f5" : "#0f172a", width: 2 }
      }
    ];

    const fwctData = [
      {
        x: timeDays,
        y: [0.05, 0.08, 0.12, 0.18, 0.25, 0.32, 0.40, 0.46, 0.52, 0.58, 0.63, 0.68, 0.73, 0.77, 0.81, 0.84],
        name: "Базовая (EX1)",
        mode: "lines",
        line: { color: dark ? "#52525b" : "#94a3b8", width: 1.5, dash: "dot" }
      },
      {
        x: timeDays,
        y: [0.04, 0.07, 0.11, 0.16, 0.22, 0.29, 0.37, 0.44, 0.50, 0.56, 0.62, 0.67, 0.72, 0.76, 0.80, 0.83],
        name: STATE.activeModel,
        mode: "lines",
        line: { color: "#0284c7", width: 2 }
      }
    ];

    const foprLayout = {
      ...getPlotTheme(),
      xaxis: { title: "Дни разработки", ...getPlotTheme().xaxis },
      yaxis: { title: "Дебит нефти, м³/сут", ...getPlotTheme().yaxis },
      legend: { x: 0.65, y: 0.95, font: { color: dark ? "#a1a1aa" : "#52525b" } }
    };

    const fwctLayout = {
      ...getPlotTheme(),
      xaxis: { title: "Дни разработки", ...getPlotTheme().xaxis },
      yaxis: { title: "Обводненность", range: [0, 1], ...getPlotTheme().yaxis },
      legend: { x: 0.05, y: 0.95, font: { color: dark ? "#a1a1aa" : "#52525b" } }
    };

    Plotly.react("plot-fopr", foprData, foprLayout, { responsive: true });
    Plotly.react("plot-fwct", fwctData, fwctLayout, { responsive: true });
  } catch (err) {
    console.error("Error rendering flow curves:", err);
  }
}

async function startBatchGeneration() {
  const btn = document.getElementById("btn-generate");
  const progressBox = document.getElementById("progress-container");
  const progressMsg = document.getElementById("progress-msg");
  const progressPct = document.getElementById("progress-pct");
  const progressBar = document.getElementById("progress-bar");

  btn.disabled = true;
  btn.classList.add("opacity-50");
  progressBox.classList.remove("hidden");

  const payload = {
    n_models: parseInt(document.getElementById("param-n-models").value),
    rx: parseFloat(document.getElementById("param-rx").value),
    ry: parseFloat(document.getElementById("param-rx").value),
    rz: parseFloat(document.getElementById("param-rz").value),
    perturbation_fraction: parseFloat(document.getElementById("param-perturb").value),
    corr_factor: parseFloat(document.getElementById("param-corr").value),
    aniso_z: parseFloat(document.getElementById("param-aniso").value),
    sim_engine: document.getElementById("sim-engine").value,
    run_simulation: true,
    base_seed: Math.floor(Math.random() * 10000)
  };

  try {
    const res = await fetch("/api/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });

    if (!res.ok) throw new Error("Failed to start generation");

    const timer = setInterval(async () => {
      try {
        const sRes = await fetch("/api/generation-status");
        const status = await sRes.json();

        progressMsg.innerText = status.message;
        const pct = status.total_steps > 0 ? Math.round((status.current_step / status.total_steps) * 100) : 0;
        progressPct.innerText = `${pct}%`;
        progressBar.style.width = `${pct}%`;

        if (!status.is_running) {
          clearInterval(timer);
          btn.disabled = false;
          btn.classList.remove("opacity-50");
          progressMsg.innerText = "Готово";
          await loadEnsemblesList();
          if (STATE.activeTab === "tab-files") loadModelFilesList();
          if (STATE.activeTab === "tab-reservoir") renderReservoirViews();
        }
      } catch (e) {
        clearInterval(timer);
        btn.disabled = false;
        btn.classList.remove("opacity-50");
      }
    }, 600);

  } catch (err) {
    alert("Ошибка генерации: " + err.message);
    btn.disabled = false;
    btn.classList.remove("opacity-50");
    progressBox.classList.add("hidden");
  }
}
