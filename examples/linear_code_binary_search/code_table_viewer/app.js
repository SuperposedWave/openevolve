const state = {
  data: null,
  cellsByKey: new Map(),
  selectedDetailId: null,
};

const elements = {
  summary: document.getElementById("summary"),
  controls: document.getElementById("controls"),
  minN: document.getElementById("minN"),
  maxN: document.getElementById("maxN"),
  minK: document.getElementById("minK"),
  maxK: document.getElementById("maxK"),
  statusFilter: document.getElementById("statusFilter"),
  resetFilters: document.getElementById("resetFilters"),
  tableStatus: document.getElementById("tableStatus"),
  matrixTable: document.getElementById("matrixTable"),
  emptyDetail: document.getElementById("emptyDetail"),
  detail: document.getElementById("detail"),
  detailTitle: document.getElementById("detailTitle"),
  detailSubtitle: document.getElementById("detailSubtitle"),
  detailMetrics: document.getElementById("detailMetrics"),
  attemptList: document.getElementById("attemptList"),
  matrixSection: document.getElementById("matrixSection"),
  generatorSection: document.getElementById("generatorSection"),
  prioritySection: document.getElementById("prioritySection"),
  hRows: document.getElementById("hRows"),
  gRows: document.getElementById("gRows"),
  prioritySource: document.getElementById("prioritySource"),
  sourcePaths: document.getElementById("sourcePaths"),
  closeDetail: document.getElementById("closeDetail"),
};

function keyFor(n, k) {
  return `${n}:${k}`;
}

function clampInt(value, min, max, fallback) {
  const parsed = Number.parseInt(value, 10);
  if (Number.isNaN(parsed)) {
    return fallback;
  }
  return Math.min(Math.max(parsed, min), max);
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function statusLabel(status) {
  const labels = {
    found: "found",
    failed: "searched, not found",
    upper_failed_after_found: "upper not found",
    unsearched: "unsearched",
  };
  return labels[status] || status;
}

function numberOrDash(value) {
  return value === null || value === undefined || value === "" ? "-" : value;
}

async function loadData() {
  const response = await fetch("code_table_data.json");
  if (!response.ok) {
    throw new Error(`Failed to load code_table_data.json: ${response.status}`);
  }
  const data = await response.json();
  state.data = data;
  state.cellsByKey = new Map(data.cells.map((cell) => [keyFor(cell.n, cell.k), cell]));
}

function renderSummary() {
  const meta = state.data.meta;
  const counts = meta.countsByStatus || {};
  elements.summary.textContent =
    `${meta.totalCells.toLocaleString()} cells · ` +
    `${(counts.found || 0).toLocaleString()} found · ` +
    `${(counts.failed || 0).toLocaleString()} searched, not found · ` +
    `${(counts.upper_failed_after_found || 0).toLocaleString()} upper failed · ` +
    `${(counts.unsearched || 0).toLocaleString()} unsearched`;
}

function currentFilters() {
  let minN = clampInt(elements.minN.value, 1, 256, 1);
  let maxN = clampInt(elements.maxN.value, 1, 256, 256);
  let minK = clampInt(elements.minK.value, 1, 256, 1);
  let maxK = clampInt(elements.maxK.value, 1, 256, 256);
  if (minN > maxN) {
    [minN, maxN] = [maxN, minN];
  }
  if (minK > maxK) {
    [minK, maxK] = [maxK, minK];
  }
  return {
    minN,
    maxN,
    minK,
    maxK,
    status: elements.statusFilter.value,
  };
}

function renderTable() {
  const filters = currentFilters();
  if (state.selectedDetailId) {
    const selectedCell = state.data.cells.find((cell) => cell.detailId === state.selectedDetailId);
    if (selectedCell && filters.status !== "all" && selectedCell.status !== filters.status) {
      state.selectedDetailId = null;
      elements.detail.hidden = true;
      elements.emptyDetail.hidden = false;
    }
  }
  const kValues = [];
  for (let k = filters.minK; k <= filters.maxK; k += 1) {
    kValues.push(k);
  }

  let visibleDataCells = 0;
  let matchedCells = 0;
  const header = [
    "<thead><tr><th scope=\"col\">n/k</th>",
    ...kValues.map((k) => `<th scope="col">${k}</th>`),
    "</tr></thead>",
  ].join("");

  const rows = [];
  for (let n = filters.minN; n <= filters.maxN; n += 1) {
    const row = [`<tr><th scope="row">${n}</th>`];
    for (const k of kValues) {
      if (k > n) {
        row.push('<td class="blank-cell" aria-label="invalid"></td>');
        continue;
      }
      const cell = state.cellsByKey.get(keyFor(n, k));
      if (!cell) {
        row.push('<td class="blank-cell" aria-label="missing"></td>');
        continue;
      }
      visibleDataCells += 1;
      const matches = filters.status === "all" || cell.status === filters.status;
      if (matches) {
        matchedCells += 1;
      }
      const selectedClass = cell.detailId && cell.detailId === state.selectedDetailId ? " active" : "";
      const filteredClass = matches ? "" : " filtered-cell";
      const classes = `data-cell status-${cell.status}${selectedClass}${filteredClass}`;
      const title =
        `n=${cell.n}, k=${cell.k}, bounds=${cell.label}, status=${statusLabel(cell.status)}`;
      if (cell.detailId && matches) {
        row.push(
          `<td class="${classes}" title="${escapeHtml(title)}">` +
            `<button class="cell-button" type="button" data-detail-id="${cell.detailId}" ` +
            `aria-label="${escapeHtml(title)}"><span class="cell-value">${escapeHtml(cell.label)}</span></button>` +
          "</td>"
        );
      } else {
        row.push(
          `<td class="${classes}" title="${escapeHtml(title)}">` +
            `<span class="cell-value">${matches ? escapeHtml(cell.label) : ""}</span>` +
          "</td>"
        );
      }
    }
    row.push("</tr>");
    rows.push(row.join(""));
  }

  elements.matrixTable.innerHTML = `${header}<tbody>${rows.join("")}</tbody>`;
  elements.tableStatus.textContent =
    `Showing ${matchedCells.toLocaleString()} matching cells within ` +
    `${visibleDataCells.toLocaleString()} valid cells in the selected range.`;
}

function renderMetric(name, value) {
  return `<div><dt>${escapeHtml(name)}</dt><dd>${escapeHtml(numberOrDash(value))}</dd></div>`;
}

function renderDetails(detailId) {
  const detail = state.data.details[detailId];
  if (!detail) {
    return;
  }
  state.selectedDetailId = detailId;
  elements.emptyDetail.hidden = true;
  elements.detail.hidden = false;

  const status = detail.completeConstruction ? "complete construction" : "no complete construction";
  elements.detailTitle.textContent = `n=${detail.n}, k=${detail.k}`;
  elements.detailSubtitle.textContent =
    detail.trivialDistance
      ? `bounds ${detail.lower === detail.upper ? detail.lower : `${detail.lower}-${detail.upper}`} · d<=2 needs no search`
      : `bounds ${detail.lower === detail.upper ? detail.lower : `${detail.lower}-${detail.upper}`} · ${status}`;

  elements.detailMetrics.innerHTML = [
    renderMetric("best d", detail.bestDistance),
    renderMetric("target d", detail.targetDistance),
    renderMetric("lower", detail.lower),
    renderMetric("upper", detail.upper),
    renderMetric("constructed", detail.metrics.constructed_columns),
    renderMetric("score", detail.metrics.combined_score),
  ].join("");

  elements.attemptList.innerHTML = detail.attempts
    .map((attempt) => {
      const score = numberOrDash(attempt.metrics && attempt.metrics.combined_score);
      const source = [attempt.sourceRoot, attempt.sourceRun].filter(Boolean).join(" / ");
      return (
        `<div class="attempt ${escapeHtml(attempt.status)}">` +
          `<strong>d=${escapeHtml(attempt.targetDistance)} · ${escapeHtml(attempt.status)}</strong>` +
          `actual=${escapeHtml(numberOrDash(attempt.actualDistance))} · score=${escapeHtml(score)}` +
          `${source ? `<br>source=${escapeHtml(source)}` : ""}` +
        "</div>"
      );
    })
    .join("") || '<div class="notice">d<=2 lower bound is satisfied without running the search.</div>';

  if (detail.completeConstruction) {
    elements.matrixSection.hidden = false;
    elements.hRows.textContent = detail.hRows.join("\n");
    elements.generatorSection.hidden = detail.gRows.length === 0;
    elements.gRows.textContent = detail.gRows.join("\n");
    elements.prioritySection.hidden = false;
    elements.prioritySource.textContent = detail.prioritySource || "No saved priority source.";
    elements.sourcePaths.textContent = JSON.stringify(
      {
        sourceRoot: detail.sourceRoot,
        sourceRun: detail.sourceRun,
      },
      null,
      2
    );
  } else {
    elements.matrixSection.hidden = false;
    elements.hRows.textContent = detail.trivialDistance
      ? "d<=2 is satisfied without search; no saved H matrix is required."
      : "No complete saved H matrix is available for this cell.";
    elements.generatorSection.hidden = true;
    elements.gRows.textContent = "";
    elements.prioritySection.hidden = true;
    elements.prioritySource.textContent = "";
    elements.sourcePaths.textContent = detail.trivialDistance && detail.attempts.length === 0
      ? "trivial d<=2 lower-bound witness"
      : JSON.stringify(
          detail.attempts.map((attempt) => ({
            sourceRoot: attempt.sourceRoot,
            sourceRun: attempt.sourceRun,
          })),
          null,
          2
        );
  }
  renderTable();
}

function closeDetail() {
  state.selectedDetailId = null;
  elements.detail.hidden = true;
  elements.emptyDetail.hidden = false;
  renderTable();
}

function resetFilters() {
  elements.minN.value = "1";
  elements.maxN.value = "256";
  elements.minK.value = "1";
  elements.maxK.value = "256";
  elements.statusFilter.value = "all";
  renderTable();
}

function bindEvents() {
  elements.controls.addEventListener("input", renderTable);
  elements.controls.addEventListener("change", renderTable);
  elements.resetFilters.addEventListener("click", resetFilters);
  elements.closeDetail.addEventListener("click", closeDetail);
  elements.matrixTable.addEventListener("click", (event) => {
    const button = event.target.closest("[data-detail-id]");
    if (!button) {
      return;
    }
    renderDetails(button.dataset.detailId);
  });
}

async function init() {
  try {
    await loadData();
    renderSummary();
    renderTable();
    bindEvents();
  } catch (error) {
    elements.summary.textContent = error.message;
    elements.tableStatus.textContent =
      "Run this page through a local web server so it can fetch code_table_data.json.";
  }
}

init();
