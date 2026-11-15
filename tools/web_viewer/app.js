const folderSelect = document.getElementById("folder-select");
const searchInput = document.getElementById("search-input");
const matchList = document.getElementById("match-list");
const matchStats = document.getElementById("match-stats");
const stageEl = document.getElementById("replay-stage");
const stagePlaceholder = document.getElementById("stage-placeholder");

let currentFolder = "";
let allMatches = [];
let selectedMatch = "";
let currentViewer = null;
let loadSequence = 0;
let folderSequence = 0;
let folderLoadError = "";
let viewerEngineDigest = "";
let mountViewer = null;

function showStageMessage(title, { isError = false, details = [] } = {}) {
  stagePlaceholder.innerHTML = "";
  stagePlaceholder.classList.toggle("is-error", isError);

  const titleEl = document.createElement("div");
  titleEl.className = "stage-placeholder-title";
  titleEl.textContent = title;
  stagePlaceholder.appendChild(titleEl);

  if (details.length > 0) {
    const detailsEl = document.createElement("div");
    detailsEl.className = "stage-placeholder-details";
    details.forEach((detail) => {
      const lineEl = document.createElement("div");
      lineEl.className = "stage-placeholder-detail";
      lineEl.textContent = detail;
      detailsEl.appendChild(lineEl);
    });
    stagePlaceholder.appendChild(detailsEl);
  }

  stagePlaceholder.style.display = "block";
}

function hideStagePlaceholder() {
  stagePlaceholder.classList.remove("is-error");
  stagePlaceholder.innerHTML = "";
  stagePlaceholder.style.display = "none";
}

async function fetchViewerEngineDigest() {
  const response = await fetch("/cartridges/ants/engine.json", { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`HTTP ${response.status} fetching /cartridges/ants/engine.json`);
  }
  const metadata = await response.json();
  if (!metadata || typeof metadata !== "object" || Array.isArray(metadata)) {
    throw new Error('Missing "engine_digest" field in /cartridges/ants/engine.json');
  }
  const rawDigest = metadata.engine_digest;
  if (rawDigest === undefined || rawDigest === null) {
    throw new Error('Missing "engine_digest" field in /cartridges/ants/engine.json');
  }
  if (typeof rawDigest !== "string") {
    throw new Error('Invalid "engine_digest" field in /cartridges/ants/engine.json');
  }
  const trimmed = rawDigest.trim();
  if (!trimmed) {
    throw new Error('Empty "engine_digest" field in /cartridges/ants/engine.json');
  }
  if (rawDigest !== trimmed) {
    throw new Error('Invalid "engine_digest" field in /cartridges/ants/engine.json: unexpected whitespace');
  }
  return rawDigest;
}

function updateUrl(folder, file, pushHistory = true) {
  const params = new URLSearchParams();
  if (folder) params.set("folder", folder);
  if (file) params.set("file", file);
  const paramStr = params.toString() ? `?${params.toString()}` : "";
  const newUrl = `${window.location.pathname}${paramStr}`;
  if (window.location.search !== paramStr) {
    if (pushHistory) {
      window.history.pushState({ folder, file }, "", newUrl);
    } else {
      window.history.replaceState({ folder, file }, "", newUrl);
    }
  }
}

function updateActiveItem() {
  const items = matchList.querySelectorAll("li");
  items.forEach((item) => {
    if (item.dataset.fileName === selectedMatch) {
      item.classList.add("active");
    } else {
      item.classList.remove("active");
    }
  });
}

function renderMatchList() {
  const query = searchInput.value.trim().toLowerCase();
  const filtered = query
    ? allMatches.filter((name) => name.toLowerCase().includes(query))
    : allMatches;

  matchList.innerHTML = "";
  if (folderLoadError) {
    matchStats.textContent = "Failed to load matches";
    const errorLi = document.createElement("li");
    errorLi.textContent = "Failed to load replay list";
    errorLi.style.cursor = "default";
    errorLi.style.color = "#f87171";
    matchList.appendChild(errorLi);
    return;
  }
  if (filtered.length === 0) {
    matchStats.textContent = allMatches.length === 0 ? "0 matches found" : `0 of ${allMatches.length} matches`;
    const emptyLi = document.createElement("li");
    emptyLi.textContent = allMatches.length === 0 ? "No replay files found" : "No matches match search";
    emptyLi.style.cursor = "default";
    emptyLi.style.color = "#64748b";
    matchList.appendChild(emptyLi);
    return;
  }

  matchStats.textContent = query
    ? `${filtered.length} of ${allMatches.length} matches`
    : `${filtered.length} matches found`;

  filtered.forEach((fileName) => {
    const li = document.createElement("li");
    li.dataset.fileName = fileName;
    li.textContent = fileName;
    if (fileName === selectedMatch) {
      li.classList.add("active");
    }
    li.addEventListener("click", () => {
      if (selectedMatch === fileName && currentViewer) {
        return;
      }
      selectMatch(fileName);
    });
    matchList.appendChild(li);
  });
}

function destroyCurrentViewer() {
  if (currentViewer) {
    try {
      currentViewer.destroy();
    } catch {
      // 清理旧实例异常
    }
    currentViewer = null;
  }
  stageEl.innerHTML = "";
  stageEl.className = "";
  delete stageEl.dataset.tbTheme;
  delete stageEl.dataset.tbChrome;
}

async function selectMatch(fileName, pushHistory = true) {
  folderSequence++;
  selectedMatch = fileName;
  updateActiveItem();
  updateUrl(currentFolder, selectedMatch, pushHistory);

  destroyCurrentViewer();
  await loadReplay(currentFolder, selectedMatch);
}

async function loadReplay(folder, file) {
  const currentRequestId = ++loadSequence;

  if (!viewerEngineDigest || typeof mountViewer !== "function") {
    destroyCurrentViewer();
    showStageMessage("Failed to initialize replay viewer: missing viewer engine_digest.", {
      isError: true,
    });
    return;
  }

  showStageMessage(`Loading ${file}...`);

  let replayData;
  try {
    const url = `/api/replay?folder=${encodeURIComponent(folder)}&file=${encodeURIComponent(file)}`;
    const response = await fetch(url, { cache: "no-store" });
    if (!response.ok) {
      throw new Error(`Server returned HTTP ${response.status}`);
    }
    replayData = await response.json();
  } catch (err) {
    if (currentRequestId !== loadSequence) return;
    destroyCurrentViewer();
    showStageMessage(`Failed to load match: ${err.message}`, { isError: true });
    return;
  }

  if (currentRequestId !== loadSequence) {
    return;
  }

  const rawReplayDigest =
    replayData && typeof replayData === "object" && !Array.isArray(replayData)
      ? replayData.engine_digest
      : undefined;
  const trimmedReplayDigest =
    typeof rawReplayDigest === "string" ? rawReplayDigest.trim() : "";

  if (!trimmedReplayDigest || rawReplayDigest !== viewerEngineDigest) {
    destroyCurrentViewer();
    const isMissingOrInvalid = !trimmedReplayDigest;
    let replayDigestDisplay;
    if (rawReplayDigest === undefined || rawReplayDigest === null) {
      replayDigestDisplay = "(missing)";
    } else if (typeof rawReplayDigest !== "string") {
      replayDigestDisplay = `(invalid: ${JSON.stringify(rawReplayDigest)})`;
    } else if (trimmedReplayDigest === "") {
      replayDigestDisplay = "(empty)";
    } else if (rawReplayDigest !== trimmedReplayDigest) {
      replayDigestDisplay = JSON.stringify(rawReplayDigest);
    } else {
      replayDigestDisplay = rawReplayDigest;
    }

    showStageMessage(
      isMissingOrInvalid
        ? `Cannot load "${file}": replay is missing engine_digest.`
        : `Cannot load "${file}": engine_digest mismatch.`,
      {
        isError: true,
        details: [
          `Replay engine_digest: ${replayDigestDisplay}`,
          `Viewer engine_digest: ${viewerEngineDigest}`,
        ],
      }
    );
    return;
  }

  destroyCurrentViewer();
  hideStagePlaceholder();

  try {
    const viewer = await mountViewer("#replay-stage", replayData, {
      tier: "stage",
      autoplay: false,
      theme: "dark",
    });
    if (currentRequestId !== loadSequence) {
      try {
        viewer.destroy();
      } catch {
        // 清理已作废实例
      }
      return;
    }
    currentViewer = viewer;
  } catch (err) {
    if (currentRequestId !== loadSequence) return;
    destroyCurrentViewer();
    showStageMessage(`Error mounting viewer: ${err.message}`, { isError: true });
  }
}

async function selectFolder(folderName, targetFile = null, pushHistory = true) {
  const currentFolderReq = ++folderSequence;
  loadSequence++;

  currentFolder = "";
  folderLoadError = "";
  folderSelect.value = folderName;
  selectedMatch = "";
  allMatches = [];
  renderMatchList();
  destroyCurrentViewer();
  showStageMessage(`Loading matches for ${folderName}...`);

  let data;
  try {
    const response = await fetch(`/api/matches?folder=${encodeURIComponent(folderName)}`);
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    data = await response.json();
  } catch (err) {
    if (currentFolderReq !== folderSequence) return;
    allMatches = [];
    folderLoadError = err.message || "Request failed";
    renderMatchList();
    destroyCurrentViewer();
    showStageMessage(`Failed to load matches for ${folderName}: ${err.message}`, {
      isError: true,
    });
    return;
  }

  if (currentFolderReq !== folderSequence) return;

  folderLoadError = "";
  currentFolder = folderName;
  allMatches = data;
  renderMatchList();

  if (targetFile && allMatches.includes(targetFile)) {
    await selectMatch(targetFile, pushHistory);
  } else {
    selectedMatch = "";
    destroyCurrentViewer();
    if (targetFile) {
      showStageMessage(`Match file "${targetFile}" not found in ${folderName}`, {
        isError: true,
      });
    } else {
      showStageMessage("Select a match to replay");
    }
    updateUrl(currentFolder, "", pushHistory);
  }
}

async function init() {
  folderSelect.disabled = true;
  searchInput.disabled = true;
  showStageMessage("Loading viewer engine metadata...");
  try {
    viewerEngineDigest = await fetchViewerEngineDigest();
    const vizModule = await import("/cartridges/ants/viz.js");
    if (!vizModule || typeof vizModule.mount !== "function") {
      throw new Error('Viewer module /cartridges/ants/viz.js does not export "mount"');
    }
    mountViewer = vizModule.mount;
  } catch (err) {
    viewerEngineDigest = "";
    mountViewer = null;
    const reason = (err && err.message) ? err.message : "Request failed";
    destroyCurrentViewer();
    folderSelect.innerHTML = "";
    const opt = document.createElement("option");
    opt.value = "";
    opt.textContent = "Viewer initialization failed";
    folderSelect.appendChild(opt);
    folderSelect.disabled = true;
    searchInput.disabled = true;
    matchStats.textContent = "Viewer unavailable";
    matchList.innerHTML = "";
    const errorLi = document.createElement("li");
    errorLi.textContent = "Engine metadata unavailable";
    errorLi.style.cursor = "default";
    errorLi.style.color = "#f87171";
    matchList.appendChild(errorLi);
    showStageMessage(`Failed to initialize replay viewer: ${reason}`, { isError: true });
    return;
  }

  let directories = [];
  let dirError = "";
  try {
    const response = await fetch("/api/directories", { cache: "no-store" });
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    directories = await response.json();
  } catch (err) {
    directories = [];
    dirError = (err && err.message) ? err.message : "Request failed";
  }

  folderSelect.innerHTML = "";
  if (dirError) {
    folderLoadError = dirError;
    const opt = document.createElement("option");
    opt.value = "";
    opt.textContent = "Failed to load folders";
    folderSelect.appendChild(opt);
    folderSelect.disabled = true;
    searchInput.disabled = true;
    renderMatchList();
    showStageMessage(`Failed to load replay folders: ${dirError}`, { isError: true });
    return;
  }
  if (directories.length === 0) {
    const opt = document.createElement("option");
    opt.value = "";
    opt.textContent = "No replay folders found";
    folderSelect.appendChild(opt);
    folderSelect.disabled = true;
    searchInput.disabled = true;
    renderMatchList();
    showStageMessage("No replay folders found");
    return;
  }

  directories.forEach((dir) => {
    const opt = document.createElement("option");
    opt.value = dir.folder;
    opt.textContent = `${dir.name} (${dir.count})`;
    folderSelect.appendChild(opt);
  });

  folderSelect.disabled = false;
  searchInput.disabled = false;

  folderSelect.addEventListener("change", async (e) => {
    const newFolder = e.target.value;
    searchInput.value = "";
    await selectFolder(newFolder, null, true);
  });

  searchInput.addEventListener("input", () => {
    renderMatchList();
  });

  window.addEventListener("popstate", async () => {
    const params = new URLSearchParams(window.location.search);
    const popFolder = params.get("folder");
    const popFile = params.get("file");
    const targetFolder = popFolder || folderSelect.options[0]?.value || "";

    searchInput.value = "";
    if (!targetFolder) {
      renderMatchList();
      return;
    }
    if (targetFolder !== currentFolder) {
      await selectFolder(targetFolder, popFile, false);
    } else {
      renderMatchList();
      if (popFile) {
        if (!allMatches.includes(popFile)) {
          loadSequence++;
          folderSequence++;
          selectedMatch = "";
          updateActiveItem();
          destroyCurrentViewer();
          showStageMessage(`Match file "${popFile}" not found in ${currentFolder}`, {
            isError: true,
          });
        } else if (popFile !== selectedMatch) {
          await selectMatch(popFile, false);
        }
      } else {
        loadSequence++;
        folderSequence++;
        selectedMatch = "";
        updateActiveItem();
        destroyCurrentViewer();
        showStageMessage("Select a match to replay");
      }
    }
  });

  const params = new URLSearchParams(window.location.search);
  const initialFolder = params.get("folder");
  const initialFile = params.get("file");

  const matchingDir = directories.find((d) => d.folder === initialFolder);
  const folderToLoad = matchingDir ? matchingDir.folder : directories[0].folder;
  const fileToLoad = (!initialFolder || matchingDir) ? initialFile : null;

  await selectFolder(folderToLoad, fileToLoad, false);
}

init();
