import { mount } from "/cartridges/ants/viz.js";

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

  stagePlaceholder.textContent = `Loading ${file}...`;
  stagePlaceholder.style.display = "block";

  let replayData;
  try {
    const url = `/api/replay?folder=${encodeURIComponent(folder)}&file=${encodeURIComponent(file)}`;
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`Server returned HTTP ${response.status}`);
    }
    replayData = await response.json();
  } catch (err) {
    if (currentRequestId !== loadSequence) return;
    stagePlaceholder.textContent = `Failed to load match: ${err.message}`;
    stagePlaceholder.style.display = "block";
    return;
  }

  if (currentRequestId !== loadSequence) {
    return;
  }

  destroyCurrentViewer();
  stagePlaceholder.style.display = "none";

  try {
    const viewer = await mount("#replay-stage", replayData, {
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
    stagePlaceholder.textContent = `Error mounting viewer: ${err.message}`;
    stagePlaceholder.style.display = "block";
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
  stagePlaceholder.textContent = `Loading matches for ${folderName}...`;
  stagePlaceholder.style.display = "block";

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
    stagePlaceholder.textContent = `Failed to load matches for ${folderName}: ${err.message}`;
    stagePlaceholder.style.display = "block";
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
      stagePlaceholder.textContent = `Match file "${targetFile}" not found in ${folderName}`;
    } else {
      stagePlaceholder.textContent = "Select a match to replay";
    }
    stagePlaceholder.style.display = "block";
    updateUrl(currentFolder, "", pushHistory);
  }
}

async function init() {
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
    if (targetFolder && targetFolder !== currentFolder) {
      await selectFolder(targetFolder, popFile, false);
    } else {
      renderMatchList();
      if (popFile) {
        if (popFile !== selectedMatch) {
          folderSequence++;
          await selectMatch(popFile, false);
        }
      } else {
        loadSequence++;
        folderSequence++;
        selectedMatch = "";
        updateActiveItem();
        destroyCurrentViewer();
        stagePlaceholder.textContent = "Select a match to replay";
        stagePlaceholder.style.display = "block";
      }
    }
  });

  let directories = [];
  try {
    const response = await fetch("/api/directories");
    if (response.ok) {
      directories = await response.json();
    }
  } catch {
    directories = [];
  }

  folderSelect.innerHTML = "";
  if (directories.length === 0) {
    const opt = document.createElement("option");
    opt.value = "";
    opt.textContent = "No replay folders found";
    folderSelect.appendChild(opt);
    renderMatchList();
    stagePlaceholder.textContent = "No replay folders found";
    stagePlaceholder.style.display = "block";
    return;
  }

  directories.forEach((dir) => {
    const opt = document.createElement("option");
    opt.value = dir.folder;
    opt.textContent = `${dir.name} (${dir.count})`;
    folderSelect.appendChild(opt);
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
