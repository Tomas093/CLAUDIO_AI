import './style.css';
import { DxfViewer } from 'dxf-viewer';
import * as THREE from 'three';

let viewer = null;
let isSelectionMode = false;
let isDragging = false;
let startScreenPos = { x: 0, y: 0 };
let currentScreenPos = { x: 0, y: 0 };
let zones = [];
let currentSelectionCoords = null;
let dxfFileObj = null;
let globalDetectionsData = null;

const canvasContainer = document.getElementById('dxf-canvas');
const fileInput = document.getElementById('file-upload');
const toggleModeBtn = document.getElementById('toggle-mode');
const selectionBox = document.getElementById('selection-box');
const toggleText = toggleModeBtn.querySelector('.text');

const zoneManagement = document.getElementById('zone-management');
const zoneNameInput = document.getElementById('zone-name');
const saveZoneBtn = document.getElementById('save-zone-btn');
const savedZonesList = document.getElementById('saved-zones-list');
const runDetectionsBtn = document.getElementById('run-detections-btn');
const bomDashboard = document.getElementById('bom-dashboard');
const bomTitle = document.getElementById('bom-title');
const bomList = document.getElementById('bom-list');

function initViewer() {
  viewer = new DxfViewer(canvasContainer, {
    clearColor: new THREE.Color('#0f172a'),
    autoResize: true,
    blackWhiteInversion: false
  });
}

fileInput.addEventListener('change', async (e) => {
  const file = e.target.files[0];
  if (!file) return;
  dxfFileObj = file;

  const url = URL.createObjectURL(file);
  
  if (!viewer) {
    initViewer();
  }
  
  try {
    await viewer.Load({ 
      url, 
      fonts: ['/fonts/Roboto-Regular.ttf']
    });
    console.log('DXF loaded successfully');
    toggleModeBtn.disabled = false;
  } catch (err) {
    console.error('Error loading DXF:', err);
    alert('Failed to load DXF file.');
  } finally {
    URL.revokeObjectURL(url);
  }
});

toggleModeBtn.addEventListener('click', () => {
  isSelectionMode = !isSelectionMode;
  
  if (isSelectionMode) {
    toggleModeBtn.classList.add('active');
    toggleText.textContent = 'Disable Selection Mode';
    if (viewer && viewer.controls) {
      viewer.controls.enabled = false;
    }
    canvasContainer.style.cursor = 'crosshair';
  } else {
    toggleModeBtn.classList.remove('active');
    toggleText.textContent = 'Enable Selection Mode';
    if (viewer && viewer.controls) {
      viewer.controls.enabled = true;
    }
    canvasContainer.style.cursor = 'default';
    hideSelectionBox();
  }
});

canvasContainer.addEventListener('pointerdown', (e) => {
  if (!isSelectionMode) return;
  e.stopImmediatePropagation();
  
  isDragging = true;
  startScreenPos = { x: e.clientX, y: e.clientY };
  
  updateSelectionBox(e.clientX, e.clientY);
  selectionBox.classList.remove('hidden');
}, true);

canvasContainer.addEventListener('pointermove', (e) => {
  if (!isSelectionMode || !isDragging) return;
  e.stopImmediatePropagation();
  updateSelectionBox(e.clientX, e.clientY);
}, true);

canvasContainer.addEventListener('pointerup', (e) => {
  if (!isSelectionMode || !isDragging) return;
  e.stopImmediatePropagation();
  isDragging = false;
  
  currentScreenPos = { x: e.clientX, y: e.clientY };
  hideSelectionBox();
  
  processSelection(startScreenPos, currentScreenPos);
}, true);

canvasContainer.addEventListener('pointerleave', () => {
  if (isDragging) {
    isDragging = false;
    hideSelectionBox();
  }
});

function updateSelectionBox(currentX, currentY) {
  const minX = Math.min(startScreenPos.x, currentX);
  const maxX = Math.max(startScreenPos.x, currentX);
  const minY = Math.min(startScreenPos.y, currentY);
  const maxY = Math.max(startScreenPos.y, currentY);
  
  selectionBox.style.left = `${minX}px`;
  selectionBox.style.top = `${minY}px`;
  selectionBox.style.width = `${maxX - minX}px`;
  selectionBox.style.height = `${maxY - minY}px`;
}

function hideSelectionBox() {
  selectionBox.classList.add('hidden');
  selectionBox.style.width = '0px';
  selectionBox.style.height = '0px';
}

function processSelection(startPos, endPos) {
  if (Math.abs(startPos.x - endPos.x) < 2 && Math.abs(startPos.y - endPos.y) < 2) {
    return;
  }

  const localStart = unprojectToWorld(startPos.x, startPos.y);
  const localEnd = unprojectToWorld(endPos.x, endPos.y);

  const origin = viewer && viewer.GetOrigin ? viewer.GetOrigin() : { x: 0, y: 0 };

  const absStartX = localStart.x + origin.x;
  const absStartY = localStart.y + origin.y;
  const absEndX = localEnd.x + origin.x;
  const absEndY = localEnd.y + origin.y;

  const minX = Math.min(absStartX, absEndX);
  const maxX = Math.max(absStartX, absEndX);
  const minY = Math.min(absStartY, absEndY);
  const maxY = Math.max(absStartY, absEndY);

  currentSelectionCoords = { 
    min_x: parseFloat(minX.toFixed(4)), 
    max_x: parseFloat(maxX.toFixed(4)), 
    min_y: parseFloat(minY.toFixed(4)), 
    max_y: parseFloat(maxY.toFixed(4)) 
  };
  
  zoneManagement.classList.remove('hidden');
  zoneNameInput.focus();
}

function unprojectToWorld(clientX, clientY) {
  if (!viewer || !viewer.GetCamera()) return { x: 0, y: 0 };
  
  const rect = canvasContainer.getBoundingClientRect();
  
  const x = ((clientX - rect.left) / rect.width) * 2 - 1;
  const y = -((clientY - rect.top) / rect.height) * 2 + 1;
  
  const vector = new THREE.Vector3(x, y, 0);
  vector.unproject(viewer.GetCamera());
  
  return {
    x: vector.x,
    y: vector.y
  };
}

saveZoneBtn.addEventListener('click', () => {
  const name = zoneNameInput.value.trim();
  if (!name || !currentSelectionCoords) return;

  const newZone = { name, ...currentSelectionCoords };
  zones.push(newZone);
  
  drawZoneBox(newZone);
  
  updateZonesList();
  
  zoneNameInput.value = '';
  currentSelectionCoords = null;
  
  runDetectionsBtn.classList.remove('hidden');
});

function drawZoneBox(zone) {
  if (!viewer || !viewer.GetScene()) return;
  
  const origin = viewer.GetOrigin ? viewer.GetOrigin() : { x: 0, y: 0 };
  
  const lx1 = zone.min_x - origin.x;
  const ly1 = zone.min_y - origin.y;
  const lx2 = zone.max_x - origin.x;
  const ly2 = zone.max_y - origin.y;
  
  const material = new THREE.LineBasicMaterial({ 
    color: 0x38bdf8, 
    linewidth: 2,
    depthTest: false,
    depthWrite: false,
    transparent: true,
    opacity: 1.0
  });
  const points = [];
  points.push(new THREE.Vector3(lx1, ly1, 0));
  points.push(new THREE.Vector3(lx2, ly1, 0));
  points.push(new THREE.Vector3(lx2, ly2, 0));
  points.push(new THREE.Vector3(lx1, ly2, 0));
  points.push(new THREE.Vector3(lx1, ly1, 0));
  
  const geometry = new THREE.BufferGeometry().setFromPoints(points);
  const line = new THREE.Line(geometry, material);
  line.userData = { isZone: true, zoneName: zone.name, zoneData: zone };
  
  viewer.GetScene().add(line);
  zone.threeObject = line;
  
  if (viewer.Render) viewer.Render();
}

function updateZonesList() {
  savedZonesList.innerHTML = '';
  zones.forEach((z, index) => {
    const li = document.createElement('li');
    li.className = 'zone-item';
    li.innerHTML = `
      <span>${z.name}</span>
      <div style="display: flex; gap: 8px;">
        <button class="run-btn btn btn-primary" data-index="${index}" style="padding: 4px 8px; font-size: 0.8rem;">Run</button>
        <button class="delete-btn" data-index="${index}">×</button>
      </div>
    `;
    savedZonesList.appendChild(li);
  });
  
  document.querySelectorAll('.run-btn').forEach(btn => {
    btn.addEventListener('click', (e) => {
      const idx = e.target.getAttribute('data-index');
      const z = zones[idx];
      runDetectionsForZones([z]);
    });
  });

  document.querySelectorAll('.delete-btn').forEach(btn => {
    btn.addEventListener('click', (e) => {
      const idx = e.target.getAttribute('data-index');
      const z = zones[idx];
      if (z.threeObject && viewer && viewer.GetScene()) {
        viewer.GetScene().remove(z.threeObject);
        z.threeObject.geometry.dispose();
        z.threeObject.material.dispose();
      }
      zones.splice(idx, 1);
      updateZonesList();
      if (zones.length === 0) runDetectionsBtn.classList.add('hidden');
      if (viewer && viewer.Render) viewer.Render();
    });
  });
}

runDetectionsBtn.addEventListener('click', () => {
  runDetectionsForZones(zones);
});

async function runDetectionsForZones(zonesToRun) {
  if (!dxfFileObj || zonesToRun.length === 0) return;
  
  runDetectionsBtn.disabled = true;
  runDetectionsBtn.innerHTML = '<span class="icon">⏳</span> Processing...';
  
  const formData = new FormData();
  formData.append('file', dxfFileObj);
  const cleanZones = zonesToRun.map(z => ({ name: z.name, min_x: z.min_x, max_x: z.max_x, min_y: z.min_y, max_y: z.max_y }));
  formData.append('zones', JSON.stringify(cleanZones));
  
  try {
    const response = await fetch('http://localhost:8000/api/detect', {
      method: 'POST',
      body: formData
    });
    
    if (!response.ok) throw new Error('API Error');
    
    const data = await response.json();
    globalDetectionsData = data;
    
    if (zonesToRun.length === 1) {
      const zName = zonesToRun[0].name;
      const zoneBom = globalDetectionsData.zones_bom[zName] || {};
      renderBOM(zoneBom, `BOM: ${zName}`);
    } else {
      renderBOM(data.global_bom, 'Global BOM');
    }
    bomDashboard.classList.remove('hidden');
    
  } catch (err) {
    console.error(err);
    alert('Failed to run detections');
  } finally {
    runDetectionsBtn.disabled = false;
    runDetectionsBtn.innerHTML = '<span class="icon">🚀</span> Run Detections';
  }
}

function renderBOM(bomObj, title) {
  bomTitle.textContent = title;
  bomList.innerHTML = '';
  
  if (!bomObj || Object.keys(bomObj).length === 0) {
    bomList.innerHTML = '<li class="bom-item" style="color: #94a3b8; font-style: italic;">No detections</li>';
    return;
  }
  
  for (const [clase, count] of Object.entries(bomObj)) {
    const li = document.createElement('li');
    li.className = 'bom-item';
    li.innerHTML = `
      <span>${clase.replace(/_/g, ' ').toUpperCase()}</span>
      <span class="bom-count">${count}</span>
    `;
    bomList.appendChild(li);
  }
}

const raycaster = new THREE.Raycaster();
raycaster.params.Line.threshold = 1.0; 

canvasContainer.addEventListener('click', (e) => {
  if (isSelectionMode || isDragging) return;
  if (!globalDetectionsData) return;
  
  const rect = canvasContainer.getBoundingClientRect();
  const x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
  const y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
  
  if (!viewer || !viewer.GetCamera() || !viewer.GetScene()) return;
  
  raycaster.setFromCamera(new THREE.Vector2(x, y), viewer.GetCamera());
  const intersects = raycaster.intersectObjects(viewer.GetScene().children, true);
  
  const zoneIntersect = intersects.find(i => i.object.userData && i.object.userData.isZone);
  
  if (zoneIntersect) {
    const zName = zoneIntersect.object.userData.zoneName;
    const zoneBom = globalDetectionsData.zones_bom[zName] || {};
    renderBOM(zoneBom, `BOM: ${zName}`);
    
    zones.forEach(z => {
      if (z.threeObject) z.threeObject.material.color.setHex(z.name === zName ? 0x10b981 : 0x38bdf8);
    });
    if(viewer.Render) viewer.Render();
  } else {
    renderBOM(globalDetectionsData.global_bom, 'Global BOM');
    zones.forEach(z => {
      if (z.threeObject) z.threeObject.material.color.setHex(0x38bdf8);
    });
    if(viewer.Render) viewer.Render();
  }
});
