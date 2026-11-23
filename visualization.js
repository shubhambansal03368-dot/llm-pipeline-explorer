/**
 * LLM Pipeline Explorer - 3D Visualization Module
 * Pure Vanilla JavaScript (ES6 Module)
 *
 * Drives interactive 3D vector-space rendering via Three.js (CDN).
 * Includes OrbitControls, smooth coordinate transitions (lerp animation),
 * 2D projected labels, nearest-neighbor dashed lines, and a graceful
 * 2D Canvas fallback if Three.js is unavailable.
 */

import { findNearestNeighbors } from './embeddings.js';

const CATEGORY_COLORS = {
  ai_tech: 0x2563eb, // blue
  royalty_people: 0x7c3aed, // violet
  vehicles: 0x0284c7, // sky
  food_fruit: 0x16a34a, // green
  animals: 0xd97706, // amber
  nature_space: 0x0d9488, // teal
  actions: 0xe11d48, // rose
  abstract_concepts: 0x4f46e5, // indigo
  common: 0x64748b, // slate
};

function getWordColor(category, index) {
  const catLower = (category || '').toLowerCase();
  for (const [key, color] of Object.entries(CATEGORY_COLORS)) {
    if (catLower.includes(key) || catLower.includes(key.replace('_', ' '))) {
      return color;
    }
  }
  const palette = [0x2563eb, 0x7c3aed, 0x0284c7, 0x059669, 0xd97706, 0x4f46e5, 0xe11d48];
  return palette[index % palette.length];
}

export class EmbeddingVisualizer {
  constructor(containerEl, options = {}) {
    this.container = containerEl;
    this.canvas = containerEl.querySelector('canvas') || document.createElement('canvas');
    if (!this.canvas.parentElement) {
      this.container.appendChild(this.canvas);
    }

    this.onSelectWord = options.onSelectWord || (() => {});
    this.embeddings = [];
    this.selectedWord = null;
    this.mode = 'spelling';

    // Display flags
    this.showLabels = true;
    this.showLines = true;
    this.showGrid = true;
    this.autoRotate = false;

    // Three.js instances
    this.isThreeSupported = typeof window.THREE !== 'undefined';
    this.scene = null;
    this.camera = null;
    this.renderer = null;
    this.controls = null;
    this.pointsGroup = null;
    this.linesGroup = null;
    this.haloMesh = null;
    this.animId = null;

    // Position interpolation (lerp)
    this.currentPositions = new Map();
    this.targetPositions = new Map();
    this.transitionProgress = 1.0;

    // HTML Label Overlay Container
    this.labelsOverlay = containerEl.querySelector('.labels-overlay');
    if (!this.labelsOverlay) {
      this.labelsOverlay = document.createElement('div');
      this.labelsOverlay.className = 'labels-overlay';
      this.container.appendChild(this.labelsOverlay);
    }

    // Tooltip Element
    this.tooltip = containerEl.querySelector('.viz-tooltip');
    if (!this.tooltip) {
      this.tooltip = document.createElement('div');
      this.tooltip.className = 'viz-tooltip hidden';
      this.container.appendChild(this.tooltip);
    }

    if (this.isThreeSupported) {
      this.initThree();
    } else {
      console.warn('Three.js CDN not found; initializing 2D Canvas Fallback.');
      this.init2DFallback();
    }
  }

  // -------------------------------------------------------------
  // Three.js 3D Initialization
  // -------------------------------------------------------------
  initThree() {
    const THREE = window.THREE;
    const width = this.container.clientWidth || 800;
    const height = this.container.clientHeight || 550;

    // Scene
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0xfbfcfe);

    // Camera
    this.camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 1000);
    this.camera.position.set(0, 10, 24);

    // Renderer
    this.renderer = new THREE.WebGLRenderer({
      canvas: this.canvas,
      antialias: true,
      alpha: true,
    });
    this.renderer.setSize(width, height);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));

    // OrbitControls
    if (THREE.OrbitControls) {
      this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
      this.controls.enableDamping = true;
      this.controls.dampingFactor = 0.05;
      this.controls.minDistance = 4;
      this.controls.maxDistance = 60;
    }

    // Lighting
    const ambient = new THREE.AmbientLight(0xffffff, 0.85);
    this.scene.add(ambient);

    const dir1 = new THREE.DirectionalLight(0xffffff, 0.75);
    dir1.position.set(20, 30, 20);
    this.scene.add(dir1);

    const dir2 = new THREE.DirectionalLight(0xe0e7ff, 0.45);
    dir2.position.set(-20, -10, -20);
    this.scene.add(dir2);

    // Grid & Axes
    const grid = new THREE.GridHelper(24, 24, 0xdbeafe, 0xf1f5f9);
    grid.position.y = -8;
    grid.name = 'gridHelper';
    this.scene.add(grid);

    const axes = new THREE.AxesHelper(3);
    axes.position.set(-10, -8, -10);
    axes.name = 'axesHelper';
    this.scene.add(axes);

    // Groups
    this.pointsGroup = new THREE.Group();
    this.scene.add(this.pointsGroup);

    this.linesGroup = new THREE.Group();
    this.scene.add(this.linesGroup);

    // Selection Halo
    const haloGeo = new THREE.RingGeometry(0.55, 0.72, 32);
    const haloMat = new THREE.MeshBasicMaterial({
      color: 0x2563eb,
      side: THREE.DoubleSide,
      transparent: true,
      opacity: 0.8,
    });
    this.haloMesh = new THREE.Mesh(haloGeo, haloMat);
    this.haloMesh.visible = false;
    this.scene.add(this.haloMesh);

    // Bind pointer events
    this.canvas.addEventListener('pointermove', (e) => this.onPointerMove(e));
    this.canvas.addEventListener('pointerdown', (e) => this.onPointerDown(e));

    // Resize observer
    this.resizeObserver = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const { width: newW, height: newH } = entry.contentRect;
        if (newW > 0 && newH > 0 && this.camera && this.renderer) {
          this.camera.aspect = newW / newH;
          this.camera.updateProjectionMatrix();
          this.renderer.setSize(newW, newH);
        }
      }
    });
    this.resizeObserver.observe(this.container);

    // Start Animation Loop
    let lastTime = performance.now();
    const animate = (time) => {
      this.animId = requestAnimationFrame(animate);
      const delta = (time - lastTime) / 1000;
      lastTime = time;

      // Position lerping
      if (this.transitionProgress < 1.0) {
        this.transitionProgress = Math.min(1.0, this.transitionProgress + delta * 2.2);
        const t = this.transitionProgress;
        const ease = t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;

        this.pointsGroup.children.forEach((mesh) => {
          const word = mesh.userData.word;
          const start = this.currentPositions.get(word);
          const target = this.targetPositions.get(word);
          if (start && target) {
            mesh.position.lerpVectors(start, target, ease);
          }
        });
      }

      if (this.controls) {
        this.controls.autoRotate = this.autoRotate;
        this.controls.update();
      }

      if (this.haloMesh && this.haloMesh.visible) {
        this.haloMesh.quaternion.copy(this.camera.quaternion);
      }

      this.renderer.render(this.scene, this.camera);
      this.updateScreenLabels();
    };

    this.animId = requestAnimationFrame(animate);
  }

  // -------------------------------------------------------------
  // Data Updates & Mesh Recreation
  // -------------------------------------------------------------
  setEmbeddings(embeddings, mode = 'spelling') {
    this.embeddings = embeddings;
    this.mode = mode;

    if (!this.isThreeSupported) {
      this.draw2DFallback();
      return;
    }

    const THREE = window.THREE;
    const oldPositions = new Map();
    this.pointsGroup.children.forEach((child) => {
      oldPositions.set(child.userData.word, child.position.clone());
    });

    // Clear old points
    while (this.pointsGroup.children.length > 0) {
      const obj = this.pointsGroup.children[0];
      this.pointsGroup.remove(obj);
      if (obj.geometry) obj.geometry.dispose();
      if (obj.material) obj.material.dispose();
    }

    const newTargetPositions = new Map();
    const sphereGeo = new THREE.SphereGeometry(0.35, 24, 24);

    embeddings.forEach((emb, index) => {
      const isSelected = this.selectedWord && this.selectedWord.toLowerCase() === emb.word.toLowerCase();
      const colorHex = getWordColor(emb.category, index);

      const sphereMat = new THREE.MeshStandardMaterial({
        color: colorHex,
        roughness: 0.25,
        metalness: 0.15,
        emissive: isSelected ? 0x2563eb : 0x000000,
        emissiveIntensity: isSelected ? 0.45 : 0,
      });

      const mesh = new THREE.Mesh(sphereGeo, sphereMat);
      mesh.userData = {
        word: emb.word,
        id: emb.id,
        embedding: emb,
        isSelected,
      };

      const targetPos = new THREE.Vector3(...emb.coordinates3D);
      newTargetPositions.set(emb.word, targetPos);

      if (oldPositions.has(emb.word)) {
        mesh.position.copy(oldPositions.get(emb.word));
      } else {
        mesh.position.copy(targetPos);
      }

      this.pointsGroup.add(mesh);
    });

    this.currentPositions = oldPositions;
    this.targetPositions = newTargetPositions;
    this.transitionProgress = oldPositions.size > 0 ? 0.0 : 1.0;

    this.updateSelectionAndLines();
  }

  setSelectedWord(word) {
    this.selectedWord = word ? word.toLowerCase() : null;

    if (!this.isThreeSupported) {
      this.draw2DFallback();
      return;
    }

    // Update emissive highlighting
    if (this.pointsGroup) {
      this.pointsGroup.children.forEach((mesh) => {
        const isSelected = this.selectedWord && mesh.userData.word.toLowerCase() === this.selectedWord;
        mesh.userData.isSelected = isSelected;
        if (mesh.material) {
          mesh.material.emissive.setHex(isSelected ? 0x2563eb : 0x000000);
          mesh.material.emissiveIntensity = isSelected ? 0.45 : 0;
        }
      });
    }

    this.updateSelectionAndLines();
  }

  updateSelectionAndLines() {
    if (!this.isThreeSupported) return;
    const THREE = window.THREE;

    // Clear lines
    while (this.linesGroup.children.length > 0) {
      const obj = this.linesGroup.children[0];
      this.linesGroup.remove(obj);
      if (obj.geometry) obj.geometry.dispose();
      if (obj.material) obj.material.dispose();
    }

    if (!this.selectedWord) {
      if (this.haloMesh) this.haloMesh.visible = false;
      return;
    }

    const targetEmb = this.embeddings.find(
      (e) => e.word.toLowerCase() === this.selectedWord
    );

    if (!targetEmb) {
      if (this.haloMesh) this.haloMesh.visible = false;
      return;
    }

    // Position selection halo
    if (this.haloMesh) {
      this.haloMesh.position.set(...targetEmb.coordinates3D);
      this.haloMesh.visible = true;
    }

    // Connect Top-3 nearest neighbors
    if (this.showLines) {
      const neighbors = findNearestNeighbors(this.selectedWord, this.embeddings, 3);
      neighbors.forEach((neighbor) => {
        const points = [
          new THREE.Vector3(...targetEmb.coordinates3D),
          new THREE.Vector3(...neighbor.coordinates3D),
        ];
        const lineGeo = new THREE.BufferGeometry().setFromPoints(points);
        const lineMat = new THREE.LineDashedMaterial({
          color: 0x3b82f6,
          dashSize: 0.4,
          gapSize: 0.2,
          linewidth: 2,
          transparent: true,
          opacity: 0.75,
        });
        const line = new THREE.Line(lineGeo, lineMat);
        line.computeLineDistances();
        this.linesGroup.add(line);
      });
    }
  }

  // -------------------------------------------------------------
  // 2D Screen Projected Typography Labels
  // -------------------------------------------------------------
  updateScreenLabels() {
    if (!this.labelsOverlay) return;

    if (!this.showLabels) {
      this.labelsOverlay.innerHTML = '';
      return;
    }

    const THREE = window.THREE;
    const width = this.container.clientWidth;
    const height = this.container.clientHeight;
    const tempVec = new THREE.Vector3();

    let html = '';
    this.pointsGroup.children.forEach((mesh) => {
      tempVec.setFromMatrixPosition(mesh.matrixWorld);
      tempVec.project(this.camera);

      // Only in front of camera
      if (tempVec.z < 1) {
        const x = (tempVec.x * 0.5 + 0.5) * width;
        const y = (-(tempVec.y * 0.5) + 0.5) * height;

        if (x >= 25 && x <= width - 25 && y >= 25 && y <= height - 25) {
          const isSelected = mesh.userData.isSelected;
          const word = mesh.userData.word;
          const cls = isSelected ? 'point-label selected' : 'point-label';
          html += `<div class="${cls}" style="left:${x}px; top:${y - 18}px;" data-word="${word}">${word}</div>`;
        }
      }
    });

    this.labelsOverlay.innerHTML = html;

    // Attach click handlers to 2D labels
    this.labelsOverlay.querySelectorAll('.point-label').forEach((el) => {
      el.addEventListener('click', (e) => {
        const word = e.target.getAttribute('data-word');
        if (word) {
          this.setSelectedWord(word);
          this.onSelectWord(word);
        }
      });
    });
  }

  // -------------------------------------------------------------
  // Pointer Interactions & Tooltip
  // -------------------------------------------------------------
  onPointerMove(e) {
    if (!this.isThreeSupported || !this.camera || !this.pointsGroup) return;

    const THREE = window.THREE;
    const rect = this.canvas.getBoundingClientRect();
    const x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
    const y = -((e.clientY - rect.top) / rect.height) * 2 + 1;

    const raycaster = new THREE.Raycaster();
    raycaster.setFromCamera(new THREE.Vector2(x, y), this.camera);
    const intersects = raycaster.intersectObjects(this.pointsGroup.children);

    if (intersects.length > 0) {
      const mesh = intersects[0].object;
      const emb = mesh.userData.embedding;
      this.canvas.style.cursor = 'pointer';
      this.showTooltip(emb, e.clientX - rect.left, e.clientY - rect.top);
    } else {
      this.canvas.style.cursor = 'grab';
      this.hideTooltip();
    }
  }

  onPointerDown(e) {
    if (!this.isThreeSupported || !this.camera || !this.pointsGroup) return;

    const THREE = window.THREE;
    const rect = this.canvas.getBoundingClientRect();
    const x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
    const y = -((e.clientY - rect.top) / rect.height) * 2 + 1;

    const raycaster = new THREE.Raycaster();
    raycaster.setFromCamera(new THREE.Vector2(x, y), this.camera);
    const intersects = raycaster.intersectObjects(this.pointsGroup.children);

    if (intersects.length > 0) {
      const mesh = intersects[0].object;
      const word = mesh.userData.word;
      this.setSelectedWord(word);
      this.onSelectWord(word);
    }
  }

  showTooltip(emb, x, y) {
    if (!this.tooltip) return;
    this.tooltip.classList.remove('hidden');
    this.tooltip.style.left = `${x + 16}px`;
    this.tooltip.style.top = `${y + 16}px`;
    this.tooltip.innerHTML = `
      <div class="tooltip-header">
        <span class="tooltip-title">"${emb.word}"</span>
        <span class="tooltip-id">ID: ${emb.id}</span>
      </div>
      <div class="tooltip-body">
        <div>Category: <strong>${emb.category}</strong></div>
        <div>Coords: [${emb.coordinates3D.map((n) => n.toFixed(2)).join(', ')}]</div>
        <div class="tooltip-vec">Vector: [${emb.vector.slice(0, 4).map((n) => n.toFixed(2)).join(', ')}...]</div>
      </div>
    `;
  }

  hideTooltip() {
    if (this.tooltip) {
      this.tooltip.classList.add('hidden');
    }
  }

  // -------------------------------------------------------------
  // Camera Controls
  // -------------------------------------------------------------
  resetCamera() {
    if (this.camera && this.controls) {
      this.camera.position.set(0, 10, 24);
      this.controls.target.set(0, 0, 0);
      this.controls.update();
    }
  }

  zoomIn() {
    if (this.camera && this.controls) {
      this.camera.position.multiplyScalar(0.8);
      this.controls.update();
    }
  }

  zoomOut() {
    if (this.camera && this.controls) {
      this.camera.position.multiplyScalar(1.25);
      this.controls.update();
    }
  }

  toggleLabels() {
    this.showLabels = !this.showLabels;
    this.updateScreenLabels();
    return this.showLabels;
  }

  toggleLines() {
    this.showLines = !this.showLines;
    this.updateSelectionAndLines();
    return this.showLines;
  }

  toggleAutoRotate() {
    this.autoRotate = !this.autoRotate;
    return this.autoRotate;
  }

  toggleGrid() {
    this.showGrid = !this.showGrid;
    if (this.scene) {
      const grid = this.scene.getObjectByName('gridHelper');
      const axes = this.scene.getObjectByName('axesHelper');
      if (grid) grid.visible = this.showGrid;
      if (axes) axes.visible = this.showGrid;
    }
    return this.showGrid;
  }

  // -------------------------------------------------------------
  // Graceful 2D Canvas Fallback (if Three.js is blocked/offline)
  // -------------------------------------------------------------
  init2DFallback() {
    const ctx = this.canvas.getContext('2d');
    this.ctx2d = ctx;

    const resize = () => {
      this.canvas.width = this.container.clientWidth;
      this.canvas.height = this.container.clientHeight || 500;
      this.draw2DFallback();
    };

    window.addEventListener('resize', resize);
    resize();
  }

  draw2DFallback() {
    if (!this.ctx2d) return;
    const ctx = this.ctx2d;
    const w = this.canvas.width;
    const h = this.canvas.height;

    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = '#f8fafc';
    ctx.fillRect(0, 0, w, h);

    // Draw center axes
    ctx.strokeStyle = '#e2e8f0';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(w / 2, 0);
    ctx.lineTo(w / 2, h);
    ctx.moveTo(0, h / 2);
    ctx.lineTo(w, h / 2);
    ctx.stroke();

    if (this.embeddings.length === 0) return;

    const scale = Math.min(w, h) / 20;

    // Draw points
    this.embeddings.forEach((emb, i) => {
      const [x3d, y3d] = emb.coordinates3D;
      const screenX = w / 2 + x3d * scale;
      const screenY = h / 2 - y3d * scale;

      const isSelected = this.selectedWord && this.selectedWord.toLowerCase() === emb.word.toLowerCase();

      ctx.beginPath();
      ctx.arc(screenX, screenY, isSelected ? 8 : 5, 0, Math.PI * 2);
      ctx.fillStyle = isSelected ? '#2563eb' : '#4f46e5';
      ctx.fill();

      if (this.showLabels) {
        ctx.fillStyle = isSelected ? '#1e3a8a' : '#334155';
        ctx.font = isSelected ? 'bold 12px Inter, sans-serif' : '11px Inter, sans-serif';
        ctx.fillText(emb.word, screenX + 8, screenY + 4);
      }
    });
  }
}
