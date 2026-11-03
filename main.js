import * as THREE from 'three';
import { FaceLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';

/*
 * 3D 視窗相框
 * ------------------------------------------------------------
 * 原理:把手機螢幕當成真實世界裡的一扇「窗戶」(位於 z = 0 的平面,
 * 單位為公分)。用前鏡頭追蹤眼睛的 3D 位置 (x, y, z),
 * 再以「非對稱視錐」(off-axis projection) 從眼睛位置穿過螢幕四個角
 * 建立投影矩陣。這樣螢幕上的畫面就會像透過窗戶看到後面的空間一樣。
 */

const WASM_URL = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm';
const MODEL_URL =
  'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task';

const DEFAULT_Z = 35; // 沒有追蹤資料時,假設眼睛在螢幕前 35 cm
const STORE_KEY = '3dframe.settings.v1';

// ---------- 設定 ----------
const minViewportPx = () => Math.min(innerWidth, innerHeight);
const isPhone = Math.min(screen.width, screen.height) < 600;
const DEFAULTS = {
  cmPerPx: isPhone ? 0.0167 : 0.026, // 每個 CSS 像素對應幾公分
  fov: 65, // 前鏡頭在影像長邊方向的視角(度)
  ipd: 6.3, // 瞳距(cm)
  gain: 1, // 效果強度(放大左右上下位移)
  depth: 25, // 場景深度(cm)
  camGap: 0.8, // 鏡頭到螢幕可視區域邊緣的距離(cm)
  smooth: 0.6,
  preview: false,
};
const S = loadSettings();

function loadSettings() {
  try {
    const saved = JSON.parse(localStorage.getItem(STORE_KEY) || '{}');
    return { ...DEFAULTS, ...saved };
  } catch {
    return { ...DEFAULTS };
  }
}
function saveSettings() {
  try {
    localStorage.setItem(STORE_KEY, JSON.stringify(S));
  } catch {
    /* 私密瀏覽等情況下忽略 */
  }
}

// ---------- DOM ----------
const $ = (s) => document.querySelector(s);
const canvas = $('#c');
const video = $('#video');
const preview = $('#preview');
const pctx = preview.getContext('2d');
const statusEl = $('#status');
const dotEl = $('#dot');

function setStatus(text, level = '') {
  statusEl.textContent = text;
  dotEl.className = level;
}

// ---------- Three.js ----------
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera();

function screenSize() {
  return { W: innerWidth * S.cmPerPx, H: innerHeight * S.cmPerPx };
}

// 非對稱視錐:眼睛在 (ex, ey, ez),螢幕是 z=0 上寬 W、高 H 的矩形
function updateCamera(eye) {
  const { W, H } = screenSize();
  const ex = eye.x * S.gain;
  const ey = eye.y * S.gain;
  const ez = Math.max(eye.z, 8);
  const near = 0.5;
  const far = 2000;
  camera.position.set(ex, ey, ez);
  camera.quaternion.identity();
  camera.updateMatrixWorld();
  const k = near / ez;
  camera.projectionMatrix.makePerspective(
    (-W / 2 - ex) * k,
    (W / 2 - ex) * k,
    (H / 2 - ey) * k,
    (-H / 2 - ey) * k,
    near,
    far,
  );
  camera.projectionMatrixInverse.copy(camera.projectionMatrix).invert();
}

// ---------- 小工具 ----------
function mulberry32(seed) {
  return () => {
    seed |= 0;
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function canvasTexture(c) {
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  t.anisotropy = 4;
  return t;
}

function radialTexture(inner, outer) {
  const c = document.createElement('canvas');
  c.width = c.height = 128;
  const x = c.getContext('2d');
  const g = x.createRadialGradient(64, 64, 0, 64, 64, 64);
  g.addColorStop(0, inner);
  g.addColorStop(1, outer);
  x.fillStyle = g;
  x.fillRect(0, 0, 128, 128);
  return canvasTexture(c);
}

// 螢幕後方的盒子:背牆 + 左右上下四面牆
function addBox(group, W, H, D, mat, { back = true, cast = false } = {}) {
  const walls = [];
  const add = (geo, pos, rot) => {
    const m = new THREE.Mesh(geo, mat);
    m.position.set(...pos);
    m.rotation.set(...rot);
    m.receiveShadow = true;
    m.castShadow = cast;
    group.add(m);
    walls.push(m);
  };
  if (back) add(new THREE.PlaneGeometry(W, H), [0, 0, -D], [0, 0, 0]);
  add(new THREE.PlaneGeometry(D, H), [-W / 2, 0, -D / 2], [0, Math.PI / 2, 0]);
  add(new THREE.PlaneGeometry(D, H), [W / 2, 0, -D / 2], [0, -Math.PI / 2, 0]);
  add(new THREE.PlaneGeometry(W, D), [0, -H / 2, -D / 2], [-Math.PI / 2, 0, 0]);
  add(new THREE.PlaneGeometry(W, D), [0, H / 2, -D / 2], [Math.PI / 2, 0, 0]);
  return walls;
}

function gridLines(W, H, D, color, opacity) {
  const p = [];
  const seg = (a, b) => p.push(...a, ...b);
  const nx = 6;
  const sx = W / nx;
  const ny = Math.max(2, Math.round(H / sx));
  const sy = H / ny;
  const nz = Math.max(2, Math.round(D / sx));
  const sz = D / nz;
  const x0 = -W / 2, x1 = W / 2, y0 = -H / 2, y1 = H / 2;
  for (let k = 0; k <= nz; k++) {
    const z = -k * sz;
    seg([x0, y0, z], [x1, y0, z]);
    seg([x1, y0, z], [x1, y1, z]);
    seg([x1, y1, z], [x0, y1, z]);
    seg([x0, y1, z], [x0, y0, z]);
  }
  for (let i = 0; i <= nx; i++) {
    const x = x0 + i * sx;
    seg([x, y0, 0], [x, y0, -D]);
    seg([x, y1, 0], [x, y1, -D]);
    seg([x, y0, -D], [x, y1, -D]);
  }
  for (let j = 0; j <= ny; j++) {
    const y = y0 + j * sy;
    seg([x0, y, 0], [x0, y, -D]);
    seg([x1, y, 0], [x1, y, -D]);
    seg([x0, y, -D], [x1, y, -D]);
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.Float32BufferAttribute(p, 3));
  return new THREE.LineSegments(
    geo,
    new THREE.LineBasicMaterial({ color, transparent: true, opacity }),
  );
}

function shadowLight(group, pos, target, size, intensity = 2.2) {
  const light = new THREE.DirectionalLight(0xffffff, intensity);
  light.position.set(...pos);
  light.target.position.set(...target);
  light.castShadow = true;
  light.shadow.mapSize.set(1024, 1024);
  const cam = light.shadow.camera;
  cam.left = cam.bottom = -size;
  cam.right = cam.top = size;
  cam.near = 1;
  cam.far = 300;
  light.shadow.bias = -0.0005;
  light.shadow.normalBias = 0.02;
  group.add(light, light.target);
  return light;
}

// ---------- 場景 1:房間(經典格線盒子 + 懸吊的球) ----------
function buildRoom(W, H, D) {
  const g = new THREE.Group();
  const wallMat = new THREE.MeshStandardMaterial({
    color: 0x161b33,
    roughness: 0.9,
    polygonOffset: true,
    polygonOffsetFactor: 1,
    polygonOffsetUnits: 1,
  });
  addBox(g, W, H, D, wallMat);
  g.add(gridLines(W, H, D, 0x3ae0ff, 0.5));

  // 螢幕邊框的亮線,強調「窗框」
  const rim = gridLines(W, H, 0.0001, 0xffffff, 0.0);
  g.add(rim);

  g.add(new THREE.HemisphereLight(0x9fb0ff, 0x10101c, 1.1));
  shadowLight(g, [W * 0.35, H * 0.55, 40], [0, -H * 0.1, -D], Math.max(W, H, D));

  const specs = [
    // x, y (相對寬高), z (相對深度), 半徑, 顏色
    [-0.26, 0.16, -0.22, 0.5, 0xff5a7a],
    [0.24, -0.06, -0.5, 0.55, 0x3ae0ff],
    [-0.04, 0.3, -0.82, 0.5, 0xffc857],
    [0.2, 0.3, -0.12, 0.38, 0x8affc1],
    [-0.22, -0.24, -0.66, 0.55, 0xb18cff],
  ];
  const balls = [];
  const stringMat = new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.35 });
  for (const [sx, sy, sz, sr, color] of specs) {
    const r = sr * W * 0.2;
    const ball = new THREE.Mesh(
      new THREE.SphereGeometry(r, 32, 20),
      new THREE.MeshStandardMaterial({ color, roughness: 0.35, metalness: 0.1 }),
    );
    const x = sx * W, y = sy * H, z = sz * D;
    ball.position.set(x, y, z);
    ball.castShadow = true;
    const sgeo = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(x, H / 2, z),
      new THREE.Vector3(x, y + r, z),
    ]);
    const string = new THREE.Line(sgeo, stringMat);
    g.add(ball, string);
    balls.push({ ball, string, y, r, phase: Math.random() * 6.28 });
  }

  // 凸出螢幕外的物體(z > 0)
  const knot = new THREE.Mesh(
    new THREE.TorusKnotGeometry(W * 0.12, W * 0.035, 160, 20),
    new THREE.MeshStandardMaterial({ color: 0xff8a3d, roughness: 0.25, metalness: 0.55 }),
  );
  knot.position.set(0, -H * 0.12, 4);
  knot.castShadow = true;
  g.add(knot);

  return {
    group: g,
    background: 0x05060a,
    fog: new THREE.Fog(0x05060a, DEFAULT_Z + D * 0.4, DEFAULT_Z + D * 2.4),
    update(t) {
      for (const b of balls) {
        const y = b.y + Math.sin(t * 0.0012 + b.phase) * H * 0.012;
        b.ball.position.y = y;
        const pos = b.string.geometry.attributes.position;
        pos.setY(1, y + b.r);
        pos.needsUpdate = true;
      }
      knot.rotation.x = t * 0.0004;
      knot.rotation.y = t * 0.0007;
    },
  };
}

// ---------- 場景 2:山景(多層剪影,深度各不相同) ----------
function ridgeTexture(seed, top, bottom, ridgeFrac, rough) {
  const c = document.createElement('canvas');
  c.width = 1024;
  c.height = 512;
  const x = c.getContext('2d');
  const rnd = mulberry32(seed);
  const ph = [0, 0, 0, 0, 0].map(() => rnd() * Math.PI * 2);
  const f = [1 + rnd(), 2.5 + rnd() * 1.5, 6 + rnd() * 3, 15 + rnd() * 6, 34 + rnd() * 10];
  const a = [0.5, 0.28 * rough, 0.14 * rough, 0.06 * rough, 0.025 * rough];
  x.beginPath();
  x.moveTo(0, 512);
  for (let i = 0; i <= 512; i++) {
    const u = i / 512;
    let h = 0;
    for (let k = 0; k < 5; k++) h += Math.sin(u * Math.PI * 2 * f[k] + ph[k]) * a[k];
    x.lineTo(u * 1024, (ridgeFrac - h * 0.16) * 512);
  }
  x.lineTo(1024, 512);
  x.closePath();
  const g = x.createLinearGradient(0, (ridgeFrac - 0.2) * 512, 0, 512);
  g.addColorStop(0, top);
  g.addColorStop(1, bottom);
  x.fillStyle = g;
  x.fill();
  return canvasTexture(c);
}

function skyTexture() {
  const c = document.createElement('canvas');
  c.width = 512;
  c.height = 1024;
  const x = c.getContext('2d');
  const g = x.createLinearGradient(0, 0, 0, 1024);
  g.addColorStop(0, '#05051a');
  g.addColorStop(0.3, '#1a1446');
  g.addColorStop(0.47, '#5a3a7e');
  g.addColorStop(0.56, '#e07a6a');
  g.addColorStop(0.62, '#ffc08a');
  g.addColorStop(1, '#ffe2b8');
  x.fillStyle = g;
  x.fillRect(0, 0, 512, 1024);
  const rnd = mulberry32(7);
  for (let i = 0; i < 260; i++) {
    const y = rnd() * 460;
    x.globalAlpha = (1 - y / 460) * (0.4 + rnd() * 0.6);
    x.fillStyle = '#fff';
    const s = rnd() < 0.08 ? 1.6 : 0.8;
    x.fillRect(rnd() * 512, y, s, s);
  }
  return canvasTexture(c);
}

function buildMountain(W, H, D) {
  const g = new THREE.Group();
  const e = DEFAULT_Z;
  // 在深度 d 的平面要做多大,頭移動時邊緣才不會露出來
  const cover = (d) => 1 + (3 * d) / 20;
  // 讓從預設位置看時,物件出現在螢幕上 screenY(相對高度)的世界座標
  const worldY = (screenY, d) => screenY * H * (e + d) / e;

  // 天空
  const dSky = D * 4;
  const skySize = Math.max(W, H) * cover(dSky) * 1.3;
  const sky = new THREE.Mesh(
    new THREE.PlaneGeometry(skySize, skySize * 2),
    new THREE.MeshBasicMaterial({ map: skyTexture(), fog: false }),
  );
  // 天空貼圖地平線在 0.6 的位置,對齊到螢幕高度 +0.05H
  sky.position.set(0, worldY(0.05, dSky) + skySize * 2 * (0.6 - 0.5), -dSky);
  g.add(sky);

  // 月亮 + 光暈
  const dMoon = D * 3.6;
  const moonR = W * 0.07 * (e + dMoon) / e;
  const moonPos = [W * 0.18 * (e + dMoon) / e, worldY(0.28, dMoon), -dMoon];
  const glow = new THREE.Mesh(
    new THREE.PlaneGeometry(moonR * 9, moonR * 9),
    new THREE.MeshBasicMaterial({
      map: radialTexture('rgba(255,230,200,0.55)', 'rgba(255,230,200,0)'),
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
    }),
  );
  glow.position.set(...moonPos);
  const moon = new THREE.Mesh(
    new THREE.CircleGeometry(moonR, 48),
    new THREE.MeshBasicMaterial({ color: 0xfff3dc }),
  );
  moon.position.set(moonPos[0], moonPos[1], moonPos[2] + 0.1);
  g.add(glow, moon);

  // 山的層次:越遠越亮、越霧
  const layers = [
    { d: 2.8, top: '#9a6f9c', bottom: '#7a5888', y: 0.06, rough: 0.6, seed: 11 },
    { d: 1.7, top: '#61477e', bottom: '#4a3668', y: -0.04, rough: 0.8, seed: 23 },
    { d: 0.95, top: '#3a2a5a', bottom: '#2a1f45', y: -0.15, rough: 1.0, seed: 37 },
    { d: 0.4, top: '#1e1634', bottom: '#120d22', y: -0.29, rough: 1.2, seed: 51 },
  ];
  for (const L of layers) {
    const d = L.d * D;
    const size = W * cover(d) * 1.4;
    const ridgeFrac = 0.3;
    const mesh = new THREE.Mesh(
      new THREE.PlaneGeometry(size, size),
      new THREE.MeshBasicMaterial({
        map: ridgeTexture(L.seed, L.top, L.bottom, ridgeFrac, L.rough),
        transparent: true,
        fog: false,
      }),
    );
    mesh.position.set(0, worldY(L.y, d) - size * (0.5 - ridgeFrac), -d);
    g.add(mesh);
  }

  // 螢火蟲:有些在螢幕後、有些飄到螢幕前
  const N = 70;
  const rnd = mulberry32(99);
  const base = new Float32Array(N * 3);
  for (let i = 0; i < N; i++) {
    const z = -D * 1.1 + rnd() * (D * 1.1 + 6);
    const spread = (e - z) / e;
    base[i * 3] = (rnd() - 0.5) * W * 1.3 * spread;
    base[i * 3 + 1] = (-0.5 + rnd() * 0.45) * H * spread;
    base[i * 3 + 2] = z;
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(base.slice(), 3));
  const flies = new THREE.Points(
    geo,
    new THREE.PointsMaterial({
      size: W * 0.06,
      map: radialTexture('rgba(255,240,170,1)', 'rgba(255,200,80,0)'),
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
    }),
  );
  g.add(flies);

  return {
    group: g,
    background: 0x05051a,
    update(t) {
      const p = geo.attributes.position;
      for (let i = 0; i < N; i++) {
        const s = i * 1.7;
        p.setXYZ(
          i,
          base[i * 3] + Math.sin(t * 0.0005 + s) * W * 0.04,
          base[i * 3 + 1] + Math.sin(t * 0.0007 + s * 2.3) * W * 0.03,
          base[i * 3 + 2] + Math.cos(t * 0.0004 + s) * 0.6,
        );
      }
      p.needsUpdate = true;
      flies.material.opacity = 0.75 + Math.sin(t * 0.003) * 0.2;
    },
  };
}

// ---------- 場景 3:照片相框 ----------
let photoTexture = null;
let photoAspect = 1;

function placeholderTexture(W, H) {
  const c = document.createElement('canvas');
  c.width = 600;
  c.height = Math.round(600 * (H / W));
  const x = c.getContext('2d');
  const g = x.createLinearGradient(0, 0, c.width, c.height);
  g.addColorStop(0, '#2b5876');
  g.addColorStop(1, '#4e4376');
  x.fillStyle = g;
  x.fillRect(0, 0, c.width, c.height);
  x.fillStyle = 'rgba(255,255,255,0.9)';
  x.textAlign = 'center';
  x.font = '600 44px system-ui, sans-serif';
  x.fillText('再點一次「照片」', c.width / 2, c.height / 2 - 10);
  x.font = '32px system-ui, sans-serif';
  x.fillText('選一張你的照片放進相框', c.width / 2, c.height / 2 + 44);
  return canvasTexture(c);
}

function buildPhoto(W, H, D) {
  const g = new THREE.Group();
  const Dp = Math.max(4, D * 0.45);
  const paper = new THREE.MeshStandardMaterial({ color: 0xece6da, roughness: 1 });
  addBox(g, W, H, Dp, paper, { back: false, cast: true });

  // 內框的薄邊條(像相框裡的卡紙)
  const matDepth = Dp * 0.35;
  const border = W * 0.06;
  const shape = new THREE.Shape();
  shape.moveTo(-W / 2, -H / 2);
  shape.lineTo(W / 2, -H / 2);
  shape.lineTo(W / 2, H / 2);
  shape.lineTo(-W / 2, H / 2);
  const hole = new THREE.Path();
  hole.moveTo(-W / 2 + border, -H / 2 + border);
  hole.lineTo(-W / 2 + border, H / 2 - border);
  hole.lineTo(W / 2 - border, H / 2 - border);
  hole.lineTo(W / 2 - border, -H / 2 + border);
  shape.holes.push(hole);
  const mat = new THREE.Mesh(
    new THREE.ShapeGeometry(shape),
    new THREE.MeshStandardMaterial({ color: 0xf7f3ec, roughness: 1 }),
  );
  mat.position.z = -matDepth;
  mat.castShadow = mat.receiveShadow = true;
  g.add(mat);

  // 照片本體
  const tex = photoTexture || placeholderTexture(W, H);
  const aspect = photoTexture ? photoAspect : W / H;
  // 「cover」填滿:裁掉多出來的部分
  const pw = W - border * 2;
  const ph = H - border * 2;
  const frameAspect = pw / ph;
  tex.repeat.set(1, 1);
  tex.offset.set(0, 0);
  if (aspect > frameAspect) {
    tex.repeat.x = frameAspect / aspect;
    tex.offset.x = (1 - tex.repeat.x) / 2;
  } else {
    tex.repeat.y = aspect / frameAspect;
    tex.offset.y = (1 - tex.repeat.y) / 2;
  }
  const photo = new THREE.Mesh(
    new THREE.PlaneGeometry(pw, ph),
    new THREE.MeshStandardMaterial({ map: tex, roughness: 0.8 }),
  );
  photo.position.z = -Dp;
  photo.receiveShadow = true;
  g.add(photo);
  // 照片後的底板
  const backing = new THREE.Mesh(new THREE.PlaneGeometry(W, H), paper);
  backing.position.z = -Dp - 0.01;
  backing.receiveShadow = true;
  g.add(backing);

  g.add(new THREE.HemisphereLight(0xffffff, 0x8a7f70, 1.6));
  shadowLight(g, [-W * 0.6, H * 0.9, 30], [0, 0, -Dp], Math.max(W, H) * 1.2, 1.8);

  return { group: g, background: 0x0d0c0a, update() {} };
}

// ---------- 場景管理 ----------
const builders = { room: buildRoom, mountain: buildMountain, photo: buildPhoto };
let sceneName = 'room';
let world = null;

function disposeGroup(group) {
  group.traverse((o) => {
    o.geometry?.dispose();
    const mats = Array.isArray(o.material) ? o.material : [o.material];
    for (const m of mats) {
      if (!m) continue;
      if (m.map && m.map !== photoTexture) m.map.dispose();
      m.dispose();
    }
    o.shadow?.map?.dispose();
  });
}

function buildWorld() {
  if (world) {
    scene.remove(world.group);
    disposeGroup(world.group);
  }
  const { W, H } = screenSize();
  world = builders[sceneName](W, H, S.depth);
  scene.add(world.group);
  scene.background = new THREE.Color(world.background);
  scene.fog = world.fog || null;
}

function setScene(name) {
  sceneName = name;
  document.querySelectorAll('#scenes button').forEach((b) => {
    b.classList.toggle('on', b.dataset.scene === name);
  });
  buildWorld();
}

// ---------- One Euro 濾波(降低抖動但保持反應快) ----------
class OneEuro {
  constructor(minCutoff = 1, beta = 0.05, dCutoff = 1) {
    this.minCutoff = minCutoff;
    this.beta = beta;
    this.dCutoff = dCutoff;
    this.x = null;
  }
  alpha(cutoff, dt) {
    const tau = 1 / (2 * Math.PI * cutoff);
    return 1 / (1 + tau / dt);
  }
  filter(x, tMs) {
    if (this.x === null) {
      this.x = x;
      this.dx = 0;
      this.t = tMs;
      return x;
    }
    const dt = Math.max(1e-3, (tMs - this.t) / 1000);
    this.t = tMs;
    const dx = (x - this.x) / dt;
    this.dx += this.alpha(this.dCutoff, dt) * (dx - this.dx);
    const cutoff = this.minCutoff + this.beta * Math.abs(this.dx);
    this.x += this.alpha(cutoff, dt) * (x - this.x);
    return this.x;
  }
}
const filters = { x: new OneEuro(), y: new OneEuro(), z: new OneEuro() };
function applySmoothing() {
  const minCutoff = 6 - S.smooth * 5.6; // 0 → 6Hz(很靈敏),1 → 0.4Hz(很平滑)
  for (const f of Object.values(filters)) f.minCutoff = minCutoff;
}

// ---------- 輸入:眼部追蹤 / 陀螺儀 / 滑鼠 ----------
let mode = 'none'; // 'face' | 'gyro' | 'pointer' | 'none'
const raw = { x: 0, y: 0, z: DEFAULT_Z };
const eye = { x: 0, y: 0, z: DEFAULT_Z };
let landmarker = null;
let lastVideoTime = -1;
let lastFaceSeen = 0;
let lastIpdPx = 0;
let lastVideoLong = 0;

async function initCamera() {
  if (!navigator.mediaDevices?.getUserMedia) {
    throw new Error(window.isSecureContext ? '此瀏覽器不支援相機' : '需要以 HTTPS 開啟');
  }
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: 'user', width: { ideal: 640 }, height: { ideal: 480 } },
    audio: false,
  });
  video.srcObject = stream;
  await video.play();
}

async function initLandmarker() {
  const fileset = await FilesetResolver.forVisionTasks(WASM_URL);
  const opts = (delegate) => ({
    baseOptions: { modelAssetPath: MODEL_URL, delegate },
    runningMode: 'VIDEO',
    numFaces: 1,
  });
  try {
    landmarker = await FaceLandmarker.createFromOptions(fileset, opts('GPU'));
  } catch (err) {
    console.warn('GPU delegate 失敗,改用 CPU', err);
    landmarker = await FaceLandmarker.createFromOptions(fileset, opts('CPU'));
  }
}

function orientationAngle() {
  const a = screen.orientation?.angle ?? window.orientation ?? 0;
  return (((a % 360) + 360) % 360) * (Math.PI / 180);
}

function trackFace(now) {
  if (!landmarker || video.readyState < 2) return;
  if (video.currentTime === lastVideoTime) return;
  lastVideoTime = video.currentTime;

  const res = landmarker.detectForVideo(video, now);
  const lm = res.faceLandmarks?.[0];
  if (S.preview) drawPreview(lm);
  if (!lm) return;

  // 虹膜中心(478 點模型的 468 / 473),沒有的話用眼角平均
  const center = (ids) => {
    let x = 0, y = 0;
    for (const i of ids) {
      x += lm[i].x;
      y += lm[i].y;
    }
    return { x: x / ids.length, y: y / ids.length };
  };
  const L = lm.length > 473 ? lm[468] : center([33, 133]);
  const R = lm.length > 473 ? lm[473] : center([362, 263]);

  const vw = video.videoWidth;
  const vh = video.videoHeight;
  const long = Math.max(vw, vh);
  const focal = long / (2 * Math.tan((S.fov * Math.PI) / 360));
  const ipdPx = Math.hypot((R.x - L.x) * vw, (R.y - L.y) * vh);
  if (ipdPx < 1) return;
  lastIpdPx = ipdPx;
  lastVideoLong = long;

  // 相機座標系下的眼睛位置(cm)。前鏡頭影像左右相反,所以 x 取負。
  const z = (S.ipd * focal) / ipdPx;
  const mx = (L.x + R.x) / 2;
  const my = (L.y + R.y) / 2;
  const camX = -((mx - 0.5) * vw * z) / focal;
  const camY = -((my - 0.5) * vh * z) / focal;

  // 前鏡頭不在螢幕中央:直立時在螢幕上緣外。依螢幕旋轉角度換算。
  const { W, H } = screenSize();
  const a = orientationAngle();
  const d = Math.max(W, H) / 2 + S.camGap;
  raw.x = camX - d * Math.sin(a);
  raw.y = camY + d * Math.cos(a);
  raw.z = THREE.MathUtils.clamp(z, 10, 150);
  lastFaceSeen = now;
}

function drawPreview(lm) {
  const vw = video.videoWidth, vh = video.videoHeight;
  if (!vw) return;
  const w = 224;
  const h = Math.round((w * vh) / vw);
  if (preview.width !== w || preview.height !== h) {
    preview.width = w;
    preview.height = h;
  }
  pctx.save();
  pctx.translate(w, 0);
  pctx.scale(-1, 1); // 鏡像,像照鏡子
  pctx.drawImage(video, 0, 0, w, h);
  if (lm) {
    pctx.fillStyle = '#3ae0ff';
    for (const i of [468, 473, 33, 133, 362, 263]) {
      if (!lm[i]) continue;
      pctx.beginPath();
      pctx.arc(lm[i].x * w, lm[i].y * h, 3, 0, Math.PI * 2);
      pctx.fill();
    }
  }
  pctx.restore();
}

// 陀螺儀:把手機轉動換算成「眼睛相對螢幕」的位移
let gyroBase = null;
function onOrientation(ev) {
  if (ev.beta == null || ev.gamma == null) return;
  if (mode === 'none') mode = 'gyro';
  if (mode !== 'gyro') return;
  if (!gyroBase) gyroBase = { beta: ev.beta, gamma: ev.gamma };
  const rad = Math.PI / 180;
  const db = THREE.MathUtils.clamp(ev.beta - gyroBase.beta, -45, 45) * rad;
  const dg = THREE.MathUtils.clamp(ev.gamma - gyroBase.gamma, -45, 45) * rad;
  raw.x = -DEFAULT_Z * Math.sin(dg) * 1.5;
  raw.y = DEFAULT_Z * Math.sin(db) * 1.5;
  raw.z = DEFAULT_Z;
}

function onPointer(ev) {
  if (mode === 'face') return;
  if (mode === 'gyro' && ev.pointerType !== 'mouse') return;
  mode = 'pointer';
  const { W, H } = screenSize();
  raw.x = (ev.clientX / innerWidth - 0.5) * W * 2;
  raw.y = -(ev.clientY / innerHeight - 0.5) * H * 1.2;
  raw.z = DEFAULT_Z;
}
canvas.addEventListener('pointermove', onPointer);
canvas.addEventListener('pointerdown', onPointer);

async function startFace() {
  $('#intro').hidden = true;
  mode = 'face';
  setStatus('啟動相機與模型…', 'warn');
  try {
    await Promise.all([initCamera(), initLandmarker()]);
    setStatus('請看著螢幕', 'warn');
  } catch (err) {
    console.error(err);
    mode = 'none';
    const msg = err?.name === 'NotAllowedError' ? '相機權限被拒' : err?.message || String(err);
    setStatus(`${msg} — 改用觸控/滑鼠`, 'err');
    startManual(false);
  }
}

async function startManual(hideIntro = true) {
  if (hideIntro) $('#intro').hidden = true;
  if (mode !== 'face') mode = 'none';
  try {
    if (typeof DeviceOrientationEvent?.requestPermission === 'function') {
      await DeviceOrientationEvent.requestPermission();
    }
  } catch {
    /* 拒絕就用觸控 */
  }
  gyroBase = null;
  addEventListener('deviceorientation', onOrientation);
  if (hideIntro) setStatus('轉動手機或拖曳畫面', 'ok');
}

// ---------- 主迴圈 ----------
let lastStatusUpdate = 0;
function frame(now) {
  requestAnimationFrame(frame);

  if (mode === 'face') {
    trackFace(now);
    const lost = now - lastFaceSeen > 700;
    if (lost) {
      // 找不到臉:慢慢回到正中央
      raw.x *= 0.96;
      raw.y *= 0.96;
      raw.z += (DEFAULT_Z - raw.z) * 0.04;
    }
    if (landmarker && now - lastStatusUpdate > 250) {
      lastStatusUpdate = now;
      if (lost) setStatus('找不到臉,請正對螢幕', 'warn');
      else setStatus(`追蹤中 · 距離 ${Math.round(eye.z)} cm`, 'ok');
    }
  }

  eye.x = filters.x.filter(raw.x, now);
  eye.y = filters.y.filter(raw.y, now);
  eye.z = filters.z.filter(raw.z, now);

  updateCamera(eye);
  world?.update(now);
  renderer.render(scene, camera);
}

// ---------- 設定面板 ----------
const sliders = {
  screen: {
    get: () => S.cmPerPx * minViewportPx(),
    set: (v) => (S.cmPerPx = v / minViewportPx()),
    fmt: (v) => v.toFixed(1),
    rebuild: true,
  },
  fov: { get: () => S.fov, set: (v) => (S.fov = v), fmt: (v) => v.toFixed(0) },
  gain: { get: () => S.gain, set: (v) => (S.gain = v), fmt: (v) => v.toFixed(2) },
  depth: { get: () => S.depth, set: (v) => (S.depth = v), fmt: (v) => v.toFixed(0), rebuild: true },
  smooth: {
    get: () => S.smooth,
    set: (v) => {
      S.smooth = v;
      applySmoothing();
    },
    fmt: (v) => v.toFixed(2),
  },
};

let rebuildTimer = 0;
function scheduleRebuild() {
  clearTimeout(rebuildTimer);
  rebuildTimer = setTimeout(buildWorld, 120);
}

function syncPanel() {
  for (const [key, s] of Object.entries(sliders)) {
    const v = s.get();
    $(`#s-${key}`).value = v;
    $(`#o-${key}`).textContent = s.fmt(v);
  }
  $('#s-preview').checked = S.preview;
  preview.hidden = !S.preview;
}

for (const [key, s] of Object.entries(sliders)) {
  $(`#s-${key}`).addEventListener('input', (ev) => {
    const v = parseFloat(ev.target.value);
    s.set(v);
    $(`#o-${key}`).textContent = s.fmt(v);
    saveSettings();
    if (s.rebuild) scheduleRebuild();
  });
}

$('#s-preview').addEventListener('change', (ev) => {
  S.preview = ev.target.checked;
  preview.hidden = !S.preview;
  saveSettings();
});

$('#calib').addEventListener('click', () => {
  if (!lastIpdPx || mode !== 'face') {
    setStatus('要先開啟眼部追蹤並偵測到臉才能校正', 'err');
    return;
  }
  // 已知距離 30 cm → 反推相機焦距 → 視角
  const focal = (30 * lastIpdPx) / S.ipd;
  S.fov = THREE.MathUtils.clamp((2 * Math.atan(lastVideoLong / (2 * focal)) * 180) / Math.PI, 30, 120);
  saveSettings();
  syncPanel();
  setStatus(`已校正:相機視角 ${S.fov.toFixed(0)}°`, 'ok');
});

$('#reset').addEventListener('click', () => {
  Object.assign(S, DEFAULTS);
  saveSettings();
  applySmoothing();
  syncPanel();
  buildWorld();
});

$('#gear').addEventListener('click', () => {
  const p = $('#panel');
  p.hidden = !p.hidden;
  if (!p.hidden) syncPanel();
});

$('#fullscreen').addEventListener('click', () => {
  const el = document.documentElement;
  if (document.fullscreenElement) document.exitFullscreen?.();
  else (el.requestFullscreen || el.webkitRequestFullscreen)?.call(el);
});

// 照片上傳
const fileInput = $('#file');
$('#upload').addEventListener('click', () => fileInput.click());
document.querySelectorAll('#scenes button').forEach((b) => {
  b.addEventListener('click', () => {
    if (b.dataset.scene === 'photo' && sceneName === 'photo') fileInput.click();
    else setScene(b.dataset.scene);
  });
});
fileInput.addEventListener('change', () => {
  const file = fileInput.files?.[0];
  if (!file) return;
  const url = URL.createObjectURL(file);
  new THREE.TextureLoader().load(
    url,
    (tex) => {
      URL.revokeObjectURL(url);
      tex.colorSpace = THREE.SRGBColorSpace;
      tex.anisotropy = 4;
      photoTexture?.dispose();
      photoTexture = tex;
      photoAspect = tex.image.width / tex.image.height;
      setScene('photo');
    },
    undefined,
    () => setStatus('無法讀取這張照片', 'err'),
  );
  fileInput.value = '';
});

// ---------- 啟動 ----------
$('#start-face').addEventListener('click', startFace);
$('#start-manual').addEventListener('click', () => startManual(true));

function resize() {
  renderer.setSize(innerWidth, innerHeight, false);
  scheduleRebuild();
}
addEventListener('resize', resize);
screen.orientation?.addEventListener?.('change', () => {
  gyroBase = null;
  resize();
});

renderer.setSize(innerWidth, innerHeight, false);
applySmoothing();
buildWorld();
requestAnimationFrame(frame);

// 方便除錯:在 console 裡可以看到 eye / raw
window.__frame = { eye, raw, S };
