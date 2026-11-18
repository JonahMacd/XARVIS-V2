// XARVIS viewer · view3d.js
// What SPHEREx can see (visible now / overall regions on the star backdrop), the main 3D scene (draw3D) and the
// cursor read-out in the bottom dock.
// Plain browser scripts sharing one global scope, loaded in order by templates/index.html:
// core.js, scene.js, draw.js, tracker.js, view3d.js, analysis.js, app.js.
'use strict';

// ============================================================ what SPHEREx can see (field of regard) ===
// Official constraints used: SPHEREx points at least 91° from the Sun (NASA/JPL Horizons), and it cannot see through
// Earth (geometry: Earth's disc seen from SPHEREx's real position). One exposure covers 3.5° x 11.3° (IPAC Explanatory
// Supplement, Table 2). The Moon, flight-direction and Earth-limb margins are also avoided, but their angles are not
// published, so they are not drawn. Sky directions are the same from SPHEREx and from Earth (the stars are at infinity).
const SUN_LIMIT_DEG = 91, FOV_DEG = [3.5, 11.3], COS_SUN = Math.cos(SUN_LIMIT_DEG*D2R);
const FOV_SR = 4*Math.asin(Math.sin(FOV_DEG[0]/2*D2R)*Math.sin(FOV_DEG[1]/2*D2R)), SR2DEG2 = (180/Math.PI)**2;
function visGeom() {                                   // directions from SPHEREx to the Sun and to Earth's centre, Earth's angular radius
  const sp = state.spherex, d = sp.earth_dist_km;
  return {sun: norm([-sp.pos[0], -sp.pos[1], -sp.pos[2]]), toE: norm([-sp.off[0], -sp.off[1], -sp.off[2]]), cosRho: Math.cos(Math.asin(Math.min(1, EARTH_R_KM/Math.max(d, EARTH_R_KM + 1))))};
}
function visibilityOf(dir, g = visGeom()) {
  const sunOK = dot(dir, g.sun) <= COS_SUN, earth = dot(dir, g.toE) >= g.cosRho;
  return {visible_now: sunOK && !earth, sun_limit_ok: sunOK, behind_earth: earth,
          reason: !sunOK ? 'closer than 91° to the Sun' : earth ? 'behind Earth right now' : 'in view'};
}
// fraction of the whole sky in each zone: 40,000 evenly spread directions (Fibonacci sphere), refreshed twice a second
const FIB = (() => { const n = 40000, a = new Float32Array(n*3), g = Math.PI*(3 - Math.sqrt(5));
  for (let i = 0; i < n; i++) { const z = 1 - 2*(i + 0.5)/n, r = Math.sqrt(1 - z*z), t = g*i; a[i*3] = r*Math.cos(t); a[i*3+1] = r*Math.sin(t); a[i*3+2] = z; } return a; })();
let skyFrac = null, skyFracT = 0;
function skyFractions() {
  const t = performance.now(); if (skyFrac && t - skyFracT < 500) return skyFrac; skyFracT = t;
  const g = visGeom(), n = FIB.length/3; let now = 0, sun = 0;
  for (let i = 0; i < n; i++) { const x = FIB[i*3], y = FIB[i*3+1], z = FIB[i*3+2];
    if (x*g.sun[0] + y*g.sun[1] + z*g.sun[2] > COS_SUN) continue; sun++;
    if (x*g.toE[0] + y*g.toE[1] + z*g.toE[2] < g.cosRho) now++; }
  return skyFrac = {now: now/n, sun: sun/n, rhoDeg: Math.acos(g.cosRho)/D2R};
}
// The zones are painted on the star backdrop with a coarse per-cell test (exact for any view, including the backdrop's
// singular point), then smoothed by the browser; the boundary circles are drawn crisply on top.
const visCv = document.createElement('canvas'), visCtx = visCv.getContext('2d');
// Fast time skips: SPHEREx circles Earth every 98 minutes, so Earth's shadow on the sky (the slice SPHEREx can't see
// right now) jumps to a new place every frame once time runs at hours per second - a flashing that is real but useless
// (and unpleasant). From VIS_FAST_FROM of an orbit per frame the "Visible now" area eases into the WHOLE area SPHEREx can
// see during its orbit (everything at least 91 deg from the Sun), which only turns slowly with the Sun; it is all of it
// from VIS_FAST_FULL. At normal speeds it is the exact slice, as before.
const VIS_FAST_FROM = 1/24, VIS_FAST_FULL = 1/8, VIS_FAST_EASE_S = 0.4;    // orbits per frame; seconds to ease in / out
const visFast = {w: 0, t: 0, frameS: 1/60};
function visFastWeight() {                                // 0: the exact slice now ... 1: the whole area seen over an orbit
  const now = performance.now(), dt = visFast.t ? Math.min((now - visFast.t)/1000, 0.5) : 0; visFast.t = now;
  if (dt > 0) visFast.frameS += (dt - visFast.frameS)*0.1;                   // (the real time per frame, smoothed)
  const orbitsPerFrame = Math.abs(simRate())*visFast.frameS*1440/((snap && snap.period_min) || 98.4);
  const goal = smooth(VIS_FAST_FROM, VIS_FAST_FULL, orbitsPerFrame);
  visFast.w = dt ? visFast.w + (goal - visFast.w)*(1 - Math.exp(-dt/VIS_FAST_EASE_S)) : goal;
  return visFast.w;
}
function drawVisibility() {
  if (!opt.cVisNow && !opt.cVisAll) return;
  const g = visGeom(), SK = skyScale(), cell = 6, gw = Math.ceil(W/cell), gh = Math.ceil(H/cell), fw = visFastWeight();
  if (visCv.width !== gw || visCv.height !== gh) { visCv.width = gw; visCv.height = gh; }
  const img = visCtx.createImageData(gw, gh), d = img.data, R0 = R[0], R1 = R[1], R2 = R[2];
  const rho = Math.acos(g.cosRho);
  for (let j = 0; j < gh; j++) for (let i = 0; i < gw; i++) {
    const u = ((i + 0.5)*cell - W/2)/SK, v = -((j + 0.5)*cell - H/2)/SK, r2 = u*u + v*v, k = 2/(1 + r2), a = u*k, b = v*k, c = -(1 - r2)/(1 + r2);
    const x = R0[0]*a + R1[0]*b + R2[0]*c, y = R0[1]*a + R1[1]*b + R2[1]*c, z = R0[2]*a + R1[2]*b + R2[2]*c;   // view -> ecliptic direction
    // soft edges: how far (in angle) this cell is inside each limit, compared with the cell's own angular size
    const wEdge = cell*k/SK;
    const mSun = Math.acos(clamp(x*g.sun[0] + y*g.sun[1] + z*g.sun[2], -1, 1)) - SUN_LIMIT_DEG*D2R;   // > 0: farther than 91 deg from the Sun
    const cSun = clamp(mSun/wEdge + 0.5, 0, 1); if (cSun <= 0) continue;
    const mE = fw < 1 ? Math.acos(clamp(x*g.toE[0] + y*g.toE[1] + z*g.toE[2], -1, 1)) - rho : 1;   // > 0: not behind Earth
    const cNow = fw + (1 - fw)*clamp(mE/wEdge + 0.5, 0, 1);                                        // (fast: Earth's shadow eased out)
    const aAll = opt.cVisAll ? 16 : 0, al = opt.cVisNow ? cSun*(34*cNow + aAll*(1 - cNow)) : cSun*aAll;   // 34 = visible now, 16 = Sun limit only
    if (al < 0.5) continue;
    const o = (j*gw + i)*4; d[o] = 0; d[o+1] = 181; d[o+2] = 226; d[o+3] = Math.round(al);
  }
  visCtx.putImageData(img, 0, 0);
  ctx.save(); ctx.imageSmoothingEnabled = true; ctx.drawImage(visCv, 0, 0, gw*cell, gh*cell); ctx.restore();
  ctx.lineWidth = 1.2;
  if (opt.cVisAll) { ctx.strokeStyle = '#ff9d6bcc'; ctx.setLineDash([7, 6]); ctx.beginPath(); drawGreatPath(smallCircle(g.sun, SUN_LIMIT_DEG, 180), SK); ctx.stroke(); ctx.setLineDash([]); }
  if (opt.cVisNow) drawVisLabel(fw > 0.5 ? {...g, cosRho: 2, noEarth: true} : g, SK, fw > 0.5);   // (cosRho 2: nothing is behind Earth)
}
// The "SPHEREx can see now" label belongs to the Visible now layer alone: it is on exactly when that layer is, whatever
// else is switched on. It is pinned to the middle of the region on the SKY: the direction opposite the Sun, the centre of
// the half of the sky that is at least 91 degrees from it. That point moves only as slowly as the Sun does. (It used to follow
// the centre of whatever part of the region was on screen, which swings wildly as SPHEREx circles Earth - 98 minutes per
// orbit, 15 orbits per second at 1 day/s - and it was placed among the constellation names' rectangles.)
// Where Earth is covering that point right now, the label slides along the sky just clear of Earth's edge, so it always
// stands on sky that can really be seen, and it moves smoothly as Earth passes (never a jump).
const LABEL_EDGE_MARGIN = 6*D2R;
function visCentre(g) {
  const a = [-g.sun[0], -g.sun[1], -g.sun[2]];
  if (g.noEarth) return a;                                                                 // (fast time: the whole area, Earth ignored)
  const e = g.toE, rho = Math.acos(g.cosRho), limit = rho + LABEL_EDGE_MARGIN, cosA = clamp(dot(a, e), -1, 1);
  if (Math.acos(cosA) >= limit) return a;                                                  // not under Earth: the middle of the region
  let t = [a[0] - cosA*e[0], a[1] - cosA*e[1], a[2] - cosA*e[2]], tn = Math.hypot(...t);
  if (tn < 1e-4) { t = cross(e, g.sun); tn = Math.hypot(...t); }                         // Earth exactly behind it: slide sideways
  t = [t[0]/tn, t[1]/tn, t[2]/tn];
  const c = [Math.cos(limit)*e[0] + Math.sin(limit)*t[0], Math.cos(limit)*e[1] + Math.sin(limit)*t[1], Math.cos(limit)*e[2] + Math.sin(limit)*t[2]];
  return dot(c, g.sun) <= COS_SUN ? c : null;                                                // (always 91+ degrees from the Sun for SPHEREx's real orbit)
}
const visLabel = {rect: null, slot: 0, ui: [], dir: null, t: 0};   // last frame's rectangle (the names keep clear of it), the spot used, the panels, the smoothed sky position
const LABEL_MARGIN = 28;          // px kept clear between the label and the edge of the screen
const LABEL_SMOOTH_S = 0.25;      // seconds: much faster than the real orbit (98 min), so invisible at normal speeds, but it averages out the Earth sweeping past when time runs at days per second
const LABEL_TEXT = 'SPHEREx can see now', LABEL_TEXT_FAST = 'SPHEREx can see (during each orbit)', LABEL_FONT = '12px system-ui', LABEL_DY = [4, -20, 28, -44, 52];     // spots, from the centre down and up
function drawVisLabel(g, SK, fast = false) {
  visLabel.rect = null;
  const text = fast ? LABEL_TEXT_FAST : LABEL_TEXT;
  const c = visCentre(g); if (!c) { visLabel.dir = null; return; }
  const now = performance.now(), dt = (now - visLabel.t)/1000; visLabel.t = now;
  if (!visLabel.dir || dt > 1) visLabel.dir = c;               // first frame, or the layer was off: no gliding
  else { const k = 1 - Math.exp(-dt/LABEL_SMOOTH_S), m = [0, 1, 2].map(i => visLabel.dir[i] + k*(c[i] - visLabel.dir[i])), n = Math.hypot(...m) || 1; visLabel.dir = [m[0]/n, m[1]/n, m[2]/n]; }
  const a = visLabel.dir, w = measure(LABEL_FONT, text), h = 13;
  const place = d => {                                       // where the label would stand on sky direction d: the first free spot, last frame's first
    const q = skyXY(toView(d), SK); if (!q) return null;     // (behind the view: no spot)
    for (let t = 0; t < LABEL_DY.length; t++) {
      const s = (visLabel.slot + t) % LABEL_DY.length, lx = q[0] - w/2, ly = q[1] + LABEL_DY[s], r = [lx, ly - h + 2, lx + w, ly + 3];
      if (r[0] >= LABEL_MARGIN && r[2] <= W - LABEL_MARGIN && r[1] >= LABEL_MARGIN && r[3] <= H - LABEL_MARGIN && !hitsRect(r, visLabel.ui)) return {lx, ly, r, s};     // on screen (not hugging the edge), not under a panel
    }
    return null;
  };
  let spot = place(a);
  if (!spot) {            // its natural place is off screen or under a panel: slide along the sky toward what is at the middle of the screen, to
                          // the first spot that is on screen and on sky that SPHEREx can really see now (moves smoothly with the view and the time)
    const c0 = skyDirAt(W/2, H/2), cosW = c0 ? clamp(dot(a, c0), -1, 1) : 1, om = Math.acos(cosW), so = Math.sin(om);
    if (c0 && so > 1e-3) {
      const at = t => { const k1 = Math.sin((1 - t)*om)/so, k2 = Math.sin(t*om)/so; return [k1*a[0] + k2*c0[0], k1*a[1] + k2*c0[1], k1*a[2] + k2*c0[2]]; };
      const good = t => { const d = at(t); return visibilityOf(d, g).visible_now ? place(d) : null; };
      const N = 50; let lo = 0, hi = -1;
      for (let i = 1; i <= N; i++) if (good(i/N)) { hi = i/N; lo = (i - 1)/N; break; }
      if (hi >= 0) { for (let k = 0; k < 8; k++) { const mid = (lo + hi)/2; if (good(mid)) hi = mid; else lo = mid; } spot = good(hi); }
    }
  }
  if (!spot) return;
  visLabel.slot = spot.s; visLabel.rect = spot.r; labelRects.push(spot.r);
  ctx.font = LABEL_FONT; ctx.fillStyle = '#5fd6f0'; ctx.fillText(text, spot.lx, spot.ly);
}

// ============================================================================ 3D drawing ===
function draw3D() {
  const s = scaleOf(zt), c = centerOf(s), fade = constellationFade(s);
  visLabel.ui = uiReserved(); baseRects = visLabel.ui.slice();
  if (!opt.cVisNow) visLabel.rect = null;
  else if (visLabel.rect) baseRects.push(visLabel.rect);                   // the constellation names keep clear of the visible-region label
  labelRects = baseRects.slice();                                          // start every frame clean, with or without the names layer (stale rectangles used to pile up)
  drawStarfield(0.75, fade);
  drawVisibility();
  drawSkyTarget();
  drawAnalysed();
  drawTrackPaths();
  drawAnaSpot();
  drawClickMark();
  const E = state.E, sp = state.spherex, wAU = W/s, orbitA = smooth(0.004, 0.03, wAU);
  drawRegions(c, s);
  if (opt.cOrbits && orbitA > 0.01) {
    ctx.lineWidth = 1; ctx.globalAlpha = orbitA;
    if (state.orbits) strokeOrbits(state.orbits, c, s, '55');
    ctx.globalAlpha = 1;
  }
  const [sx, sy] = project([0,0,0], c, s), sunR = Math.max(5, 695700/AU_KM*s), vis = sx + sunR*3.2 > 0 && sx - sunR*3.2 < W && sy + sunR*3.2 > 0 && sy - sunR*3.2 < H;
  if (vis) { if (sunR < 1500) drawGlow(sx, sy, Math.max(sunR*3.2, 18), '#ffb300'); ctx.fillStyle = '#ffd35a'; ctx.beginPath(); ctx.arc(sx, sy, sunR, 0, 7); ctx.fill(); }
  addHit(sx, sy, sunR, 0.2, () => tipHTML('Sun', 'star', [['Position','origin of the frame'], ['From Earth', distStr(Math.hypot(...E))], ['RA / Dec (J2000)', fmtRADec(norm([-E[0], -E[1], -E[2]]))]]));

  // SPHEREx orbit: far half before Earth, near half after, so Earth really occludes it
  const ringPx = sp.earth_dist_km/AU_KM*s, moonPx = 2.6e-3*s, R2 = R[2], ring = sp.ring;
  const ringHalf = front => { if (ringPx < 2.5) return; ctx.strokeStyle = front ? '#ff7a59' : '#a64a35'; ctx.lineWidth = 1.5; ctx.beginPath();
    const R0 = R[0], R1 = R[1], pen = makePen(); let px = 0, py = 0, prev = false;
    for (let i = 0; i < ring.length; i++) { const o = ring[i];
      if (((R2[0]*o[0] + R2[1]*o[1] + R2[2]*o[2]) >= 0) !== front) { prev = false; pen.gap(); continue; }
      const dx = E[0]+o[0]-c[0], dy = E[1]+o[1]-c[1], dz = E[2]+o[2]-c[2], x = W/2 + (R0[0]*dx + R0[1]*dy + R0[2]*dz)*s, y = H/2 - (R1[0]*dx + R1[1]*dy + R1[2]*dz)*s;
      if (prev) pen.line(px, py, x, y); px = x; py = y; prev = true; }
    ctx.stroke(); };
  ringHalf(false);
  if (state.moonOff && opt.cOrbits && moonPx > 4) { ctx.strokeStyle = '#8a8a8a88'; ctx.lineWidth = 1; polyline(state.moonOff, c, s, 1, E); }

  let earthXY = null;
  for (const p of state.bodies) {
    const [x, y] = project(p.pos, c, s), r = Math.max(3, p.radius_km/AU_KM*s);
    if (p.name === 'Earth') earthXY = [x, y, r];
    if (x < -r-50 || x > W+r+50 || y < -r-50 || y > H+r+50) continue;
    if (p.name === 'Earth' && r > 8) { const sv = toView(state.sun_dir), g = ctx.createRadialGradient(x + sv[0]*r*.5, y - sv[1]*r*.5, r*.05, x, y, r*1.02);
      g.addColorStop(0,'#7cc0ff'); g.addColorStop(.55,'#2b6fc4'); g.addColorStop(1,'#0a1e3d'); ctx.fillStyle = g; } else ctx.fillStyle = COLORS[p.name];
    ctx.beginPath();
    ctx.arc(x, y, Math.min(r, 1e5), 0, 7);
    ctx.fill();
    const nearSun = Math.hypot(x - sx, y - sy) < 22;
    if (p.name === 'Moon' && r < 25) { /* too close to Earth to label */ }
    else if (r < 60) { if (!nearSun || p.name === 'Earth') { ctx.fillStyle = '#9fb0d8'; ctx.font = '12px system-ui'; ctx.fillText(p.name, x + r + 5, y - r - 3); } }
    else { ctx.fillStyle = '#fff'; ctx.font = '13px system-ui'; ctx.fillText(p.name, x - 18, y + 4); }
    addHit(x, y, r + 3, p.name === 'Moon' ? 0.4 : 0.1, () => { const rows = bodyRows(p.pos, p.name === 'Earth' ? null : E, 'Earth'); if (p.name !== 'Earth') rows.push(['Radius', p.radius_km.toLocaleString() + ' km']);
      return tipHTML(p.name, p.name === 'Moon' ? 'moon' : 'planet', rows, 'Position: ' + (state.from === 'track' ? 'NASA/JPL DE440s ephemeris.' : "JPL's mean orbital elements (approximate).")); });
  }
  ringHalf(true);
  drawTracked(c, s);
  if (earthXY && earthXY[2] > 20) { const L = 7.2e-5, tip = project([E[0]+state.sun_dir[0]*L, E[1]+state.sun_dir[1]*L, E[2]+state.sun_dir[2]*L], c, s);
    ctx.strokeStyle = '#ffd35a'; ctx.setLineDash([6,5]); ctx.lineWidth = 1.2; ctx.beginPath(); ctx.moveTo(earthXY[0], earthXY[1]); ctx.lineTo(tip[0], tip[1]); ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle = '#ffd35a'; ctx.font = '12px system-ui'; ctx.fillText('to Sun', tip[0] + 6, tip[1]); }

  if (sp.showMarker) {
    const [px, py] = project(sp.pos, c, s), behind = earthXY && toView(sp.off)[2] < 0 && Math.hypot(px - earthXY[0], py - earthXY[1]) < earthXY[2];
    ctx.globalAlpha = behind ? 0.35 : 1; const pulse = 9 + 3*Math.sin(performance.now()/350);
    ctx.strokeStyle = '#ff4d4d'; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(px, py, pulse, 0, 7); ctx.stroke();
    ctx.fillStyle = '#ff4d4d'; ctx.beginPath(); ctx.arc(px, py, 3.5, 0, 7); ctx.fill();
    if (wAU < 150) { ctx.fillStyle = '#ffd0d0'; ctx.font = 'bold 13px system-ui'; ctx.fillText('SPHEREx', px + 16, py - 10); }
    ctx.globalAlpha = 1;
    addHit(px, py, 14, 0.5, () => tipHTML('SPHEREx', 'telescope', [['Heliocentric r', sp.helio_dist_au.toFixed(5) + ' AU'], ['Ecl. lon / lat', sp.ecl_lon_deg.toFixed(3) + '° / ' + sp.ecl_lat_deg.toFixed(3) + '°'],
      ['X, Y, Z (AU)', sp.pos.map(v => v.toFixed(5)).join(', ')], ['Altitude', Math.round(sp.altitude_km).toLocaleString() + ' km'], ['Orbit period', sp.period_min.toFixed(1) + ' min'],
      ['RA / Dec from Earth centre', fmtRADec(norm(sp.off))]], sp.live ? 'Live position from NASA/JPL Horizons.' : 'Approximate: position along the orbit is not known without Horizons data.'));
  }
}

// ============================================================================ cursor readout ===
// "Where am I looking?"  The sky direction under the cursor, in three coordinate systems, plus (in 3D) the point on
// the ecliptic plane the cursor is over.
function skyDirAt(x, y) {                                                    // inverse of the stereographic backdrop projection
  const SK = skyScale(), u = (x - W/2)/SK, v = -(y - H/2)/SK, r2 = u*u + v*v, k = 2/(1 + r2);
  return fromView([u*k, v*k, -(1 - r2)/(1 + r2)]);
}
function planePointAt(x, y) {                                                // cursor ray (3D view) meets the ecliptic plane z = 0
  const s = scaleOf(zt), c = centerOf(s), vx = (x - W/2)/s, vy = -(y - H/2)/s, r22 = R[2][2];
  if (Math.abs(r22) < 0.05) return null;
  const vz = -(c[2] + R[0][2]*vx + R[1][2]*vy)/r22;
  return [c[0] + R[0][0]*vx + R[1][0]*vy + R[2][0]*vz, c[1] + R[0][1]*vx + R[1][1]*vy + R[2][1]*vz];
}
function updateCoords() {                                                  // always the same three rows, so the box never changes size
  const plane = cursor && planePointAt(cursor[0], cursor[1]);
  let sk = '—';
  if (cursor) { const [ra, dec] = ecl2eq(skyDirAt(cursor[0], cursor[1])); sk = fmtRA(ra, false) + ' · ' + fmtDec(dec, false); }
  const rows = [['Sky', sk], ['Plane x, y (AU)', plane ? plane[0].toFixed(2) + ', ' + plane[1].toFixed(2) + ' · r ' + Math.hypot(plane[0], plane[1]).toFixed(2) : '—']];
  let hint = 'double-click to analyse';
  if (cursor && state) { const v = visibilityOf(skyDirAt(cursor[0], cursor[1])); hint = v.visible_now ? '<span class="ok">● SPHEREx can see this</span>' : `<span class="no">● SPHEREx can't see this</span>`; }
  if (analysisBusy()) hint = '<span class="no">● analysis running: one at a time</span>';
  setHTML('cHint', hint);
  setHTML('coordsBody', rows.map(r => `<div class="v"><span>${r[0]}</span><span>${r[1]}</span></div>`).join(''));
}
