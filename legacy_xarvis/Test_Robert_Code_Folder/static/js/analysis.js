// XARVIS viewer · analysis.js
// Go to sky coordinates (test targets from ESA Gaia DR3, typed RA/Dec) and the sky point analysis popup: visibility,
// target motion, known asteroids in the patch (JPL SB Ident) and the patch map. requestAnalysis() is the backend stub.
// Plain browser scripts sharing one global scope, loaded in order by templates/index.html:
// core.js, scene.js, draw.js, tracker.js, view3d.js, analysis.js, app.js.
'use strict';

// ======================================================================= go to sky coordinates ===
// Type RA/Dec (or pick a test target) to look at a patch of sky: the view turns to it, a marker shows the patch, and the
// analysis popup opens for it. Test targets come from ESA Gaia DR3 (static/targets.json), and are moved with their
// proper motion to the date shown; WISE 0855-0714 is too faint for Gaia and uses the challenge's coordinates.
let targets = [], skyTarget = null, viewAnim = null;
const SPHEREX_PIX = 6.15;
const patchPix = () => clamp(Math.round(+$('gPix').value || 50), 10, 2000), patchHW = () => patchPix()*SPHEREX_PIX/3600/2;   // half-width in degrees
function showPatchInfo() { const px = patchPix(), am = px*SPHEREX_PIX/60; setText('gPixInfo', `${px} × ${px} px at 6.15″ per pixel = ${am < 60 ? am.toFixed(1) + '′' : (am/60).toFixed(2) + '°'} square. Larger patches show faster-moving objects but take longer to load.`); }                                         // arcsec per pixel (IPAC Explanatory Supplement, Table 2)
function parseRA(t) {
  t = t.trim().toLowerCase().replace(/[−–]/g, '-'); if (!t) return null;
  const n = t.match(/[\d.]+/g); if (!n) return null;
  if (/[h:]/.test(t) || /\s/.test(t) || n.length > 1) { const [h, m = 0, s = 0] = n.map(Number); const v = (h + m/60 + s/3600)*15; return h < 24 && m < 60 && s < 60 ? v : null; }
  const v = Number(n[0]); return v >= 0 && v < 360 ? v : null;            // a single number: degrees
}
function parseDec(t) {
  t = t.trim().replace(/[−–]/g, '-'); if (!t) return null;
  const neg = /^-/.test(t), n = t.match(/[\d.]+/g); if (!n) return null;
  const [d, m = 0, s = 0] = n.map(Number), v = (neg ? -1 : 1)*(d + m/60 + s/3600);
  return Math.abs(v) <= 90 && m < 60 && s < 60 ? v : null;
}
function targetEqAt(T, jd) {                                      // [ra, dec] in degrees at date jd (linear proper motion, like the star field)
  if (T.pmra == null || T.epoch == null) return [T.ra, T.dec];
  const a = T.ra*D2R, d = T.dec*D2R, u = [Math.cos(d)*Math.cos(a), Math.cos(d)*Math.sin(a), Math.sin(d)], e = [-Math.sin(a), Math.cos(a), 0], nn = [-Math.sin(d)*Math.cos(a), -Math.sin(d)*Math.sin(a), Math.cos(d)];
  const yrs = (jd - 2451545.0)/365.25 + 2000 - T.epoch, k = D2R/3.6e6*yrs, v = norm([0, 1, 2].map(i => u[i] + k*(T.pmra*e[i] + T.pmdec*nn[i])));
  return [(Math.atan2(v[1], v[0])/D2R + 360) % 360, Math.asin(v[2])/D2R];
}
const targetDir = (T, jd = sim.jd) => { const [ra, dec] = targetEqAt(T, jd); return eq2ecl(ra*D2R, dec*D2R); };
async function loadTargets() {
  try { targets = (await api('/static/targets.json')).targets;
    $('gTarget').innerHTML = '<option value="">Choose a target…</option>' + targets.map((t, i) => `<option value="${i}">${esc(t.name)}</option>`).join(''); }
  catch (e) { console.warn('targets', e); }
}
function lookToward(f) {                                          // turn the view smoothly so direction f is in the middle of the screen
  spin = false; $('spin').classList.remove('on'); $('spinPanel').hidden = true;
  const f0 = norm([-R[2][0], -R[2][1], -R[2][2]]), ang = Math.acos(clamp(dot(f0, f), -1, 1)), t0 = performance.now();
  viewAnim = t => { const k = Math.min(1, (t - t0)/1300), e = k < .5 ? 2*k*k : 1 - Math.pow(-2*k + 2, 2)/2;
    let v = f; if (ang > 1e-6 && ang < Math.PI - 1e-3) { const sn = Math.sin(ang); v = [0, 1, 2].map(i => (Math.sin((1 - e)*ang)*f0[i] + Math.sin(e*ang)*f[i])/sn); }
    R = lookAt(norm(v), [0, 0, 1]); return k >= 1; };
}
function showSkyTarget(T, hw) {
  skyTarget = {T, hw}; lookToward(targetDir(T));
  $('gotoChip').innerHTML = `<span class="chip"><b>${esc(T.name)}</b><button class="ghost cf" id="gReopen" title="Open the analysis again">Open</button><button class="ghost cx" id="gClear" title="Remove the marker">✕</button></span>`;
  $('gReopen').onclick = () => openAnalysisDir(targetDir(T), T, hw); $('gClear').onclick = () => { skyTarget = null; $('gotoChip').innerHTML = ''; };
  setTimeout(() => openAnalysisDir(targetDir(T), T, hw), 1350);
}
function goToTyped() {
  const i = $('gTarget').value, hw = patchHW(), ra = parseRA($('gRA').value), dec = parseDec($('gDec').value);
  if (ra == null || dec == null) { $('gMsg').className = 'err'; $('gMsg').textContent = ra == null ? 'Right ascension not understood: try 08h55m10.8s, 08 55 10.8 or 133.795 (degrees).' : 'Declination not understood: try -7d14m42.5s, -07 14 42.5 or -7.245.'; return; }
  $('gMsg').className = 'dim';
  const T0 = i !== '' ? targets[+i] : null;
  const same = T0 && Math.abs(ra - T0.ra) < 1e-4 && Math.abs(dec - T0.dec) < 1e-4;      // untouched preset: keep its Gaia motion
  showSkyTarget(same ? T0 : {name: `RA ${fmtRA(ra, false)} · Dec ${fmtDec(dec, false)}`, ra, dec, typed: true, source: 'typed coordinates (J2000)'}, hw);
}
function drawSkyTarget() {
  if (!skyTarget) return;
  const SK = skyScale(), T = skyTarget.T, d = targetDir(T), q = skyXY(toView(d), SK); if (!q) return;
  const half = Math.max(9, skyTarget.hw*D2R*SK), pulse = 3*Math.sin(performance.now()/300);
  ctx.strokeStyle = '#5fd6f0'; ctx.lineWidth = 1.6; ctx.strokeRect(q[0] - half, q[1] - half, 2*half, 2*half);
  ctx.beginPath(); for (const [dx, dy] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) { ctx.moveTo(q[0] + dx*(half + 4 + pulse), q[1] + dy*(half + 4 + pulse)); ctx.lineTo(q[0] + dx*(half + 14 + pulse), q[1] + dy*(half + 14 + pulse)); } ctx.stroke();
  ctx.font = 'bold 12px system-ui'; const w = measure('bold 12px system-ui', T.name); ctx.fillStyle = '#02040acc'; ctx.fillRect(q[0] + half + 16, q[1] - half - 20, w + 8, 17);
  ctx.fillStyle = '#9fe6f7'; ctx.fillText(T.name, q[0] + half + 20, q[1] - half - 7);
  addHit(q[0], q[1], half + 6, 0.6, () => tipHTML(esc(T.name), 'sky patch', [['RA / Dec (J2000)', fmtRADec(d)], ['Patch', (2*skyTarget.hw*60).toFixed(0) + '′ × ' + (2*skyTarget.hw*60).toFixed(0) + '′']], 'Double-click or use "Open" to analyse it.'));
}

// ========================================================================= analysis popup ===
// Double-click: take the sky direction under the cursor (as seen from Earth: the stars are effectively at infinity, so
// the geocentric direction equals the direction drawn), package it and show it. PLACEHOLDER - nothing is sent yet.
function analysisPayload(dir, T = null, hw = 0.25) {
  const eq = ecl2eqVec(dir), [ra, dec] = ecl2eq(dir), [lam, beta] = eclLonLat(dir), [l, b] = ecl2gal(dir);
  const sp = state.spherex, elong = Math.acos(clamp(dot(dir, state.sun_dir), -1, 1))/D2R;
  const r6 = v => +v.toFixed(6);
  return {
    request: 'sky_point_analysis', version: 1,
    time: {utc: fmtDate(sim.jd) + ' UTC', jd_utc: +sim.jd.toFixed(6), simulated: !sim.live},
    target: {frame: 'geocentric, J2000 equatorial (ICRS-aligned)', ra_deg: r6(ra), dec_deg: r6(dec), unit_vector: eq.map(r6),
             spherex_visibility: visibilityOf(dir), ecliptic_lon_deg: r6(lam), ecliptic_lat_deg: r6(beta), galactic_l_deg: r6(l), galactic_b_deg: r6(b), sun_elongation_deg: +elong.toFixed(3)},
    patch: {half_width_deg: hw, size: (2*hw*60).toFixed(0) + ' x ' + (2*hw*60).toFixed(0) + ' arcmin'},
    named_target: T ? {name: T.name, source: T.source || null, gaia_dr3: T.gaia_dr3 || null, ra_deg_at_epoch: T.ra, dec_deg_at_epoch: T.dec, epoch: T.epoch ?? null,
                       pmra_mas_yr: T.pmra ?? null, pmdec_mas_yr: T.pmdec ?? null, parallax_mas: T.parallax ?? null, gaia_g_mag: T.g ?? null} : null,
    spherex: {geocentric_km_j2000_equatorial: ecl2eqVec(sp.off).map(v => +(v*AU_KM).toFixed(1)), altitude_km: +sp.altitude_km.toFixed(1), orbit_source: sp.source},
  };
}
// TODO(backend): send the payload to the analysis service, e.g.
//   const r = await fetch('/api/analysis', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload)});
// and draw the returned data inside #anaViz. Until then this does nothing and the popup shows a placeholder.
async function requestAnalysis(payload) { return null; }
let anaPayloadNow = null;
// When will SPHEREx be able to see direction d? Behind Earth: step SPHEREx around its orbit minute by minute.
// Too close to the Sun: step the Sun along Earth's orbit day by day (JPL mean elements) until it is 91 deg away.
function spxOffAt(jd) { const a = snap.a_km/AU_KM, phi = 2*Math.PI*(jd - snap.jd)*1440/snap.period_min; return [0, 1, 2].map(k => a*(Math.cos(phi)*snap.u[k] + Math.sin(phi)*snap.w[k])); }
function nextVisible(d) {
  const sunOK = jd => { const E = kepler('Earth', (jd - J2000)/36525); return dot(d, norm([-E[0], -E[1], -E[2]])) <= COS_SUN; };
  if (sunOK(sim.jd)) {
    for (let m = 1; m <= 200; m++) { const jd = sim.jd + m/1440, o = spxOffAt(jd), r = Math.hypot(...o), toE = [-o[0]/r, -o[1]/r, -o[2]/r];
      if (dot(d, toE) < Math.cos(Math.asin(Math.min(1, EARTH_R_KM/(r*AU_KM))))) return `It comes back into view in about ${m} minute${m === 1 ? '' : 's'}, as SPHEREx moves around Earth (one orbit takes ${snap.period_min.toFixed(0)} minutes).`; }
    return 'Earth blocks it for the whole of this orbit.';
  }
  for (let dd = 1; dd <= 366; dd++) if (sunOK(sim.jd + dd)) return `As Earth moves around the Sun, it comes into SPHEREx's view around ${fmtDate(sim.jd + dd, false).slice(0, 10)} (in ${dd} day${dd === 1 ? '' : 's'}).`;
  return 'It stays too close to the Sun for the next year.';
}
let clickMark = null;
function drawClickMark() {                                     // brief feedback where you double-clicked: green ring = SPHEREx can see it, red = it can't
  if (!clickMark) return; const age = (performance.now() - clickMark.t)/1000; if (age > 3) { clickMark = null; return; }
  const q = skyXY(toView(clickMark.dir), skyScale()); if (!q) return;
  ctx.globalAlpha = Math.min(1, 3 - age); ctx.lineWidth = 2; ctx.strokeStyle = clickMark.ok ? '#5fe0a0' : '#ff6b5a';
  ctx.beginPath(); ctx.arc(q[0], q[1], 12 + age*4, 0, 7); ctx.stroke();
  if (!clickMark.ok) { ctx.beginPath(); ctx.moveTo(q[0] - 7, q[1] - 7); ctx.lineTo(q[0] + 7, q[1] + 7); ctx.moveTo(q[0] + 7, q[1] - 7); ctx.lineTo(q[0] - 7, q[1] + 7); ctx.stroke(); }
  ctx.globalAlpha = 1;
}
function openAnalysis(x, y) { if (state && sky) openAnalysisDir(skyDirAt(x, y), null, patchHW()); }
let anaReq = 0;
function openAnalysisDir(dir, T, hw) {
  if (!state || !sky) return;
  const P = anaPayloadNow = analysisPayload(dir, T, hw), t = P.target;
  setText('anaTitle', T ? T.name : `RA ${fmtRA(t.ra_deg, false)} · Dec ${fmtDec(t.dec_deg, false)}`);
  const sg = v => (v < 0 ? '−' : '') + Math.abs(v).toFixed(3) + '°';
  const rows = [['Time (UTC)', P.time.utc + (P.time.simulated ? ' · simulated' : '')], ['Right ascension / declination', fmtRA(t.ra_deg) + ' / ' + fmtDec(t.dec_deg)],
                ['', t.ra_deg.toFixed(5) + '° / ' + sg(t.dec_deg)],
                ['Ecliptic longitude / latitude', t.ecliptic_lon_deg.toFixed(3) + '° / ' + sg(t.ecliptic_lat_deg)], ['Galactic longitude / latitude', t.galactic_l_deg.toFixed(3) + '° / ' + sg(t.galactic_b_deg)],
                ['Angle from the Sun', t.sun_elongation_deg.toFixed(1) + '°'], ['Can SPHEREx see it now?', t.spherex_visibility.visible_now ? 'yes' : 'no · ' + t.spherex_visibility.reason], ['Patch', P.patch.size.replace(' x ', '′ × ').replace(' arcmin', '′')]];
  $('anaRows').innerHTML = rows.map(r => `<div class="v"><span>${r[0]}</span><span>${r[1]}</span></div>`).join('');
  // the named target: who measured it and how far it moves during the SPHEREx survey
  let trk = null;
  if (T && T.pmra != null) {
    const pm = Math.hypot(T.pmra, T.pmdec)/1000, yrs = (SCIENCE_END - SCIENCE_START)/365.25, mv = pm*yrs;
    trk = SURVEY_EPOCHS.map(jd => [jd, targetEqAt(T, jd)]); trk.now = targetEqAt(T, sim.jd);
    $('anaTarget').innerHTML = [['Catalogue', `ESA Gaia DR3 ${esc(T.gaia_dr3)} · G = ${T.g.toFixed(2)}`], ['Note', esc(T.note || '')], ['Proper motion', pm.toFixed(3) + '″ per year'],
      ['Distance', T.parallax > 0 ? (1000/T.parallax).toFixed(T.parallax > 100 ? 2 : 0) + ' pc · ' + (3261.56/T.parallax).toFixed(T.parallax > 100 ? 2 : 0) + ' light-years' : '—'],
      ['Moves during the SPHEREx survey', `${mv.toFixed(1)}″ ≈ ${(mv/SPHEREX_PIX).toFixed(1)} SPHEREx pixels (6.15″)`]].map(r => `<div class="v"><span>${r[0]}</span><span>${r[1]}</span></div>`).join('');
  } else if (T) $('anaTarget').innerHTML = `<div class="v"><span>Source</span><span>${esc(T.source || '')}</span></div>` + (T.note ? `<div class="v"><span>Note</span><span>${esc(T.note)}</span></div>` : '') + (T.typed ? '' : '<div class="v"><span>Proper motion</span><span>not in Gaia: position not moved</span></div>');
  else $('anaTarget').innerHTML = '';
  $('anaJSON').textContent = JSON.stringify(P, null, 2);
  $('anaCopy').textContent = 'Copy coordinates';
  // the most important answer first: can SPHEREx see it, and if not, when
  const vis = t.spherex_visibility, nx = vis.visible_now ? null : nextVisible(dir);
  $('anaVis').className = vis.visible_now ? 'ok' : 'no';
  $('anaVis').innerHTML = vis.visible_now ? `<b>✓ SPHEREx can see this point right now</b><span>It is ${t.sun_elongation_deg.toFixed(0)}° from the Sun (the limit is 91°) and not behind Earth.</span>`
    : `<b>✕ Outside SPHEREx's view right now</b><span>${vis.behind_earth && vis.sun_limit_ok ? 'Earth is in the way at this moment.' : `It is only ${t.sun_elongation_deg.toFixed(0)}° from the Sun; SPHEREx never points closer than 91°.`} ${nx || ''}</span>`;
  const inSpan = inDataSpan(sim.jd);
  $('anaViz').hidden = !vis.visible_now || !inSpan;          // no image box outside SPHEREx's view or outside the dates it has data for
  if (!inSpan) $('anaVis').innerHTML += `<span class="span">No SPHEREx images exist for this date: its data run from ${utc(DATA_FIRST)} to ${utc(dataLatest())}. Use “Jump to…” in the time bar to go there.</span>`;
  else if (sim.jd > nowJD() - 60) $('anaVis').innerHTML += '<span class="span">Images from the last 60 days may not be released by IRSA yet.</span>';
  P.target.spherex_next_visible = nx;
  $('anaJSON').textContent = JSON.stringify(P, null, 2);
  $('anaBack').hidden = false; $('ana').scrollTop = 0; $('anaPayload').open = false; $('anaClose').focus({preventScroll: true});
  clickMark = {dir, ok: vis.visible_now, t: performance.now()};
  drawPatchMap(t.ra_deg, t.dec_deg, hw, trk, null);
  // known asteroids and comets in the patch on this date (NASA/JPL SB Ident)
  const id = ++anaReq;
  $('anaSB').innerHTML = '<div class="sbh">Known asteroids and comets in the patch</div><div class="dim">Asking NASA/JPL…</div>';
  fetch(`/api/sbident?ra=${t.ra_deg}&dec=${t.dec_deg}&jd=${sim.jd}&hw=${hw}`).then(r => r.json()).then(d => {
    if (id !== anaReq) return;
    if (d.error) { sbFailed(); return; }
    P.known_small_bodies = {count: d.count, time_utc: d.time_utc, source: d.source, objects: d.objects}; $('anaJSON').textContent = JSON.stringify(P, null, 2);
    drawPatchMap(t.ra_deg, t.dec_deg, hw, trk, d.objects);
    $('anaSB').innerHTML = `<div class="sbh">Known asteroids and comets in the patch · ${d.count}</div>` + (d.count ? `<table><tr><th>Object</th><th>V mag</th><th>Offset E, N</th><th>Motion</th></tr>` +
      d.objects.slice(0, 15).map(o => `<tr><td>${esc(o.name)}</td><td>${o.v == null ? '—' : o.v.toFixed(1)}</td><td>${o.east_arcmin.toFixed(1)}′, ${o.north_arcmin.toFixed(1)}′</td><td>${o.rate_ra_arcsec_h == null ? '—' : Math.hypot(o.rate_ra_arcsec_h, o.rate_dec_arcsec_h).toFixed(0) + '″/h'}</td></tr>`).join('') + '</table>' +
      (d.count > 15 ? `<div class="dim" style="margin-top:4px">and ${d.count - 15} fainter (all in the backend data)</div>` : '') : '<div class="dim">None on this date.</div>') +
      `<div class="dim" style="margin-top:4px">${esc(d.source)}, seen from Earth's centre at ${esc(d.time_utc)} UTC.</div>`;
  }).catch(() => { if (id === anaReq) sbFailed(); });
  // NASA/JPL (or our server) did not answer: leave the section out entirely rather than show an error
  function sbFailed() { $('anaSB').innerHTML = ''; drawPatchMap(t.ra_deg, t.dec_deg, hw, trk, []); }
  requestAnalysis(P);                                         // placeholder: returns null for now
}
// The patch as a little map (gnomonic, north up, east left like a sky image): the target's predicted positions at the
// SPHEREx survey dates (ESA Gaia proper motion) and the known asteroids and comets with their motion over 2 hours (NASA/JPL).
// SPHEREx science data: from 1 May 2025 (start of the survey) to now, until the mission ends; IRSA releases each week's
// images within 60 days of observation (IPAC SPHEREx Explanatory Supplement, sec. 2.1-2.2).
const utc = jd => fmtDate(jd, false) + ' UTC';                     // 'YYYY-MM-DD hh:mm UTC'
const DATA_FIRST = 2460796.5, dataLatest = () => Math.min(nowJD(), 2461557.5), inDataSpan = jd => jd >= DATA_FIRST && jd <= dataLatest();
const SCIENCE_START = 2460796.5, SCIENCE_END = 2461557.5, SURVEY_EPOCHS = [0, 1, 2, 3, 4].map(i => SCIENCE_START + i*182.6).filter(jd => jd <= SCIENCE_END);
function drawPatchMap(ra0, dec0, hw, trk, objs) {
  const cvs = $('anaMap'), g = cvs.getContext('2d'), Wm = cvs.width, Hm = cvs.height, S = Math.min(Wm, Hm)*0.92/(2*hw*60);   // px per arcmin
  const a0 = ra0*D2R, d0 = dec0*D2R, c0 = [Math.cos(d0)*Math.cos(a0), Math.cos(d0)*Math.sin(a0), Math.sin(d0)], e = [-Math.sin(a0), Math.cos(a0), 0], n = cross(c0, e);
  const off = (ra, dec) => { const a = ra*D2R, d = dec*D2R, u = [Math.cos(d)*Math.cos(a), Math.cos(d)*Math.sin(a), Math.sin(d)], w = dot(u, c0); return [dot(u, e)/w/D2R*60, dot(u, n)/w/D2R*60]; };
  const X = x => Wm/2 - x*S, Y = y => Hm/2 - y*S;                                                  // east to the left
  g.clearRect(0, 0, Wm, Hm);
  const box = hw*60*S; g.strokeStyle = '#34436a'; g.lineWidth = 1; g.strokeRect(Wm/2 - box, Hm/2 - box, 2*box, 2*box);
  g.setLineDash([2, 4]); g.beginPath(); g.moveTo(Wm/2, Hm/2 - box); g.lineTo(Wm/2, Hm/2 + box); g.moveTo(Wm/2 - box, Hm/2); g.lineTo(Wm/2 + box, Hm/2); g.stroke(); g.setLineDash([]);
  g.font = '11px system-ui'; g.fillStyle = '#6c7a9c'; g.fillText((2*hw*60).toFixed(0) + '′', Wm/2 + box - 26, Hm/2 + box - 6);
  const pix = SPHEREX_PIX/60*S; if (pix >= 2) { g.strokeStyle = '#4a5f94'; g.strokeRect(Wm/2 - box + 8, Hm/2 + box - 8 - pix, pix, pix); g.fillText('1 SPHEREx pixel', Wm/2 - box + 12 + pix, Hm/2 + box - 9); }
  let leg = [];
  if (objs && objs.length) {
    for (const o of objs) { const [x, y] = [o.east_arcmin, o.north_arcmin], r = o.v == null ? 2 : clamp(5.5 - (o.v - 14)*0.45, 1.5, 5.5);
      g.fillStyle = '#e2a45f'; g.beginPath(); g.arc(X(x), Y(y), r, 0, 7); g.fill();
      if (o.rate_ra_arcsec_h != null) { const dx = o.rate_ra_arcsec_h*2/60, dy = o.rate_dec_arcsec_h*2/60; g.strokeStyle = '#e2a45f99'; g.beginPath(); g.moveTo(X(x), Y(y)); g.lineTo(X(x + dx), Y(y + dy)); g.stroke(); } }
    leg.push('<span style="color:#e2a45f">●</span> known asteroids/comets (line: motion over 2 hours)');
  }
  if (trk) {
    const pts = trk.map(([jd, [ra, dec]]) => [jd, off(ra, dec)]), span = Math.max(...pts.map(p => Math.hypot(p[1][0] - pts[0][1][0], p[1][1] - pts[0][1][1])));
    const zoom = span*S < 40 && span > 0 ? Math.min(400, 40/(span*S)) : 1;                         // too small to see: magnify the track around the centre
    const cx = pts[Math.floor(pts.length/2)][1];
    const P = q => [X(cx[0] + (q[0] - cx[0])*zoom), Y(cx[1] + (q[1] - cx[1])*zoom)];
    g.strokeStyle = '#5fd6f0'; g.lineWidth = 1.2; g.beginPath(); pts.forEach(([, q], i) => { const [x, y] = P(q); i ? g.lineTo(x, y) : g.moveTo(x, y); }); g.stroke();
    pts.forEach(([jd, q], i) => { const [x, y] = P(q); g.fillStyle = '#5fd6f0'; g.beginPath(); g.arc(x, y, 3.2, 0, 7); g.fill();
      if (i === 0 || i === pts.length - 1) { g.fillStyle = '#9fe6f7'; g.fillText(fmtDate(jd, false).slice(0, 7), x + 9, y + (i === 0 ? 12 : -6)); } });
    const [nx, ny] = P(off(...trk.now)); g.strokeStyle = '#ffffff'; g.lineWidth = 1.5; g.beginPath(); g.arc(nx, ny, 7, 0, 7); g.stroke();
    leg.push('○ position on the date shown');
    leg.push('<span style="color:#5fd6f0">●</span> target at each 6-month SPHEREx sky survey (ESA Gaia proper motion)' + (zoom > 1 ? ` · track magnified ×${zoom.toFixed(0)}` : ''));
  } else { g.strokeStyle = '#5fd6f0'; g.lineWidth = 1.4; g.beginPath(); g.arc(Wm/2, Hm/2, 6, 0, 7); g.stroke(); leg.push('<span style="color:#5fd6f0">○</span> patch centre'); }
  if (objs === null) leg.push('asking NASA/JPL for known asteroids…');
  $('anaLegend').innerHTML = leg.join('<br>');
}
function closeAnalysis() { $('anaBack').hidden = true; }
