// XARVIS viewer · presence.js
// Is this a NEW visit? Then the saved analyses are cleared (POST /api/visit/new; see app.py, "visits"). The browser
// decides, because it knows which XARVIS tabs are open - a browser session cookie does not work (Chrome / Edge keep it
// after the window is closed). A new visit is the viewer opening when
//   - no other XARVIS tab answers on the 'xarvis' BroadcastChannel (background tabs answer too: messages are not
//     throttled like timers), and
//   - this tab did not just come from another XARVIS page or a reload (it notes when it left, in sessionStorage).
// Loaded by the viewer and the About page (which answers, so a second tab keeps everything).
'use strict';
const xarvisPage = (() => {
  const NAV_GAP_MS = 10000, ASK_MS = 400;
  const channel = 'BroadcastChannel' in window ? new BroadcastChannel('xarvis') : null;
  if (channel) channel.addEventListener('message', e => { if (e.data === 'anyone?') channel.postMessage('here'); });
  addEventListener('pagehide', () => { try { sessionStorage.setItem('xarvis.left', String(Date.now())); } catch (e) {} });
  function otherTabOpen() {
    if (!channel) return Promise.resolve(false);
    return new Promise(done => {
      const answer = e => { if (e.data === 'here') finish(true); };
      const timer = setTimeout(() => finish(false), ASK_MS);
      function finish(open) { clearTimeout(timer); channel.removeEventListener('message', answer); done(open); }
      channel.addEventListener('message', answer); channel.postMessage('anyone?');
    });
  }
  async function isNewVisit() {
    let left = 0; try { left = +sessionStorage.getItem('xarvis.left') || 0; } catch (e) {}
    if (Date.now() - left < NAV_GAP_MS) return false;              // from the About page, or a reload
    return !(await otherTabOpen());
  }
  async function hello(viewer) {                                    // -> {new_visit, reset}
    if (!viewer || !(await isNewVisit())) return {new_visit: false};
    try { const r = await fetch('/api/visit/new', {method: 'POST'}); return {new_visit: true, ...(await r.json())}; }
    catch (e) { console.warn('visit', e); return {new_visit: true}; }
  }
  return {hello};
})();
