/**
 * Body Language ML Engine
 * All logic lives here — triggered by the main dashboard's loadComponent() call.
 * Libraries (tf + posenet) are pre-loaded in public/index.html
 */

(function () {
    let detector = null;
    let camStream = null;
    let running = false;
    let frameLog = [];
    let smooth = { p: 85, e: 85, g: 55 };
    let noseY = [];

    // ─── Entry Point ─────────────────────────────────────────────────────────
    window.blStart = async function () {
        if (running) return stopSession();

        const btn = document.getElementById('bl-start-btn');
        if (btn) { btn.innerText = '⏹ Stop Analysis'; btn.style.background = '#ef4444'; }

        try {
            // Load posenet only once
            if (!detector) {
                setTip('all', 'Loading AI model...');
                detector = await posenet.load({
                    architecture: 'MobileNetV1',
                    outputStride: 16,
                    inputResolution: { width: 257, height: 257 },
                    multiplier: 0.75
                });
            }

            camStream = await navigator.mediaDevices.getUserMedia({ video: true });

            const container = document.getElementById('bl-video-container');
            if (!container) return;
            container.innerHTML = '';

            const video = Object.assign(document.createElement('video'), {
                autoplay: true, playsinline: true, muted: true
            });
            video.style.cssText = 'width:100%;height:100%;object-fit:cover;';
            container.appendChild(video);

            const canvas = document.createElement('canvas');
            canvas.style.cssText = 'position:absolute;top:0;left:0;width:100%;height:100%;pointer-events:none;';
            container.appendChild(canvas);

            video.srcObject = camStream;
            running = true;
            frameLog = [];
            noseY = [];

            requestAnimationFrame(() => analysisLoop(video, canvas));

        } catch (err) {
            console.error('BL Error:', err);
            resetBtn();
            alert('Camera or AI failed: ' + err.message);
        }
    };

    // ─── Stop ─────────────────────────────────────────────────────────────────
    function stopSession() {
        running = false;
        if (camStream) camStream.getTracks().forEach(t => t.stop());
        resetBtn();
        generateReport();
    }

    window.stopBodyLanguageCamera = stopSession; // legacy hook
    window.startBodyLanguageCamera = window.blStart; // legacy hook

    function resetBtn() {
        const btn = document.getElementById('bl-start-btn');
        if (btn) { btn.innerText = '🎬 Start Analysis'; btn.style.background = ''; }
    }

    // ─── Analysis Loop ────────────────────────────────────────────────────────
    async function analysisLoop(video, canvas) {
        if (!running) return;

        if (video.readyState >= 2) {
            try {
                const pose = await detector.estimateSinglePose(video, { flipHorizontal: false });
                if (pose && pose.keypoints) {
                    const raw = extractMetrics(pose.keypoints);
                    blend(raw);
                    drawSkeleton(pose.keypoints, canvas, video);
                    updateUI();
                    frameLog.push({ ...smooth });
                }
            } catch (_) {}
        }

        requestAnimationFrame(() => analysisLoop(video, canvas));
    }

    // ─── Metrics ─────────────────────────────────────────────────────────────
    function extractMetrics(kps) {
        const get = (name) => kps.find(k => k.part === name) || { position: { x: 0, y: 0 }, score: 0 };

        const nose = get('nose');
        const lS   = get('leftShoulder'),  rS = get('rightShoulder');
        const lE   = get('leftEye'),       rE = get('rightEye');
        const lW   = get('leftWrist'),     rW = get('rightWrist');

        // Bail if key points not visible
        if (nose.score < 0.15 || lS.score < 0.15 || rS.score < 0.15) {
            return { p: 0, e: 0, g: 0 };
        }

        // POSTURE — shoulder levelness (0 = perfectly level, bigger = tilt)
        const sw = Math.abs(lS.position.x - rS.position.x) || 1;
        const tilt = Math.abs(lS.position.y - rS.position.y) / sw; // 0 .. ~1
        const pScore = clamp(100 - tilt * 800);

        // EYE CONTACT — how centred nose is between the two eyes
        const eyeCx  = (lE.position.x + rE.position.x) / 2;
        const eyeW   = Math.abs(lE.position.x - rE.position.x) || 1;
        const gaze   = Math.abs(nose.position.x - eyeCx) / eyeW; // 0 centred .. ~0.5 side
        const eScore = clamp(100 - gaze * 400);

        // GESTURES — wrists raised above shoulder line?
        const gActive =
            (lW.score > 0.3 && lW.position.y < lS.position.y + 60) ||
            (rW.score > 0.3 && rW.position.y < rS.position.y + 60);
        const gScore = gActive ? 92 : 50;

        return { p: pScore, e: eScore, g: gScore };
    }

    function blend(raw) {
        const a = 0.25; // blend speed — higher = more responsive
        smooth.p = raw.p * a + smooth.p * (1 - a);
        smooth.e = raw.e * a + smooth.e * (1 - a);
        smooth.g = raw.g * a + smooth.g * (1 - a);
    }

    function clamp(v) { return Math.max(0, Math.min(100, isNaN(v) ? 0 : v)); }

    // ─── UI Update ────────────────────────────────────────────────────────────
    function updateUI() {
        const setMetric = (valId, barId, tipId, val, goodMsg, badMsg) => {
            const v = Math.round(val);
            const color = v > 80 ? '#2dd4bf' : v > 60 ? '#f59e0b' : '#ef4444';
            const el = document.getElementById(valId);
            const bar = document.getElementById(barId);
            const tip = document.getElementById(tipId);
            if (el)  { el.innerText = v + '%'; el.style.color = color; }
            if (bar) { bar.style.width = v + '%'; bar.style.background = color; }
            if (tip) tip.innerText = v > 80 ? goodMsg : badMsg;
        };

        setMetric('bl-p-val', 'bl-p-bar', 'bl-p-tip', smooth.p, 'Great posture!', 'Sit up straight & level shoulders.');
        setMetric('bl-e-val', 'bl-e-bar', 'bl-e-tip', smooth.e, 'Strong eye contact!', 'Look directly at the camera.');
        setMetric('bl-g-val', 'bl-g-bar', 'bl-g-tip', smooth.g, 'Active gestures!', 'Raise your hands when speaking.');

        const overall = Math.round((smooth.p + smooth.e + smooth.g) / 3);
        const ovEl  = document.getElementById('bl-overall');
        const ovBar = document.getElementById('bl-overall-bar');
        if (ovEl)  ovEl.innerText = overall + '%';
        if (ovBar) ovBar.style.width = overall + '%';
    }

    // ─── Skeleton Draw ────────────────────────────────────────────────────────
    function drawSkeleton(kps, canvas, video) {
        canvas.width  = canvas.clientWidth;
        canvas.height = canvas.clientHeight;
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        const sx = canvas.width  / (video.videoWidth  || 640);
        const sy = canvas.height / (video.videoHeight || 480);

        const pts = {};
        kps.forEach(kp => { pts[kp.part] = { x: kp.position.x * sx, y: kp.position.y * sy, s: kp.score }; });

        ctx.strokeStyle = '#00ff99';
        ctx.lineWidth   = 4;
        ctx.shadowBlur  = 12;
        ctx.shadowColor = '#00ff99';

        const line = (a, b) => {
            if (!pts[a] || !pts[b] || pts[a].s < 0.2 || pts[b].s < 0.2) return;
            ctx.beginPath();
            ctx.moveTo(pts[a].x, pts[a].y);
            ctx.lineTo(pts[b].x, pts[b].y);
            ctx.stroke();
        };

        line('leftShoulder',  'rightShoulder');
        line('leftShoulder',  'leftElbow');
        line('leftElbow',     'leftWrist');
        line('rightShoulder', 'rightElbow');
        line('rightElbow',    'rightWrist');

        // Face dots
        ['nose', 'leftEye', 'rightEye'].forEach(part => {
            const p = pts[part];
            if (p && p.s > 0.4) {
                ctx.fillStyle = '#ffffff';
                ctx.beginPath();
                ctx.arc(p.x, p.y, 5, 0, Math.PI * 2);
                ctx.fill();
            }
        });
    }

    // ─── Final Report ─────────────────────────────────────────────────────────
    function generateReport() {
        if (frameLog.length === 0) return;

        const avg = key => Math.round(frameLog.reduce((s, m) => s + m[key], 0) / frameLog.length);
        const p = avg('p'), e = avg('e'), g = avg('g');
        const overall = Math.round((p + e + g) / 3);

        const postureTip  = p  < 75 ? 'Keep your shoulders level and back straight.' : 'Excellent posture maintained!';
        const eyeTip      = e  < 75 ? 'Try to look directly at the camera more often.' : 'Strong, consistent eye contact!';
        const gestureTip  = g  < 60 ? 'Use more hand gestures to show confidence.' : 'Great use of gestures!';

        const el = document.getElementById('bl-report-text');
        if (el) {
            el.innerHTML = `
                <div style="background:rgba(45,212,191,0.1);border:1px solid #2dd4bf;border-radius:10px;padding:10px;margin-bottom:10px;">
                    <b style="color:#2dd4bf;">Overall: ${overall}%</b>
                </div>
                <ul style="padding-left:16px;margin:0;color:#cbd5e1;">
                    <li style="margin-bottom:6px;"><b>Posture ${p}%:</b> ${postureTip}</li>
                    <li style="margin-bottom:6px;"><b>Eye Contact ${e}%:</b> ${eyeTip}</li>
                    <li><b>Gestures ${g}%:</b> ${gestureTip}</li>
                </ul>
                <button class="btn btn-primary" style="width:100%;margin-top:12px;border-radius:10px;" onclick="location.reload()">🔄 Try Again</button>
            `;
        }
    }

    function setTip(which, msg) {
        ['p', 'e', 'g'].forEach(k => {
            const el = document.getElementById('bl-' + k + '-tip');
            if (el) el.innerText = msg;
        });
    }

})();
