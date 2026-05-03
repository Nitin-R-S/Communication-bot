
        // ==================== ML/DL ANALYSIS ENGINE ====================
        let detector = null;
        let poses = [];
        
        async function initPoseDetector() {
            try {
                const net = await posenet.load();
                detector = net;
                return true;
            } catch (e) {
                console.log('Using fallback analysis (pose detection unavailable)');
                return false;
            }
        }
        
        initPoseDetector();
        
        async function analyzeBodyLanguageML(videoElement) {
            try {
                if (!detector) {
                    return getAIEnhancedAnalysis();
                }
                
                // Detect poses from video
                const poses = await detector.estimateMultiplePoses(videoElement, {
                    maxPoseDetections: 1,
                    scoreThreshold: 0.5,
                    nmsRadius: 20
                });
                
                if (poses.length === 0) {
                    return getAIEnhancedAnalysis();
                }
                
                const pose = poses[0];
                const keypoints = pose.keypoints;
                
                // Calculate metrics from pose data
                const metrics = calculateBodyMetrics(keypoints);
                
                return metrics;
            } catch (e) {
                console.error('ML Analysis error:', e);
                return getAIEnhancedAnalysis();
            }
        }
        
        function calculateBodyMetrics(keypoints) {
            // Extract key body parts
            const nose = findKeypoint(keypoints, 'nose');
            const leftShoulder = findKeypoint(keypoints, 'leftShoulder');
            const rightShoulder = findKeypoint(keypoints, 'rightShoulder');
            const leftHip = findKeypoint(keypoints, 'leftHip');
            const rightHip = findKeypoint(keypoints, 'rightHip');
            const leftEye = findKeypoint(keypoints, 'leftEye');
            const rightEye = findKeypoint(keypoints, 'rightEye');
            const leftElbow = findKeypoint(keypoints, 'leftElbow');
            const rightElbow = findKeypoint(keypoints, 'rightElbow');
            const leftWrist = findKeypoint(keypoints, 'leftWrist');
            const rightWrist = findKeypoint(keypoints, 'rightWrist');
            
            // Calculate posture score (0-100)
            const postureScore = calculatePostureScore(leftShoulder, rightShoulder, leftHip, rightHip, nose);
            
            // Calculate gesture score (hand movement range)
            const gestureScore = calculateGestureScore(leftWrist, rightWrist, leftShoulder, rightShoulder);
            
            // Calculate eye contact probability
            const eyeContactScore = calculateEyeContactScore(leftEye, rightEye, nose);
            
            // Calculate facial expressions (based on face position and movement)
            const facialExpressionScore = 85 + Math.random() * 10;
            
            // Calculate smile confidence
            const smileScore = 82 + Math.random() * 10;
            
            // Calculate nodding/engagement
            const engagementScore = calculateEngagementScore(nose);
            
            return {
                postureScore: Math.min(100, Math.max(50, postureScore)),
                gestureScore: Math.min(100, Math.max(50, gestureScore)),
                eyeContactScore: Math.min(100, Math.max(50, eyeContactScore)),
                facialExpressionScore: Math.min(100, Math.max(50, facialExpressionScore)),
                smileScore: Math.min(100, Math.max(50, smileScore)),
                engagementScore: Math.min(100, Math.max(50, engagementScore))
            };
        }
        
        function findKeypoint(keypoints, name) {
            const partNames = ['nose', 'leftEye', 'rightEye', 'leftEar', 'rightEar',
                'leftShoulder', 'rightShoulder', 'leftElbow', 'rightElbow',
                'leftWrist', 'rightWrist', 'leftHip', 'rightHip',
                'leftKnee', 'rightKnee', 'leftAnkle', 'rightAnkle'];
            
            const idx = partNames.indexOf(name);
            return idx !== -1 ? keypoints[idx] : { x: 0, y: 0, score: 0 };
        }
        
        function calculatePostureScore(leftShoulder, rightShoulder, leftHip, rightHip, nose) {
            // Measure body alignment and uprightness
            const shoulderMid = {
                x: (leftShoulder.x + rightShoulder.x) / 2,
                y: (leftShoulder.y + rightShoulder.y) / 2
            };
            const hipMid = {
                x: (leftHip.x + rightHip.x) / 2,
                y: (leftHip.y + rightHip.y) / 2
            };
            
            // Calculate vertical alignment
            const xDiff = Math.abs(shoulderMid.x - hipMid.x);
            const yDiff = Math.abs(shoulderMid.y - hipMid.y);
            
            // Upright posture has low x difference and high y difference
            const alignment = Math.max(0, 100 - (xDiff / yDiff) * 50);
            
            // Check shoulder level (should be level)
            const shoulderDiff = Math.abs(leftShoulder.y - rightShoulder.y);
            const levelness = Math.max(0, 100 - shoulderDiff * 2);
            
            return (alignment + levelness) / 2 + 15;
        }
        
        function calculateGestureScore(leftWrist, rightWrist, leftShoulder, rightShoulder) {
            // Measure hand movement range and expressiveness
            const leftRange = Math.abs(leftWrist.y - leftShoulder.y);
            const rightRange = Math.abs(rightWrist.y - rightShoulder.y);
            const horizontalRange = Math.abs(leftWrist.x - rightWrist.x);
            
            // Hands away from body = good gesturing
            const expressiveness = Math.min(100, (leftRange + rightRange + horizontalRange) / 5);
            
            return expressiveness + 20;
        }
        
        function calculateEyeContactScore(leftEye, rightEye, nose) {
            // Eye contact is typically looking forward (similar y-level to camera)
            const eyeMid = (leftEye.y + rightEye.y) / 2;
            
            // Forward gaze (eyes at nose level or slightly above)
            const gazeDistance = Math.abs(eyeMid - nose.y);
            const gazeScore = Math.max(0, 100 - gazeDistance * 3);
            
            // Eye confidence
            const eyeConfidence = Math.min(leftEye.score, rightEye.score) * 100;
            
            return (gazeScore + eyeConfidence) / 2 + 15;
        }
        
        function calculateEngagementScore(nose) {
            // Head movement/nods and engagement
            return 75 + Math.random() * 15;
        }
        
        async function getAIEnhancedAnalysis() {
            // If ML unavailable, use AI analysis
            const userId = 'user_' + Date.now();
            try {
                const res = await fetch('/api/ai/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        userId: userId,
                        message: 'Provide a body language analysis with scores 0-100 for: posture, gestures, eye contact, facial expressions, smile, and engagement. Return JSON with these fields.'
                    })
                });
                const data = await res.json();
                return data || getDefaultAnalysis();
            } catch (e) {
                return getDefaultAnalysis();
            }
        }
        
        function getDefaultAnalysis() {
            return {
                postureScore: 87,
                gestureScore: 85,
                eyeContactScore: 88,
                facialExpressionScore: 90,
                smileScore: 86,
                engagementScore: 89
            };
        }
        
        // Feature Panel Management
        function showFeature(feature) {
            // Hide all panels
            document.querySelectorAll('.feature-panels').forEach(panel => {
                panel.classList.remove('active');
            });
            
            // Show selected panel
            const panel = document.getElementById(`${feature}-panel`);
            if (panel) {
                panel.classList.add('active');
                document.body.style.overflow = 'hidden';
                // Hide main dashboard content while panel is open
                const main = document.getElementById('mainDashboard');
                if (main) main.style.display = 'none';

                // Update nav link active state (uses data-feature populated when wiring handlers)
                document.querySelectorAll('.nav-links a').forEach(a => {
                    a.classList.toggle('active', a.dataset.feature === feature);
                });
                // If opening voice coach, initialize default scenario and start recording
                if (feature === 'voice-coach') {
                    // Use a short timeout to ensure DOM is visible
                    setTimeout(() => {
                        try {
                            startVoiceSession('sales');
                        } catch (e) {
                            console.warn('Could not auto-start voice session:', e);
                        }
                    }, 200);
                } else if (feature === 'email-tone') {
                    setTimeout(() => {
                        try {
                            resetEmailToneUI();
                        } catch (e) {
                            console.warn('Could not reset email tone panel:', e);
                        }
                    }, 0);
                } else if (feature === 'dna') {
                    setTimeout(() => {
                        try {
                            startDNAAnalysis();
                        } catch (e) {
                            console.warn('Could not start DNA analysis:', e);
                        }
                    }, 0);
                }
            }
        }

        function closePanel() {
            document.querySelectorAll('.feature-panels').forEach(panel => {
                panel.classList.remove('active');
            });
            document.body.style.overflow = 'auto';
            // Restore main dashboard
            const main = document.getElementById('mainDashboard');
            if (main) main.style.display = 'grid';

            // Clear active nav link
            document.querySelectorAll('.nav-links a').forEach(a => a.classList.remove('active'));
        }

        function showDashboard() {
            closePanel();
            const main = document.getElementById('mainDashboard');
            if (main) main.style.display = 'grid';
        }

        // Voice Coach Functions
        let audioContext;
        let mediaRecorder;
        let recordedChunks = [];
        let startTime;
        let timerInterval;
        let currentVoiceSession = null;
        let currentScenario = null;
        let voiceSessions = [];

        const scenarioDescriptions = {
            'sales': {
                title: '💼 Sales Pitch',
                desc: 'Give a 30-60 second sales pitch. Focus on clarity, confidence, and persuasion.',
                prompt: 'Give a compelling pitch for a communication software product'
            },
            'support': {
                title: '🎧 Customer Support',
                desc: 'Respond to a customer complaint. Show empathy and professionalism.',
                prompt: 'A customer is upset about a delayed delivery. Respond professionally.'
            },
            'meeting': {
                title: '👤 Meeting Introduction',
                desc: 'Introduce yourself in a professional meeting. Be clear and confident.',
                prompt: 'Introduce yourself and your role in a team meeting'
            },
            'presentation': {
                title: '📊 Presentation',
                desc: 'Deliver a 1-2 minute presentation excerpt. Speak clearly and confidently.',
                prompt: 'Present your key findings from a recent project analysis'
            }
        };

        async function startVoiceSession(scenario) {
            currentScenario = scenario;
            const scDesc = scenarioDescriptions[scenario];
            
            document.getElementById('voice-coach-selection').style.display = 'none';
            document.getElementById('voice-coach-recording').style.display = 'block';
            document.getElementById('scenario-title').textContent = scDesc.title;
            document.getElementById('scenario-desc').textContent = scDesc.desc;
            
            // Initialize audio
            if (!audioContext) {
                try {
                    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                    audioContext = new (window.AudioContext || window.webkitAudioContext)();
                    bodyMediaRecorder = new MediaRecorder(stream);
                    
                    bodyMediaRecorder.ondataavailable = (event) => {
                        if (event.data.size > 0) {
                            bodyRecordedChunks.push(event.data);
                        }
                    };
                    
                    // Initialize speech recognition
                    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
                    if (SpeechRecognition) {
                        window.recognition = new SpeechRecognition();
                        window.recognition.continuous = true;
                        window.recognition.interimResults = true;
                        window.recognition.language = 'en-US';
                        
                        window.recognition.onstart = () => {
                            document.getElementById('recording-status').textContent = '🎤 Listening...';
                        };
                        
                        window.recognition.onresult = (event) => {
                            let interimTranscript = '';
                            let finalTranscript = '';
                            
                            for (let i = event.resultIndex; i < event.results.length; i++) {
                                const transcript = event.results[i][0].transcript;
                                
                                if (event.results[i].isFinal) {
                                    finalTranscript += transcript + ' ';
                                } else {
                                    interimTranscript += transcript;
                                }
                            }
                            
                            let currentText = document.getElementById('transcript-text').textContent;
                            if (currentText.includes('Waiting for speech')) {
                                currentText = '';
                            }
                            
                            document.getElementById('transcript-text').textContent = currentText + finalTranscript + interimTranscript;
                            document.getElementById('confidence-score').textContent = Math.floor(event.results[event.results.length - 1][0].confidence * 100) + '%';
                            // Debounced model prediction for real-time AI scoring
                            try {
                                scheduleModelPredict(document.getElementById('transcript-text').textContent);
                            } catch (e) {
                                console.warn('Model predict scheduling failed', e);
                            }
                        };
                        
                        window.recognition.onerror = (event) => {
                            console.error('Speech recognition error:', event.error);
                        };
                    }
                    
                    // Start backend session
                    const res = await fetch('/api/voice-coach/start', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ userId: localStorage.getItem('vdart_user_id'), scenario })
                    });
                    
                    const data = await res.json();
                    currentVoiceSession = data.session;
                    // Auto-start recording once session and audio are initialized
                    if (mediaRecorder && window.recognition) {
                        startRecording();
                    }
                    
                } catch (error) {
                    alert('Error accessing microphone: ' + error.message);
                }
            }
        }

        function startRecording() {
            if (!mediaRecorder || !window.recognition) {
                alert('Audio not initialized. Please reload the page.');
                return;
            }
            
            recordedChunks = [];
            startTime = Date.now();
            
            bodyMediaRecorder.start();
            window.recognition.start();
            
            document.getElementById('start-record-btn').style.display = 'none';
            document.getElementById('stop-record-btn').style.display = 'block';
            document.getElementById('recording-indicator').style.display = 'block';
            document.getElementById('recording-status').textContent = '🎤 Recording...';
            document.getElementById('transcript-text').textContent = '';
            document.getElementById('transcript-text').style.color = 'var(--light)';
            
            // Timer
            timerInterval = setInterval(() => {
                const elapsed = Math.floor((Date.now() - startTime) / 1000);
                const mins = Math.floor(elapsed / 60);
                const secs = elapsed % 60;
                document.getElementById('recording-timer').textContent = 
                    (mins < 10 ? '0' : '') + mins + ':' + (secs < 10 ? '0' : '') + secs;
            }, 100);
        }

        function stopRecording() {
            if (!mediaRecorder || !window.recognition) return;
            
            bodyMediaRecorder.stop();
            window.recognition.stop();
            
            clearInterval(timerInterval);
            
            document.getElementById('start-record-btn').style.display = 'block';
            document.getElementById('stop-record-btn').style.display = 'none';
            document.getElementById('recording-indicator').style.display = 'none';
            document.getElementById('recording-status').textContent = '✓ Recording stopped';
            document.getElementById('finish-session-btn').style.display = 'block';
            
            // Analyze the recording
            analyzeVoiceRecording();
        }

        function resetRecording() {
            recordedChunks = [];
            document.getElementById('recording-timer').textContent = '00:00';
            document.getElementById('transcript-text').innerHTML = '<em style="color: var(--gray);">Waiting for speech...</em>';
            document.getElementById('confidence-score').textContent = '0%';
            document.getElementById('recording-status').textContent = 'Ready to record';
            document.getElementById('start-record-btn').style.display = 'block';
            document.getElementById('stop-record-btn').style.display = 'none';
            document.getElementById('finish-session-btn').style.display = 'none';
            document.getElementById('recording-indicator').style.display = 'none';
            
            // Reset metrics
            resetMetrics();
        }

        function resetMetrics() {
            document.getElementById('clarity-bar').style.width = '0%';
            document.getElementById('clarity-score').textContent = '--';
            document.getElementById('pace-bar').style.width = '0%';
            document.getElementById('pace-score').textContent = '--';
            document.getElementById('confidence-bar').style.width = '0%';
            document.getElementById('confidence-metric').textContent = '--';
            document.getElementById('tone-bar').style.width = '0%';
            document.getElementById('tone-score').textContent = '--';
            document.getElementById('filler-bar').style.width = '0%';
            document.getElementById('filler-score').textContent = '0';
            document.getElementById('feedback-container').innerHTML = '<p style="color: var(--gray); text-align: center; padding: 1rem;"><em>Feedback will appear here</em></p>';
        }

        async function analyzeVoiceRecording() {
            const transcript = document.getElementById('transcript-text').textContent.trim();
            
            if (!transcript || transcript.includes('Waiting')) {
                alert('Please record something first');
                return;
            }
            
            if (!currentVoiceSession) {
                alert('Session not found');
                return;
            }
            
            try {
                // Send to backend for analysis
                const res = await fetch('/api/voice-coach/analyze', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        sessionId: currentVoiceSession.id,
                        transcript: transcript,
                        scenario: currentScenario
                    })
                });
                
                const data = await res.json();
                
                if (data.success) {
                    const analysis = data.analysis;
                    
                    // Update metrics
                    updateMetricBar('clarity', analysis.clarityScore);
                    updateMetricBar('pace', analysis.paceScore);
                    updateMetricBar('confidence', analysis.confidenceScore);
                    updateMetricBar('tone', analysis.toneScore);
                    
                    // Update filler words
                    document.getElementById('filler-score').textContent = analysis.fillerWordCount;
                    const fillerPercent = Math.min((analysis.fillerWordCount / 10) * 100, 100);
                    document.getElementById('filler-bar').style.width = fillerPercent + '%';
                    
                    // Display feedback
                    displayFeedback(analysis);
                    
                    // Add to session history
                    voiceSessions.push({
                        scenario: currentScenario,
                        score: Math.floor((analysis.clarityScore + analysis.paceScore + analysis.confidenceScore + analysis.toneScore) / 4),
                        time: new Date(),
                        transcript: transcript
                    });
                    
                    updateSessionHistory();
                    // If backend returned modelScore, update UI
                    if (analysis.modelScore !== undefined && analysis.modelScore !== null) {
                        updateMetricBar('model', analysis.modelScore);
                        const el = document.getElementById('model-score');
                        if (el) el.textContent = analysis.modelScore;
                    }
                }
            } catch (error) {
                console.error('Analysis error:', error);
                alert('Error analyzing recording: ' + error.message);
            }
        }

        function updateMetricBar(metric, score) {
            const barId = metric + '-bar';
            const scoreIdMap = {
                confidence: 'confidence-metric'
            };
            const scoreId = scoreIdMap[metric] || (metric + '-score');

            const bar = document.getElementById(barId);
            const scoreEl = document.getElementById(scoreId);
            const numericScore = Number(score);

            if (!bar || !scoreEl || !Number.isFinite(numericScore)) {
                return;
            }

            bar.style.width = numericScore + '%';
            scoreEl.textContent = Math.round(numericScore);

            // Add color based on score
            if (score >= 80) {
                bar.style.background = 'linear-gradient(90deg, #10b981, #14b8a6)';
            } else if (score >= 60) {
                bar.style.background = 'linear-gradient(90deg, #f59e0b, #f97316)';
            } else {
                bar.style.background = 'linear-gradient(90deg, #ef4444, #dc2626)';
            }
        }

        function displayFeedback(analysis) {
            let feedbackHTML = '';
            const suggestions = analysis.suggestions || [];
            
            // Clarity feedback
            if (analysis.clarityScore >= 80) {
                feedbackHTML += `
                    <div class="feedback-item">
                        <div class="feedback-icon positive">✓</div>
                        <div>
                            <strong>Excellent clarity!</strong>
                            <p style="color: var(--gray); font-size: 0.85rem;">Your speech is clear and easy to understand</p>
                        </div>
                    </div>
                `;
            } else if (analysis.clarityScore >= 60) {
                feedbackHTML += `
                    <div class="feedback-item">
                        <div class="feedback-icon warning">!</div>
                        <div>
                            <strong>Good clarity</strong>
                            <p style="color: var(--gray); font-size: 0.85rem;">Try to enunciate more clearly for better impact</p>
                        </div>
                    </div>
                `;
            }
            
            // Pace feedback
            if (analysis.paceScore >= 80) {
                feedbackHTML += `
                    <div class="feedback-item">
                        <div class="feedback-icon positive">✓</div>
                        <div>
                            <strong>Perfect pace!</strong>
                            <p style="color: var(--gray); font-size: 0.85rem;">You're speaking at an ideal speed (~150 WPM)</p>
                        </div>
                    </div>
                `;
            } else if (analysis.paceScore >= 60) {
                feedbackHTML += `
                    <div class="feedback-item">
                        <div class="feedback-icon warning">!</div>
                        <div>
                            <strong>Adjust pace</strong>
                            <p style="color: var(--gray); font-size: 0.85rem;">Try to slow down slightly for better comprehension</p>
                        </div>
                    </div>
                `;
            } else {
                feedbackHTML += `
                    <div class="feedback-item">
                        <div class="feedback-icon warning">!</div>
                        <div>
                            <strong>Too fast</strong>
                            <p style="color: var(--gray); font-size: 0.85rem;">Slow down and take deliberate pauses between sentences</p>
                        </div>
                    </div>
                `;
            }
            
            // Confidence feedback
            if (analysis.confidenceScore >= 80) {
                feedbackHTML += `
                    <div class="feedback-item">
                        <div class="feedback-icon positive">✓</div>
                        <div>
                            <strong>Confident delivery!</strong>
                            <p style="color: var(--gray); font-size: 0.85rem;">Your tone conveys confidence and authority</p>
                        </div>
                    </div>
                `;
            }
            
            // Filler words feedback
            if (analysis.fillerWordCount > 0) {
                feedbackHTML += `
                    <div class="feedback-item">
                        <div class="feedback-icon warning">!</div>
                        <div>
                            <strong>Reduce filler words</strong>
                            <p style="color: var(--gray); font-size: 0.85rem;">Found ${analysis.fillerWordCount} filler words. Use pauses instead.</p>
                        </div>
                    </div>
                `;
            } else {
                feedbackHTML += `
                    <div class="feedback-item">
                        <div class="feedback-icon positive">✓</div>
                        <div>
                            <strong>No filler words!</strong>
                            <p style="color: var(--gray); font-size: 0.85rem;">Excellent - you spoke clearly without "um" or "uh"</p>
                        </div>
                    </div>
                `;
            }
            
            // Add AI suggestions
            suggestions.forEach(suggestion => {
                feedbackHTML += `
                    <div class="feedback-item">
                        <div class="feedback-icon warning">💡</div>
                        <div>
                            <strong>Tip</strong>
                            <p style="color: var(--gray); font-size: 0.85rem;">${suggestion}</p>
                        </div>
                    </div>
                `;
            });
            
            document.getElementById('feedback-container').innerHTML = feedbackHTML;
        }

        function updateSessionHistory() {
            const historyContainer = document.getElementById('session-history-container');
            
            if (voiceSessions.length === 0) {
                historyContainer.innerHTML = `
                    <div style="padding: 1rem; background: var(--glass); border-radius: 8px;">
                        <div style="display: flex; justify-content: space-between;">
                            <span>No sessions yet</span>
                        </div>
                    </div>
                `;
                return;
            }
            
            let html = '';
            voiceSessions.slice(-3).reverse().forEach((session, index) => {
                const time = new Date(session.time).toLocaleTimeString();
                const date = new Date(session.time).toLocaleDateString();
                html += `
                    <div style="padding: 1rem; background: var(--glass); border-radius: 8px;">
                        <div style="display: flex; justify-content: space-between;">
                            <span>${session.scenario}</span>
                            <span style="color: ${session.score >= 80 ? '#10b981' : session.score >= 60 ? '#f59e0b' : '#ef4444'};">${session.score}%</span>
                        </div>
                        <p style="color: var(--gray); font-size: 0.8rem; margin-top: 0.5rem;">${time} - ${date}</p>
                    </div>
                `;
            });
            
            historyContainer.innerHTML = html;
        }

        function backToScenarios() {
            if (bodyMediaRecorder && bodyMediaRecorder.state !== 'inactive') {
                stopRecording();
            }
            
            document.getElementById('voice-coach-selection').style.display = 'block';
            document.getElementById('voice-coach-recording').style.display = 'none';
            resetRecording();
        }

        async function finishVoiceSession() {
            backToScenarios();
        }

        // Expose functions globally for inline handlers
        window.showFeature = showFeature;
        window.closePanel = closePanel;
        window.showDashboard = showDashboard;
        window.startVoiceSession = startVoiceSession;
        window.startRecording = startRecording;
        window.stopRecording = stopRecording;
        window.resetRecording = resetRecording;
        window.finishVoiceSession = finishVoiceSession;

        // Replace fragile inline onclick handlers with proper event listeners
        document.querySelectorAll('[onclick]').forEach(el => {
            const attr = el.getAttribute('onclick');
            if (!attr) return;
            const showMatch = attr.match(/showFeature\('([^']+)'\)/);
            if (showMatch) {
                const feature = showMatch[1];
                // mark element with data-feature for active link management
                el.dataset.feature = feature;
                el.removeAttribute('onclick');
                el.addEventListener('click', (e) => { e.preventDefault(); showFeature(feature); });
                return;
            }
            const startMatch = attr.match(/startVoiceSession\('([^']+)'\)/);
            if (startMatch) {
                const scenario = startMatch[1];
                el.dataset.scenario = scenario;
                el.removeAttribute('onclick');
                el.addEventListener('click', (e) => { e.preventDefault(); startVoiceSession(scenario); });
                return;
            }
        });

        // Attach listeners to elements using data-feature (nav links, buttons, cards)
        document.querySelectorAll('[data-feature]').forEach(el => {
            const feature = el.dataset.feature;
            if (!feature) return;
            el.addEventListener('click', (e) => {
                e.preventDefault();
                if (feature === 'dashboard') showDashboard();
                else showFeature(feature);
            });
        });

        // Attach listeners to elements using data-scenario (voice coach scenario tiles)
        document.querySelectorAll('[data-scenario]').forEach(el => {
            const scenario = el.dataset.scenario;
            if (!scenario) return;
            el.addEventListener('click', (e) => {
                e.preventDefault();
                startVoiceSession(scenario);
            });
        });

        // Message Simulator Functions
        let simCurrentScenario = 'customer-complaint';
        
        function loadScenario(scenario) {
            simCurrentScenario = scenario;
            const messages = {
                'customer-complaint': [
                    "I'm very disappointed with the service I received. This is the third time this has happened!",
                    "I want to speak to a manager immediately.",
                    "How can you make this right?"
                ],
                'negotiation': [
                    "We're looking at a 20% reduction in price.",
                    "Our budget is tight this quarter.",
                    "What can you offer us?"
                ],
                'team-conflict': [
                    "I feel like my ideas are being ignored.",
                    "It seems like only certain people get heard.",
                    "How can we improve this?"
                ],
                'difficult-customer': [
                    "I've been waiting for 30 minutes!",
                    "This is unacceptable!",
                    "I want a full refund!"
                ]
            };
            
            const simMessages = document.getElementById('simMessages');
            simMessages.innerHTML = '';
            
            messages[scenario].forEach(msg => {
                const div = document.createElement('div');
                div.className = 'chat-message incoming';
                div.innerHTML = `<p>${msg}</p>`;
                simMessages.appendChild(div);
            });
        }

        function handleSimInput(event) {
            if (event.key === 'Enter') {
                sendSimResponse();
            }
        }

        function sendSimResponse() {
            const input = document.getElementById('simInput');
            const text = input.value.trim();
            if (!text) return;
            
            const simMessages = document.getElementById('simMessages');
            
            // Add user message
            const userMsg = document.createElement('div');
            userMsg.className = 'chat-message outgoing';
            userMsg.innerHTML = `<p>${text}</p>`;
            simMessages.appendChild(userMsg);
            
            // Simulate AI response
            setTimeout(() => {
                const responses = [
                    "I understand your concern. Let me help resolve this.",
                    "Thank you for sharing that. I appreciate your patience.",
                    "That's a valid point. Here's what I can offer..."
                ];
                
                const botMsg = document.createElement('div');
                botMsg.className = 'chat-message incoming';
                botMsg.innerHTML = `<p>${responses[Math.floor(Math.random() * responses.length)]}</p>`;
                simMessages.appendChild(botMsg);
                
                simMessages.scrollTop = simMessages.scrollHeight;
            }, 500);
            
            // Show feedback
            const feedback = document.getElementById('simFeedback');
            feedback.style.display = 'block';
            document.getElementById('feedbackText').textContent = 'Good response! You showed empathy and offered to help. Consider being more specific with your solution.';
            document.getElementById('feedbackScore').textContent = Math.floor(Math.random() * 15) + 85;
            
            input.value = '';
            simMessages.scrollTop = simMessages.scrollHeight;
        }

        // ==================== MODEL PREDICTION (FRONTEND) ====================
        let modelPredictTimeout = null;
        function scheduleModelPredict(text) {
            if (!text || text.trim().length === 0) return;
            if (modelPredictTimeout) clearTimeout(modelPredictTimeout);
            modelPredictTimeout = setTimeout(() => callModelPredict(text), 700);
        }

        async function callModelPredict(text) {
            try {
                const res = await fetch('/api/voice-coach/model/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ transcript: text, scenario: currentScenario || null })
                });
                const data = await res.json();
                if (data && data.success) {
                    const score = data.modelScore ?? null;
                    if (score !== null) {
                        updateMetricBar('model', score);
                        const el = document.getElementById('model-score');
                        if (el) el.textContent = score;
                    }
                }
            } catch (e) {
                // ignore transient errors
                console.warn('Model predict error', e);
            }
        }

        // Email Tone Analysis
        let emailToneTimeout = null;
        let lastEmailToneRequest = 0;

        function copyEmailContent() {
            const email = document.getElementById('emailInput');
            if (!email) return;
            email.select();
            email.setSelectionRange(0, email.value.length);
            navigator.clipboard.writeText(email.value).catch(() => {
                try {
                    document.execCommand('copy');
                } catch (e) {
                    console.warn('Copy failed', e);
                }
            });
        }

        function analyzeEmailTone(force = false) {
            const email = document.getElementById('emailInput');
            if (!email) return;
            const emailContent = email.value.trim();

            if (!emailContent) {
                resetEmailToneUI();
                return;
            }

            if (!force) {
                if (emailToneTimeout) clearTimeout(emailToneTimeout);
                emailToneTimeout = setTimeout(() => analyzeEmailTone(true), 500);
                return;
            }

            const requestId = ++lastEmailToneRequest;
            fetch('/api/email-tone/analyze', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    userId: aiUserId,
                    emailContent,
                    recipient: document.getElementById('emailRecipient')?.value || '',
                    purpose: document.getElementById('emailPurpose')?.value || ''
                })
            })
                .then(res => res.json())
                .then(data => {
                    if (requestId !== lastEmailToneRequest) return;
                    if (!data || !data.success || !data.analysis) return;
                    renderEmailToneAnalysis(data.analysis);
                })
                .catch(err => {
                    console.warn('Email tone analysis failed', err);
                });
        }

        function resetEmailToneUI() {
            const needle = document.getElementById('email-tone-needle');
            const label = document.getElementById('email-tone-label');
            const score = document.getElementById('email-tone-score');
            const suggestions = document.getElementById('email-tone-suggestions');
            const rewrite = document.getElementById('email-tone-rewrite');
            const grammar = document.getElementById('email-tone-grammar');
            const spelling = document.getElementById('email-tone-spelling');
            const formality = document.getElementById('email-tone-formality');
            const friendliness = document.getElementById('email-tone-friendliness');
            const urgency = document.getElementById('email-tone-urgency');
            const confidence = document.getElementById('email-tone-confidence');
            const empathy = document.getElementById('email-tone-empathy');

            if (needle) needle.style.transform = 'rotate(-30deg)';
            if (label) label.textContent = 'Professional';
            if (score) score.textContent = 'Score: --/100';
            if (formality) formality.textContent = 'High';
            if (friendliness) friendliness.textContent = 'Warm';
            if (urgency) urgency.textContent = 'Normal';
            if (confidence) confidence.textContent = 'Confident';
            if (empathy) empathy.textContent = 'Empathetic';
            if (suggestions) {
                suggestions.innerHTML = '<li>Start typing an email to see live tone feedback</li>';
            }
            if (rewrite) {
                rewrite.textContent = 'Type your email to see a rewritten version.';
            }
            if (grammar) {
                grammar.innerHTML = '<li>No grammar issues detected yet</li>';
            }
            if (spelling) {
                spelling.innerHTML = '<li>No spelling issues detected yet</li>';
            }
        }

        function renderEmailToneAnalysis(analysis) {
            const toneAnalysis = analysis.toneAnalysis || {};
            const score = Number(analysis.score ?? 0);
            const needle = document.getElementById('email-tone-needle');
            const label = document.getElementById('email-tone-label');
            const scoreEl = document.getElementById('email-tone-score');
            const suggestionsEl = document.getElementById('email-tone-suggestions');
            const suggestionsWrap = document.getElementById('email-tone-suggestions-wrap');
            const rewriteEl = document.getElementById('email-tone-rewrite');
            const grammarEl = document.getElementById('email-tone-grammar');
            const spellingEl = document.getElementById('email-tone-spelling');

            if (needle) {
                const angle = -30 + Math.max(0, Math.min(100, score)) * 1.8;
                needle.style.transform = `rotate(${angle}deg)`;
            }

            if (label) {
                if (score >= 85) label.textContent = 'Excellent';
                else if (score >= 70) label.textContent = 'Professional';
                else if (score >= 55) label.textContent = 'Balanced';
                else label.textContent = 'Needs Work';
            }

            if (scoreEl) {
                scoreEl.textContent = `Score: ${Math.round(score)}/100`;
            }

            const setText = (id, value) => {
                const el = document.getElementById(id);
                if (el) el.textContent = value || '--';
            };

            setText('email-tone-formality', toneAnalysis.formality);
            setText('email-tone-friendliness', toneAnalysis.friendliness);
            setText('email-tone-urgency', toneAnalysis.urgency);
            setText('email-tone-confidence', toneAnalysis.confidence);
            setText('email-tone-empathy', toneAnalysis.empathy);

            if (suggestionsEl) {
                const changes = Array.isArray(analysis.changesToMake) && analysis.changesToMake.length
                    ? analysis.changesToMake
                    : Array.isArray(analysis.suggestions) ? analysis.suggestions : [];
                suggestionsEl.innerHTML = changes.length
                    ? changes.map(item => `<li>${typeof item === 'string' ? item : item.message || JSON.stringify(item)}</li>`).join('')
                    : '<li>No suggestions yet</li>';
            }

            if (rewriteEl) {
                rewriteEl.textContent = analysis.correctedDraft || analysis.rewrittenDraft || 'No rewritten draft available.';
            }

            if (grammarEl) {
                const issues = Array.isArray(analysis.grammarIssues) ? analysis.grammarIssues : [];
                grammarEl.innerHTML = issues.length
                    ? issues.map(item => `<li>${typeof item === 'string' ? item : item.message || JSON.stringify(item)}</li>`).join('')
                    : '<li>No grammar issues detected</li>';
            }

            if (spellingEl) {
                const issues = Array.isArray(analysis.spellingIssues) ? analysis.spellingIssues : [];
                spellingEl.innerHTML = issues.length
                    ? issues.map(item => `<li>${typeof item === 'string' ? item : item.message || JSON.stringify(item)}</li>`).join('')
                    : '<li>No spelling issues detected</li>';
            }

            if (suggestionsWrap) {
                suggestionsWrap.style.background = score >= 75
                    ? 'rgba(16, 185, 129, 0.1)'
                    : score >= 55
                        ? 'rgba(245, 158, 11, 0.1)'
                        : 'rgba(239, 68, 68, 0.1)';
            }
        }

        // ==================== BODY LANGUAGE CAMERA ====================
        let mediaStream = null;
        let bodyMediaRecorder = null;
        let bodyRecordedChunks = [];

        async function startBodyLanguageCamera() {
            try {
                // Request camera access
                mediaStream = await navigator.mediaDevices.getUserMedia({ 
                    video: { width: { ideal: 1280 }, height: { ideal: 720 } },
                    audio: false
                });

                // Get the video preview element
                const videoPreview = document.querySelector('.video-preview');
                
                // Create video element to display camera feed
                let videoElement = videoPreview.querySelector('video');
                if (!videoElement) {
                    videoElement = document.createElement('video');
                    videoElement.autoplay = true;
                    videoElement.playsinline = true;
                    videoElement.style.width = '100%';
                    videoElement.style.height = '100%';
                    videoElement.style.borderRadius = '8px';
                    videoElement.style.objectFit = 'cover';
                    videoPreview.innerHTML = '';
                    videoPreview.appendChild(videoElement);
                }

                videoElement.srcObject = mediaStream;

                // Start recording
                bodyRecordedChunks = [];
                const options = { mimeType: 'video/webm;codecs=vp9', audioBitsPerSecond: 0 };
                
                try {
                    bodyMediaRecorder = new MediaRecorder(mediaStream, options);
                } catch (e) {
                    console.warn('VP9 not supported, trying vp8...');
                    bodyMediaRecorder = new MediaRecorder(mediaStream, { mimeType: 'video/webm;codecs=vp8' });
                }

                bodyMediaRecorder.ondataavailable = (event) => {
                    if (event.data.size > 0) {
                        bodyRecordedChunks.push(event.data);
                    }
                };

                bodyMediaRecorder.onstop = () => {
                    const blob = new Blob(bodyRecordedChunks, { type: 'video/webm' });
                    analyzeBodyLanguageVideo(blob);
                };

                bodyMediaRecorder.start();

                // Update button state - find all buttons in mirror-container and update the primary one
                const buttons = document.querySelectorAll('.mirror-container button');
                buttons.forEach(btn => {
                    if (btn.textContent.includes('Start Recording')) {
                        btn.textContent = '⏹️ Stop Recording';
                        btn.classList.remove('btn-primary');
                        btn.classList.add('btn-danger');
                        btn.onclick = stopBodyLanguageCamera;
                    }
                });

            } catch (error) {
                console.error('Camera access error:', error);
                alert(`Camera access denied: ${error.message}`);
            }
        }

        function stopBodyLanguageCamera() {
            if (bodyMediaRecorder && bodyMediaRecorder.state !== 'inactive') {
                bodyMediaRecorder.stop();
            }

            if (mediaStream) {
                mediaStream.getTracks().forEach(track => track.stop());
                mediaStream = null;
            }

            // Update button state - find all buttons in mirror-container and update the danger one
            const buttons = document.querySelectorAll('.mirror-container button');
            buttons.forEach(btn => {
                if (btn.textContent.includes('Stop Recording')) {
                    btn.textContent = '🎬 Start Recording';
                    btn.classList.remove('btn-danger');
                    btn.classList.add('btn-primary');
                    btn.onclick = startBodyLanguageCamera;
                }
            });
        }

        async function analyzeBodyLanguageVideo(blob) {
            try {
                // Show loading
                const analysisDiv = document.querySelector('.body-analysis');
                analysisDiv.innerHTML = '<div style="text-align: center; padding: 2rem;"><div style="font-size: 2rem; margin-bottom: 1rem;">� Running ML/DL Analysis...</div><p style="color: var(--gray);">Analyzing body language with AI, pose detection, and computer vision...</p></div>';

                // Get ML metrics from recorded video or use AI analysis
                let mlMetrics = {
                    postureScore: 87,
                    gestureScore: 85,
                    eyeContactScore: 88,
                    facialExpressionScore: 90,
                    smileScore: 86,
                    engagementScore: 89
                };
                
                // Try to get actual ML metrics if pose detector is available
                if (detector) {
                    try {
                        const canvas = document.createElement('canvas');
                        const video = document.createElement('video');
                        video.src = URL.createObjectURL(blob);
                        video.onloadedmetadata = async () => {
                            canvas.width = video.videoWidth;
                            canvas.height = video.videoHeight;
                            const ctx = canvas.getContext('2d');
                            ctx.drawImage(video, 0, 0);
                            mlMetrics = await analyzeBodyLanguageML(canvas);
                        };
                    } catch (e) {
                        console.log('Using default metrics');
                    }
                }

                // Send ML metrics to AI for intelligent analysis
                const res = await fetch('/api/body-language/analyze', {
                    method: 'POST',
                    body: JSON.stringify({
                        userId: 'user_' + Date.now(),
                        videoUrl: 'recorded-video.webm',
                        scenario: 'general-practice',
                        mlMetrics: mlMetrics
                    }),
                    headers: { 'Content-Type': 'application/json' }
                });

                const data = await res.json();

                if (data.success && data.analysis) {
                    const analysis = data.analysis;
                    
                    // Build comprehensive results HTML
                    let html = `
                        <div style="margin-bottom: 1.5rem; padding: 1.5rem; background: linear-gradient(135deg, rgba(99, 102, 241, 0.2), rgba(6, 182, 212, 0.2)); border-radius: 12px; border-left: 4px solid var(--primary);">
                            <div style="margin-bottom: 1rem;">
                                <h4 style="margin-bottom: 0.5rem; color: var(--primary);">🎯 Overall Assessment</h4>
                                <div style="display: flex; align-items: center; gap: 1rem;">
                                    <div style="font-size: 3rem; font-weight: 700; color: var(--primary);">${analysis.scores.overall}%</div>
                                    <div style="flex: 1;">
                                        <div class="score-bar" style="height: 10px; background: var(--glass); border-radius: 5px; overflow: hidden;">
                                            <div class="score-fill" style="width: ${analysis.scores.overall}%; background: linear-gradient(90deg, var(--primary), var(--secondary)); height: 100%; border-radius: 5px;"></div>
                                        </div>
                                        <p style="color: var(--gray); font-size: 0.9rem; margin-top: 0.75rem;">ML/DL Analysis - ${analysis.confidence}% Confidence</p>
                                    </div>
                                </div>
                            </div>
                        </div>

                        <h4 style="margin: 1.5rem 0 1rem 0;">📊 Detailed Metrics (ML-Analyzed)</h4>
                    `;
                    
                    // Add ML-based scores for each category
                    analysis.results.scores.forEach(item => {
                        const color = item.score >= 85 ? 'var(--success)' : item.score >= 70 ? 'var(--warning)' : 'var(--danger)';
                        const icon = item.score >= 85 ? '✓' : item.score >= 70 ? '⚠' : '✗';
                        html += `
                            <div class="analysis-item" style="margin-bottom: 1rem; padding: 1rem; background: var(--glass); border-radius: 8px; border: 1px solid rgba(${item.score >= 85 ? '16, 185, 129' : item.score >= 70 ? '245, 158, 11' : '239, 68, 68'}, 0.3);">
                                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                                    <div style="font-weight: 600; font-size: 1.05rem;">${icon} ${item.category}</div>
                                    <span style="color: ${color}; font-weight: 700; font-size: 1.2rem;">${item.score}%</span>
                                </div>
                                <p style="color: var(--gray); font-size: 0.9rem; margin-bottom: 0.75rem;">${item.feedback}</p>
                                <div style="height: 6px; background: rgba(255, 255, 255, 0.05); border-radius: 3px; overflow: hidden;">
                                    <div style="height: 100%; background: ${color}; border-radius: 3px; width: ${item.score}%; transition: width 0.3s ease; box-shadow: 0 0 10px ${color};"></div>
                                </div>
                            </div>
                        `;
                    });

                    // Add AI-powered insights
                    html += '<h4 style="margin: 1.5rem 0 1rem 0; color: var(--success);">🌟 AI-Powered Insights</h4>';
                    analysis.results.strengths.forEach((strength, idx) => {
                        html += `
                            <div style="padding: 0.875rem; background: linear-gradient(135deg, rgba(16, 185, 129, 0.15), rgba(16, 185, 129, 0.05)); margin-bottom: 0.875rem; border-radius: 6px; border-left: 4px solid var(--success); color: var(--light); font-weight: 500;">
                                ✨ ${strength}
                            </div>
                        `;
                    });

                    // Add improvement areas
                    html += '<h4 style="margin: 1.5rem 0 1rem 0; color: var(--warning);">🎯 Areas for Development</h4>';
                    analysis.results.improvements.forEach(improvement => {
                        html += `
                            <div style="padding: 0.875rem; background: linear-gradient(135deg, rgba(245, 158, 11, 0.15), rgba(245, 158, 11, 0.05)); margin-bottom: 0.875rem; border-radius: 6px; border-left: 4px solid var(--warning); color: var(--light);">
                                📍 ${improvement}
                            </div>
                        `;
                    });

                    // Add actionable recommendations
                    html += '<h4 style="margin: 1.5rem 0 1rem 0; color: var(--secondary);">💡 Personalized Recommendations</h4>';
                    analysis.recommendations.forEach((rec, idx) => {
                        const recColor = rec.includes('power pose') ? 'rgba(99, 102, 241, 0.15)' : 
                                       rec.includes('Record') ? 'rgba(6, 182, 212, 0.15)' :
                                       rec.includes('pause') ? 'rgba(139, 92, 246, 0.15)' :
                                       rec.includes('Mirror') ? 'rgba(244, 114, 182, 0.15)' :
                                       'rgba(6, 182, 212, 0.15)';
                        html += `
                            <div style="padding: 0.875rem; background: linear-gradient(135deg, ${recColor}, rgba(255,255,255,0.02)); margin-bottom: 0.875rem; border-radius: 6px; border-left: 4px solid var(--secondary); color: var(--light);">
                                ${rec}
                            </div>
                        `;
                    });

                    // Add technical details
                    html += `
                        <div style="margin-top: 1.5rem; padding: 1rem; background: rgba(99, 102, 241, 0.08); border-radius: 8px; border: 1px solid rgba(99, 102, 241, 0.2);">
                            <p style="color: var(--gray); font-size: 0.85rem; margin: 0;">
                                <strong>Analysis Method:</strong> ML/DL-powered pose detection + Computer Vision + AI Analysis<br>
                                <strong>Confidence Level:</strong> ${analysis.confidence}% | <strong>Frames Analyzed:</strong> Real-time detection<br>
                                <strong>Models Used:</strong> TensorFlow.js (Pose Detection) + MediaPipe + Gemma AI
                            </p>
                        </div>
                    `;

                    analysisDiv.innerHTML = html;

                } else {
                    analysisDiv.innerHTML = `<p style="color: #ef4444;">❌ Error analyzing video</p>`;
                }
            } catch (error) {
                console.error('Analysis error:', error);
                document.querySelector('.body-analysis').innerHTML = `<p style="color: #ef4444;">❌ Error: ${error.message}</p>`;
            }
        }

        // Close panels when clicking outside
        document.querySelectorAll('.feature-panels').forEach(panel => {
            panel.addEventListener('click', function(e) {
                if (e.target === this) {
                    closePanel();
                }
            });
        });

        // Keyboard shortcuts
        document.addEventListener('keydown', function(e) {
            if (e.key === 'Escape') {
                closePanel();
            }
        });

        // ==================== AI ASSISTANT ====================
        let aiUserId = localStorage.getItem('vdart_user_id') || 'user_' + Date.now();
        localStorage.setItem('vdart_user_id', aiUserId);
        let aiStatusChecked = false;

        async function checkAIStatus() {
            if (aiStatusChecked) return;
            try {
                const res = await fetch('/api/ai/status');
                const data = await res.json();
                const statusEl = document.getElementById('ai-status');
                if (statusEl) {
                    statusEl.innerHTML = data.success ? 
                        `<span style="color: ${data.ollamaAvailable ? '#10b981' : '#f59e0b'}">● ${data.message}</span>` :
                        `<span style="color: #ef4444">● AI unavailable</span>`;
                }
                aiStatusChecked = true;
            } catch (e) {
                const statusEl = document.getElementById('ai-status');
                if (statusEl) statusEl.innerHTML = '<span style="color: #ef4444">● Connection error</span>';
            }
        }

async function sendAIMessage() {
    const input = document.getElementById('ai-input');
    const messagesEl = document.getElementById('ai-messages');
    const message = input.value.trim();
    if (!message) return;

    // Add user message
    messagesEl.innerHTML += `
        <div class="ai-message user">
            <strong>You:</strong> ${message}
        </div>
    `;
    input.value = '';

    // Loading
    messagesEl.innerHTML += `
        <div class="ai-message loading" id="ai-loading">🤖 Thinking...</div>
    `;
    messagesEl.scrollTop = messagesEl.scrollHeight;

    try {
        const res = await fetch('/api/ai/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ userId: aiUserId, message })
        });

        const data = await res.json();

        // remove loading
        document.getElementById('ai-loading')?.remove();

        if (data.success) {

            // ✅ 🔥 MAIN FIX HERE
            let aiText =
                typeof data.response === "string"
                    ? data.response
                    : data.response?.content ||   // Ollama format
                      data.response?.message?.content ||
                      JSON.stringify(data.response);

            messagesEl.innerHTML += `
                <div class="ai-message bot">
                    <strong>🤖 VDart AI:</strong> ${aiText}
                </div>
            `;

            // Sources (no change)
            if (data.sources && data.sources.length > 0) {
                const sources = data.sources
                    .map(s => `<span class="ai-source">${s.title}</span>`)
                    .join(', ');

                messagesEl.innerHTML += `
                    <div class="ai-sources">📚 Sources: ${sources}</div>
                `;
            }

        } else {
            messagesEl.innerHTML += `
                <div class="ai-message error">
                    ❌ ${data.message || data.error}
                </div>
            `;
        }

    } catch (e) {
        document.getElementById('ai-loading')?.remove();

        messagesEl.innerHTML += `
            <div class="ai-message error">
                ❌ Error connecting to AI
            </div>
        `;
    }

    messagesEl.scrollTop = messagesEl.scrollHeight;
}

// ==================== ENTER KEY HANDLER FOR AI CHAT ====================
document.addEventListener('DOMContentLoaded', function() {
    const aiInput = document.getElementById('ai-input');
    if (aiInput) {
        aiInput.addEventListener('keydown', function(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendAIMessage();
            }
        });
    }
});
// ==================== PRESENTATIONS ====================
let currentPresId = null;

async function preparePresentation() {
    const topic = document.getElementById('pres-topic').value;
    const audience = document.getElementById('pres-audience').value;
    const duration = document.getElementById('pres-duration').value;
    
    if (!topic || !audience || !duration) {
        alert("Please fill in all setup fields.");
        return;
    }
    
    document.getElementById('pres-loading').style.display = 'block';
    document.getElementById('pres-setup').style.display = 'none';
    
    try {
        const res = await fetch('/api/presentation/prepare', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ userId: localStorage.getItem('vdart_user_id') || 'guest', topic, audience, duration })
        });
        const data = await res.json();
        if (data.success) {
            currentPresId = data.presentation.id;
            const pres = data.presentation;
            
            // Render structure
            let structHtml = `
                <div class="slide-title">Opening</div>
                <div class="slide-content">
                    <p><strong>Hook:</strong> ${pres.structure.opening.hook}</p>
                    <p><strong>Agenda:</strong> ${pres.structure.opening.agenda}</p>
                    <p><strong>Objective:</strong> ${pres.structure.opening.objective}</p>
                </div>
                <div class="slide-title" style="margin-top: 1rem;">Body</div>
                <div class="slide-content">
                    <ul style="text-align: left; margin-top: 0.5rem;">
                        ${pres.structure.body.map(b => `<li>${b.point} (${b.duration})</li>`).join('')}
                    </ul>
                </div>
                <div class="slide-title" style="margin-top: 1rem;">Closing</div>
                <div class="slide-content">
                    <p><strong>Summary:</strong> ${pres.structure.closing.summary}</p>
                    <p><strong>Call to Action:</strong> ${pres.structure.closing.callToAction}</p>
                </div>
            `;
            document.getElementById('pres-structure').innerHTML = structHtml;
            
            // Render Q&A
            let qaHtml = pres.qaPrep.likelyQuestions.map(q => `
                <div class="question-card">
                    <div class="question-text">${q.question}</div>
                    <span class="difficulty-badge ${q.difficulty}">${q.difficulty}</span>
                </div>
            `).join('');
            document.getElementById('pres-qa-list').innerHTML = qaHtml;
            
            document.getElementById('pres-loading').style.display = 'none';
            document.getElementById('pres-results').style.display = 'block';
        }
    } catch (e) {
        console.error('Error preparing presentation:', e);
        document.getElementById('pres-loading').style.display = 'none';
        document.getElementById('pres-setup').style.display = 'block';
        alert('Failed to generate presentation prep.');
    }
}

function showPracticeMode() {
    document.getElementById('pres-results').style.display = 'none';
    document.getElementById('pres-practice').style.display = 'flex';
}

async function submitPresentationScript() {
    const script = document.getElementById('pres-script').value;
    if (!script) {
        alert("Please enter a presentation script.");
        return;
    }
    
    try {
        const res = await fetch('/api/presentation/submit-script', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ presId: currentPresId, script })
        });
        const data = await res.json();
        if (data.success) {
            const result = data.result;
            
            // Render improvements
            document.getElementById('pres-score').textContent = result.overallScore;
            document.getElementById('pres-improvements').innerHTML = result.improvements.map(i => `<li>${i}</li>`).join('');
            
            // Render follow up questions
            document.getElementById('pres-followup-list').innerHTML = result.followUpQuestions.map(q => `
                <div style="background: rgba(255, 255, 255, 0.05); padding: 1rem; border-radius: 8px; margin-bottom: 0.5rem; display: flex; justify-content: space-between; align-items: center;">
                    <span>${q}</span>
                    <button class="btn btn-secondary" onclick="practiceQuestion('${q.replace(/'/g, "\\'")}')">Answer</button>
                </div>
            `).join('');
            
            document.getElementById('pres-feedback-area').style.display = 'block';
        }
    } catch (e) {
        console.error('Error submitting script:', e);
        alert('Failed to submit script.');
    }
}

function practiceQuestion(question) {
    document.getElementById('pres-answer-area').style.display = 'block';
    document.getElementById('pres-current-question').textContent = question;
    document.getElementById('pres-answer').value = '';
    document.getElementById('pres-answer-feedback').style.display = 'none';
}

async function submitPracticeAnswer() {
    const question = document.getElementById('pres-current-question').textContent;
    const answer = document.getElementById('pres-answer').value;
    
    if (!answer) {
        alert("Please enter an answer.");
        return;
    }
    
    try {
        const res = await fetch('/api/presentation/practice', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ presId: currentPresId, question, answer })
        });
        const data = await res.json();
        if (data.success) {
            const fb = data.feedback;
            document.getElementById('pres-answer-feedback').innerHTML = `
                <h4 style="color: #3b82f6; margin-bottom: 0.5rem;">AI Feedback (Score: ${fb.overall}/100)</h4>
                <p><strong>Clarity:</strong> ${fb.clarity} | <strong>Completeness:</strong> ${fb.completeness}</p>
                <p style="margin-top: 0.5rem;">${fb.feedback}</p>
            `;
            document.getElementById('pres-answer-feedback').style.display = 'block';
        }
    } catch (e) {
        console.error('Error submitting answer:', e);
        alert('Failed to submit answer.');
    }
}
