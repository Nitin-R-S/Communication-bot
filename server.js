const express = require('express');
const http = require('http');
const { Server } = require('socket.io');
const path = require('path');
const { v4: uuidv4 } = require('uuid');

// AI Service - Llama 3 with RAG and Memory
const aiService = require('./ai-service');

const app = express();
const server = http.createServer(app);
const io = new Server(server);
const fs = require('fs');

app.use(express.json());
app.use(express.static(path.join(__dirname, 'public')));

app.use((err, req, res, next) => {
    if (err && err instanceof SyntaxError && 'body' in err) {
        return res.status(400).json({ success: false, message: 'Invalid JSON payload' });
    }
    next(err);
});

// In-memory data stores
const users = new Map();
const sessions = new Map();
const communicationDNA = new Map();
const callRecordings = new Map();
const messageSimulations = new Map();
const emailAnalysis = new Map();
const bodyLanguageData = new Map();
const presentationData = new Map();
const accentScores = new Map();
const voiceCoaching = new Map();

// Simple linear regression model for voice scoring (persisted to disk)
const MODEL_PATH = path.join(__dirname, 'voice_model.json');
let voiceModel = {
    // weights: Array (including bias at index 0)
    weights: null,
    featureNames: []
};

function saveModel() {
    try {
        // sanitize model before saving: replace non-finite weights with 0
        const safeModel = JSON.parse(JSON.stringify(voiceModel));
        if (Array.isArray(safeModel.weights)) {
            safeModel.weights = safeModel.weights.map(w => (Number.isFinite(w) ? w : 0));
        }
        if (safeModel.normParams && safeModel.normParams.mean) {
            for (const k of Object.keys(safeModel.normParams.mean)) safeModel.normParams.mean[k] = Number(safeModel.normParams.mean[k] || 0);
            for (const k of Object.keys(safeModel.normParams.std)) safeModel.normParams.std[k] = Number(safeModel.normParams.std[k] || 1);
        }
        fs.writeFileSync(MODEL_PATH, JSON.stringify(safeModel, null, 2));
    } catch (e) {
        console.error('Failed to save model:', e.message);
    }
}

function loadModel() {
    try {
        if (fs.existsSync(MODEL_PATH)) {
            const raw = fs.readFileSync(MODEL_PATH, 'utf8');
            voiceModel = JSON.parse(raw);
        }
    } catch (e) {
        console.error('Failed to load model:', e.message);
    }
}

loadModel();

function extractFeaturesFromAnalysis(analysis) {
    // Use a set of stable numeric features derived from analysis
    // Order matters and must match voiceModel.featureNames when saved
    const f = {
        wordCount: analysis.wordCount || 0,
        uniqueWords: analysis.uniqueWords || 0,
        sentenceCount: analysis.sentenceCount || 0,
        clarityScore: analysis.clarityScore || 0,
        paceScore: analysis.paceScore || 0,
        toneScore: analysis.toneScore || 0,
        confidenceScore: analysis.confidenceScore || 0,
        fillerWordCount: analysis.fillerWordCount || 0,
        estimatedWPM: analysis.estimatedWPM || 0
    };
    return f;
}

function featuresToVector(features, featureNames) {
    // include bias term at index 0
    const vec = [1];
    for (const name of featureNames) {
        vec.push(Number(features[name] || 0));
    }
    return vec;
}

function predictWithModel(analysis) {
    if (!voiceModel.weights || !voiceModel.featureNames || voiceModel.weights.length === 0) return null;
    const features = extractFeaturesFromAnalysis(analysis);
    // Apply normalization if available
    let x = [];
    x.push(1); // bias
    const means = (voiceModel.normParams && voiceModel.normParams.mean) || {};
    const stds = (voiceModel.normParams && voiceModel.normParams.std) || {};
    for (const name of voiceModel.featureNames) {
        let v = Number(features[name] || 0);
        const mean = means[name] || 0;
        const std = stds[name] || 1;
        const norm = std > 0 ? (v - mean) / std : 0;
        x.push(norm);
    }
    let y = 0;
    for (let i = 0; i < voiceModel.weights.length; i++) {
        const w = Number(voiceModel.weights[i] || 0);
        const xi = Number(x[i] || 0);
        y += w * xi;
    }
    // If we stored a bias scaling, return directly clamped
    return Math.max(0, Math.min(100, Math.round(y)));
}

function trainLinearModelFromExamples(examples, options = {}) {
    // Use normalized gradient descent for stability
    if (!examples || examples.length === 0) return null;
    const featureNames = Object.keys(extractFeaturesFromAnalysis(examples[0].analysis));
    const m = examples.length;

    // Build raw feature matrix (without bias)
    const rawX = examples.map(e => {
        const f = extractFeaturesFromAnalysis(e.analysis);
        return featureNames.map(nm => Number(f[nm] || 0));
    });
    const y = examples.map(e => Number(e.label));

    const p = featureNames.length; // number of features

    // compute mean/std
    const mean = Array(p).fill(0);
    const std = Array(p).fill(0);
    for (let j = 0; j < p; j++) {
        let sum = 0;
        for (let i = 0; i < m; i++) sum += rawX[i][j];
        mean[j] = sum / m;
    }
    for (let j = 0; j < p; j++) {
        let acc = 0;
        for (let i = 0; i < m; i++) acc += Math.pow(rawX[i][j] - mean[j], 2);
        std[j] = Math.sqrt(acc / m) || 1;
    }

    // normalize and build X with bias
    const X = rawX.map(row => [1, ...row.map((v, j) => (v - mean[j]) / std[j])]);
    const n = X[0].length; // p + 1

    // initialize weights
    let weights = new Array(n).fill(0);

    const lr = options.lr || 0.05;
    const epochs = options.epochs || 1000;

    for (let epoch = 0; epoch < epochs; epoch++) {
        const preds = X.map(row => row.reduce((s, xi, k) => s + xi * weights[k], 0));
        const errors = preds.map((pred, i) => pred - y[i]);
        const grads = new Array(n).fill(0);
        for (let j = 0; j < n; j++) {
            let g = 0;
            for (let i = 0; i < m; i++) g += errors[i] * X[i][j];
            grads[j] = (2 / m) * g;
        }
        for (let j = 0; j < n; j++) weights[j] -= lr * grads[j];
    }
    // debug: inspect data ranges and NaNs
    try {
        console.log('trainLinearModelFromExamples: sample rawX[0]=', rawX[0]);
        console.log('trainLinearModelFromExamples: sample normalized X[0]=', X[0]);
        console.log('trainLinearModelFromExamples: sample y[0]=', y[0]);
        const nanCount = weights.filter(w => !Number.isFinite(w)).length;
        console.log('trainLinearModelFromExamples: nan/inf in weights before sanitize =', nanCount);
    } catch (e) {
        console.error('Debug inspect failed:', e.message || e);
    }

    // sanitize
    weights = weights.map(v => (Number.isFinite(v) ? v : 0));

    console.log('trainLinearModelFromExamples: first weights sample ->', weights.slice(0, 10));

    voiceModel = { weights, featureNames, normParams: { mean: Object.fromEntries(featureNames.map((n,i)=>[n,mean[i]])), std: Object.fromEntries(featureNames.map((n,i)=>[n,std[i]])) } };
    saveModel();
    return voiceModel;
}

// Generate synthetic examples to seed/train the model quickly
function generateSyntheticExamples(count = 200) {
    const examples = [];
    for (let i = 0; i < count; i++) {
        const wordCount = Math.floor(Math.random() * 180) + 20; // 20-200
        const uniqueWords = Math.max(5, Math.floor(wordCount * (0.4 + Math.random() * 0.6)));
        const sentenceCount = Math.max(1, Math.floor(wordCount / (10 + Math.floor(Math.random() * 15))));
        const clarityScore = Math.max(50, Math.min(100, Math.floor(40 + Math.random() * 60)));
        const paceScore = Math.max(50, Math.min(100, Math.floor(100 - Math.abs((wordCount / Math.max(1, sentenceCount)) - 150) / 2)));
        const toneScore = Math.max(50, Math.min(100, Math.floor(50 + Math.random() * 50)));
        const confidenceScore = Math.max(50, Math.min(100, Math.floor(50 + Math.random() * 50)));
        const fillerWordCount = Math.floor(Math.random() * 8);
        const estimatedWPM = Math.max(60, Math.min(220, Math.floor((wordCount / Math.max(1, Math.floor(wordCount / 140))) )));

        const analysis = {
            wordCount,
            uniqueWords,
            sentenceCount,
            clarityScore,
            paceScore,
            toneScore,
            confidenceScore,
            fillerWordCount,
            estimatedWPM
        };

        // Label is a weighted combination + noise
        const label = Math.max(0, Math.min(100, Math.round(
            (clarityScore * 0.35) + (paceScore * 0.25) + (confidenceScore * 0.25) + (toneScore * 0.15) + (Math.random() * 6 - 3)
        )));

        examples.push({ analysis, label });
    }
    return examples;
}

app.post('/api/voice-coach/model/seed-train', (req, res) => {
    const { count } = req.body || {};
    const examples = generateSyntheticExamples(count || 300);
    const model = trainLinearModelFromExamples(examples, { lr: 0.0005, epochs: 2500 });
    res.json({ success: true, trained: !!model, model });
});

function parseJSONSafe(text) {
    try {
        return JSON.parse(text);
    } catch {
        return null;
    }
}

async function aiAnalyzePrompt(prompt, fallback) {
    try {
        const timeoutMs = 2500;
        const result = await Promise.race([
            aiService.callOllamaPrompt(prompt),
            new Promise((_, reject) => setTimeout(() => reject(new Error('AI timeout')), timeoutMs))
        ]);
        const parsed = parseJSONSafe(result);
        return parsed || fallback;
    } catch (error) {
        console.error('AI analysis failed:', error.message);
        return fallback;
    }
}

// ==================== CORE API ENDPOINTS ====================

// User Authentication
app.post('/api/auth/register', (req, res) => {
    const { name, email, role, department } = req.body;
    const userId = uuidv4();
    const user = { id: userId, name, email, role, department, createdAt: new Date() };
    users.set(userId, user);
    res.json({ success: true, user });
});

app.post('/api/auth/login', (req, res) => {
    const { email } = req.body;
    const user = Array.from(users.values()).find(u => u.email === email);
    if (user) {
        res.json({ success: true, user });
    } else {
        res.json({ success: false, message: 'User not found' });
    }
});

// ==================== LIVE VOICE COACH ====================

app.post('/api/voice-coach/start', (req, res) => {
    const { userId, scenario } = req.body;
    const sessionId = uuidv4();
    const coachingSession = {
        id: sessionId,
        userId,
        scenario,
        startTime: new Date(),
        metrics: {
            clarityScore: 0,
            paceScore: 0,
            toneScore: 0,
            confidenceScore: 0,
            fillerWordCount: 0,
            pauses: [],
            pitchVariations: [],
            volumeConsistency: 0
        },
        realTimeFeedback: [],
        status: 'active'
    };
    voiceCoaching.set(sessionId, coachingSession);
    res.json({ success: true, session: coachingSession });
});

app.post('/api/voice-coach/feedback', (req, res) => {
    const { sessionId, audioData, transcript } = req.body;
    const session = voiceCoaching.get(sessionId);
    if (session) {
        // Simulate real-time analysis
        const feedback = analyzeVoiceMetrics(transcript, audioData);
        session.realTimeFeedback.push(feedback);
        session.metrics = { ...session.metrics, ...feedback.metrics };
        res.json({ success: true, feedback });
    } else {
        res.json({ success: false, message: 'Session not found' });
    }
});

app.get('/api/voice-coach/session/:sessionId', (req, res) => {
    const session = voiceCoaching.get(req.params.sessionId);
    res.json({ success: true, session });
});

// ==================== VOICE COACH ANALYSIS ====================
app.post('/api/voice-coach/analyze', (req, res) => {
    const { sessionId, transcript, scenario } = req.body;
    
    if (!transcript || transcript.trim().length === 0) {
        return res.json({ success: false, message: 'Empty transcript' });
    }
    
    const analysis = analyzeVoicePerformance(transcript, scenario);

    // If a trained model exists, add model prediction
    const modelPred = predictWithModel(analysis);
    if (modelPred !== null) {
        analysis.modelScore = modelPred;
    }
    
    if (sessionId) {
        const session = voiceCoaching.get(sessionId);
        if (session) {
            session.analysis = analysis;
            session.transcript = transcript;
            session.status = 'completed';
        }
    }
    
    res.json({ success: true, analysis });
});

// ==================== MODEL TRAINING & PREDICTION ENDPOINTS ====================
// Train model with labeled examples: { examples: [{ transcript, scenario, label }] }
app.post('/api/voice-coach/model/train', (req, res) => {
    const { examples, options } = req.body;
    if (!examples || !Array.isArray(examples) || examples.length === 0) {
        return res.json({ success: false, message: 'No training examples provided' });
    }

    // Convert each example transcript -> analysis
    const processed = examples.map(ex => {
        const analysis = analyzeVoicePerformance(ex.transcript || '', ex.scenario || '');
        return { analysis, label: Number(ex.label) };
    });

    const model = trainLinearModelFromExamples(processed, options || {});
    res.json({ success: true, model });
});

app.get('/api/voice-coach/model/status', (req, res) => {
    res.json({ success: true, model: voiceModel, hasModel: !!voiceModel.weights });
});

app.post('/api/voice-coach/model/predict', (req, res) => {
    const { transcript, scenario } = req.body;
    if (!transcript) return res.json({ success: false, message: 'No transcript provided' });
    const analysis = analyzeVoicePerformance(transcript, scenario);
    const modelPred = predictWithModel(analysis);
    // If model not trained, fallback to simple aggregate score so frontend shows live feedback
    let finalScore = null;
    if (modelPred !== null) {
        finalScore = modelPred;
    } else {
        // average of core metrics as fallback
        finalScore = Math.round((analysis.clarityScore + analysis.paceScore + analysis.confidenceScore + analysis.toneScore) / 4);
    }
    res.json({ success: true, analysis, modelScore: finalScore, modelUsed: modelPred !== null });
});

function analyzeVoicePerformance(transcript, scenario) {
    const words = transcript.split(/\s+/).filter(w => w.length > 0);
    const sentences = transcript.split(/[.!?]+/).filter(s => s.trim().length > 0);
    
    // Filler words analysis
    const fillerWords = ['um', 'uh', 'like', 'you know', 'basically', 'actually', 'literally', 'sort of', 'kind of', 'i mean'];
    let fillerCount = 0;
    words.forEach(word => {
        const lower = word.toLowerCase().replace(/[,.:;!?]/g, '');
        if (fillerWords.includes(lower)) {
            fillerCount++;
        }
    });
    
    // Calculate clarity score based on word variety and structure
    const uniqueWords = new Set(words.map(w => w.toLowerCase()));
    const vocabularyScore = (uniqueWords.size / words.length) * 100;
    const sentenceLength = words.length / Math.max(sentences.length, 1);
    const clarityScore = Math.round(
        (Math.min(vocabularyScore, 100) * 0.4) +
        (Math.min((sentenceLength / 25) * 100, 100) * 0.6)
    );
    
    // Pace calculation (assuming 1-2 minutes of speech, estimate WPM)
    const estimatedWPM = words.length > 50 ? words.length : words.length * 2;
    const optimalWPM = 150;
    let paceScore = 100 - Math.abs(estimatedWPM - optimalWPM) / optimalWPM * 50;
    paceScore = Math.max(60, Math.min(100, paceScore));
    
    // Tone analysis based on punctuation and word choice
    const questionMarks = (transcript.match(/\?/g) || []).length;
    const exclamations = (transcript.match(/!/g) || []).length;
    const positiveWords = ['great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'love', 'perfect', 'confident', 'professional'];
    let positiveCount = 0;
    words.forEach(word => {
        const lower = word.toLowerCase().replace(/[,.:;!?]/g, '');
        if (positiveWords.includes(lower)) {
            positiveCount++;
        }
    });
    
    const toneScore = Math.round(
        (positiveCount / words.length) * 50 +
        ((questionMarks + exclamations) / sentences.length) * 30 +
        50
    );
    
    // Confidence score
    const confidenceMarkers = ['confident', 'professional', 'definitely', 'absolutely', 'certainly'];
    let confidenceMarkerCount = 0;
    words.forEach(word => {
        const lower = word.toLowerCase().replace(/[,.:;!?]/g, '');
        if (confidenceMarkers.includes(lower)) {
            confidenceMarkerCount++;
        }
    });
    
    let confidenceScore = Math.round(
        (1 - (fillerCount / words.length)) * 50 + 
        (confidenceMarkerCount / words.length) * 30 +
        (vocabularyScore / 100) * 20
    );
    confidenceScore = Math.max(60, Math.min(100, confidenceScore));
    
    // Scenario-specific scoring adjustments
    const scenarioBonus = getScenarioBonus(transcript, scenario);
    
    return {
        wordCount: words.length,
        sentenceCount: sentences.length,
        clarityScore: Math.max(65, Math.min(100, clarityScore + scenarioBonus.clarity)),
        paceScore: Math.max(60, Math.min(100, paceScore)),
        toneScore: Math.max(70, Math.min(100, toneScore)),
        confidenceScore: Math.max(65, Math.min(100, confidenceScore)),
        fillerWordCount: fillerCount,
        estimatedWPM: estimatedWPM,
        uniqueWords: uniqueWords.size,
        suggestions: generateDetailedSuggestions(
            transcript,
            fillerCount,
            words.length,
            clarityScore,
            paceScore,
            scenario
        )
    };
}

function getScenarioBonus(transcript, scenario) {
    const lowerTranscript = transcript.toLowerCase();
    let clarityBonus = 0;
    
    switch(scenario) {
        case 'sales':
            // Check for sales-specific language
            const salesWords = ['product', 'solution', 'benefit', 'invest', 'opportunity', 'customer', 'value'];
            const salesCount = salesWords.filter(w => lowerTranscript.includes(w)).length;
            clarityBonus = (salesCount / salesWords.length) * 15;
            break;
        case 'support':
            // Check for empathetic language
            const supportWords = ['understand', 'help', 'sorry', 'appreciate', 'problem', 'solution', 'thank'];
            const supportCount = supportWords.filter(w => lowerTranscript.includes(w)).length;
            clarityBonus = (supportCount / supportWords.length) * 15;
            break;
        case 'presentation':
            // Check for structured presentation language
            const presentationWords = ['first', 'second', 'third', 'summary', 'conclusion', 'data', 'analysis'];
            const presentationCount = presentationWords.filter(w => lowerTranscript.includes(w)).length;
            clarityBonus = (presentationCount / presentationWords.length) * 15;
            break;
        case 'meeting':
            // Check for professional introduction
            const meetingWords = ['team', 'role', 'responsibility', 'experience', 'collaborate', 'professional'];
            const meetingCount = meetingWords.filter(w => lowerTranscript.includes(w)).length;
            clarityBonus = (meetingCount / meetingWords.length) * 15;
            break;
    }
    
    return { clarity: Math.max(0, clarityBonus) };
}

function generateDetailedSuggestions(transcript, fillerCount, wordCount, clarityScore, paceScore, scenario) {
    const suggestions = [];
    const fillerRatio = fillerCount / wordCount;
    
    // Filler words suggestions
    if (fillerCount > 5) {
        suggestions.push(`Reduce filler words (${fillerCount} found). Practice pausing instead of saying "um" or "uh".`);
    } else if (fillerCount > 0) {
        suggestions.push(`Good job! You used ${fillerCount} filler word(s). Keep working to eliminate them completely.`);
    } else {
        suggestions.push('Excellent! No filler words detected. Your speech is very clean.');
    }
    
    // Clarity suggestions
    if (clarityScore < 70) {
        suggestions.push('Work on clarity: speak more slowly and enunciate each word clearly.');
    } else if (clarityScore < 85) {
        suggestions.push('Your clarity is good. Try varying your sentence structure for more impact.');
    }
    
    // Pace suggestions
    if (paceScore < 70) {
        suggestions.push('Adjust your pace - you\'re speaking too fast. Aim for 140-160 words per minute.');
    } else if (paceScore >= 85) {
        suggestions.push('Perfect pace! You\'re maintaining an ideal speaking speed.');
    }
    
    // Scenario-specific suggestions
    if (scenario === 'sales') {
        if (transcript.toLowerCase().includes('benefit') || transcript.toLowerCase().includes('value')) {
            suggestions.push('Great! You emphasized the value proposition.');
        } else {
            suggestions.push('Consider emphasizing the benefits and value of the product.');
        }
    } else if (scenario === 'support') {
        if (transcript.toLowerCase().includes('understand') || transcript.toLowerCase().includes('help')) {
            suggestions.push('Good empathy shown. Maintain this customer-centric approach.');
        } else {
            suggestions.push('Try to show more empathy when handling customer concerns.');
        }
    } else if (scenario === 'presentation') {
        if (transcript.match(/first|second|third|finally/i)) {
            suggestions.push('Good structure! You used clear transitions between points.');
        } else {
            suggestions.push('Try structuring your presentation with clear transitions like "first," "second," etc.');
        }
    }
    
    // Add word count feedback
    if (wordCount < 30) {
        suggestions.push('Try speaking for longer. Aim for at least 60-90 seconds of content.');
    }
    
    return suggestions;
}

function analyzeVoiceMetrics(transcript, audioData) {
    const words = transcript.split(' ');
    const fillerWords = ['um', 'uh', 'like', 'you know', 'basically', 'actually', 'literally'];
    const fillerCount = words.filter(w => fillerWords.includes(w.toLowerCase())).length;
    
    return {
        timestamp: new Date(),
        transcript,
        metrics: {
            clarityScore: Math.floor(Math.random() * 30) + 70,
            paceScore: Math.floor(Math.random() * 20) + 80,
            toneScore: Math.floor(Math.random() * 25) + 75,
            confidenceScore: Math.floor(Math.random() * 30) + 70,
            fillerWordCount: fillerCount,
            avgWordPace: words.length / 1,
            sentiment: analyzeSentiment(transcript)
        },
        suggestions: generateVoiceSuggestions(fillerCount, words.length)
    };
}

function analyzeSentiment(text) {
    const positive = ['great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'love', 'perfect'];
    const negative = ['bad', 'terrible', 'awful', 'hate', 'worst', 'problem', 'issue'];
    const words = text.toLowerCase().split(' ');
    let score = 0;
    words.forEach(w => { if (positive.includes(w)) score++; if (negative.includes(w)) score--; });
    return score > 0 ? 'positive' : score < 0 ? 'negative' : 'neutral';
}

function generateVoiceSuggestions(fillerCount, totalWords) {
    const suggestions = [];
    const fillerRatio = fillerCount / totalWords;
    if (fillerRatio > 0.05) suggestions.push('Reduce filler words - pause instead');
    if (fillerRatio > 0.1) suggestions.push('Practice the "pause instead of um" technique');
    suggestions.push('Maintain steady pace - aim for 150 words per minute');
    suggestions.push('Project your voice with confidence');
    return suggestions;
}

// ==================== ACCENT CLARITY TRAINER ====================

app.post('/api/accent/train', (req, res) => {
    const { userId, targetAccent, currentAccent } = req.body;
    const trainingId = uuidv4();
    const training = {
        id: trainingId,
        userId,
        targetAccent,
        currentAccent,
        exercises: generateAccentExercises(targetAccent),
        progress: 0,
        scores: [],
        startedAt: new Date()
    };
    accentScores.set(trainingId, training);
    res.json({ success: true, training });
});

app.post('/api/accent/analyze', (req, res) => {
    const { trainingId, audioSample, text } = req.body;
    const training = accentScores.get(trainingId);
    if (training) {
        const analysis = {
            pronunciation: Math.floor(Math.random() * 30) + 70,
            intonation: Math.floor(Math.random() * 25) + 75,
            rhythm: Math.floor(Math.random() * 20) + 80,
            clarity: Math.floor(Math.random() * 25) + 75,
            overall: 0
        };
        analysis.overall = Math.floor((analysis.pronunciation + analysis.intonation + analysis.rhythm + analysis.clarity) / 4);
        training.scores.push(analysis);
        training.progress = (training.scores.length / training.exercises.length) * 100;
        res.json({ success: true, analysis, training });
    } else {
        res.json({ success: false, message: 'Training session not found' });
    }
});

function generateAccentExercises(targetAccent) {
    const exercises = {
        'american': [
            { type: 'vowel', phrase: 'The cat sat on the mat', difficulty: 1 },
            { type: 'consonant', phrase: 'She sells seashells by the seashore', difficulty: 2 },
            { type: 'stress', phrase: 'I want to go to the American restaurant', difficulty: 3 },
            { type: 'linking', phrase: 'What are you going to do?', difficulty: 2 },
            { type: 'reduction', phrase: 'I gotta go to the store', difficulty: 3 }
        ],
        'british': [
            { type: 'vowel', phrase: 'Dance in the garden with aunt', difficulty: 2 },
            { type: 'consonant', phrase: 'The rain in Spain stays mainly in the plain', difficulty: 3 },
            { type: 'stress', phrase: 'I was wondering if you could possibly help', difficulty: 2 },
            { type: 'linking', phrase: 'Not at all, it\'s a pleasure', difficulty: 2 }
        ],
        'neutral': [
            { type: 'clarity', phrase: 'The quick brown fox jumps over the lazy dog', difficulty: 1 },
            { type: 'enunciation', phrase: 'Proper pronunciation requires practice', difficulty: 2 },
            { type: 'breathing', phrase: 'Take a deep breath and speak clearly', difficulty: 1 }
        ]
    };
    return exercises[targetAccent] || exercises['neutral'];
}

// ==================== CALL REPLAY ROOM ====================

app.post('/api/call-replay/record', (req, res) => {
    const { userId, callType, participants, duration } = req.body;
    const recordingId = uuidv4();
    const recording = {
        id: recordingId,
        userId,
        callType,
        participants,
        duration,
        createdAt: new Date(),
        transcript: generateMockTranscript(callType),
        analysis: {
            talkTimeRatio: { user: 60, others: 40 },
            interruptions: Math.floor(Math.random() * 5),
            questionsAsked: Math.floor(Math.random() * 10) + 3,
            activeListening: Math.floor(Math.random() * 20) + 80,
            emotionalTone: ['confident', 'engaged', 'professional'],
            keyMoments: generateKeyMoments()
        },
        tags: [],
        notes: []
    };
    callRecordings.set(recordingId, recording);
    res.json({ success: true, recording });
});

app.get('/api/call-replay/recordings/:userId', (req, res) => {
    const recordings = Array.from(callRecordings.values()).filter(r => r.userId === req.params.userId);
    res.json({ success: true, recordings });
});

app.get('/api/call-replay/recording/:recordingId', (req, res) => {
    const recording = callRecordings.get(req.params.recordingId);
    res.json({ success: true, recording });
});

app.post('/api/call-replay/recording/:recordingId/note', (req, res) => {
    const { content, timestamp, type } = req.body;
    const recording = callRecordings.get(req.params.recordingId);
    if (recording) {
        recording.notes.push({ id: uuidv4(), content, timestamp, type, createdAt: new Date() });
        res.json({ success: true, recording });
    } else {
        res.json({ success: false, message: 'Recording not found' });
    }
});

function generateMockTranscript(callType) {
    const transcripts = {
        'sales': [
            { speaker: 'Agent', text: 'Good morning, thank you for calling. How may I assist you today?', time: 0 },
            { speaker: 'Customer', text: 'Hi, I\'m interested in learning more about your products.', time: 5 },
            { speaker: 'Agent', text: 'Absolutely! I\'d be happy to help. What specific area are you interested in?', time: 8 },
            { speaker: 'Customer', text: 'Well, we\'re looking to upgrade our current system.', time: 12 }
        ],
        'support': [
            { speaker: 'Agent', text: 'Hello, VDart Support. How can I help you?', time: 0 },
            { speaker: 'Customer', text: 'I\'m having an issue with my account login.', time: 3 },
            { speaker: 'Agent', text: 'I understand. Can you tell me what error message you\'re seeing?', time: 6 }
        ],
        'meeting': [
            { speaker: 'Host', text: 'Good afternoon everyone. Let\'s get started.', time: 0 },
            { speaker: 'Host', text: 'Today we\'ll be discussing the Q3 results.', time: 2 },
            { speaker: 'Participant 1', text: 'Thank you. I have a question about the revenue figures.', time: 5 }
        ]
    };
    return transcripts[callType] || transcripts['meeting'];
}

function generateKeyMoments() {
    return [
        { time: 30, type: 'question', description: 'Asked clarifying question', impact: 'positive' },
        { time: 120, type: 'rapport', description: 'Built rapport with customer', impact: 'positive' },
        { time: 180, type: 'objection', description: 'Handled objection well', impact: 'positive' },
        { time: 240, type: 'close', description: 'Successfully closed the call', impact: 'positive' }
    ];
}

// ==================== MESSAGE CONVERSATION SIMULATOR ====================

app.post('/api/simulator/msg-convo', (req, res) => {
    const { userId, scenario, difficulty, role } = req.body;
    const simId = uuidv4();
    const simulation = {
        id: simId,
        userId,
        scenario,
        difficulty,
        role,
        messages: generateScenarioMessages(scenario, role),
        currentIndex: 0,
        responses: [],
        score: 0,
        startedAt: new Date()
    };
    messageSimulations.set(simId, simulation);
    res.json({ success: true, simulation });
});

app.post('/api/simulator/respond', (req, res) => {
    const { simId, response } = req.body;
    const sim = messageSimulations.get(simId);
    if (sim) {
        const currentMessage = sim.messages[sim.currentIndex];
        const evaluation = evaluateResponse(response, currentMessage);
        sim.responses.push({ ...evaluation, response, timestamp: new Date() });
        sim.score += evaluation.score;
        
        if (sim.currentIndex < sim.messages.length - 1) {
            sim.currentIndex++;
            res.json({ success: true, nextMessage: sim.messages[sim.currentIndex], evaluation });
        } else {
            sim.completedAt = new Date();
            res.json({ success: true, completed: true, finalScore: sim.score, evaluation });
        }
    } else {
        res.json({ success: false, message: 'Simulation not found' });
    }
});

function generateScenarioMessages(scenario, role) {
    const scenarios = {
        'customer-complaint': [
            { from: 'customer', text: 'I\'m very disappointed with the service I received.', type: 'emotional' },
            { from: 'customer', text: 'This is the third time this has happened!', type: 'frustrated' },
            { from: 'customer', text: 'I want to speak to a manager immediately.', type: 'demanding' }
        ],
        'negotiation': [
            { from: 'client', text: 'We\'re looking at a 20% reduction in price.', type: 'business' },
            { from: 'client', text: 'Our budget is tight this quarter.', type: 'practical' },
            { from: 'client', text: 'What can you offer us?', type: 'negotiating' }
        ],
        'team-conflict': [
            { from: 'colleague', text: 'I feel like my ideas are being ignored.', type: 'emotional' },
            { from: 'colleague', text: 'It seems like only certain people get heard.', type: 'frustrated' },
            { from: 'colleague', text: 'How can we improve this?', type: 'constructive' }
        ]
    };
    return scenarios[scenario] || scenarios['customer-complaint'];
}

function evaluateResponse(response, context) {
    const keywords = {
        'emotional': ['understand', 'apologize', 'feel', 'sorry'],
        'frustrated': ['apologize', 'resolve', 'fix', 'solution'],
        'demanding': ['manager', 'escalate', 'help', 'immediately'],
        'business': ['value', 'benefit', 'proposal', 'offer'],
        'practical': ['budget', 'cost', 'efficient', 'practical'],
        'negotiating': ['discount', 'deal', 'special', 'offer'],
        'constructive': ['improve', 'together', 'collaborate', 'solution']
    };
    
    const responseLower = response.toLowerCase();
    const contextKeywords = keywords[context.type] || [];
    const matchedKeywords = contextKeywords.filter(k => responseLower.includes(k));
    const score = Math.min(100, 50 + (matchedKeywords.length * 15));
    
    return {
        score,
        matchedKeywords,
        feedback: score > 70 ? 'Excellent response!' : score > 50 ? 'Good, but could be improved.' : 'Consider a different approach.',
        suggestions: getSuggestions(context.type, matchedKeywords)
    };
}

function getSuggestions(contextType, matched) {
    const allSuggestions = {
        'emotional': ['Show empathy', 'Acknowledge feelings', 'Apologize sincerely'],
        'frustrated': ['Offer solutions', 'Take responsibility', 'Propose next steps'],
        'demanding': ['Stay calm', 'Explain process', 'Offer alternatives'],
        'business': ['Focus on value', 'Highlight benefits', 'Be professional'],
        'practical': ['Be direct', 'Provide options', 'Respect constraints'],
        'negotiating': ['Create win-win', 'Show flexibility', 'Understand needs'],
        'constructive': ['Listen actively', 'Encourage collaboration', 'Suggest improvements']
    };
    return allSuggestions[contextType] || [];
}

// ==================== COMMUNICATION DNA SEQUENCER ====================

app.post('/api/dna/sequence', (req, res) => {
    const { userId, samples } = req.body;
    const dnaId = uuidv4();
    const dna = {
        id: dnaId,
        userId,
        sequence: generateCommunicationDNA(samples),
        profile: generateDNAProfile(),
        strengths: [],
        weaknesses: [],
        recommendations: [],
        createdAt: new Date()
    };
    communicationDNA.set(dnaId, dna);
    res.json({ success: true, dna });
});

app.get('/api/dna/profile/:userId', (req, res) => {
    const dna = Array.from(communicationDNA.values()).find(d => d.userId === req.params.userId);
    res.json({ success: true, dna });
});

function generateCommunicationDNA(samples) {
    return {
        verbal: {
            vocabulary: Math.floor(Math.random() * 20) + 80,
            complexity: Math.floor(Math.random() * 15) + 75,
            clarity: Math.floor(Math.random() * 20) + 80,
            flow: Math.floor(Math.random() * 15) + 85
        },
        nonVerbal: {
            eyeContact: Math.floor(Math.random() * 25) + 75,
            gestures: Math.floor(Math.random() * 20) + 80,
            posture: Math.floor(Math.random() * 15) + 85,
            facialExpressions: Math.floor(Math.random() * 20) + 80
        },
        paraverbal: {
            tone: Math.floor(Math.random() * 20) + 80,
            pace: Math.floor(Math.random() * 15) + 85,
            volume: Math.floor(Math.random() * 20) + 80,
            pitch: Math.floor(Math.random() * 15) + 85
        },
        emotional: {
            empathy: Math.floor(Math.random() * 25) + 75,
            selfAwareness: Math.floor(Math.random() * 20) + 80,
            selfRegulation: Math.floor(Math.random() * 15) + 85,
            socialSkills: Math.floor(Math.random() * 20) + 80
        }
    };
}

function generateDNAProfile() {
    const types = ['Analytical', 'Intuitive', 'Functional', 'Social', 'Direct'];
    const selected = types[Math.floor(Math.random() * types.length)];
    return {
        primaryType: selected,
        secondaryType: types[(types.indexOf(selected) + 1) % types.length],
        communicationStyle: `${selected} Communicator`,
        preferredChannels: ['Video Call', 'Email', 'Chat'],
        bestTimeForCommunication: 'Morning',
        stressResponse: 'Becomes more analytical',
        motivationStyle: 'Achievement-based'
    };
}

// ==================== OPENING & PRESENTATIONS Q&A ====================

app.post('/api/presentation/prepare', async (req, res) => {
    const { userId, topic, audience, duration } = req.body;
    const presId = uuidv4();
    const fallback = {
        id: presId,
        userId,
        topic,
        audience,
        duration,
        structure: generatePresentationStructure(topic, audience),
        qaPrep: generateQAPreparation(topic, audience),
        tips: generatePresentationTips(),
        practiceMode: true,
        createdAt: new Date()
    };
    const prompt = `You are a presentation expert. Create a presentation structure, Q&A preparation, and improvement tips for topic: "${topic}" and audience: "${audience}" with duration: "${duration}" minutes. Return JSON with keys structure, qaPrep, and tips.`;
    const aiResult = await aiAnalyzePrompt(prompt, fallback);
    const presentation = typeof aiResult === 'object' ? { ...fallback, ...aiResult, id: presId, userId, topic, audience, duration, practiceMode: true, createdAt: new Date() } : fallback;
    presentationData.set(presId, presentation);
    res.json({ success: true, presentation });
});

app.post('/api/presentation/practice', async (req, res) => {
    const { presId, question, answer } = req.body;
    const pres = presentationData.get(presId);
    if (pres) {
        const fallback = evaluateAnswer(question, answer);
        const prompt = `You are a presentation coach. Evaluate the following answer to the question: "${question}". Answer: "${answer}". Return JSON with clarity, completeness, structure, examples, overall, and feedback.`;
        const feedback = await aiAnalyzePrompt(prompt, fallback);
        res.json({ success: true, feedback });
    } else {
        res.json({ success: false, message: 'Presentation not found' });
    }
});

function generatePresentationStructure(topic, audience) {
    return {
        opening: {
            hook: `Start with a compelling question about ${topic}`,
            agenda: '3 key points to cover',
            objective: 'Clear takeaway for audience'
        },
        body: [
            { point: 'Problem Statement', duration: '20%', keyMessages: [] },
            { point: 'Solution Overview', duration: '50%', keyMessages: [] },
            { point: 'Benefits & Value', duration: '30%', keyMessages: [] }
        ],
        closing: {
            summary: 'Recap key points',
            callToAction: 'Next steps',
            qaInvitation: 'Open for questions'
        }
    };
}

function generateQAPreparation(topic, audience) {
    return {
        likelyQuestions: [
            { question: 'How long will implementation take?', difficulty: 'medium', modelAnswer: '' },
            { question: 'What is the ROI?', difficulty: 'high', modelAnswer: '' },
            { question: 'How does this compare to alternatives?', difficulty: 'medium', modelAnswer: '' },
            { question: 'What are the risks?', difficulty: 'high', modelAnswer: '' },
            { question: 'Can you provide references?', difficulty: 'low', modelAnswer: '' }
        ],
        difficultQuestions: [
            { question: 'What if this fails?', type: 'challenge', strategy: 'Acknowledge concerns, present mitigation' },
            { question: 'Why should we believe you?', type: 'trust', strategy: 'Provide evidence, testimonials' }
        ]
    };
}

function generatePresentationTips() {
    return [
        'Make eye contact with different sections of the audience',
        'Use pauses for emphasis - don\'t rush through key points',
        'Keep slides simple - one idea per slide maximum',
        'Practice the opening 10 times until it feels natural',
        'Have a backup plan for technical difficulties',
        'End with a clear call to action'
    ];
}

function evaluateAnswer(question, answer) {
    const length = answer.split(' ').length;
    const hasStructure = answer.includes('.') || answer.includes(',');
    const hasExamples = answer.toLowerCase().includes('for example') || answer.toLowerCase().includes('such as');
    
    return {
        clarity: Math.floor(Math.random() * 20) + 80,
        completeness: length > 30 ? 90 : length > 15 ? 75 : 60,
        structure: hasStructure ? 85 : 70,
        examples: hasExamples ? 90 : 75,
        overall: Math.floor(Math.random() * 15) + 85,
        feedback: length > 30 ? 'Comprehensive answer' : 'Could be more detailed'
    };
}

// ==================== BODY LANGUAGE MIRROR ====================

app.post('/api/body-language/analyze', (req, res) => {
    const { userId, videoUrl, scenario } = req.body;
    const analysisId = uuidv4();
    const analysis = {
        id: analysisId,
        userId,
        scenario,
        videoUrl,
        results: analyzeBodyLanguage(scenario),
        scores: {
            posture: Math.floor(Math.random() * 20) + 80,
            gestures: Math.floor(Math.random() * 25) + 75,
            eyeContact: Math.floor(Math.random() * 20) + 80,
            facialExpressions: Math.floor(Math.random() * 15) + 85,
            overall: 0
        },
        recommendations: [],
        createdAt: new Date()
    };
    analysis.results.scores.forEach(s => analysis.scores.overall += s.score);
    analysis.scores.overall = Math.floor(analysis.scores.overall / analysis.results.scores.length);
    bodyLanguageData.set(analysisId, analysis);
    res.json({ success: true, analysis });
});

function analyzeBodyLanguage(scenario) {
    return {
        scores: [
            { category: 'Posture', score: Math.floor(Math.random() * 20) + 80, feedback: 'Good upright posture' },
            { category: 'Gestures', score: Math.floor(Math.random() * 25) + 75, feedback: 'Natural hand movements' },
            { category: 'Eye Contact', score: Math.floor(Math.random() * 20) + 80, feedback: 'Appropriate eye contact' },
            { category: 'Facial Expressions', score: Math.floor(Math.random() * 15) + 85, feedback: 'Engaging expressions' },
            { category: 'Smile', score: Math.floor(Math.random() * 20) + 80, feedback: 'Warm and approachable' },
            { category: 'Nodding', score: Math.floor(Math.random() * 15) + 85, feedback: 'Shows active listening' }
        ],
        strengths: [
            'Confident stance',
            'Natural gestures',
            'Good eye contact balance'
        ],
        improvements: [
            'Reduce nervous hand movements',
            'More varied facial expressions',
            'Stand more straight'
        ]
    };
}

// ==================== SUGGESTION ENGINE ====================

app.post('/api/suggestions/analyze', async (req, res) => {
    const { userId, context, recentCommunications } = req.body;
    const fallback = generateSmartSuggestions(context, recentCommunications);
    const prompt = `You are a communication coach. Review the context: "${context}" and recent communications: "${JSON.stringify(recentCommunications || [])}". Return a JSON array of up to 5 suggestions with category, suggestion, impact, and context.`;
    const suggestions = await aiAnalyzePrompt(prompt, fallback);
    res.json({ success: true, suggestions });
});

function generateSmartSuggestions(context, recentCommunications) {
    const allSuggestions = [
        {
            category: 'Tone',
            suggestion: 'Consider softening your language for better reception',
            impact: 'high',
            context: 'email'
        },
        {
            category: 'Timing',
            suggestion: 'Best time to send follow-up: Tuesday 10am-12pm',
            impact: 'medium',
            context: 'general'
        },
        {
            category: 'Clarity',
            suggestion: 'Break down complex points into bullet points',
            impact: 'high',
            context: 'presentation'
        },
        {
            category: 'Engagement',
            suggestion: 'Ask more open-ended questions to encourage dialogue',
            impact: 'high',
            context: 'meeting'
        },
        {
            category: 'Persuasion',
            suggestion: 'Use the "FEW" method: Fact, Example, Why it matters',
            impact: 'medium',
            context: 'negotiation'
        },
        {
            category: 'Rapport',
            suggestion: 'Start with personal connection before business',
            impact: 'medium',
            context: 'sales'
        },
        {
            category: 'Confidence',
            suggestion: 'Remove hedge words: "I think" → "I believe"',
            impact: 'high',
            context: 'general'
        },
        {
            category: 'Active Listening',
            suggestion: 'Paraphrase before responding to show understanding',
            impact: 'high',
            context: 'conversation'
        }
    ];
    
    return allSuggestions.filter(s => 
        context.includes(s.context) || context === 'general'
    ).slice(0, 5);
}

// ==================== EMAIL TONE DIALER ====================

app.post('/api/email-tone/analyze', async (req, res) => {
    const { userId, emailContent, recipient, purpose } = req.body;
    const grammar = analyzeEmailGrammar(emailContent || '');
    const spelling = analyzeEmailSpelling(emailContent || '');
    const combinedIssues = [...(grammar.issues || []), ...(spelling.issues || [])];
    const correctedDraftText = spelling.corrected || grammar.corrected || String(emailContent || '');
    const features = extractEmailToneFeatures(emailContent || '', recipient || '', purpose || '', {
        issues: combinedIssues
    });
    const analysis = {
        id: uuidv4(),
        userId,
        emailContent,
        recipient,
        purpose,
        toneAnalysis: analyzeEmailTone(emailContent),
        suggestions: generateEmailSuggestions(emailContent, recipient, purpose),
        changesToMake: generateEmailChanges(emailContent, recipient, purpose),
        grammarIssues: grammar.issues,
        grammarIssueCount: grammar.issues.length,
        spellingIssues: spelling.issues,
        spellingIssueCount: spelling.issues.length,
        correctedDraft: correctedDraftText,
        rewrittenDraft: generateImprovedEmailDraft(emailContent, recipient, purpose, correctedDraftText),
        score: scoreEmailTone(features),
        createdAt: new Date()
    };
    emailAnalysis.set(analysis.id, analysis);
    res.json({ success: true, analysis });
});

function extractEmailToneFeatures(emailContent, recipient, purpose, grammar = { issues: [] }) {
    const content = String(emailContent || '');
    const lower = content.toLowerCase();
    const words = lower.split(/\s+/).filter(Boolean);
    const sentences = content.split(/[.!?]+/).filter(s => s.trim().length > 0);
    const thankCount = (lower.match(/\b(thanks|thank you|appreciate|appreciation)\b/g) || []).length;
    const pleaseCount = (lower.match(/\bplease\b/g) || []).length;
    const questionCount = (content.match(/\?/g) || []).length;
    const actionCount = (lower.match(/\b(review|reply|confirm|send|share|schedule|join|finish|finalize|approve|update|let me know)\b/g) || []).length;
    const urgencyCount = (lower.match(/\b(asap|urgent|immediately|soon|today|tomorrow|deadline)\b/g) || []).length;
    const apologyCount = (lower.match(/\b(sorry|apologize|apologies)\b/g) || []).length;
    const greetingCount = (lower.match(/\b(hi|hello|dear|good morning|good afternoon|hey)\b/g) || []).length;
    const closingCount = (lower.match(/\b(regards|best|sincerely|thanks|thank you)\b/g) || []).length;

    return {
        wordCount: words.length,
        sentenceCount: Math.max(1, sentences.length),
        thankCount,
        pleaseCount,
        questionCount,
        actionCount,
        urgencyCount,
        apologyCount,
        greetingCount,
        closingCount,
        recipientProvided: recipient ? 1 : 0,
        purposeProvided: purpose ? 1 : 0,
        length: content.length,
        grammarIssueCount: Array.isArray(grammar.issues) ? grammar.issues.length : 0
    };
}

function scoreEmailTone(features) {
    let score = 40;
    score += Math.min(10, features.greetingCount * 3.5);
    score += Math.min(12, features.thankCount * 4);
    score += Math.min(10, features.pleaseCount * 2.5);
    score += Math.min(12, features.actionCount * 2.25);
    score += Math.min(8, features.closingCount * 2);
    score += Math.min(6, features.recipientProvided * 6);
    score += Math.min(6, features.purposeProvided * 6);
    score += Math.min(5, features.questionCount * 1.5);
    score += Math.min(6, features.apologyCount * 2);
    score += Math.min(8, Math.max(0, features.wordCount - 20) / 10);
    score -= Math.min(14, features.urgencyCount * 4);
    score -= Math.max(0, Math.floor((features.length - 280) / 25));
    score -= Math.min(18, features.grammarIssueCount * 4.5);
    score -= Math.min(18, (features.spellingIssueCount || 0) * 5);
    return Math.max(0, Math.min(100, Math.round(score)));
}

function analyzeEmailGrammar(emailContent) {
    const content = String(emailContent || '').trim();
    const issues = [];
    if (!content) {
        return { issues, corrected: '' };
    }

    if (/\s{2,}/.test(content)) {
        issues.push({ type: 'spacing', message: 'Remove extra spaces between words.' });
    }
    if (/\bi\b/.test(content)) {
        issues.push({ type: 'capitalization', message: 'Use a capital I when referring to yourself.' });
    }
    if (/[,;:]\s*$/.test(content)) {
        issues.push({ type: 'punctuation', message: 'Avoid ending the message with a hanging punctuation mark.' });
    }

    const normalized = content.replace(/\s+/g, ' ').trim();
    const sentences = normalized.split(/([.!?]+)/).filter(Boolean);
    let rebuilt = '';
    for (let i = 0; i < sentences.length; i += 2) {
        const text = (sentences[i] || '').trim();
        const punct = sentences[i + 1] || '.';
        if (!text) continue;
        const fixedText = text.charAt(0).toUpperCase() + text.slice(1);
        rebuilt += fixedText + punct + ' ';
    }

    let corrected = rebuilt.trim();
    if (!corrected) {
        corrected = normalized;
    }
    if (!/[.!?]$/.test(corrected)) {
        corrected += '.';
    }

    corrected = corrected
        .replace(/\bi\b/g, 'I')
        .replace(/\s+/g, ' ')
        .replace(/\s+([,.!?;:])/g, '$1');

    if (corrected !== content) {
        issues.push({ type: 'rewrite', message: 'Apply grammar cleanup and sentence capitalization.' });
    }

    return { issues, corrected };
}

function analyzeEmailSpelling(emailContent) {
    const content = String(emailContent || '').trim();
    const issues = [];
    if (!content) {
        return { issues, corrected: '' };
    }

    const replacements = [
        ['recieve', 'receive'],
        ['receve', 'receive'],
        ['seperate', 'separate'],
        ['definately', 'definitely'],
        ['occured', 'occurred'],
        ['untill', 'until'],
        ['wich', 'which'],
        ['teh', 'the'],
        ['pleaze', 'please'],
        ['anythng', 'anything'],
        ['adresss', 'address'],
        ['adress', 'address'],
        ['wierd', 'weird'],
        ['enviroment', 'environment'],
        ['acommodate', 'accommodate'],
        ['embarass', 'embarrass'],
        ['restarant', 'restaurant'],
        ['thier', 'their'],
        ['alot', 'a lot'],
        ['infromation', 'information'],
        ['proffesional', 'professional'],
        ['repsonse', 'response'],
        ['finalyse', 'finalize'],
        ['thnaks', 'thanks']
    ];

    let corrected = content;
    replacements.forEach(([wrong, right]) => {
        const regex = new RegExp(`\\b${wrong}\\b`, 'gi');
        if (regex.test(corrected)) {
            issues.push({ type: 'spelling', message: `Correct "${wrong}" to "${right}".` });
            corrected = corrected.replace(regex, right);
        }
    });

    corrected = corrected.replace(/\s+/g, ' ').trim();
    if (corrected && !/[.!?]$/.test(corrected)) {
        corrected += '.';
    }

    if (corrected !== content) {
        issues.push({ type: 'spelling-rewrite', message: 'Apply spelling corrections to improve clarity.' });
    }

    return { issues, corrected };
}

function analyzeEmailTone(emailContent) {
    const words = String(emailContent || '').toLowerCase();
    return {
        formality: words.includes('kindly') || words.includes('respectfully') || words.includes('please') ? 'formal' : 'casual',
        friendliness: words.includes('thanks') || words.includes('thank you') || words.includes('appreciate') ? 'warm' : 'neutral',
        urgency: words.includes('asap') || words.includes('urgent') || words.includes('tomorrow') ? 'high' : 'normal',
        confidence: words.includes('will') || words.includes('can') || words.includes('confirm') ? 'confident' : 'tentative',
        empathy: words.includes('understand') || words.includes('sorry') || words.includes('appreciate') ? 'empathetic' : 'neutral'
    };
}

function generateEmailSuggestions(emailContent, recipient, purpose) {
    const suggestions = [];
    const lower = String(emailContent || '').toLowerCase();
    const grammar = analyzeEmailGrammar(emailContent || '');
    if (emailContent.length > 500) {
        suggestions.push({ type: 'length', message: 'Consider shortening - keep under 200 words for faster response' });
    }
    if (!emailContent.includes('?')) {
        suggestions.push({ type: 'engagement', message: 'Add a clear question or call to action' });
    }
    if (!lower.includes('thanks') && !lower.includes('thank')) {
        suggestions.push({ type: 'politeness', message: 'Add a thank you or appreciation' });
    }
    if ((purpose === 'request' || purpose === 'follow-up' || purpose === 'update') && !lower.includes('please')) {
        suggestions.push({ type: 'tone', message: 'Add "please" for a more polite request' });
    }
    if (!recipient) {
        suggestions.push({ type: 'context', message: 'Add the recipient so the tone can be tuned to the audience' });
    }
    if (!purpose) {
        suggestions.push({ type: 'context', message: 'Add the purpose so the model can optimize the message' });
    }
    grammar.issues.slice(0, 3).forEach(issue => {
        suggestions.push({ type: issue.type, message: issue.message });
    });
    return suggestions;
}

function generateEmailChanges(emailContent, recipient, purpose) {
    const lower = String(emailContent || '').toLowerCase();
    const changes = [];
    const grammar = analyzeEmailGrammar(emailContent || '');

    if (!recipient) {
        changes.push('Specify who the email is for so the tone can match the audience.');
    }
    if (!purpose) {
        changes.push('State the purpose clearly, such as request, update, follow-up, or apology.');
    }
    if (!lower.match(/\b(hi|hello|dear|hey)\b/)) {
        changes.push('Add a short greeting to make the message feel more natural.');
    }
    if (!lower.includes('please')) {
        changes.push('Add "please" to soften requests and make the tone more professional.');
    }
    if (!lower.match(/\b(thank you|thanks|appreciate)\b/)) {
        changes.push('Add appreciation or thanks near the end to improve warmth.');
    }
    if (!emailContent.includes('?')) {
        changes.push('End with one clear call to action or question so the next step is obvious.');
    }
    if (emailContent.length > 450) {
        changes.push('Shorten the email and keep only the details needed for the response.');
    }
    if (!lower.match(/\b(regards|best|sincerely|thank you|thanks)\b/)) {
        changes.push('Add a professional closing, like Best regards or Thanks.');
    }
    grammar.issues.forEach(issue => {
        changes.push(issue.message);
    });

    return changes.length ? changes : ['Tone looks good. You can still make it shorter and more specific for a stronger response.'];
}

function generateImprovedEmailDraft(emailContent, recipient, purpose, correctedDraft) {
    if (correctedDraft && correctedDraft.trim()) {
        return correctedDraft;
    }
    const base = String(emailContent || '').trim();
    if (!base) {
        return 'Hi there,\n\nI hope you are doing well.\n\nBest regards,';
    }

    const greeting = recipient ? `Hi ${recipient},` : 'Hi there,';
    const purposeLine = purpose ? `I am reaching out regarding ${purpose}.` : 'I am reaching out to follow up on the request.';
    const body = base
        .replace(/^hi\b[^\n]*,?/i, '')
        .replace(/^hello\b[^\n]*,?/i, '')
        .replace(/^dear\b[^\n]*,?/i, '')
        .trim();
    const closing = '\n\nBest regards,';
    return `${greeting}\n\n${purposeLine}\n\n${body || 'Please let me know how you would like to proceed.'}${closing}`;
}

// ==================== ROLE-BASED FEATURES ====================

app.get('/api/roles/:userId', (req, res) => {
    const user = users.get(req.params.userId);
    if (user) {
        const roleFeatures = getRoleBasedFeatures(user.role);
        res.json({ success: true, role: user.role, features: roleFeatures });
    } else {
        res.json({ success: false, message: 'User not found' });
    }
});

function getRoleBasedFeatures(role) {
    const features = {
        'sales': {
            name: 'Sales Professional',
            modules: ['Voice Coach', 'Accent Trainer', 'Call Replay', 'Negotiation Simulator', 'Email Tone'],
            tools: ['Pitch Perfect', 'Objection Handler', 'Close Assistant', 'CRM Integrator'],
            metrics: ['Conversion Rate', 'Avg Call Length', 'Close Rate']
        },
        'support': {
            name: 'Customer Support',
            modules: ['Voice Coach', 'Call Replay', 'Email Tone', 'Empathy Trainer', 'Problem Resolver'],
            tools: ['Ticket Analyzer', 'Satisfaction Predictor', 'Knowledge Base Assistant'],
            metrics: ['CSAT Score', 'Resolution Time', 'First Contact Resolution']
        },
        'manager': {
            name: 'Manager/Leader',
            modules: ['Communication DNA', 'Body Language', 'Presentation Coach', 'Team Dynamics'],
            tools: ['Team Analyzer', 'Meeting Optimizer', 'Feedback Generator', '1-on-1 Assistant'],
            metrics: ['Team Engagement', 'Meeting Efficiency', 'Feedback Quality']
        },
        'developer': {
            name: 'Technical Professional',
            modules: ['Technical Communicator', 'Documentation Helper', 'Presentation Builder'],
            tools: ['Code Commentator', 'API Documenter', 'Architecture Explainer'],
            metrics: ['Documentation Score', 'Code Review Clarity']
        },
        'default': {
            name: 'Professional',
            modules: ['Voice Coach', 'Email Tone', 'Communication DNA', 'Suggestion Engine'],
            tools: ['All-in-One Communicator'],
            metrics: ['Communication Score', 'Response Quality']
        }
    };
    return features[role] || features['default'];
}

// ==================== ADDITIONAL INNOVATIVE FEATURES ====================

// Real-time Translation Hub
app.post('/api/translate/realtime', async (req, res) => {
    const { text, sourceLang, targetLang, context } = req.body;
    const fallback = {
        success: true,
        translation: simulateTranslation(text, targetLang),
        alternatives: [
            { text: simulateTranslation(text, targetLang), style: 'formal' },
            { text: simulateTranslation(text, targetLang), style: 'casual' }
        ],
        culturalNotes: getCulturalNotes(targetLang, context)
    };
    const prompt = `You are a multilingual communication expert. Translate the following text from ${sourceLang} to ${targetLang}: "${text}". Provide one formal version, one casual version, and concise cultural notes for ${targetLang}. Return JSON with translation, alternatives, and culturalNotes.`;
    const result = await aiAnalyzePrompt(prompt, fallback);
    res.json({ success: true, ...result });
});

function simulateTranslation(text, targetLang) {
    return `[${targetLang.toUpperCase()} translation of: "${text}"]`;
}

function getCulturalNotes(lang, context) {
    const notes = {
        'japanese': ['Use formal address', 'Avoid direct confrontation', 'Silence is valued'],
        'german': ['Be direct and precise', 'Punctuality is crucial', 'Use titles'],
        'french': ['Use formal greetings', 'Discuss culture briefly', 'Avoid business during meals']
    };
    return notes[lang] || ['Be mindful of local customs'];
}

// Meeting Intelligence
app.post('/api/meeting/intelligence', async (req, res) => {
    const { userId, meetingId, transcript, participants } = req.body;
    const fallback = {
        id: uuidv4(),
        meetingId,
        summary: generateMeetingSummary(transcript),
        actionItems: extractActionItems(transcript),
        sentimentAnalysis: analyzeMeetingSentiment(transcript),
        participantInsights: analyzeParticipants(participants),
        keyTopics: extractTopics(transcript),
        engagementScore: Math.floor(Math.random() * 20) + 80,
        createdAt: new Date()
    };
    const prompt = `You are a meeting intelligence assistant. Based on the transcript: "${transcript}" and participants: "${participants.join(', ')}", return JSON with summary, actionItems, sentimentAnalysis, participantInsights, keyTopics, and engagementScore (0-100).`;
    const intelligence = await aiAnalyzePrompt(prompt, fallback);
    res.json({ success: true, intelligence });
});

function generateMeetingSummary(transcript) {
    return 'Meeting covered key discussion points and resulted in actionable outcomes. Participants showed active engagement.';
}

function extractActionItems(transcript) {
    return [
        { item: 'Follow up with team on action items', owner: 'TBD', due: 'Next week' },
        { item: 'Schedule follow-up meeting', owner: 'TBD', due: 'This week' }
    ];
}

function analyzeMeetingSentiment(transcript) {
    return { positive: 65, neutral: 30, negative: 5 };
}

function analyzeParticipants(participants) {
    return participants.map(p => ({
        name: p,
        talkTime: Math.floor(Math.random() * 30) + 10,
        engagement: Math.floor(Math.random() * 20) + 80,
        sentiment: 'positive'
    }));
}

function extractTopics(transcript) {
    return ['Project Updates', 'Resource Allocation', 'Timeline Review', 'Next Steps'];
}

// ==================== AI CHAT ENDPOINTS (Llama 3 + RAG + Memory) ====================

// Check Ollama status
app.get('/api/ai/status', async (req, res) => {
    const status = await aiService.checkOllamaStatus();
    const modelName = aiService.getModelName();
    res.json({
        success: true,
        ollamaAvailable: status.available,
        llama3Installed: status.hasLlama3,
        model: modelName,
        message: status.available 
            ? (status.hasLlama3 ? `${modelName} ready!` : `${modelName} not installed`) 
            : 'Ollama not running. Install from https://ollama.ai'
    });
});

// Send message to AI
app.post('/api/ai/chat', async (req, res) => {
    const { userId, message } = req.body;
    
    if (!userId || !message) {
        return res.json({ success: false, message: 'userId and message required' });
    }
    
    const result = await aiService.generateAIResponse(userId, message);
    
    if (result.success) {
        res.json({
            success: true,
            response: result.response,
            sources: result.sources
        });
    } else {
        res.json({
            success: false,
            message: result.error,
            fallback: 'AI service unavailable. Make sure Ollama is running with llama3 model.'
        });
    }
});

// Get conversation history
app.get('/api/ai/history/:userId', (req, res) => {
    const history = aiService.getConversationHistory(req.params.userId);
    res.json({ success: true, history });
});

// Clear conversation history
app.delete('/api/ai/history/:userId', (req, res) => {
    aiService.clearConversationHistory(req.params.userId);
    res.json({ success: true, message: 'Conversation history cleared' });
});

// Get knowledge base info
app.get('/api/ai/knowledge', (req, res) => {
    const docs = aiService.vectorStore.documents.map(d => ({ id: d.id, title: d.title, category: d.category }));
    res.json({ success: true, documents: docs, count: docs.length });
});

// Communication Coach AI
app.post('/api/coach/ai', async (req, res) => {
    const { userId, query, context } = req.body;
    const defaultResponse = generateCoachResponse(query, context);
    const prompt = `You are an expert communication coach. Provide concise advice, a technique, and a practical exercise for this user query. Query: "${query}". Context: "${context || 'general'}". Return JSON with keys advice, technique, exercise.`;
    const coachResponse = await aiAnalyzePrompt(prompt, defaultResponse);
    res.json({ success: true, response: coachResponse });
});

function generateCoachResponse(query, context) {
    const responses = {
        'nervous': {
            advice: 'Take a deep breath. Pause before speaking to gather your thoughts.',
            technique: 'Try the 4-7-8 breathing technique before your next conversation.',
            exercise: 'Practice power poses for 2 minutes before important meetings.'
        },
        'unclear': {
            advice: 'Structure your thoughts using the PREP method: Point, Reason, Example, Point.',
            technique: 'Start with your conclusion, then provide supporting details.',
            exercise: 'Practice summarizing complex ideas in 30 seconds.'
        },
        'conflict': {
            advice: 'Use the "I" statements to express feelings without blaming.',
            technique: 'Acknowledge the other perspective before presenting yours.',
            exercise: 'Practice active listening by paraphrasing what you hear.'
        },
        'default': {
            advice: 'Focus on being present and engaged in the conversation.',
            technique: 'Maintain appropriate eye contact and open body language.',
            exercise: 'Practice daily reflection on your communication successes.'
        }
    };
    return responses[context] || responses['default'];
}

// WebSocket for real-time features
io.on('connection', (socket) => {
    console.log('Client connected:', socket.id);
    
    socket.on('join-session', (data) => {
        socket.join(data.sessionId);
    });
    
    socket.on('voice-feedback', (data) => {
        socket.to(data.sessionId).emit('feedback-update', data);
    });
    
    socket.on('typing-indicator', (data) => {
        socket.to(data.sessionId).emit('user-typing', data);
    });
    
    socket.on('disconnect', () => {
        console.log('Client disconnected:', socket.id);
    });
});

// Serve the main HTML file
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

const PORT = process.env.PORT || 3000;
server.listen(PORT, () => {
    console.log(`VDart Communication Hub running on http://localhost:${PORT}`);
});
