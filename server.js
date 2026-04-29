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

app.use(express.json());
app.use(express.static(path.join(__dirname, 'public')));

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

app.post('/api/presentation/prepare', (req, res) => {
    const { userId, topic, audience, duration } = req.body;
    const presId = uuidv4();
    const presentation = {
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
    presentationData.set(presId, presentation);
    res.json({ success: true, presentation });
});

app.post('/api/presentation/practice', (req, res) => {
    const { presId, question, answer } = req.body;
    const pres = presentationData.get(presId);
    if (pres) {
        const feedback = evaluateAnswer(question, answer);
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

app.post('/api/suggestions/analyze', (req, res) => {
    const { userId, context, recentCommunications } = req.body;
    const suggestions = generateSmartSuggestions(context, recentCommunications);
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

app.post('/api/email-tone/analyze', (req, res) => {
    const { userId, emailContent, recipient, purpose } = req.body;
    const analysis = {
        id: uuidv4(),
        userId,
        emailContent,
        recipient,
        purpose,
        toneAnalysis: analyzeEmailTone(emailContent),
        suggestions: generateEmailSuggestions(emailContent, recipient, purpose),
        score: Math.floor(Math.random() * 20) + 80,
        createdAt: new Date()
    };
    emailAnalysis.set(analysis.id, analysis);
    res.json({ success: true, analysis });
});

function analyzeEmailTone(emailContent) {
    const words = emailContent.toLowerCase();
    return {
        formality: words.includes('kindly') || words.includes('respectfully') ? 'formal' : 'casual',
        friendliness: words.includes('thanks') || words.includes('appreciate') ? 'warm' : 'neutral',
        urgency: words.includes('asap') || words.includes('urgent') ? 'high' : 'normal',
        confidence: words.includes('will') || words.includes('can') ? 'confident' : 'tentative',
        empathy: words.includes('understand') || words.includes('sorry') ? 'empathetic' : 'neutral'
    };
}

function generateEmailSuggestions(emailContent, recipient, purpose) {
    const suggestions = [];
    if (emailContent.length > 500) {
        suggestions.push({ type: 'length', message: 'Consider shortening - keep under 200 words for faster response' });
    }
    if (!emailContent.includes('?')) {
        suggestions.push({ type: 'engagement', message: 'Add a clear question or call to action' });
    }
    if (!emailContent.includes('thanks') && !emailContent.includes('thank')) {
        suggestions.push({ type: 'politeness', message: 'Add a thank you or appreciation' });
    }
    if (purpose === 'request' && !emailContent.includes('please')) {
        suggestions.push({ type: 'tone', message: 'Add "please" for a more polite request' });
    }
    return suggestions;
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
app.post('/api/translate/realtime', (req, res) => {
    const { text, sourceLang, targetLang, context } = req.body;
    res.json({
        success: true,
        translation: simulateTranslation(text, targetLang),
        alternatives: [
            { text: simulateTranslation(text, targetLang), style: 'formal' },
            { text: simulateTranslation(text, targetLang), style: 'casual' }
        ],
        culturalNotes: getCulturalNotes(targetLang, context)
    });
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
app.post('/api/meeting/intelligence', (req, res) => {
    const { userId, meetingId, transcript, participants } = req.body;
    const intelligence = {
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
    res.json({
        success: true,
        ollamaAvailable: status.available,
        llama3Installed: status.hasLlama3,
        model: 'llama3',
        message: status.available 
            ? (status.hasLlama3 ? 'AI Assistant ready!' : 'Ollama running but llama3 not installed')
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
app.post('/api/coach/ai', (req, res) => {
    const { userId, query, context } = req.body;
    const coachResponse = generateCoachResponse(query, context);
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