/**
 * AI Service - Groq Cloud (Llama 3.3 70B)
 * High-speed, high-intelligence free cloud AI.
 */

require('dotenv').config();
const https = require('https');

// ==================== CONFIG ====================
let GROQ_API_KEY = process.env.GROQ_API_KEY || 'YOUR_GROQ_API_KEY_HERE';
const GROQ_MODEL = 'llama-3.3-70b-versatile';
const FALLBACK_AI_ENABLED = true;

// ==================== MEMORY ====================
const conversationHistory = new Map();

// ==================== COMPANY DOCS ====================
const companyDocs = [
    {
        id: 'welcome',
        title: 'Welcome to VDart',
        content: `VDart Communication Hub provides AI-powered communication tools like voice coaching, accent training, and presentation analysis.`,
        category: 'company'
    }
];

// ==================== GROQ API CALL ====================
function callGroq(messages) {
    return new Promise((resolve, reject) => {
        if (!GROQ_API_KEY || GROQ_API_KEY === 'YOUR_GROQ_API_KEY_HERE') {
            return reject(new Error('Groq API Key missing. Please add it in .env'));
        }

        const data = JSON.stringify({
            model: GROQ_MODEL,
            messages: messages,
            temperature: 0.7,
            max_tokens: 1024
        });

        const options = {
            hostname: 'api.groq.com',
            path: '/openai/v1/chat/completions',
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${GROQ_API_KEY}`,
                'Content-Type': 'application/json',
                'Content-Length': Buffer.byteLength(data)
            }
        };

        const req = https.request(options, (res) => {
            let body = '';
            res.on('data', chunk => body += chunk);
            res.on('end', () => {
                try {
                    const result = JSON.parse(body);
                    if (result.error) return reject(new Error(result.error.message));
                    const text = result.choices?.[0]?.message?.content || "No response";
                    resolve(text);
                } catch (e) {
                    reject(new Error('Invalid Groq response'));
                }
            });
        });

        req.on('error', reject);
        req.write(data);
        req.end();
    });
}

// ==================== MAIN AI FUNCTION ====================
async function generateAIResponse(userId, userMessage) {
    const history = conversationHistory.get(userId) || [];
    
    const messages = [
        { role: "system", content: "You are VDart AI Assistant, a professional communication coach. Be concise and helpful." },
        ...history,
        { role: "user", content: userMessage }
    ];

    try {
        const response = await callGroq(messages);

        // Save history
        const newHistory = [
            ...history,
            { role: 'user', content: userMessage },
            { role: 'assistant', content: response }
        ];
        conversationHistory.set(userId, newHistory);

        return {
            success: true,
            response: response,
            sources: []
        };

    } catch (error) {
        console.error("Groq error:", error.message);

        if (FALLBACK_AI_ENABLED) {
            const fallbackResponse = generateFallbackResponse(userMessage);
            return {
                success: true,
                response: fallbackResponse,
                fallback: true,
                error: error.message
            };
        }

        return { success: false, error: error.message };
    }
}

function generateFallbackResponse(userMessage) {
    const msg = (userMessage || '').toLowerCase();
    
    const patterns = [
        { test: ['hello', 'hi'], reply: "Hello! I'm your VDart assistant. How can I help with your communication today?" },
        { test: ['bad', 'poor'], reply: "Don't worry! Communication is a skill that improves with practice. Try focusing on your pace first." },
        { test: ['accent'], reply: "Focus on 'Vowel Clarity'. Record yourself and listen to how you pronounce vowels." }
    ];

    for (const pattern of patterns) {
        if (pattern.test.some(t => msg.includes(t))) return pattern.reply;
    }

    return "I'm currently in local mode, but I can still help. What specific communication skill would you like to improve?";
}

// ==================== STATUS & HELPER ====================
async function checkOllamaStatus() {
    return { available: GROQ_API_KEY !== 'YOUR_GROQ_API_KEY_HERE', cloud: true };
}

async function callOllamaPrompt(prompt) {
    return callGroq([{ role: 'user', content: prompt }]);
}

function parseJSONSafe(text) {
    try { return JSON.parse(text); } catch { return null; }
}

module.exports = {
    generateAIResponse,
    checkOllamaStatus,
    callOllamaPrompt,
    parseJSONSafe,
    getModelName: () => GROQ_MODEL,
    getConversationHistory: (userId) => conversationHistory.get(userId) || [],
    clearConversationHistory: (userId) => conversationHistory.delete(userId),
    vectorStore: { documents: companyDocs }
};

