/**
 * AI Service - Unified Hub (Groq Cloud & Local Ollama/Gemma)
 * Uses high-speed Groq (Llama 3.3 70B) as primary, falls back to local Gemma 2B.
 */

require('dotenv').config();
const https = require('https');
const http = require('http');

// ==================== CONFIG ====================
let GROQ_API_KEY = process.env.GROQ_API_KEY || '';
const GROQ_MODEL = 'llama-3.3-70b-versatile';

const OLLAMA_HOST = 'localhost';
const OLLAMA_PORT = 11434;
const OLLAMA_MODEL = 'gemma:2b'; 

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
            return reject(new Error('Groq API Key missing'));
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

// ==================== OLLAMA API CALL ====================
function callOllama(messages) {
    return new Promise((resolve, reject) => {
        const data = JSON.stringify({
            model: OLLAMA_MODEL,
            messages: messages,
            stream: false
        });

        const options = {
            hostname: OLLAMA_HOST,
            port: OLLAMA_PORT,
            path: '/api/chat',
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Content-Length': Buffer.byteLength(data)
            }
        };

        const req = http.request(options, (res) => {
            let body = '';
            res.on('data', chunk => body += chunk);
            res.on('end', () => {
                try {
                    const result = JSON.parse(body);
                    const text = result.message?.content || "No response";
                    resolve(text);
                } catch (e) {
                    reject(new Error('Invalid Ollama response'));
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
        { role: "system", content: "You are VDart AI Assistant, a professional communication coach." },
        ...history,
        { role: "user", content: userMessage }
    ];

    try {
        // Try Groq first
        if (GROQ_API_KEY) {
            const response = await callGroq(messages);
            saveToHistory(userId, userMessage, response);
            return { success: true, response, sources: [], provider: 'Groq' };
        }
        throw new Error('No Groq API Key');
    } catch (error) {
        console.log("Switching to local Ollama/Gemma...");
        try {
            // Fallback to local Ollama
            const response = await callOllama(messages);
            saveToHistory(userId, userMessage, response);
            return { success: true, response, sources: [], provider: 'Ollama' };
        } catch (ollamaError) {
            if (FALLBACK_AI_ENABLED) {
                return { success: true, response: generateFallbackResponse(userMessage), fallback: true, provider: 'Static' };
            }
            return { success: false, error: ollamaError.message };
        }
    }
}

function saveToHistory(userId, userMessage, aiResponse) {
    const history = conversationHistory.get(userId) || [];
    const newHistory = [...history, { role: 'user', content: userMessage }, { role: 'assistant', content: aiResponse }];
    conversationHistory.set(userId, newHistory.slice(-10)); // Keep last 10 messages
}

function generateFallbackResponse(userMessage) {
    const msg = (userMessage || '').toLowerCase();
    const patterns = [
        { test: ['hello', 'hi'], reply: "Hello! I'm your VDart assistant. How can I help?" },
        { test: ['accent'], reply: "Focus on 'Vowel Clarity'. Practice recording your speech." }
    ];
    for (const pattern of patterns) {
        if (pattern.test.some(t => msg.includes(t))) return pattern.reply;
    }
    return "I'm in local fallback mode. Please check if your Groq API Key or local Ollama is active.";
}

// ==================== STATUS & HELPER ====================
async function checkOllamaStatus() {
    if (GROQ_API_KEY) return { available: true, cloud: true, provider: 'Groq' };
    
    // Check local Ollama
    return new Promise((resolve) => {
        const req = http.request({ hostname: OLLAMA_HOST, port: OLLAMA_PORT, path: '/api/tags', method: 'GET' }, (res) => {
            resolve({ available: true, cloud: false, provider: 'Ollama' });
        });
        req.on('error', () => resolve({ available: false, provider: 'None' }));
        req.end();
    });
}

async function callOllamaPrompt(prompt) {
    const messages = [{ role: 'user', content: prompt }];
    try {
        if (GROQ_API_KEY) return await callGroq(messages);
        return await callOllama(messages);
    } catch (e) {
        return await callOllama(messages);
    }
}

function parseJSONSafe(text) {
    try { return JSON.parse(text); } catch { return null; }
}

module.exports = {
    generateAIResponse,
    checkOllamaStatus,
    callOllamaPrompt,
    parseJSONSafe,
    getModelName: () => GROQ_API_KEY ? GROQ_MODEL : OLLAMA_MODEL,
    getConversationHistory: (userId) => conversationHistory.get(userId) || [],
    clearConversationHistory: (userId) => conversationHistory.delete(userId),
    vectorStore: { documents: companyDocs }
};
