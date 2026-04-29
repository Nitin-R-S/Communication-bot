/**
 * AI Service - Gemma 2B with RAG and Memory
 */

const http = require('http');

// ==================== CONFIG ====================
const OLLAMA_HOST = 'localhost';
const OLLAMA_PORT = 11434;
const OLLAMA_MODEL = 'gemma:2b'; // ✅ Using Gemma 2B

// ==================== MEMORY ====================
const conversationHistory = new Map();

// ==================== COMPANY DOCS ====================
const companyDocs = [
    {
        id: 'welcome',
        title: 'Welcome to VDart',
        content: `VDart Communication Hub provides AI-powered communication tools like voice coaching, accent training, and presentation analysis.`,
        category: 'company'
    },
    {
        id: 'services',
        title: 'Services',
        content: `We offer Voice Coach, Accent Trainer, Body Language Analyzer, Presentation Coach, and Email Analyzer.`,
        category: 'services'
    },
    {
        id: 'pricing',
        title: 'Pricing',
        content: `Free tier available. Pro plan costs $29/month. Enterprise pricing is custom.`,
        category: 'pricing'
    }
];

// ==================== SIMPLE SEARCH ====================
function getRelevantContext(query) {
    return companyDocs
        .filter(doc => query.toLowerCase().includes(doc.title.toLowerCase()))
        .map(doc => `[${doc.title}]\n${doc.content}`)
        .join('\n\n');
}

// ==================== OLLAMA CHAT ====================
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
            path: '/api/chat',   // ✅ FIXED
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

                    // ✅ IMPORTANT FIX
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

    const context = getRelevantContext(userMessage);

    const systemPrompt = `
You are VDart AI Assistant.

Use the context below to answer:

${context}

User: ${userMessage}
Assistant:
`;

    try {

        const response = await callOllama([
            { role: "user", content: systemPrompt }
        ]);

        // Save history
        const newHistory = [
            ...history,
            { role: 'user', content: userMessage },
            { role: 'assistant', content: response }
        ];

        conversationHistory.set(userId, newHistory);

        return {
            success: true,
            response: response,   // ✅ ALWAYS STRING
            sources: []
        };

    } catch (error) {

        console.error("Ollama error:", error.message);

        return {
            success: false,
            response: null,
            error: error.message
        };
    }
}

// ==================== STATUS ====================
async function checkOllamaStatus() {
    return new Promise((resolve) => {

        const req = http.request({
            hostname: OLLAMA_HOST,
            port: OLLAMA_PORT,
            path: '/api/tags',
            method: 'GET'
        }, (res) => {

            let body = '';
            res.on('data', chunk => body += chunk);

            res.on('end', () => {
                try {
                    const data = JSON.parse(body);

                    const hasLlama3 = data.models?.some(m => m.name === OLLAMA_MODEL);

                    resolve({
                        available: true,
                        hasLlama3
                    });

                } catch {
                    resolve({ available: false, hasLlama3: false });
                }
            });
        });

        req.on('error', () => resolve({ available: false, hasLlama3: false }));
        req.end();
    });
}

async function callOllamaPrompt(prompt) {
    return callOllama([{ role: 'user', content: prompt }]);
}

function parseJSONSafe(text) {
    try {
        return JSON.parse(text);
    } catch {
        return null;
    }
}

function getModelName() {
    return OLLAMA_MODEL;
}

function getConversationHistory(userId) {
    return conversationHistory.get(userId) || [];
}

function clearConversationHistory(userId) {
    conversationHistory.delete(userId);
}

const vectorStore = {
    documents: companyDocs
};

// ==================== EXPORT ====================
module.exports = {
    generateAIResponse,
    checkOllamaStatus,
    callOllamaPrompt,
    parseJSONSafe,
    getModelName,
    getConversationHistory,
    clearConversationHistory,
    vectorStore
};
