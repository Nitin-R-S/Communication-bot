/**
 * AI Service - Groq Cloud Exclusive
 * Powered by Llama 3.3 70B for high-speed, high-intelligence communication coaching.
 */

require('dotenv').config();
const https = require('https');

// ==================== CONFIG ====================
let GROQ_API_KEY = process.env.GROQ_API_KEY || '';
const GROQ_MODEL = 'llama-3.3-70b-versatile';

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
            return reject(new Error('Groq API Key missing. Please check your .env file.'));
        }

        const data = JSON.stringify({
            model: GROQ_MODEL,
            messages: messages,
            temperature: 0.7,
            max_tokens: 1500
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
                    const text = result.choices?.[0]?.message?.content || "No response from AI.";
                    resolve(text);
                } catch (e) {
                    reject(new Error('Invalid response from Groq Cloud.'));
                }
            });
        });

        req.on('error', (e) => reject(new Error(`Network error connecting to Groq: ${e.message}`)));
        req.write(data);
        req.end();
    });
}

// ==================== MAIN AI FUNCTION ====================
async function generateAIResponse(userId, userMessage) {
    const history = conversationHistory.get(userId) || [];
    const messages = [
        { 
            role: "system", 
            content: "You are VDart AI Assistant, a world-class professional communication coach. Your goal is to help users improve their professional presence, speech clarity, and writing impact. Be concise, encouraging, and highly professional." 
        },
        ...history,
        { role: "user", content: userMessage }
    ];

    try {
        const response = await callGroq(messages);
        
        // Save to history (keep last 10 turns)
        const newHistory = [...history, { role: 'user', content: userMessage }, { role: 'assistant', content: response }];
        conversationHistory.set(userId, newHistory.slice(-10));

        return { 
            success: true, 
            response, 
            sources: [], 
            provider: 'Groq Cloud AI' 
        };
    } catch (error) {
        console.error("Groq Cloud AI Error:", error.message);
        return { 
            success: false, 
            error: error.message,
            fallback: "Static fallback mode. Please check your Groq API Key."
        };
    }
}

// ==================== STATUS & HELPER ====================
async function checkOllamaStatus() {
    // Renamed but kept signature for compatibility
    return { 
        available: !!GROQ_API_KEY, 
        cloud: true, 
        provider: 'Groq Cloud AI (Llama 3.3 70B)' 
    };
}

async function callOllamaPrompt(prompt) {
    // Wrapper for callGroq
    try {
        return { 
            response: await callGroq([{ role: 'user', content: prompt }]), 
            provider: 'Groq Cloud AI' 
        };
    } catch (e) {
        throw e;
    }
}

function parseJSONSafe(text) {
    try {
        // Handle markdown-wrapped JSON if present
        const jsonMatch = text.match(/```(?:json)?\s*([\s\S]*?)```/);
        const toParse = jsonMatch ? jsonMatch[1].trim() : text.trim();
        return JSON.parse(toParse);
    } catch {
        return null;
    }
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
