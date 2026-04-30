const http = require('http');

function postJson(path, payload) {
  return new Promise((resolve, reject) => {
    const data = JSON.stringify(payload);
    const opts = {
      hostname: 'localhost',
      port: 3000,
      path,
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Content-Length': Buffer.byteLength(data)
      }
    };

    const req = http.request(opts, (res) => {
      let body = '';
      res.on('data', chunk => body += chunk);
      res.on('end', () => {
        try { resolve(JSON.parse(body)); } catch (e) { resolve(body); }
      });
    });

    req.on('error', reject);
    req.write(data);
    req.end();
  });
}

function getJson(path) {
  return new Promise((resolve, reject) => {
    http.get({ hostname: 'localhost', port: 3000, path, agent: false }, res => {
      let body = '';
      res.on('data', c => body += c);
      res.on('end', () => {
        try { resolve(JSON.parse(body)); } catch (e) { resolve(body); }
      });
    }).on('error', reject);
  });
}

(async () => {
  try {
    console.log('Calling seed-train...');
    const seed = await postJson('/api/voice-coach/model/seed-train', { count: 400 });
    console.log('seed-train result:', JSON.stringify(seed, null, 2));

    console.log('\nChecking model status...');
    const status = await getJson('/api/voice-coach/model/status');
    console.log('status:', JSON.stringify(status, null, 2));

    console.log('\nPredict sample...');
    const pred = await postJson('/api/voice-coach/model/predict', { transcript: 'This is a short demo. I am confident and clear.', scenario: 'sales' });
    console.log('predict:', JSON.stringify(pred, null, 2));
  } catch (e) {
    console.error('Error:', e.message || e);
    process.exit(1);
  }
})();
