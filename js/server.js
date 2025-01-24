const express = require('express');
const https = require('https');
const fs = require('fs');
const app = express();
const path = require('path');

// Load SSL certificates (for local development, you can generate your own)
const privateKey = fs.readFileSync(path.join(__dirname, '..', 'server.key'), 'utf8');
const certificate = fs.readFileSync(path.join(__dirname, '..', 'server.cert'), 'utf8');
const credentials = { key: privateKey, cert: certificate };

// Serve static files with the necessary headers for cross-origin isolation
app.use(express.static(path.join(__dirname, '..'), {
  setHeaders: (res) => {
    res.setHeader('Cross-Origin-Opener-Policy', 'cross-origin');
    res.setHeader('Cross-Origin-Embedder-Policy', 'require-corp');
  }
}));

// Create an HTTPS server
https.createServer(credentials, app).listen(3000, () => {
  console.log('HTTPS server running at https://localhost:3000');
});