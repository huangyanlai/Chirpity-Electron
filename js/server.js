const express = require('express');
const app = express();
const path = require('path');
const PORT = 3000;

// Serve static files from the current directory
app.use(express.static(path.join(__dirname, '..'), {
  setHeaders: (res) => {
    // Add Cross-Origin policies
    res.setHeader('Cross-Origin-Opener-Policy', 'same-origin');
    res.setHeader('Cross-Origin-Embedder-Policy', 'require-corp');
  }
}));

app.listen(PORT, () => {
  console.log(`Server running at http://localhost:${PORT}`);
});
