const express = require('express');
// const cors = require('cors');
const app = express();
const path = require('path');


// Add cross-origin isolation headers
app.use((req, res, next) => {
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Cross-Origin-Resource-Policy", "same-origin");
  res.setHeader("Cross-Origin-Opener-Policy", "same-origin");
  res.setHeader("Cross-Origin-Embedder-Policy", "require-corp");
  next();
});

// app.use(cors(corsOptions ))
// Serve static files with the necessary headers for cross-origin isolation
app.use(express.static(path.join(__dirname, '..'), {

}));

app.listen(3000, )