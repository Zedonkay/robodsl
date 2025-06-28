// Simple test script to verify extension loading
const path = require('path');

// Test if the extension files exist
const clientPath = path.join(__dirname, 'client', 'out', 'extension.js');
const serverPath = path.join(__dirname, 'server', 'out', 'server.js');

const fs = require('fs');

console.log('Testing RoboDSL Extension...');
console.log('Client path:', clientPath);
console.log('Server path:', serverPath);

if (fs.existsSync(clientPath)) {
    console.log('✅ Client extension.js exists');
} else {
    console.log('❌ Client extension.js missing');
}

if (fs.existsSync(serverPath)) {
    console.log('✅ Server server.js exists');
} else {
    console.log('❌ Server server.js missing');
}

// Test if we can require the extension
try {
    const extension = require('./client/out/extension.js');
    console.log('✅ Extension can be required');
    console.log('Exports:', Object.keys(extension));
} catch (error) {
    console.log('❌ Failed to require extension:', error.message);
}

// Test if we can require the server
try {
    const server = require('./server/out/server.js');
    console.log('✅ Server can be required');
} catch (error) {
    console.log('❌ Failed to require server:', error.message);
} 