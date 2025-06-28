const { spawn } = require('child_process');
const path = require('path');

console.log('Testing server startup...');

// Test if we can find the Python bridge
const pyScriptPath = path.join(__dirname, 'server', 'py_parser_bridge.py');
console.log('Python script path:', pyScriptPath);

// Test if the script exists
const fs = require('fs');
if (fs.existsSync(pyScriptPath)) {
    console.log('✓ Python bridge script exists');
} else {
    console.log('✗ Python bridge script not found');
    process.exit(1);
}

// Test if we can spawn Python
const py = spawn('python3', [pyScriptPath]);

let output = '';
let error = '';

py.stdout.on('data', (data) => {
    output += data.toString();
});

py.stderr.on('data', (data) => {
    error += data.toString();
});

py.on('close', (code) => {
    console.log(`Python bridge process exited with code: ${code}`);
    if (code !== 0) {
        console.error('Python bridge error output:', error);
    } else {
        console.log('✓ Python bridge works correctly');
        try {
            const diagnostics = JSON.parse(output);
            console.log('✓ Python bridge returned valid JSON:', diagnostics);
        } catch (e) {
            console.error('✗ Python bridge returned invalid JSON:', e.message);
        }
    }
});

py.on('error', (err) => {
    console.error('✗ Failed to spawn Python process:', err.message);
});

// Send test input
py.stdin.write('node test_node { parameter test_param: int = 42; }');
py.stdin.end(); 