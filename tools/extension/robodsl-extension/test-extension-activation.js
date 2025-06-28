const path = require('path');
const fs = require('fs');

console.log('Testing extension activation...');

// Check if all required files exist
const requiredFiles = [
    'client/out/extension.js',
    'server/out/server.js',
    'server/py_parser_bridge.py',
    'package.json',
    'language-configuration.json',
    'syntaxes/robodsl.tmLanguage.json'
];

console.log('Checking required files:');
requiredFiles.forEach(file => {
    if (fs.existsSync(file)) {
        console.log(`✓ ${file}`);
    } else {
        console.log(`✗ ${file} - MISSING`);
    }
});

// Check package.json for language server configuration
const packageJson = JSON.parse(fs.readFileSync('package.json', 'utf8'));
console.log('\nChecking package.json configuration:');

const hasLanguageServer = packageJson.contributes && packageJson.contributes.languages;
const hasActivationEvents = packageJson.activationEvents;
const hasMain = packageJson.main;

console.log(`✓ main: ${hasMain ? 'present' : 'MISSING'}`);
console.log(`✓ activationEvents: ${hasActivationEvents ? 'present' : 'MISSING'}`);
console.log(`✓ languages: ${hasLanguageServer ? 'present' : 'MISSING'}`);

if (hasLanguageServer) {
    console.log('  Languages configured:');
    packageJson.contributes.languages.forEach(lang => {
        console.log(`    - ${lang.id} (${lang.aliases.join(', ')})`);
    });
}

console.log('\nExtension should activate when:');
if (hasActivationEvents) {
    packageJson.activationEvents.forEach(event => {
        console.log(`  - ${event}`);
    });
} 