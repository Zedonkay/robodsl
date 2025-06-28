#!/bin/bash

echo "=== RoboDSL Extension Diagnostic ==="
echo ""

echo "1. Checking extension files..."
if [ -f "client/out/extension.js" ]; then
    echo "✅ Client extension compiled"
else
    echo "❌ Client extension not compiled"
fi

if [ -f "server/out/server.js" ]; then
    echo "✅ Server extension compiled"
else
    echo "❌ Server extension not compiled"
fi

if [ -f "syntaxes/robodsl.tmLanguage.json" ]; then
    echo "✅ Syntax highlighting configured"
else
    echo "❌ Syntax highlighting missing"
fi

if [ -f "language-configuration.json" ]; then
    echo "✅ Language configuration present"
else
    echo "❌ Language configuration missing"
fi

if [ -f "fileicons/robodsl-icon-theme.json" ]; then
    echo "✅ File icons configured"
else
    echo "❌ File icons missing"
fi

echo ""
echo "2. Testing Python parser bridge..."
if [ -f "server/py_parser_bridge.py" ]; then
    echo "✅ Python bridge exists"
    echo "Testing parser..."
    echo 'node test { parameter int x = 42 }' | python3 server/py_parser_bridge.py
    if [ $? -eq 0 ]; then
        echo "✅ Parser bridge working"
    else
        echo "❌ Parser bridge failed"
    fi
else
    echo "❌ Python bridge missing"
fi

echo ""
echo "3. Checking VSIX package..."
if [ -f "robodsl-0.1.1.vsix" ]; then
    echo "✅ VSIX package exists"
    ls -la robodsl-0.1.1.vsix
else
    echo "❌ VSIX package missing"
fi

echo ""
echo "4. Extension troubleshooting steps:"
echo "   a) Uninstall any existing RoboDSL extension"
echo "   b) Restart VS Code"
echo "   c) Install robodsl-0.1.1.vsix"
echo "   d) Restart VS Code again"
echo "   e) Open a .robodsl file"
echo "   f) Check Output panel (View → Output → RoboDSL)"
echo ""
echo "5. If issues persist:"
echo "   - Check VS Code Developer Console (Help → Toggle Developer Tools)"
echo "   - Look for error messages in the Console tab"
echo "   - Check if the extension is listed in Extensions panel" 