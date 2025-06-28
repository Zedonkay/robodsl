#!/bin/bash

echo "=== RoboDSL Extension Diagnostic ==="
echo

echo "1. Checking extension files..."
if [ -f "client/out/extension.js" ]; then
    echo "✓ Client extension compiled"
else
    echo "✗ Client extension not compiled"
fi

if [ -f "server/out/server.js" ]; then
    echo "✓ Server compiled"
else
    echo "✗ Server not compiled"
fi

if [ -f "server/py_parser_bridge.py" ]; then
    echo "✓ Python bridge exists"
else
    echo "✗ Python bridge missing"
fi

echo
echo "2. Testing Python bridge..."
echo "node test_node { parameter test_param: int = 42; }" | python3 server/py_parser_bridge.py
echo

echo "3. Testing VSIX package..."
if [ -f "robodsl-0.1.1.vsix" ]; then
    echo "✓ VSIX package exists"
    ls -lh robodsl-0.1.1.vsix
else
    echo "✗ VSIX package missing"
fi

echo
echo "4. Extension installation instructions:"
echo "   a) Uninstall any existing RoboDSL extension"
echo "   b) Restart VS Code completely"
echo "   c) Install the VSIX: code --install-extension robodsl-0.1.1.vsix"
echo "   d) Restart VS Code again"
echo "   e) Open a .robodsl file"
echo
echo "5. Troubleshooting steps:"
echo "   - Check VS Code Developer Tools (Help > Toggle Developer Tools)"
echo "   - Look for errors in the Console tab"
echo "   - Check the Output panel for 'RoboDSL Language Server'"
echo "   - Try the command palette: 'RoboDSL: Activate Extension'"
echo
echo "6. Expected behavior:"
echo "   - Extension should activate when opening .robodsl files"
echo "   - Should see 'RoboDSL Extension is activating...' message"
echo "   - Should see 'RoboDSL Language Server started successfully!' message"
echo "   - Should see syntax highlighting and icons"
echo "   - Should see diagnostics in Problems panel" 