# RoboDSL Extension - Comprehensive Debugging Guide

## 🚨 Current Status
The extension is **properly compiled and packaged** but may not be activating due to VS Code environment issues.

## 🔍 Quick Diagnosis

### 1. Extension Structure ✅
- ✅ Client extension.js exists and compiles
- ✅ Server server.js exists and compiles  
- ✅ All dependencies installed
- ✅ Syntax highlighting configured
- ✅ Language configuration set up
- ✅ Package.json properly configured

### 2. Known Issues
- VS Code CLI shows warnings on macOS
- Extension may not activate automatically
- Language server may not start

## 🛠️ Manual Testing Steps

### Step 1: Install Extension Manually
1. Open VS Code
2. Go to Extensions (Ctrl+Shift+X or Cmd+Shift+X)
3. Click "..." menu → "Install from VSIX..."
4. Select `robodsl-0.1.1.vsix` from the extension directory
5. Restart VS Code

### Step 2: Force Extension Activation
1. Open VS Code
2. Press Ctrl+Shift+P (or Cmd+Shift+P on Mac)
3. Type "Developer: Reload Window"
4. Open a `.robodsl` file
5. Check if syntax highlighting works

### Step 3: Test Commands
1. Press Ctrl+Shift+P (or Cmd+Shift+P on Mac)
2. Type "RoboDSL: Activate RoboDSL Extension"
3. You should see a notification: "RoboDSL Extension is active!"

### Step 4: Check Language Server
1. Open a `.robodsl` file
2. Go to View → Output
3. Select "RoboDSL Language Server" from dropdown
4. Look for activation messages

## 🔧 Common Fixes

### Fix 1: Force Language Association
If `.robodsl` files aren't recognized:
1. Open a `.robodsl` file
2. Click on language mode in bottom-right corner
3. Select "Configure file association for '.robodsl'"
4. Choose "RoboDSL"

### Fix 2: Enable Extension
If extension is disabled:
1. Go to Extensions (Ctrl+Shift+X)
2. Find "RoboDSL Language Support"
3. Click "Enable" if disabled
4. Reload window

### Fix 3: Check Output Panel
If nothing seems to work:
1. View → Output
2. Select "RoboDSL Language Server"
3. Look for error messages
4. Check Developer Console (Help → Toggle Developer Tools)

## 📋 Test Files

Use these files to test the extension:

### `test-debug.robodsl` (Simple Test)
```robodsl
// Test file for RoboDSL extension debugging
node test_node {
    parameter test_param: "test_value"
    namespace: "/test_namespace"
    timer test_timer: 1000
}
```

### `comprehensive_test.robodsl` (Full Features)
Contains various syntax errors and correct examples to test:
- Syntax highlighting
- Error detection
- Auto-completion
- Hover information

## 🎯 Expected Behavior

### ✅ Working Features
- **File Icons**: Custom icon for `.robodsl` files
- **Syntax Highlighting**: Keywords in color, strings in quotes
- **Auto-completion**: Suggestions for keywords and types
- **Error Detection**: Squiggly lines for syntax errors
- **Hover Information**: Documentation on hover
- **Commands**: "RoboDSL: Activate RoboDSL Extension" works

### ❌ If Extension Doesn't Work

#### No Syntax Highlighting
```
Solution:
1. Check if file has .robodsl extension
2. Force language association
3. Reload VS Code window
```

#### No Auto-completion
```
Solution:
1. Check Output panel for language server errors
2. Try manual activation command
3. Check if extension is enabled
```

#### Language Server Not Starting
```
Solution:
1. Check Developer Console for errors
2. Verify all files are compiled
3. Try reinstalling extension
```

## 🐛 Advanced Debugging

### Check Extension Logs
1. Help → Toggle Developer Tools
2. Go to Console tab
3. Look for "RoboDSL" messages
4. Check for error stack traces

### Verify File Structure
Run the debug script:
```bash
./debug-extension.sh
```

### Test Language Server Directly
The language server can't be tested outside VS Code, but you can verify:
1. Server file exists: `server/out/server.js`
2. Client file exists: `client/out/extension.js`
3. Both compile without errors

## 📞 Getting Help

If the extension still doesn't work:

1. **Check VS Code Version**: Must be 1.75.0 or higher
2. **Try Clean Install**: 
   - Uninstall extension
   - Delete extension folder
   - Reinstall from VSIX
3. **Check System Requirements**:
   - Node.js installed
   - VS Code properly installed
4. **Report Specific Errors**:
   - Copy error messages from Output panel
   - Copy error messages from Developer Console
   - Note VS Code version and OS

## 🎉 Success Indicators

The extension is working if you see:
- ✅ Custom icon for `.robodsl` files
- ✅ Syntax highlighting (keywords in color)
- ✅ Auto-completion suggestions
- ✅ Error squiggly lines
- ✅ "RoboDSL Extension is active!" notification
- ✅ Language server messages in Output panel

## 🔄 Rebuilding Extension

If you need to rebuild:
```bash
npm run compile
npm run package
```

The extension should now work properly with these debugging steps! 