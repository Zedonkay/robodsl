# RoboDSL Extension - Testing Guide

## 🧪 How to Test the Extension

### Step 1: Install the Extension
1. **Download**: Get `robodsl-0.1.1.vsix` from the extension directory
2. **Install in VS Code**:
   - Open VS Code
   - Go to Extensions (Ctrl+Shift+X)
   - Click "..." menu → "Install from VSIX..."
   - Select `robodsl-0.1.1.vsix`
   - Restart VS Code

### Step 2: Test File Icons
1. **Create a test file**: Create a new file with `.robodsl` extension
2. **Expected result**: You should see a custom RoboDSL icon for the file
3. **If icons don't appear**:
   - Go to Extensions → RoboDSL Language Support → Settings
   - Check if the extension is enabled
   - Try reloading the window (Ctrl+Shift+P → "Developer: Reload Window")

### Step 3: Test Syntax Highlighting
1. **Open `comprehensive_test.robodsl`** (included in the extension)
2. **Expected result**: 
   - Keywords should be colored (node, method, kernel, etc.)
   - Strings should be in quotes
   - Comments should be grayed out
   - C++ code blocks should have C++ syntax highlighting

### Step 4: Test Language Server
1. **Check Output Panel**:
   - View → Output
   - Select "RoboDSL Language Server" from dropdown
   - You should see: "RoboDSL Extension activating..." and "Language Server started successfully!"

2. **If language server fails**:
   - Check the Output panel for error messages
   - Look in Developer Console (Help → Toggle Developer Tools)

### Step 5: Test Intellisense
1. **Auto-completion**:
   - Type `node` and press Ctrl+Space
   - You should see suggestions for RoboDSL keywords
   - Type `std_msgs/` and see ROS type suggestions

2. **Hover information**:
   - Hover over keywords like `node`, `method`, `publisher`
   - You should see detailed documentation and examples

### Step 6: Test Error Detection
1. **Open `comprehensive_test.robodsl`**
2. **Look for squiggly lines**:
   - **Red underlines**: Syntax errors
   - **Yellow underlines**: Warnings
3. **Check error count**:
   - Look in the Output panel for "Total diagnostics: X"
   - Should show parser errors + additional validation errors

## 🔍 What to Look For

### ✅ Working Features
- **File Icons**: Custom icon for `.robodsl` files
- **Syntax Highlighting**: Proper colors for keywords, strings, comments
- **Auto-completion**: Suggestions when typing
- **Hover Information**: Documentation on hover
- **Error Detection**: Squiggly lines for syntax errors
- **Error Counting**: Console shows total number of errors

### ❌ Common Issues & Solutions

#### Icons Not Showing
```
Problem: No custom icons for .robodsl files
Solution: 
1. Check if extension is enabled
2. Reload VS Code window
3. Check if file has .robodsl extension
```

#### Language Server Not Starting
```
Problem: No intellisense, no error detection
Solution:
1. Check Output panel → "RoboDSL Language Server"
2. Look for error messages
3. Check Developer Console for detailed errors
```

#### Red Brackets/Weird Highlighting
```
Problem: All brackets are red or syntax highlighting is wrong
Solution:
1. This was fixed in the latest version
2. If still happening, check if old version is installed
3. Uninstall and reinstall the extension
```

#### No Auto-completion
```
Problem: No suggestions when typing
Solution:
1. Make sure file has .robodsl extension
2. Check if language server is running (Output panel)
3. Try Ctrl+Space to force suggestions
```

## 📊 Expected Error Count

For `comprehensive_test.robodsl`, you should see approximately:
- **Parser Errors**: 2-3 (basic syntax validation)
- **Additional Errors**: 4-5 (missing colons, quotes, braces)
- **Total**: 6-8 diagnostics

## 🐛 Debugging Steps

### 1. Check Extension Status
```
View → Output → "RoboDSL Language Server"
Look for:
✅ "RoboDSL Extension activating..."
✅ "Language client started successfully!"
❌ Any error messages
```

### 2. Check Developer Console
```
Help → Toggle Developer Tools → Console
Look for:
✅ Extension activation messages
❌ Any error messages or stack traces
```

### 3. Test Language Server Communication
```
1. Open a .robodsl file
2. Make a syntax error (e.g., remove a colon)
3. Check if squiggly lines appear
4. Check Output panel for diagnostic messages
```

### 4. Verify File Association
```
1. Create a new file with .robodsl extension
2. Check if VS Code recognizes it as RoboDSL
3. Look at bottom-right status bar for language mode
```

## 📝 Test Files Included

1. **`comprehensive_test.robodsl`**: Contains various syntax errors and correct examples
2. **`test_fixes.robodsl`**: Simple test file for basic functionality
3. **`simple-test.robodsl`**: Minimal test file

## 🎯 Success Criteria

The extension is working correctly if:
- ✅ `.robodsl` files show custom icons
- ✅ Syntax highlighting works properly (no red brackets)
- ✅ Auto-completion provides suggestions
- ✅ Hover shows documentation
- ✅ Squiggly lines appear for syntax errors
- ✅ Error count is displayed in console
- ✅ Language server starts without errors

## 🆘 If Nothing Works

1. **Uninstall and reinstall** the extension
2. **Check VS Code version** (requires 1.75.0+)
3. **Try in a clean workspace** (no other extensions)
4. **Check system requirements** (Node.js, etc.)
5. **Report specific error messages** from Output panel and Developer Console

The extension should now provide a complete language experience similar to C++ or Python extensions! 