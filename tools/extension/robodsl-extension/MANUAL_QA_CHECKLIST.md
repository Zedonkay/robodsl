# RoboDSL Extension Manual QA Checklist

## Pre-Testing Setup
- [ ] Install the extension from the VSIX package
- [ ] Open VS Code with a fresh workspace
- [ ] Create a `.robodsl` file for testing

## 1. File Icons ✅
- [ ] `.robodsl` files show the RoboDSL icon in the file explorer
- [ ] Icon appears in the file tabs
- [ ] Icon appears in the breadcrumb navigation

## 2. Syntax Highlighting ✅
- [ ] Keywords (`node`, `parameter`, `publisher`, etc.) are highlighted
- [ ] ROS message types (`std_msgs/String`, `geometry_msgs/Twist`) are highlighted
- [ ] C++ types (`int`, `std::string`, `cv::Mat`) are highlighted
- [ ] Comments (`//` and `/* */`) are properly colored
- [ ] Strings (quoted text) are highlighted
- [ ] Numbers are highlighted
- [ ] Operators (`=`, `:`, `{`, `}`) are highlighted

## 3. Language Mode ✅
- [ ] Status bar shows "RoboDSL" as the language mode
- [ ] File is recognized as RoboDSL in the bottom-right corner
- [ ] Language-specific features are available

## 4. Diagnostics (Squiggles) ✅
- [ ] **Valid syntax**: No red squiggles on correct code
- [ ] **Missing equals**: `parameter int x 42` shows error
- [ ] **Missing quotes**: `publisher /topic: std_msgs/String` shows error
- [ ] **Missing braces**: `node test_node` without `{` shows error
- [ ] **Unmatched braces**: Extra or missing `{`/`}` shows error
- [ ] **Invalid syntax**: Malformed constructs show appropriate errors

## 5. Problems Panel ✅
- [ ] Errors appear in the Problems panel (Ctrl+Shift+M)
- [ ] Error count is displayed in the Problems panel header
- [ ] Clicking on errors navigates to the correct line
- [ ] Error messages are clear and helpful

## 6. Intellisense (Auto-completion) ✅
- [ ] **Keywords**: Type `par` → suggests `parameter`
- [ ] **ROS types**: Type `std_` → suggests `std_msgs/` types
- [ ] **C++ types**: Type `std::` → suggests `std::string`, `std::vector`, etc.
- [ ] **Context-aware**: Different suggestions based on context
- [ ] **Documentation**: Hover over suggestions shows descriptions

## 7. Hover Information ✅
- [ ] **Keywords**: Hover over `node` shows documentation
- [ ] **ROS types**: Hover over `std_msgs/String` shows type info
- [ ] **C++ types**: Hover over `int` shows type info
- [ ] **Documentation**: Hover shows helpful descriptions and examples

## 8. Extension Activation ✅
- [ ] Extension activates when opening `.robodsl` files
- [ ] Extension activates when switching to RoboDSL language mode
- [ ] No activation errors in the Output panel

## 9. Commands ✅
- [ ] **Manual activation**: `Ctrl+Shift+P` → "RoboDSL: Activate" works
- [ ] **Test command**: `Ctrl+Shift+P` → "RoboDSL: Test" works
- [ ] Commands show appropriate output in the Output panel

## 10. Output Panel Logging ✅
- [ ] Open Output panel (`View` → `Output`)
- [ ] Select "RoboDSL" from the dropdown
- [ ] Verify logging shows:
  - Extension activation
  - Document validation
  - Parser integration
  - Error messages

## 11. Performance ✅
- [ ] Extension loads quickly
- [ ] Diagnostics appear within 3 seconds of typing
- [ ] No lag when typing or editing
- [ ] Memory usage is reasonable

## 12. Error Handling ✅
- [ ] **Parser errors**: Graceful handling when Python parser fails
- [ ] **Invalid files**: Appropriate error messages for corrupted files
- [ ] **Missing dependencies**: Clear error messages if Python dependencies missing
- [ ] **Network issues**: Graceful handling of any network-related errors

## 13. Integration Testing ✅
- [ ] **Valid file**: `test-files/valid-syntax.robodsl` shows no errors
- [ ] **Invalid file**: `test-files/invalid-syntax.robodsl` shows multiple errors
- [ ] **Real-world files**: Test with actual RoboDSL files from the project

## 14. Cross-Platform Testing ✅
- [ ] **macOS**: All features work correctly
- [ ] **Windows**: All features work correctly (if applicable)
- [ ] **Linux**: All features work correctly (if applicable)

## 15. Edge Cases ✅
- [ ] **Empty files**: No errors on empty `.robodsl` files
- [ ] **Large files**: Performance with large RoboDSL files
- [ ] **Special characters**: Handling of Unicode and special characters
- [ ] **Malformed content**: Graceful handling of completely invalid content

## Test Files to Use
1. **`test-files/valid-syntax.robodsl`** - Should show no errors
2. **`test-files/invalid-syntax.robodsl`** - Should show multiple errors
3. **Create your own test files** with various syntax patterns

## Expected Behaviors
- ✅ **Green checkmarks** indicate features that should work
- 🔄 **Real-time validation** as you type
- 📝 **Clear error messages** that help fix issues
- 🎯 **Context-aware suggestions** that make sense
- ⚡ **Fast response times** for all features

## Reporting Issues
If any item fails:
1. Note the specific behavior
2. Check the Output panel for error messages
3. Check the Developer Console for additional details
4. Report with steps to reproduce

## Success Criteria
- All items marked with ✅ should work correctly
- No critical errors in the Output panel
- Extension provides a smooth, professional development experience
- Real RoboDSL parser integration works reliably 