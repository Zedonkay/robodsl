RoboDSL VS Code Extension

This client extension launches the Python `robodsl-ls` language server and provides language configuration and syntax highlighting for `.robodsl` files.

Build and run locally:

```sh
cd tools/extension/robodsl2/extension
npm install
npm run compile
# Press F5 in VS Code to run Extension Development Host
```

Configuration:
- `robodsl.languageServerPath`: path to the `robodsl-ls` executable (default: `robodsl-ls` on PATH)


