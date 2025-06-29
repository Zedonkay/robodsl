"use strict";
/* --------------------------------------------------------------------------------------------
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See License.txt in the project root for license information.
 * ------------------------------------------------------------------------------------------ */
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || function (mod) {
    if (mod && mod.__esModule) return mod;
    var result = {};
    if (mod != null) for (var k in mod) if (k !== "default" && Object.prototype.hasOwnProperty.call(mod, k)) __createBinding(result, mod, k);
    __setModuleDefault(result, mod);
    return result;
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.deactivate = exports.activate = void 0;
const path = __importStar(require("path"));
const fs = __importStar(require("fs"));
const vscode_1 = require("vscode");
const node_1 = require("vscode-languageclient/node");
let client;
function activate(context) {
    console.log('RoboDSL Extension activating...');
    // Show a notification that the extension is activating
    vscode_1.window.showInformationMessage('RoboDSL Extension is activating...');
    // Register commands for debugging
    context.subscriptions.push(vscode_1.commands.registerCommand('robodsl.activate', () => {
        vscode_1.window.showInformationMessage('RoboDSL Extension is active!');
        console.log('RoboDSL Extension manually activated');
    }));
    context.subscriptions.push(vscode_1.commands.registerCommand('robodsl.test', () => {
        vscode_1.window.showInformationMessage('RoboDSL Extension test command executed!');
        console.log('RoboDSL Extension test command executed');
    }));
    // ---------- Embedded C++ / CUDA support ----------
    const EMBED_SCHEME = 'robodsl-embedded';
    // Provide virtual documents representing the embedded code sections
    const virtualDocProvider = {
        provideTextDocumentContent: (uri) => {
            const [fsPath, indexStr] = uri.path.split('#');
            const index = Number(indexStr);
            const srcDoc = vscode_1.workspace.textDocuments.find(doc => doc.uri.fsPath === fsPath);
            if (!srcDoc) {
                return '';
            }
            const codeBlocks = extractCodeBlocks(srcDoc.getText());
            return codeBlocks[index] || '';
        }
    };
    context.subscriptions.push(vscode_1.workspace.registerTextDocumentContentProvider(EMBED_SCHEME, virtualDocProvider));
    // Middleware for embedded C++/CUDA
    const embeddedMiddleware = {
        provideCompletionItem: async (document, position, contextMiddleware, token, next) => {
            if (isInsideCodeBlock(document, position)) {
                const { virtualUri, innerPos } = toVirtualCppUri(document, position);
                return vscode_1.commands.executeCommand('vscode.executeCompletionItemProvider', virtualUri, innerPos);
            }
            return next(document, position, contextMiddleware, token);
        },
        provideHover: async (document, position, token, next) => {
            if (isInsideCodeBlock(document, position)) {
                const { virtualUri, innerPos } = toVirtualCppUri(document, position);
                return vscode_1.commands.executeCommand('vscode.executeHoverProvider', virtualUri, innerPos);
            }
            return next(document, position, token);
        }
    };
    // The server is implemented in node
    const serverModuleOut = context.asAbsolutePath(path.join('server', 'out', 'server.js'));
    const serverModuleSrc = context.asAbsolutePath(path.join('server', 'src', 'server.js'));
    // Use compiled version if it exists, otherwise fallback to source
    const serverModule = fs.existsSync(serverModuleOut) ? serverModuleOut : serverModuleSrc;
    console.log('Server module path:', serverModule);
    // If the extension is launched in debug mode then the debug server options are used
    // Otherwise the run options are used
    const serverOptions = {
        run: { module: serverModule, transport: node_1.TransportKind.ipc },
        debug: {
            module: serverModule,
            transport: node_1.TransportKind.ipc,
            options: { execArgv: ['--nolazy', '--inspect=6009'] }
        }
    };
    // Options to control the language client
    const clientOptions = {
        // Register the server for robodsl documents
        documentSelector: [{ scheme: 'file', language: 'robodsl' }],
        synchronize: {
            // Notify the server about file changes to .robodsl files in the workspace
            fileEvents: vscode_1.workspace.createFileSystemWatcher('**/*.robodsl')
        },
        // Enable diagnostic collection
        diagnosticCollectionName: 'robodsl',
        // Enable progress reporting
        progressOnInitialization: true,
        middleware: embeddedMiddleware
    };
    console.log('Creating language client...');
    // Create the language client and start the client.
    client = new node_1.LanguageClient('robodslLanguageServer', 'RoboDSL Language Server', serverOptions, clientOptions);
    console.log('Starting language client...');
    // Start the client. This will also launch the server
    client.start().then(() => {
        console.log('Language client started successfully!');
        vscode_1.window.showInformationMessage('RoboDSL Language Server started successfully!');
        // Register the client for disposal
        context.subscriptions.push(client);
        // Create a status bar item to show diagnostic count
        const statusBarItem = vscode_1.window.createStatusBarItem(vscode_1.StatusBarAlignment.Left, 100);
        statusBarItem.text = 'RoboDSL: $(sync~spin)';
        statusBarItem.tooltip = 'RoboDSL diagnostics';
        statusBarItem.show();
        function refreshDiagnosticCount() {
            const allDiagnostics = vscode_1.languages.getDiagnostics();
            let count = 0;
            for (const [uri, diags] of allDiagnostics) {
                if (uri.path.endsWith('.robodsl')) {
                    count += diags.length;
                }
            }
            statusBarItem.text = `RoboDSL: ${count} problem${count === 1 ? '' : 's'}`;
        }
        // Refresh once client is ready and whenever diagnostics change
        refreshDiagnosticCount();
        context.subscriptions.push(vscode_1.languages.onDidChangeDiagnostics(refreshDiagnosticCount));
    }).catch((error) => {
        console.error('Failed to start language client:', error);
        vscode_1.window.showErrorMessage(`Failed to start RoboDSL Language Server: ${error.message}`);
    });
    function extractCodeBlocks(text) {
        const blocks = [];
        const regex = /code\s*\{([\s\S]*?)\}/g;
        let match;
        while ((match = regex.exec(text)) !== null) {
            blocks.push(match[1]);
        }
        return blocks;
    }
    function isInsideCodeBlock(document, position) {
        const text = document.getText();
        const offset = document.offsetAt(position);
        const regex = /code\s*\{([\s\S]*?)\}/g;
        let match;
        while ((match = regex.exec(text)) !== null) {
            const start = match.index;
            const end = match.index + match[0].length;
            if (offset >= start && offset <= end)
                return true;
        }
        return false;
    }
    function toVirtualCppUri(document, position) {
        const text = document.getText();
        const offset = document.offsetAt(position);
        const regex = /code\s*\{([\s\S]*?)\}/g;
        let match;
        let idx = 0;
        while ((match = regex.exec(text)) !== null) {
            const start = match.index;
            const end = match.index + match[0].length;
            if (offset >= start && offset <= end) {
                const virtualUri = vscode_1.Uri.parse(`${EMBED_SCHEME}:${document.uri.fsPath}#${idx}`);
                const innerOffset = offset - start - (match[0].indexOf('{') + 1);
                const codeText = match[1];
                const lines = codeText.substring(0, innerOffset).split('\n');
                const innerPos = new vscode_1.Position(lines.length - 1, lines[lines.length - 1].length);
                return { virtualUri, innerPos };
            }
            idx++;
        }
        // Fallback
        return { virtualUri: document.uri, innerPos: position };
    }
}
exports.activate = activate;
function deactivate() {
    console.log('RoboDSL Extension deactivating...');
    if (!client) {
        return undefined;
    }
    return client.stop();
}
exports.deactivate = deactivate;
//# sourceMappingURL=extension.js.map