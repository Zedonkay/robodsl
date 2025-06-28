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
    // The server is implemented in node
    const serverModule = context.asAbsolutePath(path.join('server', 'out', 'server.js'));
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
        progressOnInitialization: true
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
    }).catch((error) => {
        console.error('Failed to start language client:', error);
        vscode_1.window.showErrorMessage(`Failed to start RoboDSL Language Server: ${error.message}`);
    });
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