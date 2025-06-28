"use strict";
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
    console.log('RoboDSL extension is now active!');
    // The server is implemented in node
    const serverModule = context.asAbsolutePath(path.join('out', 'language-server.js'));
    console.log('Server module path:', serverModule);
    // The debug options for the server
    // --inspect=6009: runs the server in Node's Inspector mode so VS Code can attach to the server for debugging
    const debugOptions = { execArgv: ['--nolazy', '--inspect=6009'] };
    // If the extension is launched in debug mode then the debug server options are used
    // Otherwise the run options are used
    const serverOptions = {
        run: { module: serverModule, transport: node_1.TransportKind.stdio },
        debug: {
            module: serverModule,
            transport: node_1.TransportKind.stdio,
            options: debugOptions
        }
    };
    // Options to control the language client
    const clientOptions = {
        // Register the server for robodsl documents
        documentSelector: [
            { scheme: 'file', language: 'robodsl' }
        ],
        synchronize: {
            // Notify the server about file changes to .robodsl files in the workspace
            fileEvents: vscode_1.workspace.createFileSystemWatcher('**/*.robodsl')
        }
    };
    // Create the language client and start the client.
    client = new node_1.LanguageClient('robodslLanguageServer', 'RoboDSL Language Server', serverOptions, clientOptions);
    // Start the client. This will also launch the server
    client.start().then(() => {
        console.log('RoboDSL Language Server started successfully');
        vscode_1.window.showInformationMessage('RoboDSL Language Server is now active!');
    }).catch((error) => {
        console.error('Failed to start RoboDSL Language Server:', error);
        vscode_1.window.showErrorMessage('Failed to start RoboDSL Language Server: ' + error.message);
    });
}
exports.activate = activate;
function deactivate() {
    if (!client) {
        return undefined;
    }
    return client.stop();
}
exports.deactivate = deactivate;
//# sourceMappingURL=extension.js.map