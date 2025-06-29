/* --------------------------------------------------------------------------------------------
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See License.txt in the project root for license information.
 * ------------------------------------------------------------------------------------------ */

import * as path from 'path';
import * as fs from 'fs';
import { 
    workspace, 
    ExtensionContext, 
    window, 
    commands, 
    StatusBarAlignment,
    StatusBarItem,
    languages,
    Uri,
    Position,
    TextDocumentContentProvider
} from 'vscode';

import {
    LanguageClient,
    LanguageClientOptions,
    ServerOptions,
    TransportKind,
    Middleware
} from 'vscode-languageclient/node';

let client: LanguageClient;

export function activate(context: ExtensionContext) {
    console.log('RoboDSL Extension activating...');
    
    // Show a notification that the extension is activating
    window.showInformationMessage('RoboDSL Extension is activating...');
    
    // Register commands for debugging
    context.subscriptions.push(
        commands.registerCommand('robodsl.activate', () => {
            window.showInformationMessage('RoboDSL Extension is active!');
            console.log('RoboDSL Extension manually activated');
        })
    );
    
    context.subscriptions.push(
        commands.registerCommand('robodsl.test', () => {
            window.showInformationMessage('RoboDSL Extension test command executed!');
            console.log('RoboDSL Extension test command executed');
        })
    );

    // ---------- Embedded C++ / CUDA support ----------
    const EMBED_SCHEME = 'robodsl-embedded';

    // Provide virtual documents representing the embedded code sections
    const virtualDocProvider: TextDocumentContentProvider = {
        provideTextDocumentContent: (uri: Uri) => {
            const [fsPath, indexStr] = uri.path.split('#');
            const index = Number(indexStr);
            const srcDoc = workspace.textDocuments.find(doc => doc.uri.fsPath === fsPath);
            if (!srcDoc) { return ''; }
            const codeBlocks = extractCodeBlocks(srcDoc.getText());
            return codeBlocks[index] || '';
        }
    };
    context.subscriptions.push(workspace.registerTextDocumentContentProvider(EMBED_SCHEME, virtualDocProvider));

    // Middleware for embedded C++/CUDA
    const embeddedMiddleware: Middleware = {
        provideCompletionItem: async (document, position, contextMiddleware, token, next) => {
            if (isInsideCodeBlock(document, position)) {
                const { virtualUri, innerPos } = toVirtualCppUri(document, position);
                return commands.executeCommand('vscode.executeCompletionItemProvider', virtualUri, innerPos);
            }
            return next(document, position, contextMiddleware, token);
        },
        provideHover: async (document, position, token, next) => {
            if (isInsideCodeBlock(document, position)) {
                const { virtualUri, innerPos } = toVirtualCppUri(document, position);
                return commands.executeCommand('vscode.executeHoverProvider', virtualUri, innerPos);
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
    const serverOptions: ServerOptions = {
        run: { module: serverModule, transport: TransportKind.ipc },
        debug: {
            module: serverModule,
            transport: TransportKind.ipc,
            options: { execArgv: ['--nolazy', '--inspect=6009'] }
        }
    };

    // Options to control the language client
    const clientOptions: LanguageClientOptions = {
        // Register the server for robodsl documents
        documentSelector: [{ scheme: 'file', language: 'robodsl' }],
        synchronize: {
            // Notify the server about file changes to .robodsl files in the workspace
            fileEvents: workspace.createFileSystemWatcher('**/*.robodsl')
        },
        // Enable diagnostic collection
        diagnosticCollectionName: 'robodsl',
        // Enable progress reporting
        progressOnInitialization: true,
        middleware: embeddedMiddleware
    };

    console.log('Creating language client...');

    // Create the language client and start the client.
    client = new LanguageClient(
        'robodslLanguageServer',
        'RoboDSL Language Server',
        serverOptions,
        clientOptions
    );

    console.log('Starting language client...');

    // Start the client. This will also launch the server
    client.start().then(() => {
        console.log('Language client started successfully!');
        window.showInformationMessage('RoboDSL Language Server started successfully!');
        
        // Register the client for disposal
        context.subscriptions.push(client);
        
        // Create a status bar item to show diagnostic count
        const statusBarItem: StatusBarItem = window.createStatusBarItem(StatusBarAlignment.Left, 100);
        statusBarItem.text = 'RoboDSL: $(sync~spin)';
        statusBarItem.tooltip = 'RoboDSL diagnostics';
        statusBarItem.show();

        function refreshDiagnosticCount() {
            const allDiagnostics = languages.getDiagnostics();
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
        context.subscriptions.push(languages.onDidChangeDiagnostics(refreshDiagnosticCount));
    }).catch((error) => {
        console.error('Failed to start language client:', error);
        window.showErrorMessage(`Failed to start RoboDSL Language Server: ${error.message}`);
    });

    function extractCodeBlocks(text: string): string[] {
        const blocks: string[] = [];
        const regex = /code\s*\{([\s\S]*?)\}/g;
        let match;
        while ((match = regex.exec(text)) !== null) {
            blocks.push(match[1]);
        }
        return blocks;
    }

    function isInsideCodeBlock(document: any, position: Position): boolean {
        const text = document.getText();
        const offset = document.offsetAt(position);
        const regex = /code\s*\{([\s\S]*?)\}/g;
        let match;
        while ((match = regex.exec(text)) !== null) {
            const start = match.index;
            const end = match.index + match[0].length;
            if (offset >= start && offset <= end) return true;
        }
        return false;
    }

    function toVirtualCppUri(document: any, position: Position): { virtualUri: Uri; innerPos: Position } {
        const text = document.getText();
        const offset = document.offsetAt(position);
        const regex = /code\s*\{([\s\S]*?)\}/g;
        let match;
        let idx = 0;
        while ((match = regex.exec(text)) !== null) {
            const start = match.index;
            const end = match.index + match[0].length;
            if (offset >= start && offset <= end) {
                const virtualUri = Uri.parse(`${EMBED_SCHEME}:${document.uri.fsPath}#${idx}`);
                const innerOffset = offset - start - (match[0].indexOf('{') + 1);
                const codeText = match[1];
                const lines = codeText.substring(0, innerOffset).split('\n');
                const innerPos = new Position(lines.length - 1, lines[lines.length - 1].length);
                return { virtualUri, innerPos };
            }
            idx++;
        }
        // Fallback
        return { virtualUri: document.uri, innerPos: position };
    }
}

export function deactivate(): Promise<void> | undefined {
    console.log('RoboDSL Extension deactivating...');
    if (!client) {
        return undefined;
    }
    return client.stop();
} 