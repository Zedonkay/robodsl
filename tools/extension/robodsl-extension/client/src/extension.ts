/* --------------------------------------------------------------------------------------------
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See License.txt in the project root for license information.
 * ------------------------------------------------------------------------------------------ */

import * as path from 'path';
import { workspace, ExtensionContext, window, commands } from 'vscode';

import {
    LanguageClient,
    LanguageClientOptions,
    ServerOptions,
    TransportKind
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
    
    // The server is implemented in node
    const serverModule = context.asAbsolutePath(
        path.join('server', 'out', 'server.js')
    );

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
        progressOnInitialization: true
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
    }).catch((error) => {
        console.error('Failed to start language client:', error);
        window.showErrorMessage(`Failed to start RoboDSL Language Server: ${error.message}`);
    });
}

export function deactivate(): Promise<void> | undefined {
    console.log('RoboDSL Extension deactivating...');
    if (!client) {
        return undefined;
    }
    return client.stop();
} 