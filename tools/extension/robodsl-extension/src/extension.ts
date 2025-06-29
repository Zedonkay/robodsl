import * as path from 'path';
import { workspace, ExtensionContext, window } from 'vscode';

import {
    LanguageClient,
    TransportKind
} from 'vscode-languageclient/node';

let client: LanguageClient;

export function activate(context: ExtensionContext) {
    console.log('RoboDSL extension is now active!');

    // The server is implemented in node
    const serverModule = context.asAbsolutePath(path.join('server', 'out', 'server.js'));
    console.log('Server module path:', serverModule);
    
    // The debug options for the server
    // --inspect=6009: runs the server in Node's Inspector mode so VS Code can attach to the server for debugging
    const debugOptions = { execArgv: ['--nolazy', '--inspect=6009'] };

    // If the extension is launched in debug mode then the debug server options are used
    // Otherwise the run options are used
    const serverOptions = {
        run: { module: serverModule, transport: TransportKind.stdio },
        debug: {
            module: serverModule,
            transport: TransportKind.stdio,
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
            fileEvents: workspace.createFileSystemWatcher('**/*.robodsl')
        }
    };

    // Create the language client and start the client.
    client = new LanguageClient(
        'robodslLanguageServer',
        'RoboDSL Language Server',
        serverOptions,
        clientOptions
    );

    // Start the client. This will also launch the server
    client.start().then(() => {
        console.log('RoboDSL Language Server started successfully');
        window.showInformationMessage('RoboDSL Language Server is now active!');
    }).catch((error) => {
        console.error('Failed to start RoboDSL Language Server:', error);
        window.showErrorMessage('Failed to start RoboDSL Language Server: ' + error.message);
    });
}

export function deactivate(): Promise<void> | undefined {
    if (!client) {
        return undefined;
    }
    return client.stop();
} 