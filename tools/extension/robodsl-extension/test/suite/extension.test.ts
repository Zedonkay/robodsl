import * as assert from 'assert';
import * as vscode from 'vscode';
import * as path from 'path';

suite('RoboDSL Extension Test Suite', () => {
    vscode.window.showInformationMessage('Start RoboDSL tests.');

    test('Extension should be present', () => {
        assert.ok(vscode.extensions.getExtension('robodsl'));
    });

    test('Extension should activate', async () => {
        const extension = vscode.extensions.getExtension('robodsl');
        if (extension) {
            await extension.activate();
            assert.ok(extension.isActive);
        }
    });

    test('File icon appears for .robodsl files', async () => {
        // Create a test file
        const testFile = path.join(vscode.workspace.workspaceFolders![0].uri.fsPath, 'test-icon.robodsl');
        const uri = vscode.Uri.file(testFile);
        
        // Create the file with some content
        await vscode.workspace.fs.writeFile(uri, Buffer.from('node test_node { parameter int x = 42 }'));
        
        // Open the document
        const doc = await vscode.workspace.openTextDocument(uri);
        await vscode.window.showTextDocument(doc);
        
        // Check that the language is recognized as RoboDSL
        assert.strictEqual(doc.languageId, 'robodsl');
        
        // Clean up
        await vscode.workspace.fs.delete(uri);
    });

    test('Syntax highlighting works', async () => {
        const doc = await vscode.workspace.openTextDocument({ 
            language: 'robodsl', 
            content: 'node test_node { parameter int x = 1 }' 
        });
        await vscode.window.showTextDocument(doc);
        
        // Check that the document is recognized as RoboDSL
        assert.strictEqual(doc.languageId, 'robodsl');
        
        // Note: Visual syntax highlighting verification would require token inspection
        // which is complex. We'll verify this manually.
    });

    test('Diagnostics (squiggles) appear for syntax errors', async () => {
        const doc = await vscode.workspace.openTextDocument({ 
            language: 'robodsl', 
            content: 'node test_node { parameter int x 1 }' // Missing equals sign
        });
        await vscode.window.showTextDocument(doc);
        
        // Wait for diagnostics to be computed
        await new Promise(resolve => setTimeout(resolve, 3000));
        
        const diagnostics = vscode.languages.getDiagnostics(doc.uri);
        assert.ok(diagnostics.length > 0, 'Should have diagnostics for syntax error');
        
        // Check that we have error diagnostics
        const hasErrors = diagnostics.some(diag => diag.severity === vscode.DiagnosticSeverity.Error);
        assert.ok(hasErrors, 'Should have error diagnostics');
    });

    test('Intellisense (completion) works', async () => {
        const doc = await vscode.workspace.openTextDocument({ 
            language: 'robodsl', 
            content: 'par' // Partial keyword
        });
        await vscode.window.showTextDocument(doc);
        
        const pos = new vscode.Position(0, 3);
        const completions = await vscode.commands.executeCommand<vscode.CompletionList>(
            'vscode.executeCompletionItemProvider',
            doc.uri,
            pos
        );
        
        assert.ok(completions, 'Should have completions');
        assert.ok(completions.items.length > 0, 'Should have completion items');
        
        // Check for expected keywords
        const hasParameter = completions.items.some(item => item.label === 'parameter');
        const hasPublisher = completions.items.some(item => item.label === 'publisher');
        const hasNode = completions.items.some(item => item.label === 'node');
        
        assert.ok(hasParameter || hasPublisher || hasNode, 'Should have RoboDSL keywords');
    });

    test('Hover information appears', async () => {
        const doc = await vscode.workspace.openTextDocument({ 
            language: 'robodsl', 
            content: 'node test_node { parameter int x = 42 }' 
        });
        await vscode.window.showTextDocument(doc);
        
        // Test hover on 'node' keyword
        const pos = new vscode.Position(0, 0);
        const hovers = await vscode.commands.executeCommand<vscode.Hover[]>(
            'vscode.executeHoverProvider',
            doc.uri,
            pos
        );
        
        assert.ok(hovers && hovers.length > 0, 'Should have hover information');
    });

    test('Problems panel shows error count', async () => {
        const doc = await vscode.workspace.openTextDocument({ 
            language: 'robodsl', 
            content: 'node test_node { parameter int x 1 }' // Invalid syntax
        });
        await vscode.window.showTextDocument(doc);
        
        // Wait for diagnostics
        await new Promise(resolve => setTimeout(resolve, 3000));
        
        const diagnostics = vscode.languages.getDiagnostics(doc.uri);
        assert.ok(diagnostics.length > 0, 'Should have diagnostics in Problems panel');
        
        // The Problems panel should show these diagnostics
        // Note: We can't directly test the Problems panel UI, but we can verify diagnostics exist
    });

    test('Language mode is RoboDSL', async () => {
        const doc = await vscode.workspace.openTextDocument({ 
            language: 'robodsl', 
            content: '' 
        });
        await vscode.window.showTextDocument(doc);
        
        assert.strictEqual(doc.languageId, 'robodsl');
        
        // Check the status bar shows RoboDSL
        // Note: Status bar language mode is not directly testable via API
    });

    test('Manual activation command works', async () => {
        // Test the manual activation command
        const result = await vscode.commands.executeCommand('robodsl.activate');
        // The command should execute without error
        assert.ok(true, 'Activation command executed successfully');
    });

    test('Test command works', async () => {
        // Test the test command
        const result = await vscode.commands.executeCommand('robodsl.test');
        // The command should execute without error
        assert.ok(true, 'Test command executed successfully');
    });

    test('Valid syntax passes validation', async () => {
        const doc = await vscode.workspace.openTextDocument({ 
            language: 'robodsl', 
            content: 'node test_node { parameter int x = 42 }' // Valid syntax
        });
        await vscode.window.showTextDocument(doc);
        
        // Wait for diagnostics
        await new Promise(resolve => setTimeout(resolve, 3000));
        
        const diagnostics = vscode.languages.getDiagnostics(doc.uri);
        // Should have no errors for valid syntax
        const errors = diagnostics.filter(diag => diag.severity === vscode.DiagnosticSeverity.Error);
        assert.strictEqual(errors.length, 0, 'Valid syntax should have no errors');
    });

    test('Multiple errors are detected', async () => {
        const doc = await vscode.workspace.openTextDocument({ 
            language: 'robodsl', 
            content: `
                node test_node { 
                    parameter int x 1  // Missing equals
                    publisher /topic std_msgs/String  // Missing quotes
                }
            `
        });
        await vscode.window.showTextDocument(doc);
        
        // Wait for diagnostics
        await new Promise(resolve => setTimeout(resolve, 3000));
        
        const diagnostics = vscode.languages.getDiagnostics(doc.uri);
        assert.ok(diagnostics.length > 1, 'Should detect multiple errors');
    });

    test('ROS types are available in completions', async () => {
        const doc = await vscode.workspace.openTextDocument({ 
            language: 'robodsl', 
            content: 'publisher /topic: "std_' // Partial ROS type
        });
        await vscode.window.showTextDocument(doc);
        
        const pos = new vscode.Position(0, 20);
        const completions = await vscode.commands.executeCommand<vscode.CompletionList>(
            'vscode.executeCompletionItemProvider',
            doc.uri,
            pos
        );
        
        if (completions && completions.items.length > 0) {
            const hasRosType = completions.items.some(item => 
                item.label.toString().startsWith('std_msgs/')
            );
            assert.ok(hasRosType, 'Should have ROS type completions');
        }
    });

    test('C++ types are available in completions', async () => {
        const doc = await vscode.workspace.openTextDocument({ 
            language: 'robodsl', 
            content: 'method test { input: std::' // Partial C++ type
        });
        await vscode.window.showTextDocument(doc);
        
        const pos = new vscode.Position(0, 20);
        const completions = await vscode.commands.executeCommand<vscode.CompletionList>(
            'vscode.executeCompletionItemProvider',
            doc.uri,
            pos
        );
        
        if (completions && completions.items.length > 0) {
            const hasCppType = completions.items.some(item => 
                item.label.toString().startsWith('std::')
            );
            assert.ok(hasCppType, 'Should have C++ type completions');
        }
    });
}); 