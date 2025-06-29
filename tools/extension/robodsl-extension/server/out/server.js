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
const node_1 = require("vscode-languageserver/node");
const vscode_languageserver_textdocument_1 = require("vscode-languageserver-textdocument");
const child_process_1 = require("child_process");
const path = __importStar(require("path"));
// Create a connection for the server
const connection = (0, node_1.createConnection)(node_1.ProposedFeatures.all);
// Create a text document manager
const documents = new node_1.TextDocuments(vscode_languageserver_textdocument_1.TextDocument);
let hasConfigurationCapability = false;
let hasWorkspaceFolderCapability = false;
let hasDiagnosticRelatedInformationCapability = false;
// Keywords and completions from the grammar
const keywords = [
    'node', 'cuda_kernels', 'kernel', 'method', 'parameter', 'remap', 'namespace',
    'flag', 'lifecycle', 'timer', 'client', 'publisher', 'subscriber', 'service',
    'action', 'include', 'input', 'output', 'code', 'in', 'out', 'inout',
    'block_size', 'grid_size', 'shared_memory', 'use_thrust', 'qos', 'onnx_model',
    'device', 'optimization', 'pipeline', 'stage', 'config', 'true', 'false'
];
const rosTypes = [
    'std_msgs/String', 'std_msgs/Int32', 'std_msgs/Float64', 'std_msgs/Bool',
    'geometry_msgs/Twist', 'geometry_msgs/Pose', 'geometry_msgs/Point',
    'sensor_msgs/Image', 'sensor_msgs/LaserScan', 'sensor_msgs/PointCloud2',
    'nav_msgs/Odometry', 'nav_msgs/Path', 'nav_msgs/OccupancyGrid'
];
const cppTypes = [
    'int', 'float', 'double', 'bool', 'char', 'string', 'std::string',
    'std::vector', 'std::array', 'std::map', 'std::unordered_map',
    'cv::Mat', 'cv::Point', 'cv::Point2f', 'cv::Point3f',
    'Eigen::Vector3d', 'Eigen::Matrix3d', 'Eigen::Quaterniond'
];
// Python parser bridge integration
async function getDiagnosticsFromPython(text) {
    return new Promise((resolve, reject) => {
        try {
            const pyScriptPath = path.join(__dirname, '..', 'py_parser_bridge.py');
            console.log('Python script path:', pyScriptPath);
            const py = (0, child_process_1.spawn)('python3', [pyScriptPath]);
            let output = '';
            let error = '';
            py.stdout.on('data', (data) => {
                output += data.toString();
            });
            py.stderr.on('data', (data) => {
                error += data.toString();
            });
            py.on('close', (code) => {
                console.log(`Python bridge process exited with code: ${code}`);
                if (code !== 0) {
                    console.error('Python bridge error output:', error);
                    reject(new Error(`Python bridge failed with code ${code}: ${error}`));
                }
                else {
                    try {
                        console.log('Python bridge output:', output);
                        const diagnostics = JSON.parse(output);
                        resolve(diagnostics);
                    }
                    catch (e) {
                        console.error('Failed to parse Python output:', e);
                        reject(new Error(`Failed to parse Python output: ${e.message}`));
                    }
                }
            });
            py.on('error', (err) => {
                console.error('Failed to spawn Python process:', err);
                reject(new Error(`Failed to spawn Python process: ${err.message}`));
            });
            // Send text to Python script
            py.stdin.write(text);
            py.stdin.end();
        }
        catch (error) {
            console.error('Error in getDiagnosticsFromPython:', error);
            reject(error);
        }
    });
}
// Convert Python diagnostics to VS Code diagnostics
function convertToVSCodeDiagnostics(pythonDiagnostics, textDocument) {
    const diagnostics = [];
    const totalLines = textDocument.getText().split('\n');
    for (const pyDiag of pythonDiagnostics) {
        const line = Math.max(0, pyDiag.line || 0);
        const character = Math.max(0, pyDiag.column || 0);
        const lineText = totalLines[line] || '';
        const range = node_1.Range.create(line, character, line, Math.min(lineText.length, character + 1));
        const severity = pyDiag.level === 'error' ? node_1.DiagnosticSeverity.Error : node_1.DiagnosticSeverity.Warning;
        const diagnostic = {
            range,
            message: pyDiag.message,
            severity,
            source: 'robodsl'
        };
        diagnostics.push(diagnostic);
    }
    return diagnostics;
}
connection.onInitialize((params) => {
    console.log('RoboDSL Language Server initializing...');
    try {
        const capabilities = params.capabilities;
        // Does the client support the `workspace/configuration` request?
        // If not, we will fall back using global settings.
        hasConfigurationCapability = !!(capabilities.workspace && !!capabilities.workspace.configuration);
        hasWorkspaceFolderCapability = !!(capabilities.workspace && !!capabilities.workspace.workspaceFolders);
        hasDiagnosticRelatedInformationCapability = !!(capabilities.textDocument &&
            capabilities.textDocument.publishDiagnostics &&
            capabilities.textDocument.publishDiagnostics.relatedInformation);
        const result = {
            capabilities: {
                textDocumentSync: node_1.TextDocumentSyncKind.Incremental,
                // Tell the client that this server supports code completion.
                completionProvider: {
                    resolveProvider: true,
                    triggerCharacters: ['.', ':', ' ', '\n']
                },
                // Tell the client that this server supports hover.
                hoverProvider: true,
                // Tell the client that this server supports diagnostics.
                diagnosticProvider: {
                    interFileDependencies: false,
                    workspaceDiagnostics: false
                },
                // Add semantic tokens provider for better syntax highlighting
                semanticTokensProvider: {
                    legend: {
                        tokenTypes: ['keyword', 'type', 'function', 'variable', 'string', 'comment', 'operator', 'number'],
                        tokenModifiers: ['declaration', 'definition', 'readonly', 'static', 'deprecated']
                    },
                    range: false,
                    full: {
                        delta: false
                    }
                }
            }
        };
        if (hasWorkspaceFolderCapability) {
            result.capabilities.workspace = {
                workspaceFolders: {
                    supported: true
                }
            };
        }
        console.log('RoboDSL Language Server initialized with capabilities:', result.capabilities);
        return result;
    }
    catch (error) {
        console.error('Error during server initialization:', error);
        throw error;
    }
});
connection.onInitialized(() => {
    console.log('RoboDSL Language Server initialized!');
    if (hasConfigurationCapability) {
        // Register for all configuration changes.
        connection.client.register(node_1.DidChangeConfigurationNotification.type, undefined);
    }
    if (hasWorkspaceFolderCapability) {
        connection.workspace.onDidChangeWorkspaceFolders(_event => {
            connection.console.log('Workspace folder change event received.');
        });
    }
});
// The global settings, used when the `workspace/configuration` request is not supported by the client.
// Please note that this is not the case when using this server with the client provided in this example
// but could happen with other clients.
const defaultSettings = { maxNumberOfProblems: 1000 };
let globalSettings = defaultSettings;
// Cache the settings of all open documents
const documentSettings = new Map();
connection.onDidChangeConfiguration(change => {
    if (hasConfigurationCapability) {
        // Reset all cached document settings
        documentSettings.clear();
    }
    else {
        globalSettings = ((change.settings.robodslLanguageServer || defaultSettings));
    }
    // Revalidate all open text documents
    documents.all().forEach(validateTextDocument);
});
function getDocumentSettings(resource) {
    if (!hasConfigurationCapability) {
        return Promise.resolve(globalSettings);
    }
    let result = documentSettings.get(resource);
    if (!result) {
        result = connection.workspace.getConfiguration({
            scopeUri: resource,
            section: 'robodslLanguageServer'
        });
        documentSettings.set(resource, result);
    }
    return result;
}
// Only keep settings for open documents
documents.onDidClose(e => {
    documentSettings.delete(e.document.uri);
});
// The content of a text document has changed. This event is emitted
// when the text document first opened or when its content has changed.
documents.onDidChangeContent(change => {
    validateTextDocument(change.document);
});
async function validateTextDocument(textDocument) {
    console.log(`Validating document: ${textDocument.uri}`);
    // In this simple example we get the settings for every validate run.
    const settings = await getDocumentSettings(textDocument.uri);
    // The validator creates diagnostics for all syntax errors.
    const text = textDocument.getText();
    const diagnostics = [];
    try {
        // Get diagnostics from Python parser
        const pythonDiagnostics = await getDiagnosticsFromPython(text);
        const vsCodeDiagnostics = convertToVSCodeDiagnostics(pythonDiagnostics, textDocument);
        diagnostics.push(...vsCodeDiagnostics);
        console.log(`Found ${diagnostics.length} diagnostics`);
    }
    catch (error) {
        console.error('Error getting diagnostics from Python parser:', error);
        // Add a diagnostic for the parser error
        const diagnostic = {
            range: node_1.Range.create(0, 0, 0, 1),
            message: `Parser error: ${error.message}`,
            severity: node_1.DiagnosticSeverity.Error,
            source: 'robodsl'
        };
        diagnostics.push(diagnostic);
    }
    // Send the computed diagnostics to VS Code.
    connection.sendDiagnostics({ uri: textDocument.uri, diagnostics });
}
connection.onDidChangeWatchedFiles(_change => {
    // Monitored files have change in VS Code
    connection.console.log('We received a file change event');
});
// This handler provides the initial list of completion items.
connection.onCompletion((_textDocumentPosition) => {
    // The pass parameter contains the position of the text document in
    // which code complete got requested. For the example we ignore this
    // info and always provide the same completion items.
    const completions = [];
    // Add keywords
    keywords.forEach(keyword => {
        completions.push({
            label: keyword,
            kind: node_1.CompletionItemKind.Keyword,
            detail: `RoboDSL keyword: ${keyword}`,
            documentation: getKeywordDocumentation(keyword)
        });
    });
    // Add ROS types
    rosTypes.forEach(rosType => {
        completions.push({
            label: rosType,
            kind: node_1.CompletionItemKind.Class,
            detail: `ROS message type: ${rosType}`,
            documentation: `ROS message type for ${rosType}`
        });
    });
    // Add C++ types
    cppTypes.forEach(cppType => {
        completions.push({
            label: cppType,
            kind: node_1.CompletionItemKind.TypeParameter,
            detail: `C++ type: ${cppType}`,
            documentation: `C++ type for ${cppType}`
        });
    });
    return completions;
});
// This handler resolves additional information for the item selected in
// the completion list.
connection.onCompletionResolve((item) => {
    if (item.data === 1) {
        item.detail = 'TypeScript details',
            item.documentation = 'TypeScript documentation';
    }
    else if (item.data === 2) {
        item.detail = 'JavaScript details',
            item.documentation = 'JavaScript documentation';
    }
    return item;
});
connection.onHover((params) => {
    const textDocument = documents.get(params.textDocument.uri);
    if (!textDocument) {
        return null;
    }
    const position = params.position;
    const offset = textDocument.offsetAt(position);
    const text = textDocument.getText();
    const lines = text.split('\n');
    const line = lines[position.line];
    // Get word at position
    const wordRange = getWordRangeAtPosition(line, position.character);
    if (!wordRange) {
        return null;
    }
    const word = line.substring(wordRange.start, wordRange.end);
    const hoverInfo = getHoverInformation(word);
    if (hoverInfo) {
        return {
            contents: {
                kind: 'markdown',
                value: hoverInfo
            }
        };
    }
    return null;
});
function getWordRangeAtPosition(line, character) {
    const wordRegex = /\b\w+\b/g;
    let match;
    while ((match = wordRegex.exec(line)) !== null) {
        const start = match.index;
        const end = start + match[0].length;
        if (character >= start && character <= end) {
            return { start, end };
        }
    }
    return null;
}
function getKeywordDocumentation(keyword) {
    const documentation = {
        'node': 'Defines a ROS2 node with its configuration, parameters, and behavior.',
        'parameter': 'Declares a parameter with type, name, and default value.',
        'publisher': 'Declares a publisher for a specific topic and message type.',
        'subscriber': 'Declares a subscriber for a specific topic and message type.',
        'service': 'Declares a service client for a specific service type.',
        'action': 'Declares an action client for a specific action type.',
        'method': 'Defines a C++ method with input/output parameters and code.',
        'cuda_kernels': 'Defines CUDA kernels for GPU acceleration.',
        'kernel': 'Defines a single CUDA kernel with configuration and code.',
        'onnx_model': 'Defines an ONNX model for machine learning inference.',
        'pipeline': 'Defines a processing pipeline with multiple stages.',
        'timer': 'Defines a timer callback with period and configuration.',
        'namespace': 'Sets the namespace for the node.',
        'remap': 'Remaps topics from one path to another.',
        'flag': 'Sets boolean flags for node configuration.',
        'lifecycle': 'Configures lifecycle settings for the node.',
        'include': 'Includes external files or headers.'
    };
    return documentation[keyword] || `RoboDSL keyword: ${keyword}`;
}
function getHoverInformation(word) {
    const doc = getKeywordDocumentation(word);
    if (doc) {
        return `**${word}**\n\n${doc}`;
    }
    // Check if it's a ROS type
    if (rosTypes.includes(word)) {
        return `**${word}**\n\nROS message type for ${word}`;
    }
    // Check if it's a C++ type
    if (cppTypes.includes(word)) {
        return `**${word}**\n\nC++ type for ${word}`;
    }
    return null;
}
// Make the text document manager listen on the connection
// for open, change and close text document events
documents.listen(connection);
// Listen on the connection
connection.listen();
//# sourceMappingURL=server.js.map