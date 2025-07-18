"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
const node_1 = require("vscode-languageserver/node");
const vscode_languageserver_textdocument_1 = require("vscode-languageserver-textdocument");
// Create a connection for the server
const connection = (0, node_1.createConnection)(node_1.ProposedFeatures.all);
// Create a text document manager
const documents = new node_1.TextDocuments(vscode_languageserver_textdocument_1.TextDocument);
let hasConfigurationCapability = false;
let hasWorkspaceFolderCapability = false;
let hasDiagnosticRelatedInformationCapability = false;
// Keywords and completions
const keywords = [
    'node', 'cuda_kernels', 'kernel', 'method', 'parameter', 'remap', 'namespace',
    'flag', 'lifecycle', 'timer', 'client', 'publisher', 'subscriber', 'service',
    'action', 'include', 'input', 'output', 'code', 'in', 'out', 'inout',
    'block_size', 'grid_size', 'shared_memory', 'use_thrust', 'qos', 'onnx_model',
    'device', 'optimization', 'pipeline', 'stage', 'config', 'project_name',
    'parameter_callbacks', 'lifecycle_config', 'queue_size', 'autostart',
    'cleanup_on_shutdown', 'oneshot', 'no_autostart', 'true', 'false'
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
const qosSettings = [
    'reliability', 'durability', 'history', 'depth', 'deadline',
    'lifespan', 'liveliness', 'liveliness_lease_duration'
];
const qosValues = [
    'reliable', 'best_effort', 'transient_local', 'volatile',
    'keep_last', 'keep_all', 'automatic', 'manual_by_topic'
];
// Simple parser for basic syntax validation
class RoboDSLParser {
    constructor() {
        this.errors = [];
    }
    parse(text) {
        this.errors = [];
        const lines = text.split('\n');
        // Use a stack-based approach for proper bracket matching
        const bracketStack = [];
        let inString = false;
        let inComment = false;
        let inBlockComment = false;
        let stringChar = '';
        for (let i = 0; i < lines.length; i++) {
            const line = lines[i];
            const lineNumber = i + 1;
            const trimmedLine = line.trim();
            // Skip empty lines
            if (trimmedLine === '') {
                continue;
            }
            // Process the line character by character for proper context awareness
            for (let j = 0; j < line.length; j++) {
                const char = line[j];
                const nextChar = j < line.length - 1 ? line[j + 1] : '';
                // Handle comments
                if (!inString && !inBlockComment) {
                    if (char === '/' && nextChar === '/') {
                        // Line comment - skip rest of line
                        break;
                    }
                    else if (char === '/' && nextChar === '*') {
                        // Start block comment
                        inBlockComment = true;
                        j++; // Skip next character
                        continue;
                    }
                }
                if (inBlockComment) {
                    if (char === '*' && nextChar === '/') {
                        // End block comment
                        inBlockComment = false;
                        j++; // Skip next character
                        continue;
                    }
                    continue; // Skip everything in block comment
                }
                // Handle strings
                if (!inComment && !inBlockComment) {
                    if ((char === '"' || char === "'") && !inString) {
                        // Start string
                        inString = true;
                        stringChar = char;
                        continue;
                    }
                    else if (inString && char === stringChar) {
                        // End string
                        inString = false;
                        stringChar = '';
                        continue;
                    }
                }
                // Only process brackets if not in string or comment
                if (!inString && !inComment && !inBlockComment) {
                    if (char === '{') {
                        bracketStack.push({
                            type: '{',
                            line: lineNumber,
                            column: j
                        });
                    }
                    else if (char === '}') {
                        if (bracketStack.length === 0) {
                            // Extra closing brace
                            this.errors.push({
                                line: lineNumber,
                                column: j,
                                message: 'Unexpected closing brace }'
                            });
                        }
                        else {
                            const lastBracket = bracketStack.pop();
                            if (lastBracket.type !== '{') {
                                // Mismatched bracket type
                                this.errors.push({
                                    line: lineNumber,
                                    column: j,
                                    message: `Mismatched bracket: expected ${lastBracket.type === '[' ? ']' : ')'} but found }`
                                });
                            }
                        }
                    }
                    else if (char === '[') {
                        bracketStack.push({
                            type: '[',
                            line: lineNumber,
                            column: j
                        });
                    }
                    else if (char === ']') {
                        if (bracketStack.length === 0) {
                            this.errors.push({
                                line: lineNumber,
                                column: j,
                                message: 'Unexpected closing bracket ]'
                            });
                        }
                        else {
                            const lastBracket = bracketStack.pop();
                            if (lastBracket.type !== '[') {
                                this.errors.push({
                                    line: lineNumber,
                                    column: j,
                                    message: `Mismatched bracket: expected ${lastBracket.type === '{' ? '}' : ')'} but found ]`
                                });
                            }
                        }
                    }
                    else if (char === '(') {
                        bracketStack.push({
                            type: '(',
                            line: lineNumber,
                            column: j
                        });
                    }
                    else if (char === ')') {
                        if (bracketStack.length === 0) {
                            this.errors.push({
                                line: lineNumber,
                                column: j,
                                message: 'Unexpected closing parenthesis )'
                            });
                        }
                        else {
                            const lastBracket = bracketStack.pop();
                            if (lastBracket.type !== '(') {
                                this.errors.push({
                                    line: lineNumber,
                                    column: j,
                                    message: `Mismatched bracket: expected ${lastBracket.type === '{' ? '}' : ']'} but found )`
                                });
                            }
                        }
                    }
                }
            }
            // Check for syntax errors in this line
            this.checkLineSyntax(trimmedLine, lineNumber, i);
        }
        // Check for unmatched opening brackets
        bracketStack.forEach(bracket => {
            this.errors.push({
                line: bracket.line,
                column: bracket.column,
                message: `Unmatched opening ${bracket.type}`
            });
        });
        return this.errors.length === 0;
    }
    checkLineSyntax(line, lineNumber, lineIndex) {
        // Check for missing colons after keywords
        const keywordPattern = /\b(node|parameter|timer|remap|namespace|flag|publisher|subscriber|service|client|action|method|kernel|input|output|code|block_size|grid_size|shared_memory|use_thrust|device|optimization)\b/;
        const match = line.match(keywordPattern);
        if (match && !line.includes(':') && !line.includes('{') && !line.includes('=')) {
            const keyword = match[0];
            const column = line.indexOf(keyword);
            this.errors.push({
                line: lineNumber,
                column: column,
                message: `Missing colon (:) after keyword '${keyword}'`
            });
        }
        // Check for invalid syntax patterns
        if (line.includes('node') && !line.includes('{') && !line.includes(':')) {
            this.errors.push({
                line: lineNumber,
                column: line.indexOf('node'),
                message: 'Node definition must be followed by opening brace {'
            });
        }
        // Check for missing quotes in topic paths
        if (line.includes('publisher') || line.includes('subscriber') || line.includes('service') || line.includes('client') || line.includes('action')) {
            if (line.includes(':') && !line.includes('"')) {
                this.errors.push({
                    line: lineNumber,
                    column: line.indexOf(':'),
                    message: 'Topic type must be quoted'
                });
            }
        }
    }
    getErrors() {
        return this.errors;
    }
}
const parser = new RoboDSLParser();
connection.onInitialize((params) => {
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
            // Tell the client that this server supports go to definition.
            definitionProvider: true,
            // Tell the client that this server supports document symbols.
            documentSymbolProvider: true
        }
    };
    if (hasWorkspaceFolderCapability) {
        result.capabilities.workspace = {
            workspaceFolders: {
                supported: true
            }
        };
    }
    return result;
});
connection.onInitialized(() => {
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
const defaultSettings = { maxNumberOfProblems: 1000, enableBracketValidation: true };
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
    const settings = await getDocumentSettings(textDocument.uri);
    const text = textDocument.getText();
    const diagnostics = [];
    // Parse the document
    const isValid = parser.parse(text);
    const errors = parser.getErrors();
    // Convert parser errors to diagnostics, respecting the bracket validation setting
    errors.forEach(error => {
        // Skip bracket validation errors if disabled
        if (!settings.enableBracketValidation &&
            (error.message.includes('brace') ||
                error.message.includes('bracket') ||
                error.message.includes('parenthesis') ||
                error.message.includes('Unmatched') ||
                error.message.includes('Unexpected'))) {
            return;
        }
        diagnostics.push({
            severity: node_1.DiagnosticSeverity.Error,
            range: {
                start: { line: error.line - 1, character: error.column },
                end: { line: error.line - 1, character: error.column + 1 }
            },
            message: error.message,
            source: 'robodsl'
        });
    });
    // Send the computed diagnostics to VSCode.
    connection.sendDiagnostics({ uri: textDocument.uri, diagnostics });
}
connection.onDidChangeWatchedFiles(_change => {
    // Monitored files have change in VSCode
    connection.console.log('We received a file change event');
});
// This handler provides the initial list of completion items.
connection.onCompletion((_textDocumentPosition) => {
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
    rosTypes.forEach(type => {
        completions.push({
            label: type,
            kind: node_1.CompletionItemKind.Class,
            detail: `ROS message type: ${type}`,
            documentation: `ROS message type for ${type.split('/')[1]}`
        });
    });
    // Add C++ types
    cppTypes.forEach(type => {
        completions.push({
            label: type,
            kind: node_1.CompletionItemKind.TypeParameter,
            detail: `C++ type: ${type}`,
            documentation: `C++ data type`
        });
    });
    // Add QoS settings
    qosSettings.forEach(setting => {
        completions.push({
            label: setting,
            kind: node_1.CompletionItemKind.Property,
            detail: `QoS setting: ${setting}`,
            documentation: `Quality of Service setting for ROS topics`
        });
    });
    // Add QoS values
    qosValues.forEach(value => {
        completions.push({
            label: value,
            kind: node_1.CompletionItemKind.Value,
            detail: `QoS value: ${value}`,
            documentation: `Quality of Service value`
        });
    });
    return completions;
});
// This handler resolves additional information for the item selected in
// the completion list.
connection.onCompletionResolve((item) => {
    if (item.data === 1) {
        item.detail = 'TypeScript details';
        item.documentation = 'TypeScript documentation';
    }
    else if (item.data === 2) {
        item.detail = 'JavaScript details';
        item.documentation = 'JavaScript documentation';
    }
    return item;
});
connection.onHover((params) => {
    const document = documents.get(params.textDocument.uri);
    if (!document) {
        return null;
    }
    const position = params.position;
    const text = document.getText();
    const lines = text.split('\n');
    const line = lines[position.line];
    // Find the word at the cursor position
    const wordRange = getWordRangeAtPosition(line, position.character);
    if (!wordRange) {
        return null;
    }
    const word = line.substring(wordRange.start, wordRange.end);
    // Provide hover information based on the word
    const hoverInfo = getHoverInformation(word);
    if (hoverInfo) {
        return {
            contents: {
                kind: 'markdown',
                value: hoverInfo
            },
            range: {
                start: { line: position.line, character: wordRange.start },
                end: { line: position.line, character: wordRange.end }
            }
        };
    }
    return null;
});
connection.onDefinition((params) => {
    const document = documents.get(params.textDocument.uri);
    if (!document) {
        return null;
    }
    const position = params.position;
    const text = document.getText();
    const lines = text.split('\n');
    const line = lines[position.line];
    // Find the word at the cursor position
    const wordRange = getWordRangeAtPosition(line, position.character);
    if (!wordRange) {
        return null;
    }
    const word = line.substring(wordRange.start, wordRange.end);
    // Look for definitions in the document
    for (let i = 0; i < lines.length; i++) {
        const currentLine = lines[i];
        if (currentLine.includes(`node ${word}`) ||
            currentLine.includes(`method ${word}`) ||
            currentLine.includes(`kernel ${word}`) ||
            currentLine.includes(`parameter ${word}`)) {
            return {
                uri: params.textDocument.uri,
                range: {
                    start: { line: i, character: 0 },
                    end: { line: i, character: currentLine.length }
                }
            };
        }
    }
    return null;
});
connection.onDocumentSymbol((params) => {
    const document = documents.get(params.textDocument.uri);
    if (!document) {
        return [];
    }
    const symbols = [];
    const text = document.getText();
    const lines = text.split('\n');
    for (let i = 0; i < lines.length; i++) {
        const line = lines[i];
        // Find node definitions
        const nodeMatch = line.match(/node\s+([a-zA-Z_][a-zA-Z0-9_]*)/);
        if (nodeMatch) {
            symbols.push({
                name: nodeMatch[1],
                kind: node_1.SymbolKind.Class,
                range: {
                    start: { line: i, character: 0 },
                    end: { line: i, character: line.length }
                },
                selectionRange: {
                    start: { line: i, character: line.indexOf(nodeMatch[1]) },
                    end: { line: i, character: line.indexOf(nodeMatch[1]) + nodeMatch[1].length }
                }
            });
        }
        // Find method definitions
        const methodMatch = line.match(/method\s+([a-zA-Z_][a-zA-Z0-9_]*)/);
        if (methodMatch) {
            symbols.push({
                name: methodMatch[1],
                kind: node_1.SymbolKind.Method,
                range: {
                    start: { line: i, character: 0 },
                    end: { line: i, character: line.length }
                },
                selectionRange: {
                    start: { line: i, character: line.indexOf(methodMatch[1]) },
                    end: { line: i, character: line.indexOf(methodMatch[1]) + methodMatch[1].length }
                }
            });
        }
        // Find kernel definitions
        const kernelMatch = line.match(/kernel\s+([a-zA-Z_][a-zA-Z0-9_]*)/);
        if (kernelMatch) {
            symbols.push({
                name: kernelMatch[1],
                kind: node_1.SymbolKind.Function,
                range: {
                    start: { line: i, character: 0 },
                    end: { line: i, character: line.length }
                },
                selectionRange: {
                    start: { line: i, character: line.indexOf(kernelMatch[1]) },
                    end: { line: i, character: line.indexOf(kernelMatch[1]) + kernelMatch[1].length }
                }
            });
        }
    }
    return symbols;
});
// Helper functions
function getWordRangeAtPosition(line, character) {
    const wordRegex = /[a-zA-Z_][a-zA-Z0-9_]*/g;
    let match;
    while ((match = wordRegex.exec(line)) !== null) {
        if (character >= match.index && character <= match.index + match[0].length) {
            return {
                start: match.index,
                end: match.index + match[0].length
            };
        }
    }
    return null;
}
function getKeywordDocumentation(keyword) {
    const documentation = {
        'node': 'Defines a ROS2 node with its configuration and methods',
        'method': 'Defines a C++ method within a node',
        'kernel': 'Defines a CUDA kernel for GPU acceleration',
        'parameter': 'Defines a configurable parameter for the node',
        'timer': 'Defines a periodic timer for the node',
        'publisher': 'Defines a ROS2 publisher',
        'subscriber': 'Defines a ROS2 subscriber',
        'service': 'Defines a ROS2 service',
        'client': 'Defines a ROS2 service client',
        'action': 'Defines a ROS2 action',
        'include': 'Includes external files or libraries',
        'input': 'Defines an input parameter for a method or kernel',
        'output': 'Defines an output parameter for a method or kernel',
        'code': 'Contains C++ or CUDA code blocks',
        'qos': 'Quality of Service configuration for ROS topics',
        'cuda_kernels': 'Block containing CUDA kernel definitions',
        'onnx_model': 'Defines an ONNX model for machine learning inference',
        'pipeline': 'Defines a processing pipeline with multiple stages',
        'stage': 'Defines a stage within a pipeline',
        'project_name': 'Defines the name of a project',
        'parameter_callbacks': 'Defines callbacks for node parameters',
        'lifecycle_config': 'Defines the lifecycle configuration for a node',
        'queue_size': 'Defines the queue size for a ROS topic',
        'autostart': 'Defines whether a node should start automatically',
        'cleanup_on_shutdown': 'Defines whether a node should clean up on shutdown',
        'oneshot': 'Defines whether a node should run only once',
        'no_autostart': 'Defines whether a node should not start automatically'
    };
    return documentation[keyword] || `RoboDSL keyword: ${keyword}`;
}
function getHoverInformation(word) {
    const hoverInfo = {
        'node': '**Node Definition**\n\nDefines a ROS2 node with its configuration, parameters, and methods.\n\n```robodsl\nnode MyNode {\n  parameter int rate: 10\n  method process {\n    input: sensor_msgs/Image image\n    code: {\n      // C++ code here\n    }\n  }\n}\n```',
        'method': '**Method Definition**\n\nDefines a C++ method within a node that processes data.\n\n```robodsl\nmethod process {\n  input: sensor_msgs/Image image\n  output: std_msgs/String result\n  code: {\n    // C++ implementation\n  }\n}\n```',
        'kernel': '**CUDA Kernel**\n\nDefines a CUDA kernel for GPU-accelerated processing.\n\n```robodsl\nkernel process_gpu {\n  input: float* data, int size\n  block_size: (256, 1, 1)\n  code: {\n    // CUDA kernel code\n  }\n}\n```',
        'publisher': '**ROS2 Publisher**\n\nDefines a publisher that sends messages to a topic.\n\n```robodsl\npublisher /output_topic: "std_msgs/String"\n```',
        'subscriber': '**ROS2 Subscriber**\n\nDefines a subscriber that receives messages from a topic.\n\n```robodsl\nsubscriber /input_topic: "sensor_msgs/Image"\n```',
        'parameter': '**Node Parameter**\n\nDefines a configurable parameter for the node.\n\n```robodsl\nparameter int rate: 10\nparameter string model_path: "/path/to/model"\n```',
        'timer': '**Timer**\n\nDefines a periodic timer that triggers at specified intervals.\n\n```robodsl\ntimer process_timer: 100ms\n```',
        'qos': '**Quality of Service**\n\nConfigures the quality of service for ROS topics.\n\n```robodsl\nqos {\n  reliability: reliable\n  durability: transient_local\n  history: keep_last\n  depth: 10\n}\n```'
    };
    return hoverInfo[word] || null;
}
// Make the text document manager listen on the connection
// for open, change and close text document events
documents.listen(connection);
// Listen on the connection
connection.listen();
//# sourceMappingURL=language-server.js.map