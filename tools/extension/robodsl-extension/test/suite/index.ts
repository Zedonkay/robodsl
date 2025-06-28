import * as path from 'path';
import { runTests } from 'vscode-test';

export function run(): Promise<number> {
    return runTests({
        extensionDevelopmentPath: path.resolve(__dirname, '../../'),
        extensionTestsPath: path.resolve(__dirname, './suite')
    });
} 