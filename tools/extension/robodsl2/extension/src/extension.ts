import * as path from 'path';
import * as fs from 'fs';
import * as child_process from 'child_process';
import * as vscode from 'vscode';
import { workspace, ExtensionContext } from 'vscode';
import {
  LanguageClient,
  LanguageClientOptions,
  ServerOptions,
  Executable,
} from 'vscode-languageclient/node';

let client: LanguageClient | undefined;

export async function activate(context: ExtensionContext) {
  const cfg = workspace.getConfiguration('robodsl');
  const configuredPath = cfg.get<string>('languageServerPath', '');

  // Resolve workspace root (first opened folder)
  const firstFolder = workspace.workspaceFolders && workspace.workspaceFolders.length > 0
    ? workspace.workspaceFolders[0].uri.fsPath
    : undefined;

  // Compute fallback python executable and run args
  const pythonCmd = process.env.PYTHON_EXECUTABLE || 'python3';
  const serverModule = 'robodsl_ls.server';
  const toolsRoot = path.resolve(context.extensionPath, '..'); // tools/extension/robodsl2

  // Helper: check if a command exists in PATH or is an absolute path that exists
  function commandExists(cmd: string): boolean {
    try {
      if (!cmd) return false;
      if (path.isAbsolute(cmd)) {
        return fs.existsSync(cmd);
      }
      const whichCmd = process.platform === 'win32' ? 'where' : 'which';
      child_process.execSync(`${whichCmd} ${cmd}`, { stdio: 'ignore' });
      return true;
    } catch {
      return false;
    }
  }

  // Decide on server launch strategy
  let exec: Executable;
  if (configuredPath && commandExists(configuredPath)) {
    exec = {
      command: configuredPath,
      args: [],
      options: { env: process.env },
    };
  } else if (commandExists('robodsl-ls')) {
    exec = {
      command: 'robodsl-ls',
      args: [],
      options: { env: process.env },
    };
  } else {
    // Fallback: python -m robodsl_ls.server with PYTHONPATH pointing to repo src
    const env = { ...process.env };
    // Prepend repo src to PYTHONPATH so server can import robodsl
    const repoRoot = firstFolder || path.resolve(toolsRoot, '..', '..');
    const srcPath = path.join(repoRoot, 'src');
    env.PYTHONPATH = srcPath + (env.PYTHONPATH ? path.delimiter + env.PYTHONPATH : '');
    exec = {
      command: pythonCmd,
      args: ['-u', '-m', serverModule],
      options: {
        env,
        cwd: toolsRoot,
      },
    };
    void vscode.window.showInformationMessage('RoboDSL: falling back to Python language server (python -m robodsl_ls.server). You can configure robodsl.languageServerPath to a robodsl-ls executable.');
  }

  const serverOptions: ServerOptions = {
    run: exec,
    debug: exec,
  };

  const clientOptions: LanguageClientOptions = {
    documentSelector: [{ scheme: 'file', language: 'robodsl' }],
    synchronize: {
      fileEvents: workspace.createFileSystemWatcher('**/*.robodsl'),
    },
  };

  client = new LanguageClient(
    'robodsl',
    'RoboDSL Language Server',
    serverOptions,
    clientOptions,
  );

  try {
    await client.start();
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    void vscode.window.showErrorMessage(`RoboDSL Language Server client: couldn't create connection to server. ${msg}`);
    throw err;
  }
  context.subscriptions.push({ dispose: () => { void client?.stop(); } });

  // Register a lightweight file decoration so users keep their current icon theme
  // but .robodsl files are still visibly marked as RoboDSL.
  const decorationProvider: vscode.FileDecorationProvider = {
    onDidChangeFileDecorations: new vscode.EventEmitter<vscode.Uri | vscode.Uri[] | undefined>().event,
    provideFileDecoration(uri: vscode.Uri): vscode.ProviderResult<vscode.FileDecoration> {
      if (uri.fsPath.toLowerCase().endsWith('.robodsl')) {
        return {
          badge: 'R',
          tooltip: 'RoboDSL file',
        };
      }
      return undefined;
    },
  };
  const decoDisposable = vscode.window.registerFileDecorationProvider(decorationProvider);
  context.subscriptions.push(decoDisposable);
}

export async function deactivate(): Promise<void> {
  if (!client) {
    return;
  }
  await client.stop();
}


