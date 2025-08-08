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
  console.log('RoboDSL: Extension activating...');
  
  const cfg = workspace.getConfiguration('robodsl');
  const configuredPath = cfg.get<string>('languageServerPath', '');

  // Resolve workspace root (first opened folder)
  const firstFolder = workspace.workspaceFolders && workspace.workspaceFolders.length > 0
    ? workspace.workspaceFolders[0].uri.fsPath
    : undefined;

  // Fallback: if no workspace folder, try to infer from extension path
  const inferredWorkspace = firstFolder || (() => {
    // Extension is in tools/extension/robodsl2, so go up 3 levels to get project root
    const extensionDir = context.extensionPath;
    if (extensionDir.includes('tools/extension/robodsl2')) {
      return path.resolve(extensionDir, '..', '..', '..');
    }
    return undefined;
  })();

  console.log('RoboDSL: Workspace folder:', firstFolder);
  console.log('RoboDSL: Inferred workspace:', inferredWorkspace);
  console.log('RoboDSL: Extension path:', context.extensionPath);
  console.log('RoboDSL: Configured server path:', configuredPath);

  // Compute fallback python executable and run args
  // Try to detect virtual environment Python first
  function getPythonCommand(): string {
    console.log('RoboDSL: Detecting Python...');
    
    // Check for virtual environment
    if (process.env.VIRTUAL_ENV) {
      const venvPython = path.join(process.env.VIRTUAL_ENV, 'bin', 'python');
      console.log('RoboDSL: Checking VIRTUAL_ENV:', venvPython);
      if (fs.existsSync(venvPython)) {
        console.log('RoboDSL: Using VIRTUAL_ENV Python');
        return venvPython;
      }
    }
    
    // Check for conda environment
    if (process.env.CONDA_DEFAULT_ENV && process.env.CONDA_PREFIX) {
      const condaPython = path.join(process.env.CONDA_PREFIX, 'bin', 'python');
      console.log('RoboDSL: Checking conda Python:', condaPython);
      if (fs.existsSync(condaPython)) {
        console.log('RoboDSL: Using conda Python');
        return condaPython;
      }
    }
    
    // Check for local .venv in workspace
    const workspaceRoot = firstFolder || inferredWorkspace;
    if (workspaceRoot) {
      const localVenv = path.join(workspaceRoot, '.venv', 'bin', 'python');
      console.log('RoboDSL: Checking local .venv:', localVenv);
      if (fs.existsSync(localVenv)) {
        console.log('RoboDSL: Using local .venv Python');
        return localVenv;
      }
      // Also check common venv names
      for (const venvName of ['venv', 'env', '.env']) {
        const venvPath = path.join(workspaceRoot, venvName, 'bin', 'python');
        console.log('RoboDSL: Checking', venvName, ':', venvPath);
        if (fs.existsSync(venvPath)) {
          console.log('RoboDSL: Using', venvName, 'Python');
          return venvPath;
        }
      }
    } else {
      console.log('RoboDSL: No workspace folder detected');
    }
    
    // Check VS Code Python extension setting
    const pythonConfig = workspace.getConfiguration('python');
    const configuredPython = pythonConfig.get<string>('defaultInterpreterPath') || 
                            pythonConfig.get<string>('pythonPath');
    if (configuredPython && fs.existsSync(configuredPython)) {
      console.log('RoboDSL: Using VS Code configured Python:', configuredPython);
      return configuredPython;
    }
    
    // Fallback to system python
    const fallback = process.env.PYTHON_EXECUTABLE || 'python3';
    console.log('RoboDSL: Using fallback Python:', fallback);
    return fallback;
  }
  
  const pythonCmd = getPythonCommand();
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
    // Check if the configured path is a Python interpreter or a language server executable
    if (configuredPath.includes('python') || configuredPath.endsWith('python3')) {
      // It's a Python interpreter, use it to run the server module
      console.log('RoboDSL: Using configured Python interpreter:', configuredPath);
      const env = { ...process.env };
      const repoRoot = firstFolder || inferredWorkspace || path.resolve(toolsRoot, '..', '..');
      const srcPath = path.join(repoRoot, 'src');
      const langServerPath = toolsRoot;
      const pythonPaths = [langServerPath, srcPath];
      env.PYTHONPATH = pythonPaths.join(path.delimiter) + 
        (env.PYTHONPATH ? path.delimiter + env.PYTHONPATH : '');
      
      exec = {
        command: configuredPath,
        args: ['-u', '-m', serverModule],
        options: {
          env,
          cwd: toolsRoot,
        },
      };
    } else {
      // It's a language server executable
      console.log('RoboDSL: Using configured language server executable:', configuredPath);
      exec = {
        command: configuredPath,
        args: [],
        options: { env: process.env },
      };
    }
  } else if (commandExists('robodsl-ls')) {
    exec = {
      command: 'robodsl-ls',
      args: [],
      options: { env: process.env },
    };
  } else {
    // Fallback: python -m robodsl_ls.server with PYTHONPATH pointing to both repo src and language server
    const env = { ...process.env };
    
    // Add both the repo src and the language server directory to PYTHONPATH
    const repoRoot = firstFolder || inferredWorkspace || path.resolve(toolsRoot, '..', '..');
    const srcPath = path.join(repoRoot, 'src');
    const langServerPath = toolsRoot; // tools/extension/robodsl2 contains robodsl_ls/
    
    const pythonPaths = [langServerPath, srcPath];
    env.PYTHONPATH = pythonPaths.join(path.delimiter) + 
      (env.PYTHONPATH ? path.delimiter + env.PYTHONPATH : '');
    
    exec = {
      command: pythonCmd,
      args: ['-u', '-m', serverModule],
      options: {
        env,
        cwd: toolsRoot,
      },
    };
    
    // Check if Python and required modules are available
    try {
      child_process.execSync(`"${pythonCmd}" -c "import sys; sys.path.insert(0, '${langServerPath}'); import robodsl_ls.server"`, 
        { stdio: 'ignore', env });
      console.log(`RoboDSL: Using Python at ${pythonCmd}`);
    } catch (error) {
      console.error(`RoboDSL: Failed to import server with Python at ${pythonCmd}`);
      
      // Try to get more detailed error info
      try {
        const detailedError = child_process.execSync(
          `"${pythonCmd}" -c "import sys; sys.path.insert(0, '${langServerPath}'); import robodsl_ls.server"`, 
          { encoding: 'utf8', env }
        );
      } catch (detailedErr) {
        console.error('Detailed error:', detailedErr);
      }
      
      const isVenvActive = !!process.env.VIRTUAL_ENV || !!process.env.CONDA_DEFAULT_ENV;
      const venvMsg = isVenvActive ? 
        ' Make sure your virtual environment is activated and contains the dependencies.' : 
        ' Consider activating a virtual environment with the required packages.';
      
      void vscode.window.showErrorMessage(
        `RoboDSL: Python language server dependencies not found in ${pythonCmd}.${venvMsg} Install with: pip install pygls>=1.2.1 lark>=1.1.5`
      );
      throw new Error('Language server dependencies missing');
    }
    
    console.log('RoboDSL: Using Python fallback language server');
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
    console.log('RoboDSL Language Server started successfully');
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    console.error('RoboDSL Language Server failed to start:', msg);
    
    // Provide helpful error messages based on common issues
    if (msg.includes('ENOEXEC')) {
      void vscode.window.showErrorMessage(
        `RoboDSL Language Server: The configured server path is not executable. Please check your robodsl.languageServerPath setting or ensure Python dependencies are installed.`
      );
    } else if (msg.includes('ENOENT')) {
      void vscode.window.showErrorMessage(
        `RoboDSL Language Server: Command not found. Please install Python dependencies: pip install pygls>=1.2.1 lark>=1.1.5`
      );
    } else {
      void vscode.window.showErrorMessage(`RoboDSL Language Server: ${msg}`);
    }
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


