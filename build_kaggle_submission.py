"""Validate and zip kaggle_submission_pkg/ into submission.zip.

The package directory is the single source of truth - this script copies nothing
into it (the old version re-injected Titanic-specific files on every build).
"""
import os
import re
import sys
import shutil
import zipfile

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

ROOT = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.join(ROOT, 'kaggle_submission_pkg')
REQUIRED = [
    'agent.yaml',
    'prompts/system.md',
    'skills/ml-expert/SKILL.md',
    'skills/ml-expert/scripts/run_pipeline.py',
]
# ADK injects instruction text as a template: a bare {identifier} raises
# "Context variable not found" and kills the session at startup.
BRACE_RE = re.compile(r'\{[A-Za-z_][A-Za-z0-9_]*\}')


def validate():
    errors = []
    for rel in REQUIRED:
        if not os.path.exists(os.path.join(PKG, rel)):
            errors.append(f'missing required file: {rel}')
    for dirpath, _, files in os.walk(PKG):
        for name in files:
            path = os.path.join(dirpath, name)
            rel = os.path.relpath(path, PKG)
            try:
                text = open(path, encoding='utf-8').read()
            except UnicodeDecodeError:
                errors.append(f'binary file not allowed in package: {rel}')
                continue
            if rel.endswith(('.md', '.yaml', '.yml')):
                hits = BRACE_RE.findall(text)
                if hits:
                    errors.append(f'bare-brace template pattern in {rel}: {sorted(set(hits))}')
    return errors


def build():
    console.print(Panel('[bold cyan]KAGGLE AUTONOMOUS AGENT SUBMISSION BUILDER[/bold cyan]', box=box.DOUBLE, border_style='cyan'))

    errors = validate()
    if errors:
        for e in errors:
            console.print(f'[bold red]FAIL[/bold red] {e}')
        sys.exit(1)

    zip_name = 'ml_expert_agent_v7.zip'
    zip_output_path = os.path.join(ROOT, zip_name)
    if os.path.exists(zip_output_path):
        os.remove(zip_output_path)
    with zipfile.ZipFile(zip_output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for dirpath, _, files in os.walk(PKG):
            for name in files:
                path = os.path.join(dirpath, name)
                zipf.write(path, os.path.relpath(path, PKG))

    # Also keep a copy as submission.zip for backward compatibility
    shutil.copy(zip_output_path, os.path.join(ROOT, 'submission.zip'))

    table = Table(title='submission.zip contents', box=box.ROUNDED)
    table.add_column('Archive Path', style='bold cyan')
    table.add_column('Size (Bytes)', style='bold green')
    with zipfile.ZipFile(zip_output_path) as zipf:
        for info in zipf.infolist():
            table.add_row(info.filename, f'{info.file_size:,}')
    console.print(table)
    console.print(Panel(f'[bold green]SUCCESS: {zip_output_path} ({os.path.getsize(zip_output_path) / 1024:.1f} KB)[/bold green]', border_style='green'))


if __name__ == '__main__':
    build()
