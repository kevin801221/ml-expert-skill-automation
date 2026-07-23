import os
import shutil
import zipfile
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

console = Console()

def build_submission():
    console.print(Panel("[bold cyan]📦 KAGGLE AUTONOMOUS AGENT SUBMISSION BUILDER[/bold cyan]\n[dim]Compiling Google ADK submission.zip for Autonomous Agent Prediction (Beta)[/dim]", box=box.DOUBLE, border_style="cyan"))

    root_dir = '/Users/kevinluo/ml-expert-skill-make'
    pkg_dir = os.path.join(root_dir, 'kaggle_submission_pkg')
    skill_dst_dir = os.path.join(pkg_dir, 'skills', 'ml-expert')
    
    # 1. Prepare Directory Structure
    os.makedirs(os.path.join(skill_dst_dir, 'scripts'), exist_ok=True)
    os.makedirs(os.path.join(skill_dst_dir, 'resources'), exist_ok=True)

    # 2. Copy SKILL.md and assets into package
    skill_src_file = os.path.join(root_dir, '.agents', 'skills', 'ml-expert', 'SKILL.md')
    if os.path.exists(skill_src_file):
        shutil.copy(skill_src_file, os.path.join(skill_dst_dir, 'SKILL.md'))

    # Copy helper scripts
    src_files = [
        ('src/features.py', 'scripts/features.py'),
        ('src/models/pytorch_net.py', 'scripts/pytorch_net.py'),
        ('src/train_ensemble.py', 'scripts/train_ensemble.py')
    ]
    for src, dst in src_files:
        s_path = os.path.join(root_dir, src)
        d_path = os.path.join(skill_dst_dir, dst)
        if os.path.exists(s_path):
            os.makedirs(os.path.dirname(d_path), exist_ok=True)
            shutil.copy(s_path, d_path)

    # Copy domain knowledge resources
    spec_src = os.path.join(root_dir, 'ML_SPEC.md')
    if os.path.exists(spec_src):
        shutil.copy(spec_src, os.path.join(skill_dst_dir, 'resources', 'domain_knowledge.md'))

    # 3. Create submission.zip Archive
    zip_output_path = os.path.join(root_dir, 'submission.zip')
    if os.path.exists(zip_output_path):
        os.remove(zip_output_path)

    with zipfile.ZipFile(zip_output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(pkg_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, pkg_dir)
                zipf.write(file_path, arcname)

    # 4. Display Archive Contents Table
    archive_table = Table(title="📁 Built submission.zip Archive Structure", box=box.ROUNDED)
    archive_table.add_column("Archive File Path", style="bold cyan")
    archive_table.add_column("Size (Bytes)", style="bold green")

    with zipfile.ZipFile(zip_output_path, 'r') as zipf:
        for info in zipf.infolist():
            archive_table.add_row(info.filename, f"{info.file_size:,}")

    console.print(archive_table)
    
    zip_size_kb = os.path.getsize(zip_output_path) / 1024
    console.print(Panel(f"[bold green]✨ SUCCESS: Built {zip_output_path} ({zip_size_kb:.2f} KB)[/bold green]\n[dim]Ready for direct submission to Kaggle Autonomous Agent Prediction (Beta)![/dim]", border_style="green"))

if __name__ == '__main__':
    build_submission()
