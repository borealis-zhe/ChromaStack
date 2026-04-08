#!/usr/bin/env python3
"""
Nuitka Packaging Script for ChromaStack Application

This script generates a standalone executable with no console window.
It includes all necessary dependencies and resources.
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """Main packaging function."""
    # Get the current directory
    current_dir = Path(__file__).parent
    
    # Set paths
    main_script = current_dir / "chromastack_gui.py"
    output_dir = current_dir / "dist" / "nuitka"
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=== ChromaStack Nuitka Packaging ===")
    print(f"Main script: {main_script}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Check if main script exists
    if not main_script.exists():
        print(f"Error: Main script {main_script} not found!")
        sys.exit(1)
    
    # Nuitka command options
    nuitka_options = [
        sys.executable,  # Use the current Python interpreter
        "-m", "nuitka",
        
        # Standalone executable settings
        "--standalone",
        "--windows-console-mode=force",  # Always show console window to see errors
        
        # Include data directories with relative target paths
        f"--include-data-dir={current_dir / 'GUI' / 'frontend' / 'dist'}=GUI/frontend/dist",
        f"--include-data-dir={current_dir / 'GUI' / 'backend'}=GUI/backend",
        f"--include-data-dir={current_dir / 'config'}=config",
        
        # Output settings
        f"--output-dir={output_dir}",
        f"--output-filename=ChromaStack.exe",
        
        # Plugin settings
        "--enable-plugin=pywebview",
        
        # Follow imports
        "--follow-imports",
        
        # No cache to avoid permission issues
        "--disable-cache=all",
        
        # Main script
        str(main_script)
    ]
    
    print("Running Nuitka packaging with options:")
    for option in nuitka_options:
        print(f"  {option}")
    print()
    
    try:
        # Run Nuitka packaging
        subprocess.run(nuitka_options, check=True)
        
        print("\n✅ Packaging completed successfully!")
        print(f"\n📦 Standalone executable created at:")
        print(f"   {output_dir / 'ChromaStack.exe'}")
        print("\n📋 Packaging details:")
        print("   - Standalone executable: Yes")
        print("   - Console window: Disabled")
        print("   - All dependencies included")
        print("   - Frontend and backend resources included")
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Packaging failed with error: {e}")
        print("\n📋 Troubleshooting tips:")
        print("1. Make sure you're running this script with administrator privileges")
        print("2. Ensure all dependencies are installed: pip install -r requirements.txt")
        print("3. Try running the script in a virtual environment")
        print("4. If you still encounter cache errors, try manually creating the cache directory:")
        print("   mkdir -p %LOCALAPPDATA%\\Nuitka\\Nuitka\\Cache")
        sys.exit(1)
    except FileNotFoundError:
        print("\n❌ Nuitka not found! Please install it first:")
        print("   pip install nuitka")
        sys.exit(1)

if __name__ == "__main__":
    main()