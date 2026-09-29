"""
tools/convert_dwg_to_dxf.py
Headless DWG to DXF converter utilizing AutoCAD Core Console (accoreconsole.exe).
Converts Autodesk DWG files to clean ASCII DXF files without graphical user interface.
"""

import os
import sys
import argparse
import subprocess
import tempfile
from pathlib import Path
import ezdxf

DEFAULT_ACCORECONSOLE = r"C:\Program Files\Autodesk\AutoCAD 2027\accoreconsole.exe"
DEFAULT_PRODUCT = "ACAD_E"


def find_accoreconsole(custom_path: str = None) -> Path:
    """Locates the accoreconsole executable."""
    if custom_path:
        p = Path(custom_path).resolve()
        if p.exists():
            return p
        raise FileNotFoundError(f"Specified accoreconsole not found: {custom_path}")

    env_path = os.environ.get("ACCORECONSOLE_PATH")
    if env_path and Path(env_path).exists():
        return Path(env_path).resolve()

    candidates = [
        Path(DEFAULT_ACCORECONSOLE),
        Path(r"C:\Program Files\Autodesk\AutoCAD 2026\accoreconsole.exe"),
        Path(r"C:\Program Files\Autodesk\AutoCAD 2025\accoreconsole.exe"),
        Path(r"C:\Program Files\Autodesk\AutoCAD 2024\accoreconsole.exe")
    ]
    for c in candidates:
        if c.exists():
            return c

    raise FileNotFoundError(
        f"AutoCAD Core Console (accoreconsole.exe) could not be located in standard paths."
    )


def convert_dwg_to_dxf(
    dwg_path: str,
    dxf_path: str,
    precision: int = 16,
    timeout: int = 60,
    accore_path: str = None,
    product: str = DEFAULT_PRODUCT,
    verbose: bool = True
) -> bool:
    """
    Converts a DWG file to a clean DXF using accoreconsole.exe.

    Args:
        dwg_path: Path to the input .dwg file.
        dxf_path: Path to the output .dxf file.
        precision: Decimal precision for ASCII DXF (0-16).
        timeout: Execution timeout in seconds.
        accore_path: Optional explicit path to accoreconsole.exe.
        product: Product switch for AutoCAD edition (e.g. 'ACAD_E').
        verbose: Print progress and diagnostic messages.

    Returns:
        True if conversion succeeded and produced a valid DXF, False otherwise.
    """
    input_p = Path(dwg_path).resolve()
    output_p = Path(dxf_path).resolve()

    if not input_p.exists():
        raise FileNotFoundError(f"Input DWG does not exist: {input_p}")

    output_p.parent.mkdir(parents=True, exist_ok=True)
    if output_p.exists():
        output_p.unlink()

    accore_exe = find_accoreconsole(accore_path)

    # AutoCAD script commands require forward slashes for path separators
    output_str = str(output_p).replace("\\", "/")

    script_content = (
        f'_DXFOUT\n'
        f'"{output_str}"\n'
        f'{precision}\n'
        f'_QUIT\n'
        f'_N\n'
    )

    scr_file = None
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".scr", delete=False, encoding="utf-8") as f:
            f.write(script_content)
            scr_file = Path(f.name)

        cmd = [
            str(accore_exe),
            "/product", product,
            "/i", str(input_p),
            "/s", str(scr_file)
        ]

        if verbose:
            print(f"[*] Converting DWG to DXF:")
            print(f"    Input:  {input_p}")
            print(f"    Output: {output_p}")
            print(f"    Engine: {accore_exe} (/product {product})")

        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout
        )

        if proc.returncode != 0:
            if verbose:
                print(f"[!] accoreconsole returned code {proc.returncode}")
                if proc.stderr:
                    print(f"    STDERR: {proc.stderr[:500]}")
            return False

        if not output_p.exists() or output_p.stat().st_size == 0:
            if verbose:
                print(f"[!] Output DXF was not created or is empty.")
            return False

        # Validate resulting DXF with ezdxf
        doc = ezdxf.readfile(str(output_p))
        msp = doc.modelspace()
        entity_count = len(list(msp))
        if verbose:
            print(f"[+] Conversion successful!")
            print(f"    Size: {output_p.stat().st_size / 1024.0:.2f} KB")
            print(f"    DXF Version: {doc.dxfversion}")
            print(f"    Entities in ModelSpace: {entity_count}")

        return True

    except subprocess.TimeoutExpired:
        if verbose:
            print(f"[!] Timeout of {timeout}s expired during conversion.")
        return False
    except Exception as e:
        if verbose:
            print(f"[!] Error during DWG->DXF conversion: {e}")
        return False
    finally:
        if scr_file and scr_file.exists():
            try:
                scr_file.unlink()
            except OSError:
                pass


def main():
    parser = argparse.ArgumentParser(
        description="Headless DWG to clean DXF converter using AutoCAD Core Console."
    )
    parser.add_argument("input_dwg", type=str, help="Input DWG file path")
    parser.add_argument("output_dxf", type=str, help="Output DXF file path")
    parser.add_argument("--precision", type=int, default=16, help="Decimal precision (0-16, default: 16)")
    parser.add_argument("--timeout", type=int, default=60, help="Conversion timeout in seconds (default: 60)")
    parser.add_argument("--product", type=str, default=DEFAULT_PRODUCT, help="AutoCAD product switch (default: ACAD_E)")
    parser.add_argument("--accore-path", type=str, default=None, help="Explicit path to accoreconsole.exe")

    args = parser.parse_args()

    success = convert_dwg_to_dxf(
        dwg_path=args.input_dwg,
        dxf_path=args.output_dxf,
        precision=args.precision,
        timeout=args.timeout,
        accore_path=args.accore_path,
        product=args.product,
        verbose=True
    )

    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
