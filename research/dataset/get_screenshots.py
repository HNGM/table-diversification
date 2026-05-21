import sys
sys.path.append(".")
from pathlib import Path
import win32com.client
import time
from PIL import ImageGrab
import argparse
import json
import win32gui
import win32ui
import win32con
from ctypes import windll
from tqdm import tqdm


def take_excel_screenshot(excel_path: Path, output_path: Path):
    """
    Take a screenshot of an Excel file and save it as PNG.
    Scales the view to fit all columns on screen.
    
    Args:
        excel_path: Path to the Excel file
        output_path: Path where the PNG screenshot will be saved
    """
    excel = None
    workbook = None
    
    try:
        # Create Excel application instance
        excel = win32com.client.Dispatch("Excel.Application")
        excel.Visible = True  # Make visible for rendering
        excel.DisplayAlerts = False
        
        # Open the workbook
        workbook = excel.Workbooks.Open(str(excel_path.absolute()))
        
        # Get the active worksheet
        worksheet = workbook.ActiveSheet
        
        # Get the used range to determine columns and rows
        used_range = worksheet.UsedRange
        last_column = used_range.Columns(used_range.Columns.Count).Column
        last_row = used_range.Rows(used_range.Rows.Count).Row
        
        # Auto-fit all column widths to ensure full column names are visible
        used_range.Columns.AutoFit()
        
        # Maximize the Excel window
        excel.WindowState = -4137  # xlMaximized
        
        # Wait for window to maximize
        time.sleep(1)
        
        # Try to fit all columns by adjusting zoom
        first_cell = worksheet.Cells(1, 1)
        last_col_cell = worksheet.Cells(1, last_column)
        
        for zoom_level in [100, 85, 70, 60, 50, 40, 30, 25]:
            excel.ActiveWindow.Zoom = zoom_level
            time.sleep(0.4)
            
            # Check if we can fit all columns
            try:
                last_col_cell.Select()
                time.sleep(0.2)
                first_cell.Select()
                time.sleep(0.2)
                break
            except:
                continue
        
        # Ensure we're at the top-left
        worksheet.Cells(1, 1).Select()
        excel.ActiveWindow.ScrollRow = 1
        excel.ActiveWindow.ScrollColumn = 1
        
        # Wait for rendering
        time.sleep(1)
        
        # Calculate visible range for export
        # Get visible rows (approximate based on zoom)
        visible_rows = min(last_row, 50)  # Capture up to 50 rows or all rows
        
        # Define the range to export (all columns, visible rows)
        export_range = worksheet.Range(
            worksheet.Cells(1, 1),
            worksheet.Cells(visible_rows, last_column)
        )
        
        # Copy the range as a picture
        export_range.CopyPicture(Appearance=1, Format=2)  # xlScreen, xlBitmap
        
        # CRITICAL: Wait for clipboard to have the data
        time.sleep(1.0)
        
        # Add the picture to the worksheet directly
        picture = worksheet.Pictures().Paste()
        
        # Wait for picture to be created
        time.sleep(0.5)
        
        # Position it far to the right so it doesn't interfere
        picture.Top = 0
        picture.Left = 10000
        
        # Wait for positioning to complete
        time.sleep(0.3)
        
        # Get picture dimensions
        pic_width = picture.Width
        pic_height = picture.Height
        
        # Export the picture
        temp_output = str(output_path.absolute())
        picture.Copy()
        
        # CRITICAL: Wait for clipboard to have the picture data
        time.sleep(1.0)
        
        # Use a chartobject in the worksheet instead of a chart sheet
        chart_obj = worksheet.ChartObjects().Add(0, 0, pic_width, pic_height)
        
        # Wait for chart to be created
        time.sleep(0.3)
        
        # Paste into chart
        chart_obj.Chart.Paste()
        
        # Wait for paste to complete
        time.sleep(0.5)
        
        # Export the chart
        chart_obj.Chart.Export(temp_output, "PNG")
        
        # Clean up
        chart_obj.Delete()
        picture.Delete()
        
        print(f"Screenshot saved: {output_path}")
        
    except Exception as e:
        print(f"Error processing {excel_path}: {str(e)}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Clean up
        try:
            if workbook:
                workbook.Close(SaveChanges=False)
            if excel:
                excel.Quit()
        except:
            pass
        
        # Extra time to ensure Excel closes
        time.sleep(0.5)


def process_directory(data_dir: Path):
    """
    Process all Excel files in the directory and subdirectories.
    
    Args:
        data_dir: Root directory to search for Excel files
    """
    if not data_dir.exists():
        print(f"Error: Directory {data_dir} does not exist")
        return
    
    # Find all Excel files
    excel_files = list(data_dir.glob("**/*.xlsx"))
    
    if not excel_files:
        print(f"No Excel files found in {data_dir}")
        return
    
    print(f"Found {len(excel_files)} Excel file(s)")
    
    # Report how many PNGs already exist vs need to be generated
    existing_pngs = [f for f in excel_files if f.with_suffix('.png').exists()]
    to_generate = [f for f in excel_files if not f.with_suffix('.png').exists()]
    print(f"  - Already have PNG: {len(existing_pngs)}")
    print(f"  - Need to generate: {len(to_generate)}")
    
    # Process each Excel file that needs a PNG, with a tqdm progress bar
    progress = tqdm(to_generate, desc="Generating PNGs", unit="file")
    for excel_file in progress:
        output_path = excel_file.with_suffix('.png')
        progress.set_postfix_str(excel_file.name)

        # Take screenshot
        take_excel_screenshot(excel_file, output_path)

        # Longer delay between files to ensure Excel fully closes
        time.sleep(1.0)
    
    print(f"\nCompleted processing {len(excel_files)} files")


def process_file_list(excel_files: list[Path]) -> None:
    """
    Process an explicit list of Excel files (instead of scanning a directory).
    Skips files whose PNG already exists and shows a tqdm progress bar.
    """
    if not excel_files:
        print("No Excel files provided")
        return

    print(f"Received {len(excel_files)} Excel file(s)")

    # Separate not-found / already-have / to-generate
    not_found = [f for f in excel_files if not f.exists()]
    present = [f for f in excel_files if f.exists()]
    existing_pngs = [f for f in present if f.with_suffix('.png').exists()]
    to_generate = [f for f in present if not f.with_suffix('.png').exists()]

    print(f"  - Missing on disk : {len(not_found)}")
    print(f"  - Already have PNG: {len(existing_pngs)}")
    print(f"  - Need to generate: {len(to_generate)}")
    if not_found:
        print("  Sample missing files:")
        for m in not_found[:5]:
            print(f"    {m}")

    progress = tqdm(to_generate, desc="Generating PNGs", unit="file")
    for excel_file in progress:
        output_path = excel_file.with_suffix('.png')
        progress.set_postfix_str(excel_file.name)
        take_excel_screenshot(excel_file, output_path)
        time.sleep(1.0)

    print(f"\nCompleted processing {len(to_generate)} files")


def load_files_from_jsons(json_paths: list[Path], repo_root: Path) -> list[Path]:
    """
    Load unique `data_file` entries from one or more JSON files and resolve
    them to absolute paths relative to `repo_root`. Only .xlsx files are kept.
    """
    files: set[str] = set()
    for jp in json_paths:
        data = json.loads(jp.read_text(encoding="utf-8"))
        for item in data:
            df = item.get("data_file")
            if df and df.lower().endswith(".xlsx"):
                files.add(df)
    resolved: list[Path] = []
    for df in sorted(files):
        p = Path(df)
        if not p.is_absolute():
            p = (repo_root / p).resolve()
        resolved.append(p)
    return resolved

def main(data_dir: Path | None = None, json_files: list[Path] | None = None,
         repo_root: Path | None = None) -> None:
    """
    Main function to generate PNG screenshots for Excel files.

    If `json_files` is provided, only the xlsx files listed in those JSONs
    (via the `data_file` key) are processed. Otherwise, `data_dir` is
    scanned recursively for xlsx files.
    """
    if json_files:
        repo_root = repo_root or Path(__file__).resolve().parents[2]
        print(f"Loading file list from {len(json_files)} JSON file(s)")
        print(f"Resolving paths relative to: {repo_root}")
        excel_files = load_files_from_jsons(json_files, repo_root)
        process_file_list(excel_files)
    else:
        assert data_dir is not None, "Either data_dir or json_files must be provided"
        print(f"Scanning directory: {data_dir}")
        process_directory(data_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Take screenshots of Excel files')
    parser.add_argument('--data_dir', type=str,
                       default=r"C:\repo\table-diversification\dev_test\Diversification\External_Source\workbooks",
                       help='Directory containing Excel files (ignored if --json_files is given)')
    parser.add_argument('--json_files', type=str, nargs='+', default=None,
                       help='One or more JSON files whose `data_file` entries identify the xlsx files to screenshot')
    parser.add_argument('--repo_root', type=str, default=None,
                       help='Repo root for resolving relative data_file paths (defaults to the repo containing this script)')

    args = parser.parse_args()
    json_paths = [Path(p) for p in args.json_files] if args.json_files else None
    repo_root = Path(args.repo_root) if args.repo_root else None
    main(data_dir=Path(args.data_dir) if not json_paths else None,
         json_files=json_paths,
         repo_root=repo_root)



