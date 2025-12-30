import sys
sys.path.append(".")
from pathlib import Path
import win32com.client
import time
from PIL import ImageGrab
import argparse
import win32gui
import win32ui
import win32con
from ctypes import windll


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
    
    # Process each Excel file
    for idx, excel_file in enumerate(excel_files, 1):
        print(f"\nProcessing {idx}/{len(excel_files)}: {excel_file.name}")
        
        # Create output path (same location, same name, .png extension)
        output_path = excel_file.with_suffix('.png')
        
        # Take screenshot (will overwrite if exists)
        take_excel_screenshot(excel_file, output_path)
        
        # Longer delay between files to ensure Excel fully closes
        time.sleep(1.0)
    
    print(f"\nCompleted processing {len(excel_files)} files")


def main(data_dir: Path):
    """
    Main function to process Excel files and create screenshots.
    
    Args:
        data_dir: Directory containing Excel files
    """
    print(f"Scanning directory: {data_dir}")
    process_directory(data_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Take screenshots of Excel files')
    parser.add_argument('--data_dir', type=str, 
                       default=r"C:\repo\table-diversification\dev_test\Diversification\External_Source\workbooks",
                       help='Directory containing Excel files')
    
    args = parser.parse_args()
    main(Path(args.data_dir))



