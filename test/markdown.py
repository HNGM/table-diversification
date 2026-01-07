import sys
sys.path.append(".")
from src.utils.data_preview import get_data_preview_markdown
from pathlib import Path

if __name__ == "__main__":
    data_file = Path(r"C:\repo\table-diversification\dev_test\Diversification\Self Created Dataset\Manual Created Diversified Dataset\Disturbed Diversifications\University_Grades\University_Grades_original_1.xlsx")
    markdown_content = get_data_preview_markdown(data_file)
    print(markdown_content)