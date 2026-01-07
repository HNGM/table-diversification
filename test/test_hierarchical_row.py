import pandas as pd
if __name__ == "__main__":
    df = pd.read_excel(r"C:\repo\table-diversification\dev_test\Diversification\Self Created Dataset\Manual Created Diversified Dataset\Disturbed Diversifications\HR_Timesheets\HR_Timesheets_original_0.xlsx")
    print(df.head())