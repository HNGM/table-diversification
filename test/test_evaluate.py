import sys
sys.path.append(".")

x = {
    "index": "Retail_Sales_Orders_original_3",
    "query": "What percentage of orders were delivered within two days of the order date?",
    "answer": "62.5",
    "dtype": "float",
    "type": None,
    "data_file": "dev_test/Diversification/Self Created Dataset/Manual Created Diversified Dataset/Normal Diversifications/Retail_Sales_Orders/Retail_Sales_Orders_original.xlsx",
    "image_file": "dev_test/Diversification/Self Created Dataset/Manual Created Diversified Dataset/Normal Diversifications/Retail_Sales_Orders/Retail_Sales_Orders_original.png",
    "diversification_type": "original",
    "eval": [
      {
        "agent_response": {
          "answer": 54.166666666666664,
          "dtype": "float"
        },
        "raw_response": [
          {
            "role": "user",
            "content": "\n[INST]Below is the table data. You need to write a Python program that reads the data (`Retail_Sales_Orders_original.xlsx`) to solve the provided question.\n\n\n\nDATA CONTEXT:\n| OrderID | OrderDate | CustomerID | Region | ProductID | ProductName | Category | Quantity | UnitPrice | DiscountRate | GrossAmount | NetAmount | ShippingMethod | ShipDate |\n| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n| SO-2025001 | 2025-07-02 | C1001 | East | P400 | Coffee Beans | Grocery | 5 | 15 | 0.1 | 75 | 67.5 | Express | 2025-07-03 |\n| SO-2025002 | 2025-07-03 | C1002 | South | P500 | Notebook Pack | Stationery | 5 | 6 | 0.1 | 30 | 27 | Standard | 2025-07-07 |\n| SO-2025003 | 2025-07-04 | C1003 | North | P300 | Office Chair | Furniture | 3 | 120 | 0.15 | 360 | 306 | Express | 2025-07-05 |\n| SO-2025004 | 2025-07-05 | C1004 | South | P300 | Office Chair | Furniture | 6 | 120 | 0 | 720 | 720 | Express | 2025-07-06 |\n| SO-2025005 | 2025-07-06 | C1005 | South | P400 | Coffee Beans | Grocery | 6 | 15 | 0.05 | 90 | 85.5 | Express | 2025-07-07 |\n| SO-2025006 | 2025-07-07 | C1006 | West | P400 | Coffee Beans | Grocery | 5 | 15 | 0 | 75 | 75 | Express | 2025-07-08 |\n| SO-2025007 | 2025-07-08 | C1007 | North | P500 | Notebook Pack | Stationery | 4 | 6 | 0 | 24 | 24 | Standard | 2025-07-12 |\n| SO-2025008 | 2025-07-09 | C1008 | West | P200 | Headphones | Electronics | 4 | 60 | 0.15 | 240 | 204 | Standard | 2025-07-12 |\n| SO-2025009 | 2025-07-10 | C1009 | West | P300 | Office Chair | Furniture | 4 | 120 | 0.1 | 480 | 432 | Express | 2025-07-11 |\n| SO-2025010 | 2025-07-11 | C1010 | East | P100 | Laptop | Electronics | 3 | 850 | 0 | 2550 | 2550 | Standard | 2025-07-13 |\n| SO-2025011 | 2025-07-12 | C1011 | West | P100 | Laptop | Electronics | 7 | 850 | 0.05 | 5950 | 5652.5 | Standard | 2025-07-15 |\n| SO-2025012 | 2025-07-13 | C1012 | South | P200 | Headphones | Electronics | 2 | 60 | 0 | 120 | 120 | Standard | 2025-07-16 |\n| SO-2025013 | 2025-07-14 | C1013 | West | P400 | Coffee Beans | Grocery | 4 | 15 | 0.1 | 60 | 54 | Standard | 2025-07-16 |\n| SO-2025014 | 2025-07-15 | C1014 | West | P300 | Office Chair | Furniture | 6 | 120 | 0 | 720 | 720 | Express | 2025-07-16 |\n| SO-2025015 | 2025-07-16 | C1015 | South | P400 | Coffee Beans | Grocery | 2 | 15 | 0.05 | 30 | 28.5 | Express | 2025-07-17 |\n| SO-2025016 | 2025-07-17 | C1016 | North | P200 | Headphones | Electronics | 4 | 60 | 0.05 | 240 | 228 | Standard | 2025-07-20 |\n| SO-2025017 | 2025-07-18 | C1017 | South | P200 | Headphones | Electronics | 4 | 60 | 0.05 | 240 | 228 | Express | 2025-07-19 |\n| SO-2025018 | 2025-07-19 | C1018 | West | P400 | Coffee Beans | Grocery | 6 | 15 | 0.1 | 90 | 81 | Standard | 2025-07-22 |\n| SO-2025019 | 2025-07-20 | C1019 | West | P400 | Coffee Beans | Grocery | 1 | 15 | 0.05 | 15 | 14.25 | Standard | 2025-07-22 |\n| SO-2025020 | 2025-07-21 | C1020 | West | P200 | Headphones | Electronics | 7 | 60 | 0 | 420 | 420 | Express | 2025-07-22 |\n| SO-2025021 | 2025-07-22 | C1021 | West | P100 | Laptop | Electronics | 4 | 850 | 0.15 | 3400 | 2890 | Standard | 2025-07-24 |\n| SO-2025022 | 2025-07-23 | C1022 | North | P500 | Notebook Pack | Stationery | 7 | 6 | 0 | 42 | 42 | Standard | 2025-07-27 |\n| SO-2025023 | 2025-07-24 | C1023 | West | P100 | Laptop | Electronics | 1 | 850 | 0.15 | 850 | 722.5 | Express | 2025-07-25 |\n| SO-2025024 | 2025-07-25 | C1024 | West | P300 | Office Chair | Furniture | 3 | 120 | 0 | 360 | 360 | Standard | 2025-07-29 |\n\nQuestion: What percentage of orders were delivered within two days of the order date?\n\nEnclose the python code in a python code block as shown below:\n```python\n<python code>\n```\n\n- You should not use placeholders like `data.csv` in your code. Use the actual file name provided in the data context for reading the data `Retail_Sales_Orders_original.xlsx`.\n- The python code should be complete and run end to end without errors.\n- In case there are errors you will be provided with the error logs and you should fix the code accordingly.\n[/INST]\n"
          },
          {
            "role": "assistant",
            "content": " import pandas as pd\n\n# Load the data from the excel file\ndf = pd.read_excel('Retail_Sales_Orders_original.xlsx')\n\n# Convert the 'OrderDate' column to datetime format\ndf['OrderDate'] = pd.to_datetime(df['OrderDate'])\n\n# Filter the data where the 'ShipDate' is within 2 days of the 'OrderDate'\ndf_filtered = df[df['ShipDate'] >= df['OrderDate'] + pd.DateOffset(days=2)]\n\n# Calculate the percentage of orders delivered within 2 days\npercentage_delivered = (df_filtered.shape[0] / df.shape[0]) * 100\n\nprint(percentage_delivered)\n"
          },
          {
            "role": "user",
            "content": "Code block executed successfully:\n54.166666666666664"
          }
        ],
        "eval": False,
        "tools": 0
      }
    ]
  }


from research.evaluation.evaluate import evaluate
eval = evaluate(
    gt_answer=x['answer'],
    gt_dtype=x['dtype'],
    pred_answer=x['eval'][0]['agent_response']['answer'],
    pred_dtype=x['eval'][0]['agent_response']['dtype'],
)
print(eval)