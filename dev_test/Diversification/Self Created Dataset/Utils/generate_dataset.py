#!/usr/bin/env python3
"""
Generate compact, realistic Excel datasets for pipeline testing.

Files created in the current working directory:
- Retail_Sales_Orders.xlsx       (Sheet: Orders)
- HR_Timesheets.xlsx             (Sheet: Timesheets)
- Logistics_Shipments.xlsx       (Sheet: Shipments)
- Clinic_Visits.xlsx             (Sheet: Visits)
- Personal_Finance_Transactions.xlsx (Sheet: Transactions)
- University_Grades.xlsx         (Sheet: Grades)

Optionally also creates:
- Queries_and_Answers_Guide.csv
- Queries_and_Answers_Guide.json

Usage:
    python generate_test_workbooks.py
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import pandas as pd


def gen_retail_sales_orders(seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    orders = []
    start = datetime(2025, 7, 1)
    regions = ['North', 'South', 'East', 'West']
    products = [
        ('P100', 'Laptop', 'Electronics'),
        ('P200', 'Headphones', 'Electronics'),
        ('P300', 'Office Chair', 'Furniture'),
        ('P400', 'Coffee Beans', 'Grocery'),
        ('P500', 'Notebook Pack', 'Stationery'),
    ]
    unit_price = {'P100': 850.0, 'P200': 60.0, 'P300': 120.0, 'P400': 15.0, 'P500': 6.0}

    for i in range(1, 25):  # 24 rows
        order_date = start + timedelta(days=i)
        pid, pname, cat = products[np.random.randint(0, len(products))]
        qty = np.random.randint(1, 8)
        discount = np.round(np.random.choice([0, 0.05, 0.1, 0.15]), 2)
        region = np.random.choice(regions)
        shipping = np.random.choice(['Standard', 'Express'])
        ship_days = 1 if shipping == 'Express' else np.random.randint(2, 5)
        ship_date = order_date + timedelta(days=ship_days)

        gross = qty * unit_price[pid]
        net = round(gross * (1 - discount), 2)

        orders.append({
            'OrderID': f"SO-2025{i:03d}",
            'OrderDate': order_date.date().isoformat(),
            'CustomerID': f"C{1000+i}",
            'Region': region,
            'ProductID': pid,
            'ProductName': pname,
            'Category': cat,
            'Quantity': qty,
            'UnitPrice': unit_price[pid],
            'DiscountRate': discount,
            'GrossAmount': gross,
            'NetAmount': net,
            'ShippingMethod': shipping,
            'ShipDate': ship_date.date().isoformat(),
        })

    cols = ['OrderID', 'OrderDate', 'CustomerID', 'Region', 'ProductID', 'ProductName',
            'Category', 'Quantity', 'UnitPrice', 'DiscountRate', 'GrossAmount',
            'NetAmount', 'ShippingMethod', 'ShipDate']
    return pd.DataFrame(orders)[cols]


def gen_hr_timesheets(seed: int = 43) -> pd.DataFrame:
    np.random.seed(seed)
    projects = ['Apollo', 'Zephyr', 'Hermes']
    roles = ['Engineer', 'Designer', 'Analyst']
    ts_rows = []
    start = datetime(2025, 8, 4)
    rate = {'Engineer': 65, 'Designer': 55, 'Analyst': 50}

    for i in range(20):
        day = start + timedelta(days=i)
        role = np.random.choice(roles)
        hours = np.random.choice([6, 7.5, 8, 9, 10])
        overtime = max(0, hours - 8)
        billable = bool(np.random.choice([1, 1, 1, 0]))
        row = {
            'EntryID': f"TS-{i+1:03d}",
            'Date': day.date().isoformat(),
            'EmployeeID': f"E{200+i}",
            'Role': role,
            'Project': np.random.choice(projects),
            'HoursWorked': hours,
            'OvertimeHours': overtime,
            'Billable': billable,
            'BillRateUSD': rate[role],
            'WorkLocation': np.random.choice(['Onsite', 'Remote']),
        }
        ts_rows.append(row)

    df = pd.DataFrame(ts_rows)
    df['BillableAmount'] = (df['HoursWorked'] * df['BillRateUSD']).round(2)
    return df


def gen_logistics_shipments(seed: int = 44) -> pd.DataFrame:
    np.random.seed(seed)
    carriers = ['DHL', 'FedEx', 'UPS']
    ship_rows = []
    start = datetime(2025, 5, 10)
    for i in range(25):
        ship_date = start + timedelta(days=i)
        priority = np.random.choice(['Standard', 'Priority'])
        transit = np.random.randint(3, 9) if priority == 'Standard' else np.random.randint(1, 5)
        ship_rows.append({
            'ShipmentID': f"SH-{1050+i}",
            'ShipDate': ship_date.date().isoformat(),
            'Origin': np.random.choice(['BLR', 'DEL', 'BOM']),
            'Destination': np.random.choice(['NYC', 'LHR', 'SIN', 'SYD']),
            'WeightKg': np.round(np.random.uniform(2.5, 28.0), 1),
            'VolumeM3': np.round(np.random.uniform(0.02, 0.18), 3),
            'Carrier': np.random.choice(carriers),
            'Priority': priority,
            'TransitDays': transit,
            'DeliveredDate': (ship_date + timedelta(days=transit)).date().isoformat(),
            'Status': 'Delivered',
            'CostUSD': np.round(50 + np.random.uniform(5, 60) + (10 if priority == 'Priority' else 0), 2),
        })
    return pd.DataFrame(ship_rows)


def gen_clinic_visits(seed: int = 45) -> pd.DataFrame:
    np.random.seed(seed)
    conds = ['Hypertension', 'Diabetes', 'Back Pain', 'Flu', 'Allergy']
    visit_rows = []
    start = datetime(2025, 3, 1)
    for i in range(18):
        vd = start + timedelta(days=np.random.randint(0, 60))
        visit_rows.append({
            'VisitID': f"V-{300+i}",
            'VisitDate': vd.date().isoformat(),
            'PatientID': f"P-{1200 + np.random.randint(0, 40)}",
            'Age': np.random.randint(18, 82),
            'Gender': np.random.choice(['F', 'M']),
            'Condition': np.random.choice(conds),
            'BP_Systolic': np.random.randint(105, 165),
            'BP_Diastolic': np.random.randint(65, 100),
            'MedicationPrescribed': np.random.choice(['Yes', 'No']),
            'Doctor': np.random.choice(['Dr. Rao', 'Dr. Singh', 'Dr. Iyer']),
            'ConsultationFeeINR': np.random.choice([400, 500, 600, 700]),
            'FollowUpRequired': np.random.choice(['Yes', 'No']),
            'Notes': np.random.choice(['-', 'Lifestyle advice', 'Diet plan', 'Physio referral']),
        })
    return pd.DataFrame(visit_rows)


def gen_personal_finance(seed: int = 46) -> pd.DataFrame:
    np.random.seed(seed)
    cat = ['Groceries', 'Dining', 'Transport', 'Rent', 'Utilities', 'Health', 'Entertainment']
    acct = ['Credit Card', 'Debit Card', 'UPI']
    rows = []
    start = datetime(2025, 6, 1)
    for i in range(30):
        d = start + timedelta(days=np.random.randint(0, 60))
        amount = np.round(np.random.uniform(100, 3000), 2)
        cashback = np.round(amount * np.random.choice([0, 0.01, 0.02, 0.05]), 2)
        rows.append({
            'TxnID': f"T-{5000+i}",
            'Date': d.date().isoformat(),
            'Merchant': np.random.choice(['Amazon', 'BigBasket', 'Swiggy', 'Zomato', 'Uber', 'Flipkart', 'Apollo Pharmacy', 'Reliance']),
            'Category': np.random.choice(cat),
            'PaymentMethod': np.random.choice(acct),
            'AmountINR': amount,
            'CashbackINR': cashback,
            'NetAmountINR': amount - cashback,
            'Recurring': 'Yes' if np.random.rand() < 0.2 else 'No',
            'City': np.random.choice(['Bengaluru', 'Delhi', 'Mumbai']),
            'Notes': np.random.choice(['', 'Promo applied', 'Monthly bill']),
        })
    return pd.DataFrame(rows)


def gen_university_grades(seed: int = 47) -> pd.DataFrame:
    np.random.seed(seed)
    subjects = ['ML', 'DSA', 'DBMS', 'OS', 'NLP', 'CV']
    rows = []
    for i in range(22):
        mid = np.random.randint(30, 50)
        final = np.random.randint(35, 50)
        project = np.random.randint(15, 25)
        quizzes = np.random.randint(5, 15)
        total = mid + final + project + quizzes
        letter = 'A' if total >= 90 else ('B' if total >= 80 else ('C' if total >= 70 else ('D' if total >= 60 else 'F')))
        rows.append({
            'RecordID': f"G-{i+1:03d}",
            'StudentID': f"S{100+i}",
            'Semester': 'Spring-2025',
            'Course': np.random.choice(subjects),
            'Professor': np.random.choice(['Prof. Mehta', 'Prof. Banerjee', 'Prof. Krishnan']),
            'Midterm(50)': mid,
            'Final(50)': final,
            'Project(25)': project,
            'Quizzes(15)': quizzes,
            'Total(140)': total,
            'LetterGrade': letter,
            'Pass': total >= 70,
        })
    return pd.DataFrame(rows)


def write_xlsx(df: pd.DataFrame, path: Path, sheet_name: str) -> None:
    with pd.ExcelWriter(path, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)


def build_query_answer_manifest(dfs: dict) -> tuple[pd.DataFrame, dict]:
    """
    Build a compact manifest with queries and descriptions of expected answers.
    (For deterministic numeric answers you can recompute them from the generated files.)
    """
    rows = []
    j = {}

    # Retail
    rows += [
        ('Retail_Sales_Orders.xlsx',
         'Total NetAmount by Region (descending).',
         'Group by Region, sum NetAmount, sort DESC.'),
        ('Retail_Sales_Orders.xlsx',
         'Which ProductName has the highest total Quantity sold?',
         'Group by ProductName, sum Quantity; return argmax.'),
        ('Retail_Sales_Orders.xlsx',
         'Average DiscountRate for Express vs Standard.',
         'Group by ShippingMethod; mean DiscountRate.'),
        ('Retail_Sales_Orders.xlsx',
         'What percentage of orders were delivered within 2 days?',
         'Mean of (ShipDate - OrderDate) in days <= 2.'),
    ]

    # Timesheets
    rows += [
        ('HR_Timesheets.xlsx',
         'Billable revenue by Project and Role.',
         'Filter Billable=True; sum BillableAmount by Project,Role.'),
        ('HR_Timesheets.xlsx',
         'Employee with highest total OvertimeHours.',
         'Group by EmployeeID; sum OvertimeHours; pick max.'),
        ('HR_Timesheets.xlsx',
         'Average HoursWorked for Remote vs Onsite.',
         'Group by WorkLocation; mean HoursWorked.'),
    ]

    # Logistics
    rows += [
        ('Logistics_Shipments.xlsx',
         'Average TransitDays by Carrier and Priority.',
         'Group by Carrier,Priority; mean TransitDays.'),
        ('Logistics_Shipments.xlsx',
         'Top 3 destinations by total shipped WeightKg.',
         'Group by Destination; sum WeightKg; top 3.'),
        ('Logistics_Shipments.xlsx',
         'On-time rate (Priority<=4 days else <=6 days).',
         'Compute rule-based boolean and average.'),
    ]

    # Clinic
    rows += [
        ('Clinic_Visits.xlsx',
         'Median BP_Systolic by Condition.',
         'Group by Condition; median BP_Systolic.'),
        ('Clinic_Visits.xlsx',
         'Share of visits that required FollowUp, by Doctor.',
         'Mean of (FollowUpRequired == \"Yes\") by Doctor.'),
        ('Clinic_Visits.xlsx',
         'Average ConsultationFeeINR by Gender and Condition.',
         'Group by Gender,Condition; mean fee.'),
    ]

    # Finance
    rows += [
        ('Personal_Finance_Transactions.xlsx',
         'Monthly spend (NetAmountINR) by Category.',
         'Extract month; group by Month,Category; sum NetAmountINR.'),
        ('Personal_Finance_Transactions.xlsx',
         'PaymentMethod with highest total CashbackINR.',
         'Group by PaymentMethod; sum CashbackINR; argmax.'),
        ('Personal_Finance_Transactions.xlsx',
         'Fraction of spend that is Recurring.',
         'Sum NetAmountINR where Recurring==Yes divided by total.'),
    ]

    # Grades
    rows += [
        ('University_Grades.xlsx',
         'Average Total(140) per Course and pass rate.',
         'Group by Course; mean Total and mean Pass.'),
        ('University_Grades.xlsx',
         'Distribution of LetterGrade (A–F).',
         'Value counts of LetterGrade.'),
        ('University_Grades.xlsx',
         'Professor with highest average Project(25) score.',
         'Group by Professor; mean Project(25); argmax.'),
    ]

    guide_df = pd.DataFrame(rows, columns=['Dataset', 'Query', 'ExpectedAnswerDescription'])

    # Also emit a JSON-friendly structure
    for r in rows:
        dataset, query, desc = r
        j.setdefault(dataset, []).append({'query': query, 'expected': desc})

    return guide_df, j


def main(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Generate dataframes
    retail_df = gen_retail_sales_orders()
    ts_df = gen_hr_timesheets()
    ship_df = gen_logistics_shipments()
    clinic_df = gen_clinic_visits()
    finance_df = gen_personal_finance()
    grades_df = gen_university_grades()

    # Write Excel files (to current working directory)
    write_xlsx(retail_df, out_dir / 'Retail_Sales_Orders.xlsx', 'Orders')
    write_xlsx(ts_df, out_dir / 'HR_Timesheets.xlsx', 'Timesheets')
    write_xlsx(ship_df, out_dir / 'Logistics_Shipments.xlsx', 'Shipments')
    write_xlsx(clinic_df, out_dir / 'Clinic_Visits.xlsx', 'Visits')
    write_xlsx(finance_df, out_dir / 'Personal_Finance_Transactions.xlsx', 'Transactions')
    write_xlsx(grades_df, out_dir / 'University_Grades.xlsx', 'Grades')

    # Optional: write query/answer guide
    guide_df, guide_json = build_query_answer_manifest({
        'Retail': retail_df, 'Timesheets': ts_df, 'Shipments': ship_df,
        'Clinic': clinic_df, 'Finance': finance_df, 'Grades': grades_df
    })
    guide_df.to_csv(out_dir / 'Queries_and_Answers_Guide.csv', index=False)
    with open(out_dir / 'Queries_and_Answers_Guide.json', 'w', encoding='utf-8') as f:
        json.dump(guide_json, f, indent=2, ensure_ascii=False)

    print("✅ Generated files in:", out_dir.resolve())
    for fname in [
        'Retail_Sales_Orders.xlsx',
        'HR_Timesheets.xlsx',
        'Logistics_Shipments.xlsx',
        'Clinic_Visits.xlsx',
        'Personal_Finance_Transactions.xlsx',
        'University_Grades.xlsx',
        'Queries_and_Answers_Guide.csv',
        'Queries_and_Answers_Guide.json',
    ]:
        print("  -", fname)


if __name__ == "__main__":
    # Write to the current working directory by default.
    # Change to Path('some/other/folder') if needed.
    main(out_dir=Path("."))
