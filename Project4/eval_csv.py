import csv

def check_factors(csv_file):
    with open(csv_file, newline='') as f:
        reader = csv.reader(f)
        header = next(reader)  # skip header

        for line_num, row in enumerate(reader, start=2):
            number_expr = row[0].strip()
            factors_str = row[1].strip()

            try:
                number = eval(number_expr)
                factors = eval(factors_str)
                product = 1
                for f in factors:
                    product *= f

                if number == product:
                    print(f"Line {line_num}: ✅ {number_expr} == product(factors)")
                else:
                    print(f"Line {line_num}: ❌ {number_expr} != product(factors)")
                    print(f"  → {number} != {product}")

            except Exception as e:
                print(f"Line {line_num}: Error evaluating row: {e}")

# Usage
check_factors("factors_reduced.csv")
