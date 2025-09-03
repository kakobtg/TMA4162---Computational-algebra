import pandas as pd
import matplotlib.pyplot as plt

def csv_to_latex(csv_file):
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Convert DataFrame to LaTeX table format
    latex_table = df.to_latex(index=False, escape=False)
    
    # Print the LaTeX table
    print(latex_table)

# Input CSV file
csv_file = "lattice_results.csv"
csv_to_latex(csv_file)


def csv_to_latex_and_plot(csv_file):
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Convert DataFrame to LaTeX table format
    latex_table = df.to_latex(index=False, escape=False)
    
    # Print the LaTeX table
    print(latex_table)
    
    # Plot Babai Error (Original) and Babai Error (LLL) against Dim
    plt.figure(figsize=(10, 6))
    plt.plot(df["Dim"], df["Babai Error (Original)"], label="Babai Error (Original)", marker='o')
    plt.plot(df["Dim"], df["Babai Error (LLL)"], label="Babai Error (LLL)", marker='s')
    plt.plot(df["Dim"], df["Kannan Error (LLL)"], label="Kannan Error (LLL)", marker='^')
    plt.plot(df["Dim"], df["Babai Improvement (LLL)"], label="Babai Improvement (LLL)", marker='^')
    plt.plot(df["Dim"], df["Shortest Norm (Original)"], label="Shortest Norm (Original)", marker='^')
    plt.plot(df["Dim"], df["Shortest Norm (LLL)"], label="Shortest Norm (LLL)", marker='^')
    plt.xlabel("Dimension")
    plt.ylabel("Error")
    plt.yscale('log')
    plt.title("Errors in Different Lattice Reduction Methods")
    plt.legend()
    plt.grid()
    plt.show()
    
    plt.figure(figsize=(10, 6))
    plt.plot(df["Dim"], df["Babai Time"], label="Babai Time", marker='o')
    plt.plot(df["Dim"], df["LLL Time"], label="LLL Time", marker='s')
    plt.plot(df["Dim"], df["Kannan Time"], label="Kannan Time", marker='^')
    plt.plot(df["Dim"], df["Shortest Time"], label="Shortest Time", marker='^')
    plt.xlabel("Dimension")
    plt.ylabel("Seconds")
    plt.yscale('log')
    plt.title("Times for different algorithms")
    plt.legend()
    plt.grid()
    plt.show()

# Input CSV file
csv_to_latex_and_plot(csv_file)
