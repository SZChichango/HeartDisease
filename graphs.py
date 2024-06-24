#!/usr/bin/env python3

import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


try:
    # Connect to the SQLite database
    conn = sqlite3.connect("hearts.db")
    print("Database connected successfully")

    with conn:

        # Function to plot graphs
        def plot_distribution(conn, vars, target):
            fig, axes = plt.subplots(nrows=len(vars), figsize=(14, 6 * len(vars)))
            axes = axes.flatten()

            for i, var in enumerate(vars):
                query = f"SELECT {var}, {target} FROM hearts"
                df = pd.read_sql_query(query, conn)

                # Create a count plot
                sns.countplot(x=var, hue=target, data=df, ax=axes[i])
                axes[i].set_title(f'Distribution of {var} by {target}')
                axes[i].set_xlabel(var)
                axes[i].set_ylabel('Count')
                axes[i].legend(title=target)

            # Adjust layout
            plt.tight_layout()

            # Show plot
            plt.show()
        # Target
        target = 'target'
        # Categorical Variables
        categorical_variables = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
        # Numerical Variables
        numerical_variables = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

        #         Creating Graphs
        plot_distribution(conn, categorical_variables, target)
        plot_distribution(conn, numerical_variables, target)




except sqlite3.Error as error:
    print("Error while connecting to SQLite:", error)


finally:
    if conn:
        conn.close()
        print("Database connection is closed")
