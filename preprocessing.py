import pandas as pd
import sqlite3

def preprocess_data(csv_file, db_file='hearts.db', table_name='hearts'):
    """Reads a CSV file, preprocesses the data, and stores it in an SQLite3 database."""

    conn = None
    try:
        # Read CSV file with semicolon delimiter
        df = pd.read_csv(csv_file, delimiter=';')

        # Check if 'age' column exists
        if 'age' not in df.columns:
            raise KeyError("'age' column not found in the CSV file")

        # Data Cleaning and Preprocessing
        df = df.dropna()  # Drop rows with missing values
        df = df[df['age'] > 0]  # Drop rows with non-positive age values

        # Convert categorical columns to integers if necessary
        # Assuming 'sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', and 'thal' are categorical
        categorical_columns = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
        for col in categorical_columns:
            if col in df.columns:
                df[col] = df[col].astype(int)
            else:
                raise KeyError(f"'{col}' column not found in the CSV file")

        # Connect to SQLite3 database
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()

        # Create table (if it doesn't exist)
        cursor.execute(f'''
            CREATE TABLE IF NOT EXISTS {table_name} (
                age INTEGER,
                sex INTEGER,
                cp INTEGER,
                trestbps INTEGER,
                chol INTEGER,
                fbs INTEGER,
                restecg INTEGER,
                thalach INTEGER,
                exang INTEGER,
                oldpeak REAL,
                slope INTEGER,
                ca INTEGER,
                thal INTEGER,
                target INTEGER
            )
        ''')

        # Insert data into table
        for _, row in df.iterrows():
            cursor.execute(f'''
                INSERT INTO {table_name} (age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, target)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', row.values.tolist())  # Pass the entire row as a list of values

        # Commit changes and close connection
        conn.commit()
        print(f"Data from '{csv_file}' preprocessed and stored in '{db_file}' table '{table_name}' successfully!")

    except (sqlite3.Error, KeyError, pd.errors.EmptyDataError) as e:
        print(f"An error occurred: {e}")

    finally:
        if conn:
            conn.close()

# Call the function
preprocess_data('heart.csv')  # Replace 'heart.csv' with your actual file path
