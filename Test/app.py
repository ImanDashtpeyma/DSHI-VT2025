# Import necessary libraries
import pandas as pd

# Load the datasets
math_df = pd.read_csv("./Data/student-mat.csv")
portuguese_df = pd.read_csv("./Data/student-por.csv")

# Add a new column 'course' to each dataset
math_df['course'] = 'mat'
portuguese_df['course'] = 'por'

# Combine the two datasets
combined_df = pd.concat([math_df, portuguese_df])

# Save the combined dataset to a new CSV file
combined_df.to_csv("./Data/combined_student_data.csv", index=False)

print("Combined dataset saved as 'combined_student_data.csv'")