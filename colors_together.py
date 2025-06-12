import pandas as pd
import numpy as np
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# --- Configuration ---
data_file_path = 'RD COLOR AI DATA_1.xlsx'
#excel_sheet_name = 'Sheet1'

# Define tint columns you are interested in for the filtering logic
# This list already excludes RED2.
TINT_COLUMNS = ['WHITE', 'BLACK', 'RED', 'BLUE', 'GREEN', 'YELLOW']
LAB_COLUMNS = ['L', 'a', 'b']
# This list will also not include RED2 because TINT_COLUMNS doesn't.
ALL_NECESSARY_COLUMNS_FOR_PROCESSING = TINT_COLUMNS + LAB_COLUMNS

# Define what "effectively zero" means for a tint value
EFFECTIVELY_ZERO_THRESHOLD = 0.01

# --- Step 1: Load the Excel file, Filter based on RED2, and Drop RED2 ---
#print(f"Attempting to load data from: {data_file_path}, Sheet: {excel_sheet_name}")
print(f"Attempting to load data from: {data_file_path}")
df = None
try:
    #df_loaded = pd.read_excel(data_file_path, sheet_name=excel_sheet_name)
    df_loaded = pd.read_excel(data_file_path)
    print(f"Successfully loaded data. Initial rows: {len(df_loaded)}")

    df = df_loaded.copy()  # Work on a copy

    # --- NEW: Step 1a - Remove rows where RED2 > 0 ---
    if 'RED2' in df.columns:
        # First, ensure RED2 is numeric for comparison
        df['RED2'] = pd.to_numeric(df['RED2'], errors='coerce')
        # It's good practice to handle potential NaNs created by to_numeric if RED2 was non-numeric
        # However, for this specific filter, we are interested in values > threshold.
        # Rows where RED2 became NaN won't satisfy RED2 > EFFECTIVELY_ZERO_THRESHOLD.

        initial_rows_before_red2_filter = len(df)
        # Keep rows where RED2 is NOT greater than the threshold (i.e., RED2 is zero or negligible)
        df = df[~(df['RED2'] > EFFECTIVELY_ZERO_THRESHOLD)]
        rows_removed = initial_rows_before_red2_filter - len(df)
        if rows_removed > 0:
            print(f"Removed {rows_removed} rows where 'RED2' > {EFFECTIVELY_ZERO_THRESHOLD}.")
    else:
        print("INFO: 'RED2' column not found in the loaded data. No rows removed based on RED2 values.")

    # --- NEW: Step 1b - Drop 'RED2' column if it exists ---
    if 'RED2' in df.columns:
        df.drop(columns=['RED2'], inplace=True)
        print("Dropped 'RED2' column from the DataFrame.")
    # --- END OF NEW SECTION ---

    # --- Basic Cleaning and Type Conversion (based on ALL_NECESSARY_COLUMNS_FOR_PROCESSING) ---
    missing_cols = [col for col in ALL_NECESSARY_COLUMNS_FOR_PROCESSING if col not in df.columns]
    if missing_cols:
        print(f"ERROR: The following columns needed for processing are missing: {missing_cols}")
        print(f"Available columns after RED2 handling: {df.columns.tolist()}")
        df = None

    if df is not None:
        for col in ALL_NECESSARY_COLUMNS_FOR_PROCESSING:
            # Ensure column exists before trying to convert (it should, based on missing_cols check)
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            else:  # Should not happen if missing_cols check is comprehensive
                print(f"Critical Error: Column {col} expected but not found before to_numeric.")
                df = None
                break

    if df is not None:
        original_row_count = len(df)
        df.dropna(subset=ALL_NECESSARY_COLUMNS_FOR_PROCESSING, inplace=True)
        cleaned_row_count = len(df)
        if original_row_count > cleaned_row_count:
            print(
                f"Removed {original_row_count - cleaned_row_count} rows due to missing/non-numeric values in actively processed columns (post RED2 handling).")

        if df.empty:
            print(f"No data remaining after cleaning for columns: {ALL_NECESSARY_COLUMNS_FOR_PROCESSING}.")
            df = None

except FileNotFoundError:
    print(f"ERROR: File not found at '{data_file_path}'.")
    print("Please ensure the Excel file exists at the specified path.")
except ImportError:
    print("Error: The 'openpyxl' library is required to read Excel files. Please run: pip install openpyxl")
except Exception as e:
    print(f"An error occurred while loading the data: {e}")

# Function to check if a row belongs to a specific model
def filter_rows(df, input_cols):
    color_sum = df[color_cols].sum(axis=1)
    input_sum = df[input_cols].sum(axis=1)

    return df[
        (input_sum > 0) & (color_sum == input_sum) |
        (color_sum == 0)  # all color columns are zero
        ]


# Function to build and train simple neural network
def build_and_train_difference_model(X_diff, y, activation1='tanh', output_activation='sigmoid'):
    model = Sequential([
        Dense(64, activation="relu", input_shape=(1,)),
        Dense(128,  activation=tf.keras.layers.LeakyReLU(alpha=0.001)),
        Dense(1, activation=output_activation)
    ])
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_diff, y, epochs=500, verbose=0)
    return model


# Proceed only if data loading and initial cleaning were successful
if df is not None and not df.empty:
    print(f"\nData ready for filtering. Number of rows after RED2 handling and cleaning: {len(df)}")
    print(f"DataFrame columns available for filtering: {df.columns.tolist()}")

    # --- Step 2: Data Preparation - Filtering (operates on TINT_COLUMNS list) ---
    # The rest of this section remains the same as before, as TINT_COLUMNS
    # already excludes RED2, and df no longer contains the RED2 column or rows
    # where RED2 had a significant value.

    is_zero_masks = {}
    for tint_col in TINT_COLUMNS:
        if tint_col in df.columns:
            is_zero_masks[tint_col] = df[tint_col].abs() < EFFECTIVELY_ZERO_THRESHOLD
        else:
            print(
                f"Warning: Tint column '{tint_col}' for zero masking not found in DataFrame. This may affect filtering.")

    is_significant_masks = {}
    for tint_col in TINT_COLUMNS:
        if tint_col in df.columns:
            is_significant_masks[tint_col] = df[tint_col].abs() >= EFFECTIVELY_ZERO_THRESHOLD
        else:
            print(
                f"Warning: Tint column '{tint_col}' for significant masking not found in DataFrame. This may affect filtering.")

    single_significant_tint_indices = pd.Index([])

    for i, target_tint_col in enumerate(TINT_COLUMNS):
        if target_tint_col not in is_significant_masks:  # Check if mask was created
            print(f"Skipping isolation check for {target_tint_col} as its significance mask is missing.")
            continue

        condition1 = is_significant_masks[target_tint_col]
        condition2_others_zero = pd.Series(True, index=df.index)

        valid_other_tints_for_check = True
        for other_tint_col in TINT_COLUMNS:
            if other_tint_col != target_tint_col:
                if other_tint_col not in is_zero_masks:  # Check if mask for other tint exists
                    print(
                        f"Warning: Mask for 'other tint' {other_tint_col} missing. Cannot reliably perform isolation check for {target_tint_col}.")
                    valid_other_tints_for_check = False
                    break
                condition2_others_zero &= is_zero_masks[other_tint_col]

        if not valid_other_tints_for_check:
            continue

        current_tint_isolated_rows = df[condition1 & condition2_others_zero]
        single_significant_tint_indices = single_significant_tint_indices.union(current_tint_isolated_rows.index)

    print(
        f"\nNumber of rows found where exactly one tint (from TINT_COLUMNS list) is significant: {len(single_significant_tint_indices)}")

    all_tints_in_list_are_zero_condition = pd.Series(True, index=df.index)
    all_tints_list_columns_present_for_zero_check = True
    for tint_col in TINT_COLUMNS:
        if tint_col not in is_zero_masks:  # Check if mask exists
            print(f"Warning: Mask for tint {tint_col} missing. Cannot reliably perform 'all tints zero' check.")
            all_tints_list_columns_present_for_zero_check = False
            break
        all_tints_in_list_are_zero_condition &= is_zero_masks[tint_col]

    all_tints_zero_rows_indices = pd.Index([])
    if all_tints_list_columns_present_for_zero_check:
        all_tints_zero_rows_indices = df[all_tints_in_list_are_zero_condition].index
        print(
            f"Number of rows found where all tints in TINT_COLUMNS list are effectively zero: {len(all_tints_zero_rows_indices)}")
    else:
        print("Skipping 'all tints zero' check due to missing tint column masks.")

    final_indices_to_keep = single_significant_tint_indices.union(all_tints_zero_rows_indices)

    if not final_indices_to_keep.empty:
        df_filtered = df.loc[final_indices_to_keep].sort_index()
    else:
        df_filtered = pd.DataFrame(columns=df.columns)  # Create empty df with same columns if no rows to keep

    print(f"\n--- Data Filtering Complete ---")
    print(f"Total number of rows kept: {len(df_filtered)}")

    if not df_filtered.empty:
        print("\n--- Displaying all filtered data (df_filtered) ---")
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', None)

        print(df_filtered)

        print("\nBreakdown of kept rows:")
        if all_tints_list_columns_present_for_zero_check:
            print(
                f"  - Rows where all specified tints in TINT_COLUMNS are effectively zero: {len(all_tints_zero_rows_indices)}")  # This count is from before final union

        # Recalculate counts based on the final df_filtered for accurate breakdown
        actual_all_zero_in_filtered = 0
        if all_tints_list_columns_present_for_zero_check:
            temp_all_zero_cond_filtered = pd.Series(True, index=df_filtered.index)
            for tc in TINT_COLUMNS:
                temp_all_zero_cond_filtered &= (df_filtered[tc].abs() < EFFECTIVELY_ZERO_THRESHOLD)
            actual_all_zero_in_filtered = len(df_filtered[temp_all_zero_cond_filtered])
            print(f"  - Actual rows in filtered set where all TINT_COLUMNS are zero: {actual_all_zero_in_filtered}")

        for tint_col in TINT_COLUMNS:
            if tint_col not in df_filtered.columns:
                continue
            cond1 = df_filtered[tint_col].abs() >= EFFECTIVELY_ZERO_THRESHOLD
            cond2_others_zero = pd.Series(True, index=df_filtered.index)

            valid_check = True
            for other_tc in TINT_COLUMNS:
                if other_tc != tint_col:
                    if other_tc not in df_filtered.columns:
                        valid_check = False
                        break
                    cond2_others_zero &= (df_filtered[other_tc].abs() < EFFECTIVELY_ZERO_THRESHOLD)
            if not valid_check:
                continue

            count = len(df_filtered[cond1 & cond2_others_zero])
            if count > 0:
                print(f"  - Rows in filtered set where only {tint_col} is significant: {count} rows")
    else:
        print("No rows met the filtering criteria OR data became empty after RED2 processing.")
else:
    print("\nSkipping filtering step due to issues in data loading or initial cleaning.")
# Color columns
color_cols = ['WHITE', 'BLACK', 'RED', 'GREEN', 'YELLOW', 'BLUE']

class CustomScaler:
    def __init__(self):
        self.max_y = None
        self.min_x = None
        self.modified_y_at_min_x = None

    def fit_transform(self, X, y):
        print(f"X values: {X}")
        print(f"y values: {y}")

        # Step 1: Find maximum of y values
        self.max_y = np.max(y)
        print(f"Max y value is: {self.max_y}")

        # Step 2: Subtract max_y from each y value
        modified_y = y - self.max_y
        print(f"After subtracting max y: {modified_y}")

        # Step 3: Find minimum X value
        self.min_x = np.min(X)
        print(f"Minimum X value is: {self.min_x}")

        # Step 4: Find y value corresponding to minimum X
        min_x_index = np.where(X == self.min_x)[0][0]
        self.modified_y_at_min_x = modified_y[min_x_index]
        print(f"Modified y value at min X: {self.modified_y_at_min_x}")

        # Step 5: Divide each modified y value by modified_y_at_min_x
        final = modified_y / abs(self.modified_y_at_min_x)
        print(f"Final scaled values: {final}")

        return final

    def inverse_transform(self, y_scaled):
        return (y_scaled * abs(self.modified_y_at_min_x)) + self.max_y
# Define model specs
models = {
    'l_model': {
        'inputs': ['WHITE', 'BLACK'],
        'output': 'L'
    },
    'a_model': {
        'inputs': ['RED', 'GREEN'],
        'output': 'a'
    },
    'b_model': {
        'inputs': ['YELLOW', 'BLUE'],
        'output': 'b'
    }
}

trained_models = {}

# Process each model
for name, spec in models.items():
    print(name)
    filtered_df = filter_rows(df_filtered, spec['inputs'])
    X = filtered_df[spec['inputs']].values
    y = filtered_df[spec['output']].values
    print(f"{X}, {y}")
    scaler = CustomScaler()
    y_scaled = scaler.fit_transform(X.flatten(), y)
    y = y_scaled

    if len(X) > 0 :
        print(f"Training {name} on {len(X)} rows")
        # Compute input difference
        input_diff = (filtered_df[spec['inputs'][0]] - filtered_df[spec['inputs'][1]]).values
        target = y
        #print(input_diff,target.min(),target.max())
        activation = 'linear'
        model = build_and_train_difference_model(input_diff, target, output_activation=activation)
        trained_models[name] = model

        _, X_eval, _, y_eval = train_test_split(input_diff, target, test_size=0.8, random_state=42)
        y_pred_eval = model.predict(X_eval).flatten()
        if (spec['output'] == 'L'):
            y_pred_eval = scaler_y.inverse_transform(y_pred_eval.reshape(-1, 1))

        mse = mean_squared_error(y_eval, y_pred_eval)
        print(f"{name} MSE on 80% subset: {mse:.4f}")

        # Predict on full input and plot
        y_pred_full = model.predict(input_diff).flatten()
        print("=" * 60)
        print(f"validating {name} trained on {len(X)} rows ")
        print("input values:")
        print(input_diff)
        print("Actual values:")
        print(target)
        print("predicted values:")
        print(y_pred_full)
        print("="*60)

        # Plot
        plt.figure(figsize=(8, 5))
        plt.scatter(input_diff, target, label="Actual", color="blue")
        plt.scatter(input_diff, y_pred_full, label="Predicted", color="red", marker='x')
        plt.plot(sorted(input_diff.flatten()), sorted(y_pred_full), color='red', linestyle='--', alpha=0.5)
        plt.xlabel(f"{spec['inputs'][0]} - {spec['inputs'][1]}")
        plt.ylabel(spec['output'])
        plt.title(f"{name.upper()}: Actual vs Predicted")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"plots/colors together/{name}.png")
        plt.show()
    else:
        print(f"Not enough data to train {name}")

#
# import pandas as pd
# import numpy as np
# import os
# import warnings
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
#
# # from sklearn.model_selection import train_test_split # Not used in the final model training loop as per user script
# # import tensorflow as tf # Sequential and Dense are imported from tensorflow.keras
# # from tensorflow import keras # Sequential and Dense are imported from tensorflow.keras
# # from tensorflow.keras import layers # Sequential and Dense are imported from tensorflow.keras
#
# # Suppress potential warnings
# warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
# warnings.filterwarnings('ignore', category=FutureWarning)
#
# # --- Configuration ---
# data_file_path = 'RD COLOR AI DATA.xlsx'
# excel_sheet_name = 'Sheet1'
#
# # Define tint columns you are interested in for the filtering logic
# # This list already excludes RED2.
# TINT_COLUMNS = ['WHITE', 'BLACK', 'RED', 'BLUE', 'GREEN', 'YELLOW']
# LAB_COLUMNS = ['L', 'a', 'b']
# # This list will also not include RED2 because TINT_COLUMNS doesn't.
# ALL_NECESSARY_COLUMNS_FOR_PROCESSING = TINT_COLUMNS + LAB_COLUMNS
#
# # Define what "effectively zero" means for a tint value
# EFFECTIVELY_ZERO_THRESHOLD = 0.01
#
# # --- Step 1: Load the Excel file, Filter based on RED2, and Drop RED2 ---
# print(f"Attempting to load data from: {data_file_path}, Sheet: {excel_sheet_name}")
# df = None
# try:
#     df_loaded = pd.read_excel(data_file_path, sheet_name=excel_sheet_name)
#     print(f"Successfully loaded data. Initial rows: {len(df_loaded)}")
#
#     df = df_loaded.copy()  # Work on a copy
#
#     # --- Step 1a - Remove rows where RED2 > EFFECTIVELY_ZERO_THRESHOLD ---
#     if 'RED2' in df.columns:
#         print("\n--- Processing 'RED2' column ---")
#         df['RED2'] = pd.to_numeric(df['RED2'], errors='coerce')
#
#         initial_rows_before_red2_filter = len(df)
#         # Keep rows where RED2 is NOT greater than the threshold OR RED2 is NaN (will be dropped later if coerce created it)
#         df = df[~(df['RED2'] > EFFECTIVELY_ZERO_THRESHOLD)]
#         rows_removed = initial_rows_before_red2_filter - len(df)
#         if rows_removed > 0:
#             print(f"Removed {rows_removed} rows where 'RED2' > {EFFECTIVELY_ZERO_THRESHOLD}.")
#         else:
#             print(
#                 f"No rows removed based on 'RED2' > {EFFECTIVELY_ZERO_THRESHOLD} condition (or RED2 was not present/already zero/NaN).")
#
#     # --- Step 1b - Drop 'RED2' column if it exists ---
#     if 'RED2' in df.columns:
#         df.drop(columns=['RED2'], inplace=True, errors='ignore')  # errors='ignore' if already dropped or not there
#         if 'RED2' not in df_loaded.columns and 'RED2' not in df.columns:  # If RED2 was not in original
#             pass  # Already handled by the first check
#         else:
#             print("Dropped 'RED2' column from the DataFrame.")
#     else:  # If RED2 was not in df.columns to begin with
#         print("INFO: 'RED2' column not found in the loaded data. No column dropped.")
#     # --- END OF RED2 PROCESSING ---
#
#     print(f"\n--- Basic Cleaning on Essential Columns (post RED2 handling) ---")
#     # Ensure ALL_NECESSARY_COLUMNS_FOR_PROCESSING only contains columns that actually exist in df
#     current_essential_cols = [col for col in ALL_NECESSARY_COLUMNS_FOR_PROCESSING if col in df.columns]
#     missing_essential = [col for col in ALL_NECESSARY_COLUMNS_FOR_PROCESSING if col not in df.columns]
#     if missing_essential:
#         print(
#             f"Warning: The following defined essential columns are missing from the DataFrame and will be ignored: {missing_essential}")
#         print(f"Available columns for cleaning: {df.columns.tolist()}")
#
#     if not current_essential_cols:
#         print("ERROR: No essential columns (from TINT_COLUMNS + LAB_COLUMNS, excluding RED2) found for processing.")
#         df = None
#
#     if df is not None:
#         for col in current_essential_cols:  # Use only existing essential columns
#             df[col] = pd.to_numeric(df[col], errors='coerce')
#
#         original_row_count = len(df)
#         df.dropna(subset=current_essential_cols, inplace=True)
#         cleaned_row_count = len(df)
#         if original_row_count > cleaned_row_count:
#             print(
#                 f"Removed {original_row_count - cleaned_row_count} rows due to missing/non-numeric values in processed columns.")
#
#         if df.empty:
#             print(f"No data remaining after cleaning for columns: {current_essential_cols}.")
#             df = None
#
# except FileNotFoundError:
#     print(f"ERROR: File not found at '{data_file_path}'.")
#     print("Please ensure the Excel file exists at the specified path.")
#     df = None
# except ImportError:
#     print("Error: The 'openpyxl' library is required to read Excel files. Please run: pip install openpyxl")
#     df = None
# except Exception as e:
#     print(f"An error occurred while loading the data: {e}")
#     df = None
#
#
# # This function is defined by the user.
# # It filters rows based on whether the sum of all tints equals the sum of the model-specific input tints,
# # OR if all tints are zero. This implies other tints (not in input_cols) must be zero.
# def filter_rows(df_input, input_cols, all_tint_cols, zero_threshold):
#     """
#     Filters rows for specialized model training.
#     Keeps rows where:
#     1. All tints in all_tint_cols are effectively zero.
#     OR
#     2. Only the tints in input_cols are significant, and all other tints in all_tint_cols are effectively zero.
#     """
#     if df_input is None or df_input.empty:
#         return pd.DataFrame(columns=df_input.columns if df_input is not None else [])
#
#     # Ensure all necessary columns exist in the input DataFrame
#     if not all(col in df_input.columns for col in all_tint_cols):
#         print(
#             f"Warning in filter_rows: Not all tint columns ({all_tint_cols}) found in input DataFrame. Filtering might be incorrect.")
#         # Return empty if critical columns are missing to avoid errors
#         missing_in_filter = [col for col in all_tint_cols if col not in df_input.columns]
#         if any(col in input_cols for col in missing_in_filter):  # If an input_col itself is missing
#             return pd.DataFrame(columns=df_input.columns)
#         # Proceed with available columns, but be aware of potential issues
#         all_tint_cols = [col for col in all_tint_cols if col in df_input.columns]
#         input_cols = [col for col in input_cols if col in df_input.columns]
#
#     # Sum of all specified tint columns
#     all_tints_sum = df_input[all_tint_cols].abs().sum(axis=1)
#
#     # Sum of the model-specific input tint columns
#     input_tints_sum = df_input[input_cols].abs().sum(axis=1)
#
#     # Condition 1: All tints in all_tint_cols are effectively zero
#     cond_all_zero = all_tints_sum < zero_threshold
#
#     # Condition 2: Only the input_cols are significant.
#     # This means the sum of all tints should be approximately equal to the sum of input tints,
#     # AND the input_tints_sum itself should be significant.
#     cond_only_inputs_sig = (input_tints_sum >= zero_threshold) & \
#                            ((all_tints_sum - input_tints_sum).abs() < zero_threshold)
#
#     return df_input[cond_all_zero | cond_only_inputs_sig]
#
#
# # Function to build and train simple neural network
# def build_and_train_model(X, y):
#     model = Sequential([
#         Dense(64, activation='relu', input_shape=(X.shape[1],)),
#         Dense(128, activation='relu'),
#         Dense(1, activation='sigmoid')  # regression output
#     ])
#     model.compile(optimizer='adam', loss='mse', metrics=['mean_absolute_error'])  # Added MAE for monitoring
#     print(f"  Training model with {X.shape[0]} samples, input shape {X.shape[1]}...")
#     model.fit(X, y, epochs=100, batch_size=2, verbose=0)  # Kept batch_size from previous versions
#     return model
#
#
# # Proceed only if data loading and initial cleaning were successful
# if df is not None and not df.empty:
#     print(f"\nData ready for specific filtering (df). Number of rows: {len(df)}")
#
#     # --- Step 2: Initial broad filtering (as per user's script structure) ---
#     # This creates df_filtered: rows where exactly one tint from TINT_COLUMNS is significant,
#     # OR all tints from TINT_COLUMNS are zero.
#     print(f"\n--- Applying Initial Broad Filter (Step 2 from user script) ---")
#     is_zero_masks = {}
#     for tint_col in TINT_COLUMNS:  # TINT_COLUMNS is ['WHITE', 'BLACK', 'RED', 'BLUE', 'GREEN', 'YELLOW']
#         if tint_col in df.columns:
#             is_zero_masks[tint_col] = df[tint_col].abs() < EFFECTIVELY_ZERO_THRESHOLD
#         else:
#             is_zero_masks[tint_col] = pd.Series(False, index=df.index)  # Should not happen if cleaning worked
#
#     is_significant_masks = {}
#     for tint_col in TINT_COLUMNS:
#         if tint_col in df.columns:
#             is_significant_masks[tint_col] = df[tint_col].abs() >= EFFECTIVELY_ZERO_THRESHOLD
#         else:
#             is_significant_masks[tint_col] = pd.Series(False, index=df.index)
#
#     single_significant_tint_indices = pd.Index([])
#     for i, target_tint_col in enumerate(TINT_COLUMNS):
#         if not is_significant_masks[target_tint_col].any():
#             continue
#         condition1 = is_significant_masks[target_tint_col]
#         condition2_others_zero = pd.Series(True, index=df.index)
#         for other_tint_col in TINT_COLUMNS:
#             if other_tint_col != target_tint_col:
#                 condition2_others_zero &= is_zero_masks[other_tint_col]
#         current_tint_isolated_rows = df[condition1 & condition2_others_zero]
#         single_significant_tint_indices = single_significant_tint_indices.union(current_tint_isolated_rows.index)
#
#     print(
#         f"Number of rows where exactly one tint (from TINT_COLUMNS) is significant: {len(single_significant_tint_indices)}")
#
#     all_tints_in_list_are_zero_condition = pd.Series(True, index=df.index)
#     for tint_col in TINT_COLUMNS:
#         all_tints_in_list_are_zero_condition &= is_zero_masks[tint_col]
#     all_tints_zero_rows_indices = df[all_tints_in_list_are_zero_condition].index
#     print(f"Number of rows where all tints in TINT_COLUMNS are effectively zero: {len(all_tints_zero_rows_indices)}")
#
#     final_indices_to_keep = single_significant_tint_indices.union(all_tints_zero_rows_indices)
#
#     if not final_indices_to_keep.empty:
#         df_filtered_step2 = df.loc[final_indices_to_keep].sort_index()  # Renamed to avoid confusion
#     else:
#         df_filtered_step2 = pd.DataFrame(columns=df.columns)
#
#     print(f"\n--- Initial Broad Filtering Complete ---")
#     print(f"Total number of rows in df_filtered_step2: {len(df_filtered_step2)}")
#
#     if not df_filtered_step2.empty:
#         # --- Displaying the df_filtered_step2 data ---
#         # print("\n--- Displaying all data from df_filtered_step2 ---")
#         # pd.set_option('display.max_rows', None)
#         # pd.set_option('display.max_columns', None)
#         # pd.set_option('display.width', None)
#         # pd.set_option('display.max_colwidth', None)
#         # print(df_filtered_step2)
#         pass  # Displaying all can be very long, user can uncomment if needed.
#     else:
#         print("No rows after initial broad filtering. Cannot proceed with model training.")
#
#     # --- Model Specific Filtering and Training ---
#     if not df_filtered_step2.empty:
#         # color_cols is used by filter_rows, should be the same as TINT_COLUMNS
#         # TINT_COLUMNS is already defined globally.
#
#         model_specs = {
#             'L_Model': {
#                 'inputs': ['WHITE', 'BLACK'],
#                 'output': 'L',
#                 'predictions': [(1.2, 0.0), (0.0, 0.6)]  # (WHITE, BLACK)
#             },
#             'a_Model': {
#                 'inputs': ['RED', 'GREEN'],
#                 'output': 'a',
#                 'predictions': [(1.1, 0.0), (0.0, 0.5)]  # (RED, GREEN)
#             },
#             'b_Model': {
#                 'inputs': ['YELLOW', 'BLUE'],
#                 'output': 'b',
#                 'predictions': [(1.5, 0.0), (0.0, 0.9)]  # (YELLOW, BLUE)
#             }
#         }
#
#         trained_models_dict = {}
#
#         for model_name, spec in model_specs.items():
#             print(f"\n--- Processing for {model_name} ---")
#
#             # Apply the user's filter_rows function to the result of the Step 2 filtering
#             df_for_model_training = filter_rows(df_filtered_step2, spec['inputs'], TINT_COLUMNS,
#                                                 EFFECTIVELY_ZERO_THRESHOLD)
#
#             print(f"Number of rows selected by filter_rows for {model_name}: {len(df_for_model_training)}")
#
#             if not df_for_model_training.empty and len(df_for_model_training) >= 5:  # Need enough for train/val split
#                 X = df_for_model_training[spec['inputs']].values
#                 y = df_for_model_training[spec['output']].values
#
#                 # Splitting the specifically filtered data for this model
#                 # Using a fixed 0.2 test_size as per user's previous requests for model evaluation.
#                 # If all data is for training, this split is for Keras's internal validation.
#                 # The user script does not show an explicit train_test_split here,
#                 # but model.fit usually takes validation_data or validation_split.
#                 # For simplicity and to use all data for training as requested,
#                 # we'll pass all X, y to model.fit and let it use validation_split.
#
#                 # X_train_model, X_test_model, y_train_model, y_test_model = train_test_split(
#                 #     X, y, test_size=0.2, random_state=42
#                 # )
#                 # print(f"  Training {model_name} on {len(X_train_model)} rows, testing on {len(X_test_model)} rows.")
#                 # trained_models_dict[model_name] = build_and_train_model(X_train_model, y_train_model)
#
#                 # Using all filtered data for training, Keras handles validation split
#                 trained_models_dict[model_name] = build_and_train_model(X, y)
#                 print(f"Successfully trained {model_name}.")
#
#                 # Make specific predictions
#                 print(f"\n  Predictions for {model_name}:")
#                 for pred_input_tuple in spec['predictions']:
#                     pred_input_array = np.array([list(pred_input_tuple)])
#                     prediction = trained_models_dict[model_name].predict(pred_input_array, verbose=0)
#                     print(
#                         f"    Input {dict(zip(spec['inputs'], pred_input_tuple))}: Predicted {spec['output']} = {prediction[0][0]:.2f}")
#             else:
#                 print(f"Not enough data (found {len(df_for_model_training)}) to train {model_name}. Needs at least 5.")
#                 trained_models_dict[model_name] = None
#     else:
#         print("\nScript terminated early due to issues in data loading or initial cleaning (df is None or empty).")
#
# print("\n--- Script Finished ---")
