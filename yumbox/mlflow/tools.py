import json

import pandas as pd


def flatten_json_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flatten JSON string columns into separate columns.

    Args:
        df: DataFrame with potential JSON columns

    Returns:
        DataFrame with flattened JSON columns
    """
    df_flattened = df.copy()

    # Check each column for JSON-like strings
    for col in df.columns:
        if df[col].dtype == "object":
            # Try to parse first non-null value to see if it's JSON
            sample_val = (
                df[col].dropna().iloc[0] if not df[col].dropna().empty else None
            )

            if sample_val and isinstance(sample_val, str):
                try:
                    # Try to parse as JSON
                    json.loads(sample_val)
                    is_json_column = True
                except (json.JSONDecodeError, TypeError):
                    is_json_column = False

                if is_json_column:
                    # Parse all JSON values in this column
                    json_data = []
                    for idx, val in df[col].items():
                        if pd.isna(val) or val == "":
                            json_data.append({})
                        else:
                            try:
                                json_data.append(json.loads(val))
                            except (json.JSONDecodeError, TypeError):
                                json_data.append({})

                    # Convert to DataFrame and flatten
                    if json_data:
                        json_df = pd.json_normalize(json_data)
                        # Prefix column names with original column name
                        json_df.columns = [
                            f"{col}.{subcol}" for subcol in json_df.columns
                        ]
                        json_df.index = df.index

                        # Drop original column and add flattened columns
                        df_flattened = df_flattened.drop(columns=[col])
                        df_flattened = pd.concat([df_flattened, json_df], axis=1)

    return df_flattened
