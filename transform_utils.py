"""
Utilities for data transformation and datatype mapping between different database systems.
"""
import pandas as pd
import numpy as np
import datetime
import logging
import uuid
import re
import json
from typing import Dict, List, Any, Tuple, Optional

logger = logging.getLogger(__name__)

# --- Datatype Mapping Dictionaries ---

MYSQL_TO_SQLSERVER_TYPE_MAP = {
    'TINYINT': 'TINYINT',
    'SMALLINT': 'SMALLINT',
    'MEDIUMINT': 'INT', # Mapped to standard INT
    'INT': 'INT',
    'INTEGER': 'INT', # Alias for INT
    'BIGINT': 'BIGINT',
    'FLOAT': 'FLOAT', # Defaults to FLOAT(53) in SQL Server
    'DOUBLE': 'FLOAT', # Map to FLOAT(53)
    'DECIMAL': 'DECIMAL', # Precision/scale might need adjustment
    'NUMERIC': 'NUMERIC', # Alias for DECIMAL
    'CHAR': 'CHAR',
    'VARCHAR': 'VARCHAR',
    'TINYTEXT': 'VARCHAR(255)', # Map to reasonable VARCHAR
    'TEXT': 'VARCHAR(MAX)',
    'MEDIUMTEXT': 'VARCHAR(MAX)',
    'LONGTEXT': 'VARCHAR(MAX)',
    'TINYBLOB': 'VARBINARY(255)', # Map to reasonable VARBINARY
    'BLOB': 'VARBINARY(MAX)',
    'MEDIUMBLOB': 'VARBINARY(MAX)',
    'LONGBLOB': 'VARBINARY(MAX)',
    'BINARY': 'BINARY',
    'VARBINARY': 'VARBINARY',
    'DATE': 'DATE',
    'DATETIME': 'DATETIME2', # Use modern DATETIME2
    'TIMESTAMP': 'DATETIME2', # Map MySQL TIMESTAMP to DATETIME2
    'TIME': 'TIME',
    'YEAR': 'SMALLINT', # Store year as a number
    'ENUM': 'VARCHAR(255)', # Store ENUM as string
    'SET': 'VARCHAR(255)', # Store SET as string
    'JSON': 'NVARCHAR(MAX)', # Use NVARCHAR for JSON strings
    'BIT': 'BIT'
}

MYSQL_TO_SNOWFLAKE_TYPE_MAP = {
    'TINYINT': 'NUMBER(3,0)',
    'SMALLINT': 'SMALLINT',
    'MEDIUMINT': 'INTEGER',
    'INT': 'INTEGER',
    'INTEGER': 'INTEGER',
    'BIGINT': 'BIGINT',
    'FLOAT': 'FLOAT',
    'DOUBLE': 'DOUBLE',
    'DECIMAL': 'NUMBER', # Precision/scale might need adjustment
    'NUMERIC': 'NUMBER',
    'CHAR': 'CHAR',
    'VARCHAR': 'VARCHAR', # Snowflake handles length up to 16MB
    'TINYTEXT': 'STRING',
    'TEXT': 'STRING',
    'MEDIUMTEXT': 'STRING',
    'LONGTEXT': 'STRING',
    'TINYBLOB': 'BINARY',
    'BLOB': 'BINARY',
    'MEDIUMBLOB': 'BINARY',
    'LONGBLOB': 'BINARY',
    'BINARY': 'BINARY',
    'VARBINARY': 'BINARY',
    'DATE': 'DATE',
    'DATETIME': 'TIMESTAMP_NTZ', # Default to timezone-naive
    'TIMESTAMP': 'TIMESTAMP_LTZ', # Default MySQL TIMESTAMP often implies session TZ
    'TIME': 'TIME',
    'YEAR': 'SMALLINT',
    'ENUM': 'STRING',
    'SET': 'STRING',
    'JSON': 'VARIANT', # Use Snowflake's native VARIANT
    'BIT': 'BOOLEAN' # Map BIT to BOOLEAN
}

SQLSERVER_TO_SNOWFLAKE_TYPE_MAP = {
    'BIT': 'BOOLEAN',
    'TINYINT': 'NUMBER(3,0)',
    'SMALLINT': 'SMALLINT',
    'INT': 'INTEGER',
    'BIGINT': 'BIGINT',
    'DECIMAL': 'NUMBER',
    'NUMERIC': 'NUMBER',
    'FLOAT': 'FLOAT', # Precision maps reasonably
    'REAL': 'FLOAT', # REAL is approx single-precision float
    'MONEY': 'NUMBER(19,4)',
    'SMALLMONEY': 'NUMBER(10,4)',
    'CHAR': 'CHAR',
    'VARCHAR': 'VARCHAR',
    'TEXT': 'STRING',
    'NCHAR': 'CHAR', # Map NCHAR to CHAR
    'NVARCHAR': 'VARCHAR', # Map NVARCHAR to VARCHAR
    'NTEXT': 'STRING',
    'BINARY': 'BINARY',
    'VARBINARY': 'BINARY',
    'IMAGE': 'BINARY', # Deprecated IMAGE type
    'DATE': 'DATE',
    'DATETIME': 'TIMESTAMP_NTZ', # Older DATETIME
    'DATETIME2': 'TIMESTAMP_NTZ', # Newer DATETIME2
    'SMALLDATETIME': 'TIMESTAMP_NTZ', # Lower precision DATETIME
    'DATETIMEOFFSET': 'TIMESTAMP_TZ', # Includes timezone offset
    'TIME': 'TIME',
    'UNIQUEIDENTIFIER': 'VARCHAR(36)', # Store GUID as string
    'XML': 'STRING', # Store XML as string (or VARIANT if parsing desired)
    'SQL_VARIANT': 'VARIANT', # Maps well to VARIANT
    # TIMESTAMP/ROWVERSION is tricky, usually for concurrency, map to BINARY or handle differently
    'TIMESTAMP': 'BINARY(8)'
}

# Mapping for generic types inferred from CSV/Pandas
GENERIC_TO_SNOWFLAKE_TYPE_MAP = {
    'VARCHAR': 'VARCHAR', # Use Snowflake VARCHAR (default max length)
    'TEXT': 'STRING',
    'INT': 'INTEGER',
    'INTEGER': 'INTEGER',
    'BIGINT': 'BIGINT',
    'SMALLINT': 'SMALLINT',
    'TINYINT': 'NUMBER(3,0)',
    'FLOAT': 'FLOAT',
    'DOUBLE': 'DOUBLE', # Map generic DOUBLE to Snowflake DOUBLE
    'REAL': 'FLOAT',
    'DECIMAL': 'NUMBER',
    'NUMERIC': 'NUMBER',
    'BOOLEAN': 'BOOLEAN', # Map generic BOOLEAN to Snowflake BOOLEAN
    'DATE': 'DATE',
    'TIMESTAMP': 'TIMESTAMP_NTZ', # Map generic TIMESTAMP to default Snowflake TZ-naive
    'DATETIME': 'TIMESTAMP_NTZ',
    'TIME': 'TIME',
    'BINARY': 'BINARY',
    'VARBINARY': 'BINARY',
    'UUID': 'VARCHAR(36)', # Map UUID-like strings
}

GENERIC_TO_SQLSERVER_TYPE_MAP = {
    'VARCHAR': 'VARCHAR', # Need length handling
    'TEXT': 'VARCHAR(MAX)',
    'INT': 'INT',
    'INTEGER': 'INT',
    'BIGINT': 'BIGINT',
    'SMALLINT': 'SMALLINT',
    'TINYINT': 'TINYINT',
    'FLOAT': 'FLOAT', # SQL Server FLOAT is FLOAT(53) by default
    'DOUBLE': 'FLOAT', # Map generic DOUBLE to SQL Server FLOAT(53)
    'REAL': 'REAL', # SQL Server REAL is approx FLOAT(24)
    'DECIMAL': 'DECIMAL', # Need precision/scale
    'NUMERIC': 'NUMERIC',
    'BOOLEAN': 'BIT', # Map generic BOOLEAN to SQL Server BIT
    'DATE': 'DATE',
    'TIMESTAMP': 'DATETIME2', # Map generic TIMESTAMP to DATETIME2
    'DATETIME': 'DATETIME2',
    'TIME': 'TIME',
    'BINARY': 'VARBINARY', # Need length handling
    'VARBINARY': 'VARBINARY',
    'UUID': 'UNIQUEIDENTIFIER', # SQL Server has native UUID type
}


# --- Regex for Parsing Data Types ---

DATATYPE_PARAMS_REGEX = {
    # Captures base_type, length/max, scale
    'mysql': r'^([a-zA-Z]+)(?:\s*\(\s*(\d+)(?:\s*,\s*(\d+))?\s*\))?(?:\s+UNSIGNED)?(?:\s+ZEROFILL)?',
    'sqlserver': r'^([a-zA-Z_]+)(?:\s*\(\s*(\d+|max)(?:\s*,\s*(\d+))?\s*\))?',
    'snowflake': r'^([a-zA-Z]+)(?:\s*\(\s*(\d+)(?:\s*,\s*(\d+))?\s*\))?',
    'generic': r'^([a-zA-Z]+)(?:\s*\(\s*(\d+)(?:,\s*(\d+))?\s*\))?' # For types like VARCHAR(n) inferred from blob
}

def extract_type_params(data_type: str, db_type: str) -> Tuple[str, Optional[str], Optional[str]]:
    """
    Extract base type and parameters (length/max, precision/scale) from a data type string.
    Returns (base_type, length_or_max, scale)
    Handles case-insensitivity and potential extra keywords like UNSIGNED.
    """
    if not data_type: return ('', None, None) # Handle empty input

    regex_pattern = DATATYPE_PARAMS_REGEX.get(db_type)
    if not regex_pattern:
        logger.warning(f"No regex pattern defined for db_type '{db_type}'. Treating type as atomic.")
        # Basic fallback: treat whole string as base type if no pattern
        return data_type.strip().lower(), None, None

    # Match against the beginning of the lowercased, stripped string
    match = re.match(regex_pattern, data_type.strip(), re.IGNORECASE)
    if not match:
        # If no match with parameters, maybe it's just the base type
        # Check if the whole string is alphabetic
        base_type_only = data_type.strip()
        if re.match(r'^[a-zA-Z_]+$', base_type_only):
             return base_type_only.lower(), None, None
        else:
             # Cannot parse cleanly
             logger.warning(f"Could not parse data type '{data_type}' for db_type '{db_type}'. Using original.")
             return data_type.strip().lower(), None, None # Return lowercased original on failure

    # Extract groups - adjust index based on regex definition if needed
    groups = match.groups()
    base_type = groups[0].lower() if groups[0] else ''
    length_or_max = groups[1].lower() if len(groups) > 1 and groups[1] else None
    scale = groups[2] if len(groups) > 2 and groups[2] else None

    return base_type, length_or_max, scale


def get_target_datatype(source_type: str, source_db: str, target_db: str) -> str:
    """
    Map source data type to target data type based on source and target databases.
    Handles basic length/precision transfer where applicable.

    Args:
        source_type: Source data type (e.g., 'varchar(255)', 'BIGINT', 'DOUBLE')
        source_db: Source database type ('mysql', 'sqlserver', 'snowflake', 'azure_blob')
        target_db: Target database type ('mysql', 'sqlserver', 'snowflake')

    Returns:
        Target data type as a string (e.g., 'VARCHAR(255)', 'BIGINT')
    """
    # Determine the source type system ('mysql', 'sqlserver', 'generic' for blob)
    source_system = source_db
    if source_db == 'azure_blob':
        source_system = 'generic' # Use a generic type system for blob-inferred types

    # Extract components from source type string
    base_type, length_or_max, scale = extract_type_params(source_type, source_system)
    if not base_type:
        logger.error(f"Could not extract base type from source type '{source_type}' for system '{source_system}'. Returning default.")
        return 'VARCHAR' if target_db != 'snowflake' else 'STRING' # Safe default

    # Select the appropriate mapping dictionary
    type_map = None
    if source_system == 'mysql' and target_db == 'sqlserver':
        type_map = MYSQL_TO_SQLSERVER_TYPE_MAP
    elif source_system == 'mysql' and target_db == 'snowflake':
        type_map = MYSQL_TO_SNOWFLAKE_TYPE_MAP
    elif source_system == 'sqlserver' and target_db == 'snowflake':
        type_map = SQLSERVER_TO_SNOWFLAKE_TYPE_MAP
    elif source_system == 'generic' and target_db == 'snowflake':
        type_map = GENERIC_TO_SNOWFLAKE_TYPE_MAP
    elif source_system == 'generic' and target_db == 'sqlserver':
         type_map = GENERIC_TO_SQLSERVER_TYPE_MAP

    # --- Determine target_base_type ---
    target_base_type = base_type.upper() # Default to source base type (uppercase for consistency)
    if type_map:
        # Use upper() for matching keys in the map consistently
        mapped_type = type_map.get(base_type.upper()) # Check if key exists
        if mapped_type:
             target_base_type = mapped_type # Use mapped type
        # else: keep target_base_type as base_type.upper() (from default above)
    else:
        # No specific mapping dictionary found
        logger.warning(f"No specific type mapping dictionary found for {source_system} ({source_db}) to {target_db}. Using source base type '{target_base_type}'.")
        # target_base_type is already set to base_type.upper()

    # --- Handle length and precision/scale ---
    target_type_final = target_base_type # Start with the determined base type

    # Only append parameters if they were present in the source *and* make sense for the target type
    if length_or_max:
        # String/Binary types (VARCHAR, CHAR, BINARY, VARBINARY, STRING)
        if target_base_type in ['VARCHAR', 'STRING', 'CHAR', 'NVARCHAR', 'NCHAR', 'BINARY', 'VARBINARY']:
            # SQL Server MAX length handling
            if target_db == 'sqlserver':
                 is_max = length_or_max.lower() == 'max'
                 is_digit = length_or_max.isdigit()
                 is_large = is_digit and int(length_or_max) > 8000 # Common limit for non-MAX
                 # Use MAX if source was MAX or exceeds typical limit
                 if is_max or is_large:
                     # Only apply MAX to types that support it
                     if target_base_type in ['VARCHAR', 'NVARCHAR', 'VARBINARY']:
                         target_type_final = f"{target_base_type}(MAX)"
                     # else: keep base type without params if MAX not supported (e.g., for CHAR)
                 elif is_digit: # Specify length if it's a digit and not large
                     target_type_final = f"{target_base_type}({length_or_max})"
                 # else: keep base type if length wasn't digit or max (e.g., error in parsing?)

            # Snowflake: Usually let it default, but specify if length provided & valid
            elif target_db == 'snowflake':
                 if length_or_max.isdigit() and int(length_or_max) <= 16777216:
                     target_type_final = f"{target_base_type}({length_or_max})" # Specify if valid length given
                 # else: Use base type (VARCHAR/STRING) for 'max' or invalid/too large length

            # MySQL or other generic DBs
            else:
                 if length_or_max.lower() == 'max': # Handle MAX keyword if applicable
                      target_type_final = f"{target_base_type}(MAX)" # Check if target DB uses MAX keyword
                 elif length_or_max.isdigit():
                      target_type_final = f"{target_base_type}({length_or_max})"
                 # else: keep base type

        # Decimal/Numeric types (DECIMAL, NUMERIC, NUMBER)
        elif target_base_type in ['DECIMAL', 'NUMBER', 'NUMERIC']:
             # Ensure length and scale are digits if provided
             valid_length = length_or_max and length_or_max.isdigit()
             valid_scale = scale and scale.isdigit()

             if valid_length and valid_scale:
                 target_type_final = f"{target_base_type}({length_or_max},{scale})"
             elif valid_length: # Only length/precision provided
                 target_type_final = f"{target_base_type}({length_or_max})" # Assume scale 0 or DB default
             # else: use base type without parameters if length/scale are invalid/missing

        # Other types with potential precision (e.g., FLOAT, TIME, TIMESTAMP) - less common to transfer directly
        # Add specific handling here if needed, e.g., for DATETIME2(p) or TIME(p) in SQL Server
        elif target_base_type in ['FLOAT'] and target_db == 'sqlserver':
             # MySQL FLOAT(p) maps differently than SQL Server FLOAT(p)
             # Usually better to map MySQL FLOAT/DOUBLE to SQL Server FLOAT without precision
             pass # Keep base type 'FLOAT'
        elif target_base_type in ['TIME', 'DATETIME2', 'DATETIMEOFFSET'] and target_db == 'sqlserver':
             # If source had precision (scale for TIME, length for DATETIME2/OFFSET)
             # SQL Server uses 'length' from source TIME(p) as precision 'p'
             precision_digit = scale if base_type == 'time' else length_or_max
             if precision_digit and precision_digit.isdigit() and 0 <= int(precision_digit) <= 7:
                  target_type_final = f"{target_base_type}({precision_digit})"
             # else: use base type without precision

    logger.debug(f"Mapped {source_db}:{source_type} -> {target_db}:{target_type_final}")
    return target_type_final


def create_target_table_ddl(
    table_name: str,
    columns: List[Dict[str, Any]],
    target_db: str,
    source_db: str # Needed for type mapping
) -> str:
    """
    Create DDL statement for the target table based on source table schema.

    Args:
        table_name: Name of the target table (should be appropriately quoted if needed)
        columns: List of column definition dictionaries from source. Expected keys:
                 'name': column name (str)
                 'type': source data type string (str)
                 'nullable': boolean indicating if column allows nulls
                 'key': 'PRI' if primary key (str, optional)
                 'default': default value string (str, optional)
                 'extra': 'auto_increment' or 'identity' info (str, optional)
        target_db: Target database type ('mysql', 'sqlserver', 'snowflake')
        source_db: Source database type ('mysql', 'sqlserver', 'azure_blob')

    Returns:
        DDL statement string for creating the target table.
    """
    # Basic quoting for table name - adjust if more complex quoting needed
    quoted_table_name = f'"{table_name}"' if target_db == 'snowflake' else f'[{table_name}]'

    if target_db == 'snowflake':
        # Snowflake's IF NOT EXISTS is standard
        create_stmt = f"CREATE TABLE IF NOT EXISTS {quoted_table_name} (\n"
    else:
        # Standard SQL Server / MySQL CREATE TABLE
        create_stmt = f"CREATE TABLE {quoted_table_name} (\n"

    column_defs = []
    pk_column_names = [] # Store PK column names separately

    for col in columns:
        col_name = col.get('name')
        source_type = col.get('type')

        if not col_name or not source_type:
             logger.warning(f"Skipping column due to missing name or type: {col}")
             continue

        # Quote column names
        quoted_col_name = f'"{col_name}"' if target_db == 'snowflake' else f'[{col_name}]'

        # Map data type
        target_type = get_target_datatype(source_type, source_db, target_db)

        # Determine nullability
        # Source schema uses boolean True for nullable
        is_nullable = col.get('nullable', True) # Default to nullable if missing
        null_spec = "NULL" if is_nullable else "NOT NULL"

        # Handle auto-increment/identity columns
        col_def_parts = [quoted_col_name, target_type]
        extra_info = col.get('extra', '').lower()
        is_identity = 'auto_increment' in extra_info or 'identity' in extra_info

        if is_identity:
            if target_db == 'sqlserver':
                col_def_parts.append("IDENTITY(1,1)")
            elif target_db == 'snowflake':
                 # AUTOINCREMENT or IDENTITY can be used, AUTOINCREMENT is simpler
                col_def_parts.append("AUTOINCREMENT")
            elif target_db == 'mysql':
                col_def_parts.append("AUTO_INCREMENT")
            # If identity, usually implies NOT NULL, but check source schema just in case
            col_def_parts.append(null_spec)
        else:
            # Append null spec for non-identity columns
            col_def_parts.append(null_spec)

            # Handle default values (needs careful quoting and type checking)
            # default_value = col.get('default')
            # if default_value is not None:
            #     # Quoting defaults is tricky: needs '' for strings, nothing for numbers/keywords
            #     # This is a simplified placeholder - robust default handling is complex
            #     col_def_parts.append(f"DEFAULT {default_value}") # WARNING: May need proper quoting/conversion
            pass # Default value handling omitted for simplicity/safety


        column_defs.append(" ".join(col_def_parts))

        # Collect primary key columns
        if col.get('key', '').upper() == 'PRI':
            pk_column_names.append(quoted_col_name)


    # Add primary key constraint if defined
    if pk_column_names:
        pk_def = f"    PRIMARY KEY ({', '.join(pk_column_names)})"
        column_defs.append(pk_def)

    # Join column definitions
    create_stmt += ",\n".join(f"    {d}" for d in column_defs)
    create_stmt += "\n)"

    # Add table options based on database type
    if target_db == 'snowflake':
        create_stmt += ";"
    elif target_db == 'mysql':
        # Example: Add standard MySQL options
        create_stmt += " ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;"
    elif target_db == 'sqlserver':
        create_stmt += ";" # Standard terminator

    logger.info(f"Generated DDL for {target_db}:\n{create_stmt}")
    return create_stmt


def transform_data(
    df: pd.DataFrame,
    transformations: Dict[str, Any] = None
) -> pd.DataFrame:
    """
    Apply declarative transformations to the DataFrame.
    (Not currently used by migrate.py's interactive flow)

    Args:
        df: Input DataFrame
        transformations: Dictionary of transformation settings, e.g.:
            {
                'column_rename': {'old_name': 'new_name', ...},
                'column_drop': ['col1', 'col2', ...],
                'expressions': {'col_name': 'df[col] * 2', ...}, # Modify existing
                'calculated_columns': {'new_col': 'df["colA"] + df["colB"]', ...}, # Add new
                'filters': 'df["value"] > 100 and df["category"] == "A"' # Pandas query string
            }

    Returns:
        Transformed DataFrame
    """
    if not transformations:
        return df

    logger.info(f"Applying declarative transformations: {list(transformations.keys())}")
    # Create a copy to avoid modifying the original
    result_df = df.copy()

    # Apply column rename first
    if 'column_rename' in transformations and isinstance(transformations['column_rename'], dict):
        logger.debug(f"Renaming columns: {transformations['column_rename']}")
        result_df.rename(columns=transformations['column_rename'], inplace=True, errors='ignore')

    # Apply expressions to modify existing columns
    # WARNING: Using eval can be a security risk if expressions come from untrusted sources.
    # Here it assumes expressions are defined safely within the tool's configuration.
    if 'expressions' in transformations and isinstance(transformations['expressions'], dict):
        for col, expr_str in transformations['expressions'].items():
            if col in result_df.columns:
                try:
                    logger.debug(f"Applying expression to column '{col}': {expr_str}")
                    # Use pandas eval method for potentially safer/faster evaluation within df context
                    # result_df[col] = result_df.eval(expr_str)
                    # Or use Python's eval directly (more flexible but riskier)
                    result_df[col] = eval(expr_str, {'pd': pd, 'np': np}, {'df': result_df})
                except Exception as e:
                    logger.error(f"Error applying expression to column {col}: {expr_str} -> {e}", exc_info=True)
            else:
                logger.warning(f"Column '{col}' not found for applying expression.")

    # Add calculated columns
    if 'calculated_columns' in transformations and isinstance(transformations['calculated_columns'], dict):
        for new_col, expr_str in transformations['calculated_columns'].items():
            try:
                 logger.debug(f"Creating calculated column '{new_col}': {expr_str}")
                 # result_df[new_col] = result_df.eval(expr_str)
                 result_df[new_col] = eval(expr_str, {'pd': pd, 'np': np}, {'df': result_df})
            except Exception as e:
                logger.error(f"Error creating calculated column {new_col}: {expr_str} -> {e}", exc_info=True)

    # Apply filters
    if 'filters' in transformations and isinstance(transformations['filters'], str):
        try:
            filter_query = transformations['filters']
            logger.debug(f"Applying filter query: {filter_query}")
            # Use pandas query method
            original_rows = len(result_df)
            result_df = result_df.query(filter_query)
            logger.info(f"Filter applied. Rows reduced from {original_rows} to {len(result_df)}.")
        except Exception as e:
            logger.error(f"Error applying filter query '{transformations['filters']}': {e}", exc_info=True)

    # Drop columns last
    if 'column_drop' in transformations and isinstance(transformations['column_drop'], list):
        cols_to_drop = [col for col in transformations['column_drop'] if col in result_df.columns]
        if cols_to_drop:
            logger.debug(f"Dropping columns: {cols_to_drop}")
            result_df.drop(columns=cols_to_drop, inplace=True)

    return result_df


def sanitize_column_values(df: pd.DataFrame, target_db: str) -> pd.DataFrame:
    """
    Sanitize column values based on common data type constraints.
    (Example: string truncation). Not heavily used currently.

    Args:
        df: Input DataFrame
        target_db: Target database type

    Returns:
        DataFrame with sanitized values
    """
    logger.debug(f"Sanitizing column values for target DB: {target_db}")
    result_df = df.copy()

    # Process each column
    for col in result_df.columns:
        # Handle string truncation for databases with known varchar limits (example: SQL Server non-MAX)
        # This is a basic example; more specific type checks might be needed.
        if pd.api.types.is_string_dtype(result_df[col].dtype) or result_df[col].dtype == 'object':
            if target_db == 'sqlserver':
                # Example: Truncate strings potentially exceeding 8000 bytes for standard VARCHAR
                # This doesn't account for VARCHAR(MAX) or NVARCHAR differences.
                # More robust sanitization would require checking the *target column type*.
                limit = 8000
                try:
                    # Check if any string exceeds limit
                    if result_df[col].astype(str).str.len().max() > limit:
                         logger.warning(f"Column '{col}' contains strings longer than {limit} chars. Truncating for SQL Server (basic check).")
                         # Apply truncation only to strings, leave others (like None) alone
                         result_df[col] = result_df[col].apply(
                            lambda x: x[:limit] if isinstance(x, str) and len(x) > limit else x
                         )
                except Exception as e:
                     logger.warning(f"Could not check/truncate length for column {col}: {e}")

            # Add rules for other DBs if necessary (e.g., older MySQL limits)

    # Note: NaN/NaT replacement is handled in prepare_dataframe_for_db

    return result_df


def detect_column_types(df: pd.DataFrame, target_db: str) -> Dict[str, str]:
    """
    Detect appropriate column types for the target database based on DataFrame content.
    Provides a *guess* based on data, useful for initial table creation from DataFrame.

    Args:
        df: Input DataFrame
        target_db: Target database type ('mysql', 'sqlserver', 'snowflake')

    Returns:
        Dictionary mapping column names to suggested target database data types.
    """
    logger.info(f"Detecting target types for {target_db} based on DataFrame content.")
    type_map = {}
    source_db_for_mapping = 'generic' # Use generic -> target mapping

    for col in df.columns:
        dtype = df[col].dtype
        inferred_generic_type = 'VARCHAR' # Default

        # --- Infer Generic Type from Pandas Dtype ---
        if pd.api.types.is_integer_dtype(dtype):
            min_val = df[col].min()
            max_val = df[col].max()
            if pd.isna(min_val) and pd.isna(max_val): # All nulls
                 inferred_generic_type = 'INT' # Default integer
            elif min_val >= -128 and max_val <= 127: inferred_generic_type = 'TINYINT'
            elif min_val >= -32768 and max_val <= 32767: inferred_generic_type = 'SMALLINT'
            elif min_val >= -2147483648 and max_val <= 2147483647: inferred_generic_type = 'INT'
            else: inferred_generic_type = 'BIGINT'
        elif pd.api.types.is_float_dtype(dtype):
            # Check if it looks like it only contains integers
            try:
                 if (df[col].dropna() == df[col].dropna().astype(np.int64)).all():
                      # It's float but maybe should be int? Use BIGINT for safety.
                      inferred_generic_type = 'BIGINT'
                 else:
                      inferred_generic_type = 'DOUBLE' # Use DOUBLE for float types
            except (TypeError, ValueError): # Handle non-numeric floats if any
                  inferred_generic_type = 'DOUBLE'
        elif pd.api.types.is_bool_dtype(dtype):
            inferred_generic_type = 'BOOLEAN'
        elif pd.api.types.is_datetime64_any_dtype(dtype): # Catches datetime64[ns] and datetime64[ns, TZ]
            inferred_generic_type = 'TIMESTAMP'
        elif pd.api.types.is_timedelta64_dtype(dtype):
             inferred_generic_type = 'VARCHAR' # Store timedelta as string representation
        elif pd.api.types.is_string_dtype(dtype) or dtype == 'object':
            # Try to be more specific for strings/objects
            non_null_series = df[col].dropna()
            if non_null_series.empty:
                 inferred_generic_type = 'VARCHAR(255)' # Default for empty object column
            else:
                 # Check if it looks like dates
                 try:
                     pd.to_datetime(non_null_series, errors='raise')
                     # If conversion works without error, assume date/timestamp
                     # Check if time part is always zero for DATE type
                     times = pd.to_datetime(non_null_series).dt.time
                     if (times == datetime.time(0, 0)).all():
                          inferred_generic_type = 'DATE'
                     else:
                          inferred_generic_type = 'TIMESTAMP'
                 except (ValueError, TypeError):
                     # Not dates, treat as string
                     try:
                         max_len = non_null_series.astype(str).str.len().max()
                         max_len = int(max_len * 1.2) if pd.notna(max_len) else 0
                         # Cap length reasonably
                         max_len_cap = 8000 if target_db == 'sqlserver' else 16777216
                         max_len = min(max_len, max_len_cap) if max_len > 0 else 255
                         inferred_generic_type = f'VARCHAR({max_len})'
                         # Check if SQL Server target might need MAX
                         if target_db == 'sqlserver' and max_len > 8000:
                              inferred_generic_type = 'VARCHAR(MAX)'
                     except Exception:
                          inferred_generic_type = 'VARCHAR(MAX)' if target_db == 'sqlserver' else 'STRING'

        # --- Map Generic Type to Target DB Type ---
        type_map[col] = get_target_datatype(inferred_generic_type, source_db_for_mapping, target_db)

    logger.debug(f"Detected target types: {type_map}")
    return type_map


def create_table_from_dataframe(
    df: pd.DataFrame,
    table_name: str,
    target_db: str
) -> str:
    """
    Generate a CREATE TABLE statement based on a DataFrame's content.
    Uses detect_column_types to guess appropriate types.

    Args:
        df: Input DataFrame
        table_name: Target table name
        target_db: Target database type ('mysql', 'sqlserver', 'snowflake')

    Returns:
        CREATE TABLE statement string.
    """
    if df.empty:
         logger.warning("Input DataFrame is empty. Cannot generate DDL from content.")
         # Optionally return a minimal DDL or raise error
         return f"-- Cannot generate DDL: Input DataFrame for table '{table_name}' is empty."

    column_types = detect_column_types(df, target_db)

    # Use the generic DDL builder with the detected types
    # We need to format the detected types into the schema dictionary format
    schema_def = []
    for col_name, target_type in column_types.items():
         schema_def.append({
             'name': col_name,
             'type': target_type, # Use the *already mapped* target type directly
             'nullable': True # Assume nullable when creating from DF
         })

    # Call create_target_table_ddl, but source_db isn't really relevant here
    # since types are already mapped. Pass 'generic' or target_db itself.
    # However, create_target_table_ddl expects a *source* type for re-mapping.
    # Let's modify the approach: directly build the DDL here.

    logger.info(f"Generating DDL for table '{table_name}' based on DataFrame content for {target_db}.")

    # Basic quoting for table name
    quoted_table_name = f'"{table_name}"' if target_db == 'snowflake' else f'[{table_name}]'

    if target_db == 'snowflake':
        create_stmt = f"CREATE TABLE IF NOT EXISTS {quoted_table_name} (\n"
    else:
        create_stmt = f"CREATE TABLE {quoted_table_name} (\n"

    column_defs = []
    for col_name, target_type in column_types.items():
        quoted_col_name = f'"{col_name}"' if target_db == 'snowflake' else f'[{col_name}]'
        # When creating from DF, assume all columns are nullable for simplicity
        null_spec = "NULL"
        column_defs.append(f"    {quoted_col_name} {target_type} {null_spec}")

    # Note: No PK, default, or identity info inferred from DataFrame directly
    create_stmt += ",\n".join(column_defs)
    create_stmt += "\n)"

    # Add table options based on database type
    if target_db == 'snowflake':
        create_stmt += ";"
    elif target_db == 'mysql':
        create_stmt += " ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;"
    elif target_db == 'sqlserver':
        create_stmt += ";"

    logger.debug(f"Generated DDL from DataFrame:\n{create_stmt}")
    return create_stmt


def prepare_dataframe_for_db(df: pd.DataFrame, target_db: str) -> pd.DataFrame:
    """
    Prepare DataFrame for database insertion by adjusting data types and formats.
    Handles NaT/NaN/NA conversion to None suitable for DBs.

    Args:
        df: Input DataFrame
        target_db: Target database type ('mysql', 'sqlserver', 'snowflake')

    Returns:
        Prepared DataFrame ready for insertion (e.g., via executemany).
    """
    if df.empty:
        return df # Nothing to prepare

    logger.debug(f"Preparing DataFrame for target DB: {target_db}")
    result_df = df.copy()

    # --- Data Type Conversions for Connectors ---

    # Handle dates and datetimes first
    datetime_cols = result_df.select_dtypes(include=['datetime64[ns]', 'datetime64', 'datetimetz']).columns
    if not datetime_cols.empty:
        logger.debug(f"Converting datetime columns: {list(datetime_cols)}")
        for col in datetime_cols:
            # Convert NaT to None before formatting
            result_df[col] = result_df[col].replace({pd.NaT: None})
            if target_db == 'sqlserver':
                # pyodbc executemany often prefers ISO strings for DATETIME2
                result_df[col] = result_df[col].apply(
                    lambda x: x.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3] if pd.notnull(x) else None
                )
            elif target_db == 'snowflake':
                # snowflake-connector pyformat binding needs strings for timestamps
                result_df[col] = result_df[col].apply(
                    lambda x: x.strftime('%Y-%m-%d %H:%M:%S.%f') if pd.notnull(x) else None
                )
            # MySQL connector handles Python datetimes well usually

    # Handle boolean data
    # Use pandas nullable boolean type if present
    bool_cols = result_df.select_dtypes(include=['bool', 'boolean']).columns
    if not bool_cols.empty:
        logger.debug(f"Converting boolean columns: {list(bool_cols)}")
        for col in bool_cols:
             # Convert pandas <NA> to None before potential type cast
             result_df[col] = result_df[col].replace({pd.NA: None})
             if target_db == 'sqlserver':
                 # SQL Server uses bit (0/1). Allow None for NULLs.
                 result_df[col] = result_df[col].apply(lambda x: int(x) if pd.notnull(x) else None)
             # Snowflake/MySQL connectors handle Python True/False/None for BOOLEAN

    # Handle potential JSON-like objects for specific targets
    # Example for SQL Server NVARCHAR(MAX) - convert Python dicts/lists to JSON strings
    if target_db == 'sqlserver':
        object_cols = result_df.select_dtypes(include=['object']).columns
        if not object_cols.empty:
             logger.debug(f"Checking object columns for JSON conversion (SQL Server): {list(object_cols)}")
             for col in object_cols:
                 # Check only if column actually contains non-primitive types (dict/list)
                 # This check can be slow on large data, use cautiously or profile
                 try:
                     contains_dicts_lists = result_df[col].apply(lambda x: isinstance(x, (dict, list))).any()
                     if contains_dicts_lists:
                          logger.info(f"Converting column '{col}' with dicts/lists to JSON strings for SQL Server.")
                          result_df[col] = result_df[col].apply(
                              lambda x: json.dumps(x) if isinstance(x, (dict, list)) else x
                          )
                 except Exception as json_check_err:
                     logger.warning(f"Could not perform dict/list check on column '{col}': {json_check_err}")
                 # Basic validation of strings that look like JSON omitted for performance, assume valid if needed

    # --- Final Cleanup for DB Insertion ---
    logger.debug("Replacing final NA/NaN values with None.")
    # Replace any remaining Pandas NA types or Numpy NaN with None across the whole DataFrame
    # This should catch anything missed above and ensure Nones are passed to DB for NULLs
    # Using fillna(None) might be slightly safer if dtypes need preserving, but replace is broader.
    # Convert to object first to ensure `None` is used, not `nan` for numeric types.
    try:
         result_df = result_df.astype(object).replace({pd.NA: None, np.nan: None})
    except Exception as final_na_err:
        logger.error(f"Error during final NA replacement: {final_na_err}. Returning intermediate df.")
        # Fallback: return the df before this step if it causes issues

    # Ensure object columns that only contain None still have dtype object
    # (Not strictly necessary after above step, but ensures consistency)
    # for col in result_df.select_dtypes(include=['object']).columns:
    #     if result_df[col].isnull().all():
    #         result_df[col] = result_df[col].astype(object)

    logger.debug("DataFrame preparation complete.")
    return result_df


def preview_dataframe(df: pd.DataFrame, max_rows: int = 5) -> pd.DataFrame:
    """
    Return a preview (head) of the DataFrame.

    Args:
        df: Input DataFrame
        max_rows: Maximum number of rows to preview

    Returns:
        Preview DataFrame (first max_rows)
    """
    if df is None: return pd.DataFrame() # Handle None input
    if df.empty:
        return df

    return df.head(max_rows)


def analyze_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze DataFrame and return basic statistics.

    Args:
        df: Input DataFrame

    Returns:
        Dictionary with DataFrame statistics.
    """
    if df is None or df.empty:
        return {"status": "empty", "rows": 0, "columns": 0}

    logger.info(f"Analyzing DataFrame: {len(df)} rows, {len(df.columns)} columns")
    try:
        # Basic statistics
        stats = {
            "rows": len(df),
            "columns": len(df.columns),
            "column_types": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "memory_usage_mb": round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2),
            "null_counts": {col: int(count) for col, count in df.isna().sum().items()},
        }
        stats["null_percentage"] = {col: round((count / stats["rows"]) * 100, 2) if stats["rows"] > 0 else 0
                                    for col, count in stats["null_counts"].items()}


        # Add column-specific statistics where appropriate
        numeric_stats = {}
        numeric_cols = df.select_dtypes(include=np.number).columns
        if not numeric_cols.empty:
             logger.debug(f"Calculating numeric stats for: {list(numeric_cols)}")
             for col in numeric_cols:
                 # Drop NA before calculating stats to avoid errors/warnings
                 col_data = df[col].dropna()
                 if not col_data.empty:
                     numeric_stats[col] = {
                         "min": float(col_data.min()),
                         "max": float(col_data.max()),
                         "mean": float(col_data.mean()),
                         "median": float(col_data.median()),
                         "std_dev": float(col_data.std())
                     }
                 else:
                     numeric_stats[col] = {"min": None, "max": None, "mean": None, "median": None, "std_dev": None}
             if numeric_stats: stats["numeric_stats"] = numeric_stats


        # String/Object columns statistics
        string_stats = {}
        # Include category as well, treat like object for these stats
        object_cols = df.select_dtypes(include=['object', 'string', 'category']).columns
        if not object_cols.empty:
             logger.debug(f"Calculating string/object stats for: {list(object_cols)}")
             for col in object_cols:
                 col_data = df[col].dropna()
                 if not col_data.empty:
                      unique_count = col_data.nunique()
                      max_len = col_data.astype(str).str.len().max() if unique_count > 0 else 0
                      string_stats[col] = {
                          "unique_values": int(unique_count),
                          "max_length": int(max_len) if pd.notna(max_len) else 0
                      }
                      # Maybe add top N most frequent values if useful?
                      # if unique_count < 100: # Only if cardinality is low
                      #      string_stats[col]["top_values"] = col_data.value_counts().head(5).to_dict()
                 else:
                      string_stats[col] = {"unique_values": 0, "max_length": 0}
             if string_stats: stats["string_stats"] = string_stats

        # Datetime columns statistics
        datetime_stats = {}
        datetime_cols = df.select_dtypes(include=['datetime64[ns]', 'datetimetz']).columns
        if not datetime_cols.empty:
             logger.debug(f"Calculating datetime stats for: {list(datetime_cols)}")
             for col in datetime_cols:
                  col_data = df[col].dropna()
                  if not col_data.empty:
                      datetime_stats[col] = {
                          "min_date": str(col_data.min()),
                          "max_date": str(col_data.max())
                      }
                  else:
                      datetime_stats[col] = {"min_date": None, "max_date": None}
             if datetime_stats: stats["datetime_stats"] = datetime_stats

        return stats

    except Exception as e:
        logger.error(f"Error during DataFrame analysis: {e}", exc_info=True)
        return {"status": "error", "message": str(e)}


def optimize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optimize DataFrame memory usage by downcasting numeric types and
    converting low-cardinality strings to categorical type.

    Args:
        df: Input DataFrame

    Returns:
        Optimized DataFrame (potentially using less memory).
    """
    if df.empty: return df

    logger.info("Optimizing DataFrame memory usage...")
    original_mem = round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2)
    result_df = df.copy()

    # Downcast numeric columns
    for col in result_df.select_dtypes(include=['integer']).columns:
        try:
            result_df[col] = pd.to_numeric(result_df[col], downcast='integer')
        except Exception as e: # Handle potential errors if data doesn't fit narrower type
             logger.warning(f"Could not downcast integer column '{col}': {e}")

    for col in result_df.select_dtypes(include=['float']).columns:
        try:
            result_df[col] = pd.to_numeric(result_df[col], downcast='float')
        except Exception as e:
             logger.warning(f"Could not downcast float column '{col}': {e}")

    # Convert string columns with few unique values to categorical
    # Use a threshold: e.g., if unique count is less than 50% of total rows
    # Adjust threshold based on typical data and performance needs
    unique_threshold_ratio = 0.5
    string_cols = result_df.select_dtypes(include=['object', 'string']).columns
    total_rows = len(result_df)

    if total_rows > 0: # Avoid division by zero
        for col in string_cols:
            try:
                num_unique = result_df[col].nunique()
                if num_unique / total_rows < unique_threshold_ratio:
                    logger.debug(f"Converting column '{col}' to categorical ({num_unique} unique values).")
                    result_df[col] = result_df[col].astype('category')
            except Exception as e: # Handle potential errors like mixed types within object column
                logger.warning(f"Could not analyze/convert column '{col}' to categorical: {e}")

    optimized_mem = round(result_df.memory_usage(deep=True).sum() / (1024 * 1024), 2)
    logger.info(f"DataFrame memory usage reduced from {original_mem} MB to {optimized_mem} MB.")

    return result_df


def split_dataframe(df: pd.DataFrame, batch_size: int = 10000) -> List[pd.DataFrame]:
    """
    Split a large DataFrame into smaller batches (list of DataFrames).

    Args:
        df: Input DataFrame
        batch_size: Maximum number of rows in each batch

    Returns:
        List of DataFrame batches. Returns list containing the original DataFrame
        if its size is less than or equal to batch_size.
    """
    if df is None: return []
    total_rows = len(df)
    if total_rows == 0: return [] # Return empty list for empty df
    if total_rows <= batch_size:
        return [df.copy()] # Return as list with one item (copy)

    logger.info(f"Splitting DataFrame ({total_rows} rows) into batches of size {batch_size}.")
    batches = []
    for i in range(0, total_rows, batch_size):
        batches.append(df.iloc[i:min(i + batch_size, total_rows)].copy()) # Ensure last batch doesn't exceed total_rows

    logger.info(f"Created {len(batches)} batches.")
    return batches

# --- Helper for migrate.py's simpler transformation UI (if needed) ---
# This function is kept for potential future use or reference, but migrate.py
# now uses df.rename directly for the 'column_mapping' type.
def apply_basic_transformations(
    df: pd.DataFrame,
    transformation_type: str,
    transformations: Any # Could be dict for rename, list of dicts for custom
) -> pd.DataFrame:
    """
    Applies basic transformations based on the simple UI options in migrate.py.
    DEPRECATED in favor of direct df.rename in migrate.py for clarity. Kept for reference.
    """
    if transformation_type == "none" or not transformations:
        return df

    logger.info(f"Applying basic transformations of type: {transformation_type}")
    result_df = df.copy()

    if transformation_type == "column_mapping" and isinstance(transformations, dict):
        logger.debug(f"Applying column renames: {transformations}")
        result_df.rename(columns=transformations, inplace=True)

    elif transformation_type == "custom" and isinstance(transformations, list):
        logger.debug(f"Applying custom functions: {transformations}")
        for transform in transformations:
            col = transform.get('column')
            func = transform.get('function')
            if not col or not func or col not in result_df.columns:
                logger.warning(f"Skipping invalid custom transform: {transform}")
                continue

            try:
                if func == 'uppercase':
                    result_df[col] = result_df[col].astype(str).str.upper()
                elif func == 'lowercase':
                    result_df[col] = result_df[col].astype(str).str.lower()
                elif func == 'trim':
                    result_df[col] = result_df[col].astype(str).str.strip()
                elif func == 'replace':
                    old = transform.get('old_value')
                    new = transform.get('new_value')
                    if old is not None and new is not None:
                        # Be careful with types here - replace works best on strings
                        result_df[col] = result_df[col].astype(str).str.replace(str(old), str(new), regex=False)
                    else:
                        logger.warning(f"Missing old/new value for replace transform on {col}")
                elif func == 'date_format':
                    fmt = transform.get('format')
                    if fmt:
                        # Convert to datetime first (coerce errors), then format
                        result_df[col] = pd.to_datetime(result_df[col], errors='coerce').dt.strftime(fmt)
                    else:
                        logger.warning(f"Missing format for date_format transform on {col}")
                else:
                     logger.warning(f"Unsupported custom function '{func}' for column '{col}'")

            except Exception as e:
                logger.error(f"Error applying custom transform {transform} to column {col}: {e}", exc_info=True)

    return result_df