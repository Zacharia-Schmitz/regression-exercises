import os
import pandas as pd
import env as e


def get_zillow():
    """
    This function reads the Zillow data from a cached CSV file if it exists,
    or from a SQL database if it doesn't exist. If it doesn't exist, it will
    create the csv for future usage It then renames the columns to more
    descriptive names and drops all Null values.

    Args:
    - None

    Returns:
    - pandas dataframe
    """
    # Name of cached CSV file
    filename = "zillow.csv"
    # If cached data exists, read from CSV file
    if os.path.isfile(filename):
        df = pd.read_csv(filename)
    # Otherwise, read from SQL database
    else:
        df = pd.read_sql(
            """SELECT yearbuilt,
                                   bedroomcnt,
                                   bathroomcnt,
                                   calculatedfinishedsquarefeet,
                                   taxvaluedollarcnt,
                                   taxamount,
                                   fips
                            FROM properties_2017
                            WHERE propertylandusetypeid = 261""",  # 261 is single family residential id
            f"mysql+pymysql://{e.user}:{e.password}@{e.host}/zillow",
        )
        # Cache data locally
        df.to_csv(filename, index=False)
    # Rename columns
    df = df.rename(
        columns={
            "yearbuilt": "year",
            "bedroomcnt": "beds",
            "bathroomcnt": "baths",
            "calculatedfinishedsquarefeet": "sqfeet",
            "taxvaluedollarcnt": "tax_value",
            "taxamount": "prop_tax",
            "fips": "county",
        }
    )
    # Drop nulls, since it is less than 1%
    # Before Drop NA: 2,152,863
    # After Drop NA: 2,140,235
    # Total Dropped: 12,628 (0.006)
    df = df.dropna()
    return df
