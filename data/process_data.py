import sys
import pandas as pd
from sqlalchemy.engine import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Loads and merges the two datasets
    
    Param messages_filepath: string path for messages dataset.
    Param categories_filepath: string path for categories dataset.
    
    Returns the two dataframes merged.
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.join(categories, lsuffix='_messages', rsuffix='_categories')
    return df





def clean_data(df):    
    """    
    Cleans the dataframe df by dropping duplicated columns and rows and changing formats.

    Param df: Dataframe object of the two merged datasets.

    Returns the dataframe cleaned.
    """
    df.drop(columns=['id_categories'], inplace=True)
    categories = df['categories'].str.split(pat=';', expand=True)
    row = categories.iloc[0,].values
    category_colnames = [val[:-2] for val in row]
    categories.columns = category_colnames

    for column in categories:

        categories[column] = categories[column].astype(str).str[-1]
        
        categories[column] = categories[column].astype('int64')

    df.drop(columns=['categories'], inplace=True)

    df = pd.concat([df, categories], axis=1)

    df.drop_duplicates(inplace=True)
    
    return df




def save_data(df, database_filename):
    """    
    Cleans the dataframe df by dropping duplicated columns and rows and changing formats.

    Param df: Dataframe object of the two merged datasets.
    Param database_filename: name of the database.
    """
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('Cleaned_disaster_data', engine, index=False)  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()