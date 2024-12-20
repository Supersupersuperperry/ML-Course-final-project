import os
import pandas as pd
from sklearn.model_selection import train_test_split

def load_dataset(data_dir):
    # Load the three text files: imdb_labelled.txt, amazon_cells_labelled.txt, yelp_labelled.txt
    imdb_file = os.path.join(data_dir, "imdb_labelled.txt")
    amazon_file = os.path.join(data_dir, "amazon_cells_labelled.txt")
    yelp_file = os.path.join(data_dir, "yelp_labelled.txt")
    
    # Each file is separated by tabs and contains sentence and label columns
    imdb_df = pd.read_csv(imdb_file, sep='\t', header=None, names=["sentence", "label"])
    amazon_df = pd.read_csv(amazon_file, sep='\t', header=None, names=["sentence", "label"])
    yelp_df = pd.read_csv(yelp_file, sep='\t', header=None, names=["sentence", "label"])

    # Concatenate all three datasets into one
    data_df = pd.concat([imdb_df, amazon_df, yelp_df], ignore_index=True)
    
    return data_df

def preprocess_text(text):
    # Strip whitespace and convert to lowercase
    text = text.strip().lower()
    return text

if __name__ == "__main__":
    # Adjust data_dir based on your actual project structure
    data_dir = "../data/dataset"
    
    # Load the combined dataset
    df = load_dataset(data_dir)
    
    # Apply basic text preprocessing
    df['sentence'] = df['sentence'].apply(preprocess_text)
    
    # Shuffle the dataset
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    
    # Split data into train (70%) and temp (30%)
    train_df, temp_df = train_test_split(df, test_size=0.30, random_state=42, stratify=df['label'])
    
    # Split the temp set equally into val (15%) and test (15%)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['label'])
    
    # Print the size of each split for verification
    print("Train size:", len(train_df))
    print("Val size:", len(val_df))
    print("Test size:", len(test_df))
    
    # Create a processed data directory if it does not exist
    processed_dir = "../data/processed"
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)
    
    # Save the splits as CSV files
    train_df.to_csv(os.path.join(processed_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(processed_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(processed_dir, "test.csv"), index=False)
    
    print("Data preprocessing complete and files saved")
