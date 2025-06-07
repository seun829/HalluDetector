"""
Analyze hallucination patterns in GPT/Llama responses and visualize results.
Uses PyTorch for feature extraction and creative graphing.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch import nn

# Load and preprocess data
def load_data(filepaths):
    """
    Load and combine multiple labeled responses CSV files for analysis.
    Skips files that are empty, missing, or have no columns.
    """
    dfs = []
    for filepath in filepaths:
        if not os.path.exists(filepath):
            print(f"Warning: File {filepath} does not exist. Skipping.")
            continue
        if os.stat(filepath).st_size == 0:
            print(f"Warning: File {filepath} is empty. Skipping.")
            continue
        try:
            df = pd.read_csv(filepath)
            if df.empty:
                print(f"Warning: File {filepath} contains no data after reading. Skipping.")
                continue
            # Add source column (using basename)
            df['source'] = os.path.basename(filepath).replace('_labeled.csv', '')
            dfs.append(df)
        except pd.errors.EmptyDataError:
            print(f"Warning: File {filepath} has no columns to parse. Skipping.")
            continue
    if not dfs:
        raise ValueError("No valid files to process.")
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Add basic features
    combined_df['prompt_length'] = combined_df['prompt'].apply(lambda x: len(str(x)))
    combined_df['response_length'] = combined_df['model_response'].apply(lambda x: len(str(x)))
    return combined_df

# Analyze accuracy by question type
def analyze_by_question_type(df):
    """
    Group by question type and calculate accuracy.
    """
    if 'question_type' not in df.columns:
        print("No 'question_type' column found. Skipping this analysis.")
        return None

    grouped = df.groupby('question_type')['is_correct'].mean().reset_index()
    grouped.rename(columns={'is_correct': 'accuracy'}, inplace=True)
    return grouped

# Analyze accuracy by template
def analyze_by_template(df):
    """
    Group by template and calculate accuracy.
    """
    grouped = df.groupby('template')['is_correct'].mean().reset_index()
    grouped.rename(columns={'is_correct': 'accuracy'}, inplace=True)
    return grouped

# Analyze patterns based on specific words or phrases in prompts
def analyze_by_keywords(df, keywords):
    """
    Analyze accuracy based on the presence of specific keywords in the prompts.
    """
    results = []
    for keyword in keywords:
        df[f'contains_{keyword}'] = df['prompt'].str.contains(keyword, case=False, na=False)
        accuracy = df[df[f'contains_{keyword}']]['is_correct'].mean()
        results.append({'keyword': keyword, 'accuracy': accuracy})
    return pd.DataFrame(results)

# Use PyTorch for feature extraction
class FeatureExtractor(nn.Module):
    def __init__(self, input_dim):
        super(FeatureExtractor, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )

    def forward(self, x):
        return self.fc(x)

def extract_features(df):
    """
    Use PyTorch to extract features from the data.
    """
    # Example features: prompt length, response length, and keyword presence
    feature_columns = ['prompt_length', 'response_length'] + [col for col in df.columns if col.startswith('contains_')]
    features = df[feature_columns].fillna(0).values
    features = torch.tensor(features, dtype=torch.float32)

    # Initialize and apply the feature extractor
    input_dim = features.shape[1]
    model = FeatureExtractor(input_dim)
    with torch.no_grad():
        extracted_features = model(features).numpy()

    # Add extracted features back to the DataFrame
    for i in range(extracted_features.shape[1]):
        df[f'extracted_feature_{i}'] = extracted_features[:, i]
    return df

# Visualize results
def visualize_results(df, x_col, y_col, title, xlabel, ylabel):
    """
    Create a bar plot for the given data.
    """
    plt.figure(figsize=(10, 6))
    sns.barplot(x=x_col, y=y_col, data=df, palette="viridis")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

# Main analysis
def main():
    # Filepaths for the labeled response datasets
    filepaths = [
        "data/processed/responses_easy_labeled.csv",
        "data/processed/responses_auto-generated_labeled.csv",
        "data/processed/responses_hard_labeled.csv"
    ]

    # Load and preprocess data
    df = load_data(filepaths)

    # Analyze accuracy by question type
    question_type_analysis = analyze_by_question_type(df)
    if question_type_analysis is not None:
        print(question_type_analysis)
        visualize_results(
            question_type_analysis,
            x_col='question_type',
            y_col='accuracy',
            title='Accuracy by Question Type',
            xlabel='Question Type',
            ylabel='Accuracy'
        )

    # Analyze accuracy by template
    template_analysis = analyze_by_template(df)
    print(template_analysis)
    visualize_results(
        template_analysis,
        x_col='template',
        y_col='accuracy',
        title='Accuracy by Template',
        xlabel='Template',
        ylabel='Accuracy'
    )

    # Analyze accuracy by specific keywords
    keywords = ['capital', 'who', 'what', 'when', 'why', 'how']
    keyword_analysis = analyze_by_keywords(df, keywords)
    print(keyword_analysis)
    visualize_results(
        keyword_analysis,
        x_col='keyword',
        y_col='accuracy',
        title='Accuracy by Keywords in Prompts',
        xlabel='Keyword',
        ylabel='Accuracy'
    )

    # Extract features using PyTorch
    df = extract_features(df)

    # Visualize extracted features (example: correlation heatmap)
    extracted_feature_cols = [col for col in df.columns if col.startswith('extracted_feature')]
    feature_corr = df[extracted_feature_cols + ['is_correct']].corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(feature_corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap of Extracted Features and Accuracy")
    plt.show()

if __name__ == "__main__":
    main()