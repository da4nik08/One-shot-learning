import pandas as pd


def train_test_split(capt_df, target_name):
    class_counts = capt_df[target_name].value_counts()
    val_data = pd.DataFrame()
    train_data = capt_df.copy()

    for cls, count in class_counts.items():
        if count <= 5:
            continue
        elif count <= 10:
            n_samples = 2
        elif count <= 20:
            n_samples = 3
        elif count <= 100:
            n_samples = round(count * 0.1)
        else:
            n_samples = round(count * 0.1)
            
        # Select n_samples from the class for validation
        val_samples = train_data[train_data[target_name] == cls].sample(n_samples, random_state=42)
        val_data = pd.concat([val_data, val_samples])
        train_data = train_data.drop(val_samples.index)    # Remove selected instances from train set

        return train_data, val_data