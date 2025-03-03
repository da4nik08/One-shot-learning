import pandas as pd


def get_captured(dataset_path, target_name):
    df = pd.read_csv(siamese_config['dataset']['path_img_metadata_ru'])
    df = df[~df[target_name].str.contains('unknown', case=False, na=False)]         # delete unknown equipment
    df_filtered = df[df['file'].str.contains(r'\bcapt\b', case=False, na=False)]    # Get only captured equipment, because
                                                        # others have many instances with poor quality and are very damaged
    target_name = ['dataset']['target_name']
    capt_df = df_filtered[df_filtered.groupby(target_name)[target_name].transform('count') > 19]   # Filtet category that 
                                                                                        # contains less than 19 examples
    return capt_df