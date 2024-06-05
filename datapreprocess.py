import os
from glob import glob
import shutil
import wfdb
import numpy as np
import pandas as pd
from scipy.io import loadmat

def combine_dataset(data_dir, output_dir):
    """
    Combine dataset files from multiple subdirectories into a single directory.
    """
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Iterate through each subdirectory ('g1', 'g2', ..., 'gn')
    for sub_dir in glob(os.path.join(data_dir, 'g*')):
        # Find all '.hea' and '.mat' files in subdirectories
        hea_files = glob(os.path.join(sub_dir, '*.hea'))
        mat_files = glob(os.path.join(sub_dir, '*.mat'))

        # Copy each file to the output directory
        for file_path in hea_files + mat_files:
            # Get the base name of the file
            base_name = os.path.basename(file_path)
            # Construct the destination path
            dest_path = os.path.join(output_dir, base_name)
            # Copy the file
            shutil.copy(file_path, dest_path)
    
    print(f"Dataset files copied to {output_dir}")

def gen_reference_csv(data_dir, reference_csv):
    """
    Generate a reference CSV file with metadata from the ECG files.
    """
    if not os.path.exists(reference_csv):
        recordpaths = glob(os.path.join(data_dir, '*.hea'))
        results = []
        for recordpath in recordpaths:
            patient_id = os.path.basename(recordpath)[:-4]
            _, meta_data = wfdb.rdsamp(recordpath[:-4])
            sample_rate = meta_data['fs']
            signal_len = meta_data['sig_len']
            age = meta_data['comments'][0]
            sex = meta_data['comments'][1]
            dx = meta_data['comments'][2]
            age = age[5:] if age.startswith('Age: ') else np.NaN
            sex = sex[5:] if sex.startswith('Sex: ') else 'Unknown'
            dx = dx[4:] if dx.startswith('Dx: ') else ''
            results.append([patient_id, sample_rate, signal_len, age, sex, dx])
        df = pd.DataFrame(data=results, columns=['patient_id', 'sample_rate', 'signal_len', 'age', 'sex', 'dx'])
        df.sort_values('patient_id').to_csv(reference_csv, index=None)
    print(f"Reference CSV generated at {reference_csv}")

def gen_label_csv(label_csv, reference_csv, dx_dict, classes):
    """
    Generate a label CSV file with the corresponding diagnosis codes.
    """
    if not os.path.exists(label_csv):
        results = []
        df_reference = pd.read_csv(reference_csv)
        for _, row in df_reference.iterrows():
            patient_id = row['patient_id']
            dxs = [dx_dict.get(code, '') for code in row['dx'].split(',')]
            labels = [0] * len(classes)
            for idx, label in enumerate(classes):
                if label in dxs:
                    labels[idx] = 1
            results.append([patient_id] + labels)
        df = pd.DataFrame(data=results, columns=['patient_id'] + classes)
        n = len(df)
        folds = np.zeros(n, dtype=np.int8)
        for i in range(10):
            start = int(n * i / 10)
            end = int(n * (i + 1) / 10)
            folds[start:end] = i + 1
        df['fold'] = np.random.permutation(folds)
        columns = df.columns
        df['keep'] = df[classes].sum(axis=1)
        df = df[df['keep'] > 0]
        df[columns].to_csv(label_csv, index=None)
    print(f"Label CSV generated at {label_csv}")

def main():
    # Set the data directory (Make sure to adjust this path to your dataset location)
    data_dir = '/home/asus/Documents/cpsc_2018/datas'

    # Set the output directory (Kaggle's writable directory)
    output_dir = data_dir

    # Define the diagnosis codes and the corresponding classes
    dx_dict = {
        '426783006': 'SNR',  # Normal sinus rhythm
        '164889003': 'AF',   # Atrial fibrillation
        '270492004': 'IAVB', # First-degree atrioventricular block
        '164909002': 'LBBB', # Left bundle branch block
        '713427006': 'RBBB', # Complete right bundle branch block
        '59118001': 'RBBB',  # Right bundle branch block
        '284470004': 'PAC',  # Premature atrial contraction
        '63593006': 'PAC',   # Supraventricular premature beats
        '164884008': 'PVC',  # Ventricular ectopics
        '429622005': 'STD',  # ST-segment depression
        '164931005': 'STE',  # ST-segment elevation
    }
    classes = ['SNR', 'AF', 'IAVB', 'LBBB', 'RBBB', 'PAC', 'PVC', 'STD', 'STE']

    # Combine dataset files
    combine_dataset('/home/asus/Documents/cpsc_2018', output_dir)

    # Generate reference CSV
    reference_csv = os.path.join(output_dir, 'reference.csv')
    gen_reference_csv(output_dir, reference_csv)

    # Generate labels CSV
    label_csv = os.path.join(output_dir, 'labels.csv')
    gen_label_csv(label_csv, reference_csv, dx_dict, classes)

# Run the main function
if __name__ == "__main__":
    main()
