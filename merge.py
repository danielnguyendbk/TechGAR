import pandas as pd
import os

csv_files = {
    'pklot_coco': 'pklot_coco_train.csv',
    'pklot_xml': 'pklot_xml_stats.csv', 
    'cnrpark': 'cnrpark_stats.csv'
}

master_data = []
for name, csv_file in csv_files.items():
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
        df['dataset'] = name
        
        # Chuẩn hóa tên cột 'label' thành 'status' (nếu có)
        if 'label' in df.columns:
            df['status'] = df['label']
            
        if 'status_num' not in df.columns:
            df['status_num'] = df['status'].map({'empty':0, 'free':0, 'Empty':0, 'busy':1, 'occupied':1, 'Occupied':1})

        master_data.append(df[['dataset', 'status', 'status_num'] + [col for col in df.columns if 'bbox' in col.lower() or col in ['img', 'file', 'image_file', 'cam']]])
        print(f"{name}: {len(df)} records, {df['status_num'].value_counts(normalize=True).to_dict()}")

master = pd.concat(master_data, ignore_index=True)
master.to_csv('master_parking_dataset.csv', index=False)
print(f"\n✅ MASTER CSV: {len(master)} tổng spots từ 3 datasets")
print(master.groupby('dataset')['status_num'].value_counts(normalize=True))
print(master.head())
