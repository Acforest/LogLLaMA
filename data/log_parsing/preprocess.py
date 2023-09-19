import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from benchmark import log_2k_info


if __name__ == '__main__':
    sample_data = []
    num_shot = 32
    for dataset in log_2k_info.keys():
        print(dataset)

        log_df = pd.read_csv(os.path.join(dataset, f'{dataset}_2k.log_structured_corrected.csv'))

        train_df, test_df = train_test_split(log_df, test_size=0.2, random_state=1234)
        test_df, dev_df = train_test_split(test_df, test_size=0.5, random_state=1234)

        instruction = 'As a log parser, you will be given a log message, and you need to replace variables with <*> ' \
                      'to extract the corresponding template.'

        instruction_with_example = 'As a log parser, you will be given a log message, and you need to replace variables with <*> ' \
                                   'to extract the corresponding template. For example:\n' \
                                   'Input: BLOCK* NameSystem.addStoredBlock: blockMap updated: 10.250.5.237:50010 is added to blk_-476906696485288376 size 67108864\n' \
                                   'Output: BLOCK* NameSystem.addStoredBlock: blockMap updated: <*> is added to <*> size <*>'

        log_sample = log_df.sample(n=num_shot)
        for _, row in log_sample.iterrows():
            sample_data.append(
                {'instruction': instruction, 'input': row['Content'], 'output': row['EventTemplate']}
            )

        all_data = [
            {'instruction': instruction, 'input': row['Content'], 'output': row['EventTemplate']}
            for _, row in log_df.iterrows()
        ]
        with open(os.path.join(dataset, f'{dataset}_2k.json'), 'w', encoding='utf-8') as f:
            json.dump(all_data, f, indent=4)

        all_data_with_example = [
            {'instruction': instruction_with_example, 'input': row['Content'], 'output': row['EventTemplate']}
            for _, row in log_df.iterrows()
        ]
        with open(os.path.join(dataset, f'{dataset}_2k_with_example.json'), 'w', encoding='utf-8') as f:
            json.dump(all_data_with_example, f, indent=4)

        all_data = [
            {'instruction': instruction, 'input': row['Content'], 'output': row['EventTemplate']}
            for _, row in log_df.iterrows()
        ]
        with open(os.path.join(dataset, f'{dataset}_2k.json'), 'w', encoding='utf-8') as f:
            json.dump(all_data, f, indent=4)

        train_data = [
            {'instruction': instruction, 'input': row['Content'], 'output': row['EventTemplate']}
            for _, row in train_df.iterrows()
        ]
        with open(os.path.join(dataset, 'train.json'), 'w', encoding='utf-8') as f:
            json.dump(train_data, f, indent=4)

        test_data = [
            {'instruction': instruction, 'input': row['Content'], 'output': row['EventTemplate']}
            for _, row in test_df.iterrows()
        ]
        with open(os.path.join(dataset, 'test.json'), 'w', encoding='utf-8') as f:
            json.dump(test_data, f, indent=4)

        dev_data = [
            {'instruction': instruction, 'input': row['Content'], 'output': row['EventTemplate']}
            for _, row in dev_df.iterrows()
        ]
        with open(os.path.join(dataset, 'dev.json'), 'w', encoding='utf-8') as f:
            json.dump(dev_data, f, indent=4)

    with open(f'{num_shot}-shot.json', 'w', encoding='utf-8') as f:
        json.dump(sample_data, f, indent=4)
