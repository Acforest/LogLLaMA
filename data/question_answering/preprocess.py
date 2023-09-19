import json
import os.path


if __name__ == '__main__':
    dataset_names = ['HDFS', 'OpenSSH', 'Spark']
    file_names = ['qa.json.train', 'qa.json.test', 'qa.json.val']
    all_train_data, all_test_data, all_dev_data = [], [], []
    for dataset_name in dataset_names:
        for file_name in file_names:
            data = []
            path = os.path.join(dataset_name, file_name)
            with open(path, 'r', encoding='utf-8') as f:
                for line in f.readlines():
                    contents = json.loads(line)
                    contents['instruction'] = contents.pop('Question')
                    contents['input'] = contents.pop('RawLog')
                    contents['output'] = contents.pop('Answer')
                    data.append(contents)
                    if file_name == 'qa.json.train':
                        all_train_data.append(contents)
                    elif file_name == 'qa.json.test':
                        all_test_data.append(contents)
                    else:
                        all_dev_data.append(contents)
            if file_name == 'qa.json.train':
                save_path = os.path.join(dataset_name, 'trian.json')
            elif file_name == 'qa.json.test':
                save_path = os.path.join(dataset_name, 'test.json')
            else:
                save_path = os.path.join(dataset_name, 'dev.json')
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4)
    with open('./qa_train.json', 'w', encoding='utf-8') as f:
        json.dump(all_train_data, f, indent=4)
    with open('./qa_test.json', 'w', encoding='utf-8') as f:
        json.dump(all_test_data, f, indent=4)
    with open('./qa_dev.json', 'w', encoding='utf-8') as f:
        json.dump(all_dev_data, f, indent=4)
