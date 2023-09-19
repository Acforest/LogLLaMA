import os
# import re
import pandas as pd
import replicate
from tqdm import tqdm
# from collections import Counter
# from benchmark import log_2k_info
# from evaluation.utils.common import correct_single_template

if __name__ == '__main__':
    # Testing
    log_name = "HDFS"
    input_path = f'./data/log_parsing/{log_name}'
    output_path = f"./outputs/{log_name}"
    log_json_path = os.path.join(input_path, f'{log_name}_2k.json')

    replicate.Client(api_token='r8_XXR2UOl8DKoxMJzH8ELqPa2FbdRRGdJ0PP2OY')

    df = pd.read_json(log_json_path, orient="records")
    df.insert(loc=3, column="response", value='')
    df.rename(columns={"output": "groundtruth"}, inplace=True)

    os.makedirs(output_path, exist_ok=True)
    mode = 'single'
    model_name = "replicate/llama-2-70b-chat:2c1608e18606fad2812020dc541930f2d0495ce32eee50074220b87300bc16e1"
    resp_list = []
    prefix_prompt = '''
    <s>[INST] <<SYS>>\n
    You will be provided with a log message delimited by backticks. You must abstract variables with `{placeholders}` to 
    extract the corresponding template. Print the input log’s template delimited by backticks.\n
    <</SYS>>\n
    Log message: `{input}` [/INST]
    '''
    if mode == 'group':
        input_list = []
        batch_size = 10
        with open(os.path.join(output_path, f'{log_name}_response.txt'), 'w', encoding='utf-8') as f:
            for index, row in tqdm(df.iterrows(), total=df.shape[0], desc='Infering'):
                input_list.append(f'({index})\t' + row['input'])
                if (index + 1) % batch_size == 0:
                    resp = replicate.run(
                        model_name,
                        input={"prompt": prefix_prompt.replace('{input}', '\n'.join(input_list))}
                    )
                    f.write(resp + '\n')
                    resp_list.append(resp)
                    input_list.clear()
            if input_list:
                resp = replicate.run(
                    model_name,
                    input={"prompt": prefix_prompt.replace('{input}', row['input'])}
                )
                f.write(resp + '\n')
                resp_list.append(resp)
                input_list.clear()
    else:
        with open(os.path.join(output_path, f'{log_name}_response.txt'), 'w', encoding='utf-8') as f:
            for index, row in tqdm(df.iterrows(), total=df.shape[0], desc='Infering'):
                resp = replicate.run(
                    model_name,
                    input={"prompt": prefix_prompt.replace('{input}', row["input"])}
                )
                resp = ''.join(resp)
                print(resp)
                f.write(resp + '\n')
                resp_list.append(resp)

    df['response'] = resp_list
    df.to_csv(os.path.join(output_path, f'{log_name}_response.csv'), index=False)

    # setting = log_2k_info[log_name]
    # # log_df = log_to_dataframe(os.path.join(input_path, setting["log_file"]), setting['log_format'])
    # log_df = pd.read_csv(os.path.join(input_path, f'{setting["log_file"]}_structured_corrected.csv'))
    # log_df['EventTemplate'] = df['prediction'].copy()
    #
    # # Begin postprocess
    # param_regex = [
    #     r'{([ :_#.\-\w\d]+)}',
    #     r'{}'
    # ]
    # content = log_df.Content.tolist()
    # template = log_df.EventTemplate.tolist()
    # for i in range(len(content)):
    #     c = content[i]
    #     t = str(template[i])
    #     for r in param_regex:
    #         # print(r)
    #         t = re.sub(r, "<*>", t)
    #         # print(t)
    #     # if "{{}}" in t:
    #     #     print(t)
    #     #     print(re.sub(r'\{\{}}', "<*>", t))
    #     #     print(re.sub(r'{{}}', "<*>", t))
    #     template[i] = correct_single_template(t)
    # log_df.EventTemplate = pd.Series(template)
    #
    # unique_templates = sorted(Counter(template).items(), key=lambda k: k[1], reverse=True)
    # temp_df = pd.DataFrame(unique_templates, columns=['EventTemplate', 'Occurrences'])
    # # temp_df.sort_values(by=["Occurrences"], ascending=False, inplace=True)
    #
    # # End postprocess
    #
    # event_map = dict()
    # for event_id, event_template in enumerate(temp_df['EventTemplate']):
    #     event_map[event_template] = event_id
    # log_df['EventId'] = log_df['EventTemplate'].apply(lambda x: event_map[x])
    #
    # log_df.to_csv(os.path.join(output_path, f'{setting["log_file"]}_structured_logllama.csv'), index=False)
    # temp_df.to_csv(os.path.join(output_path, f'{setting["log_file"]}_templates_logllama.csv'), index=False)