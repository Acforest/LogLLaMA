import re
import os
import pandas as pd
from benchmark import log_to_dataframe, log_2k_info
from collections import Counter


def correct_single_template(template, user_strings=None):
    """Apply all rules to process a template.

    DS (Double Space)
    BL (Boolean)
    US (User String)
    DG (Digit)
    PS (Path-like String)
    WV (Word concatenated with Variable)
    DV (Dot-separated Variables)
    CV (Consecutive Variables)

    """

    boolean = {'true', 'false'}
    default_strings = {'null', 'root', 'admin'}
    path_delimiters = {  # reduced set of delimiters for tokenizing for checking the path-like strings
        r'\s', r'\,', r'\!', r'\;', r'\:',
        r'\=', r'\|', r'\"', r'\'',
        r'\[', r'\]', r'\(', r'\)', r'\{', r'\}'
    }
    token_delimiters = path_delimiters.union({  # all delimiters for tokenizing the remaining rules
        r'\.', r'\-', r'\+', r'\@', r'\#', r'\$', r'\%', r'\&',
    })

    if user_strings:
        default_strings = default_strings.union(user_strings)

    # apply DS
    template = template.strip()
    template = re.sub(r'\s+', ' ', template)

    # apply PS
    p_tokens = re.split('(' + '|'.join(path_delimiters) + ')', template)
    new_p_tokens = []
    for p_token in p_tokens:
        if re.match(r'^(\/[^\/]+)+$', p_token):
            p_token = '<*>'
        new_p_tokens.append(p_token)
    template = ''.join(new_p_tokens)

    # tokenize for the remaining rules
    tokens = re.split('(' + '|'.join(token_delimiters) + ')', template)  # tokenizing while keeping delimiters
    new_tokens = []
    for token in tokens:
        # apply BL, US
        for to_replace in boolean.union(default_strings):
            if token.lower() == to_replace.lower():
                token = '<*>'

        # apply DG
        if re.match(r'^\d+$', token):
            token = '<*>'

        # apply WV
        if re.match(r'^[^\s\/]*<\*>[^\s\/]*$', token):
            if token != '<*>/<*>':  # need to check this because `/` is not a deliminator
                token = '<*>'

        # collect the result
        new_tokens.append(token)

    # make the template using new_tokens
    template = ''.join(new_tokens)

    # Substitute consecutive variables only if separated with any delimiter including "." (DV)
    while True:
        prev = template
        template = re.sub(r'<\*>\.<\*>', '<*>', template)
        if prev == template:
            break

    # Substitute consecutive variables only if not separated with any delimiter including space (CV)
    # NOTE: this should be done at the end
    while True:
        prev = template
        template = re.sub(r'<\*><\*>', '<*>', template)
        if prev == template:
            break

    while "#<*>#" in template:
        template = template.replace("#<*>#", "<*>")

    while "<*>:<*>" in template:
        template = template.replace("<*>:<*>", "<*>")
    return template


def postprocess(df):
    return df['response'].apply(lambda row: row.split('Log template: ')[-1].split('[END]')[0].split('[BEGIN]')[-1].strip())


if __name__ == '__main__':
    # dataset = 'BGL'
    for dataset in log_2k_info.keys():
        print(dataset)
        setting = log_2k_info[dataset]

        input_path = f'./data/log_parsing/{dataset}'
        response_path = f'./outputs/{dataset}'
        output_path = f"./outputs/{dataset}"

        os.makedirs(output_path, exist_ok=True)
        log_df = pd.read_csv(os.path.join(input_path, f'{setting["log_file"]}_structured.csv'))
        try:
            res_df = pd.read_csv(os.path.join(output_path, f'{dataset}_response.csv'))
        except:
            continue

        log_df['EventTemplate'] = postprocess(res_df)

        # post process
        param_regex = [
            r'<\w+>',  # replace <num>, <hex> ... with <*>
            r'<([ :_#.\-\w\d]+)>',
            r'<>'
        ]
        content = log_df.Content.tolist()
        template = log_df.EventTemplate.tolist()
        for i in range(len(content)):
            c = content[i]
            t = str(template[i])
            for r in param_regex:
                t = re.sub(r, "<*>", t)
            template[i] = correct_single_template(t)
        log_df.EventTemplate = pd.Series(template)

        unique_templates = sorted(Counter(template).items(), key=lambda k: k[1], reverse=True)
        temp_df = pd.DataFrame(unique_templates, columns=['EventTemplate', 'Occurrences'])
        # temp_df.sort_values(by=["Occurrences"], ascending=False, inplace=True)

        event_map = dict()
        for event_id, event_template in enumerate(temp_df['EventTemplate']):
            event_map[event_template] = event_id
        log_df['EventId'] = log_df['EventTemplate'].apply(lambda x: event_map[x])

        log_df.to_csv(os.path.join(output_path, f'{setting["log_file"]}_structured_logllama-3.csv'), index=False)
        temp_df.to_csv(os.path.join(output_path, f'{setting["log_file"]}_templates_logllama-3.csv'), index=False)
