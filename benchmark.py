import os
import re
import pandas as pd

from evaluation.utils.evaluator_main import evaluator


def generate_logformat_regex(log_format):
    """ Function to generate regular expression to split log messages
    """
    headers = []
    splitters = re.split(r'(<[^<>]+>)', log_format)
    regex = ''
    for k in range(len(splitters)):
        if k % 2 == 0:
            splitter = re.sub(' +', '\\\s+', splitters[k])
            regex += splitter
        else:
            header = splitters[k].strip('<').strip('>')
            regex += '(?P<%s>.*?)' % header
            headers.append(header)
    regex = re.compile('^' + regex + '$')
    return headers, regex


def log_to_dataframe(log_file, log_format):
    """ Function to transform log file to dataframe
    """
    headers, regex = generate_logformat_regex(log_format)
    log_messages = []
    line_count = 0
    with open(log_file, 'r', encoding='utf-8', errors='ignore') as fin:
        for line in fin.readlines():
            try:
                match = regex.search(line.strip())
                message = [match.group(header) for header in headers]
                log_messages.append(message)
                line_count += 1
            except Exception as _:
                pass
    logdf = pd.DataFrame(log_messages, columns=headers)
    logdf.insert(0, 'LineId', None)
    logdf['LineId'] = [i + 1 for i in range(line_count)]
    return logdf


log_2k_info = {
    'HDFS': {
        'log_file': 'HDFS_2k.log',
        'log_format': '<Date> <Time> <Pid> <Level> <Component>: <Content>',
    },

    'Hadoop': {
        'log_file': 'Hadoop_2k.log',
        'log_format': '<SessionId> <Date> <Time> <Level> \[<Process>\] <Component>: <Content>',
    },

    'Spark': {
        'log_file': 'Spark_2k.log',
        'log_format': '<Date> <Time> <Level> <Component>: <Content>',
    },

    'Zookeeper': {
        'log_file': 'Zookeeper_2k.log',
        'log_format': '<Date> <Time> - <Level>  \[<Node>:<Component>@<Id>\] - <Content>',
    },

    'BGL': {
        'log_file': 'BGL_2k.log',
        'log_format': '<Label> <Timestamp> <Date> <Node> <Time> <NodeRepeat> <Type> <Component> <Level> <Content>',
    },

    'HPC': {
        'log_file': 'HPC_2k.log',
        'log_format': '<LogId> <Node> <Component> <State> <Time> <Flag> <Content>',
    },

    'Thunderbird': {
        'log_file': 'Thunderbird_2k.log',
        'log_format': '<Label> <Timestamp> <Date> <User> <Month> <Day> <Time> <Location> <Component>(\[<PID>\])?: <Content>',
    },

    'Windows': {
        'log_file': 'Windows_2k.log',
        'log_format': '<Date> <Time>, <Level>                  <Component>    <Content>',
    },

    'Linux': {
        'log_file': 'Linux_2k.log',
        'log_format': '<Month> <Date> <Time> <Level> <Component>(\[<PID>\])?: <Content>',
    },

    'Android': {
        'log_file': 'Android_2k.log',
        'log_format': '<Date> <Time>  <Pid>  <Tid> <Level> <Component>: <Content>',
    },

    'HealthApp': {
        'log_file': 'HealthApp_2k.log',
        'log_format': '<Time>\|<Component>\|<Pid>\|<Content>',
    },

    'Apache': {
        'log_file': 'Apache_2k.log',
        'log_format': '\[<Time>\] \[<Level>\] <Content>',
    },

    'Proxifier': {
        'log_file': 'Proxifier_2k.log',
        'log_format': '\[<Time>\] <Program> - <Content>',
    },

    'OpenSSH': {
        'log_file': 'OpenSSH_2k.log',
        'log_format': '<Date> <Day> <Time> <Component> sshd\[<Pid>\]: <Content>',
    },

    'OpenStack': {
        'log_file': 'OpenStack_2k.log',
        'log_format': '<Logrecord> <Date> <Time> <Pid> <Level> <Component> \[<ADDR>\] <Content>',
    },

    'Mac': {
        'log_file': 'Mac_2k.log',
        'log_format': '<Month>  <Date> <Time> <User> <Component>\[<PID>\]( \(<Address>\))?: <Content>',
    }
}

if __name__ == '__main__':
    bechmark_result = []
    avg_ga, avg_pa, avg_ed = 0, 0, 0
    for dataset, setting in log_2k_info.items():
        print(f'\n=== Evaluation on {dataset} ===')
        input_dir = f'./data/log_parsing/{dataset}'
        output_dir = f'./outputs/{dataset}'
        log_file = os.path.basename(setting['log_file'])

        try:
            GA, PA, ED, FTA, PTA, RTA, OG, UG, MX, unseen_PA, no_unseen = evaluator(
                groundtruth=os.path.join(input_dir, log_file + '_structured_corrected.csv'),
                parsedresult=os.path.join(output_dir, log_file + '_structured_logllama-3.csv')
            )
            bechmark_result.append([dataset, GA, PA, ED])
            avg_ga += GA
            avg_pa += PA
            avg_ed += ED
        except Exception as _:
            pass

    bechmark_result.append(["Average",
                            avg_ga / len(log_2k_info.keys()),
                            avg_pa / len(log_2k_info.keys()),
                            avg_ed / len(log_2k_info.keys())])

    print('\n=== Overall evaluation results ===')
    df_result = pd.DataFrame(bechmark_result,
                             columns=['Dataset', 'Group Accuracy', 'Parsing Accuracy', 'Edit distance'])
    df_result.set_index('Dataset', inplace=True)
    print(df_result)
    df_result.T.to_csv('./outputs/logllama-3_benchmark_result.csv')
