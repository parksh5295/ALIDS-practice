import numpy as np
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.utils import shuffle


def preprocess_wustlIIoT(filename, classification_scenario):
    data = pd.read_csv(filename)
    label_2_class = data['Target']
    label_multi_class_encoded, _ = pd.factorize(data['Traffic'])
    drop_columns = ['StartTime', 'LastTime', 'SrcAddr', 'DstAddr', 'sIpId', 'dIpId', 'Traffic', 'Target']
    data.drop(drop_columns, axis=1, inplace=True)
    scaler = MinMaxScaler()
    scaled_df = scaler.fit_transform(data)
    my_label = label_2_class if classification_scenario == '1' else label_multi_class_encoded
    le = LabelEncoder()
    my_label_encoded = le.fit_transform(my_label)
    X_train, X_test, y_train, y_test = train_test_split(scaled_df, my_label_encoded, test_size=0.3, random_state=42, stratify=my_label_encoded)
    X_train_p, X_val, y_train_p, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)
    return X_train_p, X_val, X_test, y_train_p, y_val, y_test, le

def preprocess_xiiot(filename, scenario_option):
    data = pd.read_csv(filename, low_memory=False)
    my_label_series = data.iloc[:, 61] if scenario_option == '1' else data.iloc[:, 60]
    features = data.iloc[:, :59]
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(features)
    le = LabelEncoder()
    my_label_encoded = le.fit_transform(my_label_series)
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_features, my_label_encoded, test_size=0.3, random_state=42, stratify=my_label_encoded
    )
    X_train_p, X_val, y_train_p, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)
    return X_train_p, X_val, X_test, y_train_p, y_val, y_test, le

def preprocess_edgeIIoTset(filename, classification_scenario, low_memory=False):
    def encode_text_dummy(df, name):
        dummies = pd.get_dummies(df[name], prefix=name)
        df = pd.concat([df, dummies], axis=1)
        df.drop(name, axis=1, inplace=True)
        return df
    data = pd.read_csv(filename, low_memory=low_memory)
    data.dropna(axis=0, how='any', inplace=True)
    data.drop_duplicates(subset=None, keep="first", inplace=True)
    data = shuffle(data, random_state=42)
    for col in ['http.request.method', 'http.referer', 'http.request.version', 'dns.qry.name.len', 'mqtt.conack.flags', 'mqtt.protoname', 'mqtt.topic']:
        if col in data.columns:
            data = encode_text_dummy(data, col)
    two_categories = {k: ('Attack' if k != 'Normal' else 'Normal') for k in data['Attack_type'].unique()}
    data['2_class'] = data['Attack_type'].map(two_categories)
    mapping = {
    'Normal': 'Normal',
    'DDoS_UDP': 'DDoS', 'DDoS_ICMP': 'DDoS', 'DDoS_TCP': 'DDoS', 'DDoS_HTTP': 'DDoS',
    'Ransomware': 'Malware', 'Backdoor': 'Malware','Uploading': 'Malware',
    'Vulnerability_scanner': 'Reconnaissance', 'Port_Scanning': 'Reconnaissance', 'Fingerprinting': 'Reconnaissance',
    'SQL_injection': 'Injection', 'XSS': 'Injection',
    'Password': 'Password Cracking',
    'MITM': 'MITM'
    }
    data['Attack_type'] = data['Attack_type'].map(mapping)
    my_label_series = data['2_class'] if classification_scenario == '1' else data['Attack_type']
    drop_columns = [
        "frame.time", "ip.src_host", "ip.dst_host", "arp.src.proto_ipv4", "arp.dst.proto_ipv4",
        "http.file_data", "http.request.full_uri", "icmp.transmit_timestamp", "http.request.uri.query",
        "tcp.options", "tcp.payload", "tcp.srcport", "tcp.dstport", "udp.port", "mqtt.msg",
        "Attack_label", "Attack_type", "2_class"
    ]
    if '6_class' in data.columns: drop_columns.append('6_class')
    data.drop([col for col in drop_columns if col in data.columns], axis=1, inplace=True)
    data.columns = [re.sub(r'[^a-zA-Z0-9_]', '_', col) for col in data.columns]
    features = data.drop(columns=[col for col in data.columns if 'Attack_type' in col or '2_class' in col], errors='ignore')
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(features)
    le = LabelEncoder()
    my_label_encoded = le.fit_transform(my_label_series)
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_features, my_label_encoded, test_size=0.3, random_state=42, stratify=my_label_encoded
    )
    X_train_p, X_val, y_train_p, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)
    return X_train_p, X_val, X_test, y_train_p, y_val, y_test, le

def preprocess_nftoniot(filename, classification_scenario):
    """ Preprocesses the NF-To-N-IoT-v3 dataset for a given scenario. """
    data_full = pd.read_csv(filename)
    data = data_full.sample(frac=0.1, random_state=42)
    features = data.drop(columns=['Label', 'Attack'])
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(features)
    label_series = data['Label'] if classification_scenario == '1' else data['Attack']
    le = LabelEncoder()
    my_label_encoded = le.fit_transform(label_series)
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_features, my_label_encoded, test_size=0.3, random_state=42, stratify=my_label_encoded
    )
    X_train_p, X_val, y_train_p, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)
    return X_train_p, X_val, X_test, y_train_p, y_val, y_test, le