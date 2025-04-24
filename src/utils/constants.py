# Project-level constants, including API keys and directories
# Note: importing this file has the side effect of loading a configuration file
from pathlib import Path
import yaml


##############################
# Model Configurations
##############################
model_name = 'llama2'
base_model = 'meta-llama/Llama-2-7b-hf'

######################
#   hyperparameter   #
######################
learning_rate = 2e-5
num_epochs = 5
batch_size = 1
seed = 42
threshold = 3
strategy = 'min'

##############################
# API keys
##############################
OPENAI_API_KEY = ''


##############################
# Perspective API
##############################
DISCOVERY_URL = "https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1"
PERSPECTIVE_API_KEY = ""
PERSPECTIVE_API_LEN_LIMIT = 20480

# All attributes can be found here:
# https://github.com/conversationai/perspectiveapi/blob/master/2-api/models.md
PERSPECTIVE_API_ATTRIBUTES = (
    'TOXICITY',
    'SEVERE_TOXICITY',
    'IDENTITY_ATTACK',
    'INSULT',
    'THREAT',
    'PROFANITY',
)
PERSPECTIVE_API_ATTRIBUTES_LOWER = tuple(a.lower() for a in PERSPECTIVE_API_ATTRIBUTES)


##############################
# non value datasets
##############################
non_value_datasets = [
    'alpaca',
    'grammar',
    'samsum',
    'vanilla',
    'dolly'
]


##############################
# Target Groups
##############################
group_list = [
    # 'Albania',
    # 'Belgium',
    # 'Bulgaria',
    # 'Cyprus',
    # 'Czech',
    # 'Denmark',
    # 'Estonia',
    # 'Finland',
    # 'France',
    # 'Hungary',
    # 'Iceland',
    # 'Ireland',
    # 'Israel',
    # 'Italy',
    # 'Kosovo',
    # 'Lithuania',
    # 'Netherlands',
    # 'Norway',
    # 'Poland',
    # 'Portugal',
    # 'Russia',
    # 'Slovakia',
    # 'Slovenia',
    # 'Spain',
    # 'Sweden',
    # 'Switzerland',
    # 'Ukraine',
    # 'United_Kingdom',
    'Group_1',
    'Group_2',
    'Group_3',
    'Group_4',
    'Group_5',
    'Group_6',
    'Group_7',
    'Group_8',
    'Group_9',
    'Group_10',
    'Group_11',
    'Group_12',
    'Group_13',
    'Group_14',
    'Group_15',
    'Group_16',
    'Group_17',
    'Group_18',
    'Group_19',
    'Group_20',
    'Group_21',
    'Group_22',
    'Group_23',
    'Group_24',
    'Group_25',
    'Group_26',
    'Group_27',
    'Group_28',
    'Group_29',
    'Group_30',
    'Group_31',
    'Group_32',
    'Group_33',
    'Group_34',
    'Group_35',
    'Group_36', 
    'Group_37',
    'Group_38',
    'Group_39',
    'Group_40',
    'Group_41',
    'Group_42',
    'Group_43',
    'Group_44',
    'Group_45',
    'Group_46',
    'Group_47',
    'Group_48',
    'Group_49',
    'Group_50',
    'Group_51',
    'Group_52',
    'Group_53',
    'Group_54',
    'Group_55',
    'Group_56',
    'Group_57',
    'Group_58',
    'Group_59',
    'Group_60', 
    'Group_61', 
    'Group_62', 
    'Group_63', 
    'Group_64', 
    'Group_65', 
    'Group_66', 
    'Group_67', 
    'Group_68', 
    'Group_69', 
    'Group_70', 
    'Group_71', 
    'Group_72', 
    'Group_73', 
    'Group_74', 
    'Group_75', 
    'Group_76', 
    'Group_77', 
    'Group_78', 
    'Group_79', 
    'Group_80', 
    'Group_81', 
    'Group_82', 
    'Group_83',
    'Group_84', 
    'Group_85', 
    'Group_86', 
    'Group_87', 
    'Group_88', 
    'Group_89', 
    'Group_90', 
    'Group_91', 
    'Group_92', 
    'Group_93', 
    'Group_94', 
    'Group_95', 
    'Group_96', 
    'Group_97', 
    'Group_98', 
    'Group_99',
    'Group_100'
    ]