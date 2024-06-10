import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the data files
school_list_A = pd.read_csv('path_to_school_list_A.tsv', sep='\t')
jilla = pd.read_csv('path_to_jilla.tsv', sep='\t')
school_list_B = pd.read_csv('path_to_school_list_B.tsv', sep='\t')

# Normalize district names using the district mapping file
district_mapping = jilla.set_index('जिल्ला')['district'].to_dict()
school_list_A['district_romanized'] = school_list_A['district1'].map(district_mapping)
school_list_B['district_normalized'] = school_list_B['district'].str.strip().str.lower()
