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

# Function to match schools using TF-IDF vectorization and cosine similarity
def match_schools_vectorized(school_list_A, school_list_B):
    matches = []
    vectorizer = TfidfVectorizer().fit(pd.concat([school_list_A['velthuis'], school_list_B['name']]).unique())

    for idx, school_a in school_list_A.iterrows():
        # Filter Source B schools by district
        district_schools_b = school_list_B[school_list_B['district_normalized'] == school_a['district_romanized']]
        
        # If there are no matching districts, continue to next school
        if district_schools_b.empty:
            continue

        # Vectorize school names
        tfidf_a = vectorizer.transform([school_a['velthuis']])
        tfidf_b = vectorizer.transform(district_schools_b['name'])

        # Compute cosine similarity
        similarities = cosine_similarity(tfidf_a, tfidf_b).flatten()

        # Get the best match
        best_match_idx = similarities.argmax()
        best_match_score = similarities[best_match_idx]

        # If a good match is found, add to matches
        if best_match_score > 0.8:  # Adjust the threshold as needed
            matched_school_b = district_schools_b.iloc[best_match_idx]
            matches.append((school_a['school_id'], matched_school_b['school_id']))
    
    return matches
