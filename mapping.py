import pandas as pd
from fuzzywuzzy import process, fuzz

# Load the data files
school_list_A = pd.read_csv('path_to_school_list_A.tsv', sep='\t')
jilla = pd.read_csv('path_to_jilla.tsv', sep='\t')
school_list_B = pd.read_csv('path_to_school_list_B.tsv', sep='\t')

# Normalize district names using the district mapping file
district_mapping = jilla.set_index('जिल्ला')['district'].to_dict()
school_list_A['district_romanized'] = school_list_A['district1'].map(district_mapping)
school_list_B['district_normalized'] = school_list_B['district'].str.strip().str.lower()

# Function to match schools by district and name using fuzzy matching with batching
def match_schools_batched_fuzzy(school_list_A, school_list_B, batch_size=1000):
    matches = []
    
    for idx, school_a in school_list_A.iterrows():
        # Filter Source B schools by district
        district_schools_b = school_list_B[school_list_B['district_normalized'] == school_a['district_romanized']]
        
        # If there are no matching districts, continue to next school
        if district_schools_b.empty:
            continue
        
        # Fuzzy match the school name in Romanized script with the English names in Source B
        best_match = process.extractOne(
            school_a['velthuis'], 
            district_schools_b['name'], 
            scorer=fuzz.token_set_ratio
        )
        
        # If a good match is found, add to matches
        if best_match and best_match[1] > 80:  # Adjust the threshold as needed
            matched_school_b = district_schools_b[district_schools_b['name'] == best_match[0]].iloc[0]
            matches.append((school_a['school_id'], matched_school_b['school_id']))
    
    return matches

# Break down school_list_A into smaller batches
batch_size = 1000
num_batches = len(school_list_A) // batch_size + 1
all_matches = []

for i in range(num_batches):
    batch_start = i * batch_size
    batch_end = min((i + 1) * batch_size, len(school_list_A))
    batch_a = school_list_A.iloc[batch_start:batch_end]
    batch_matches = match_schools_batched_fuzzy(batch_a, school_list_B)
    all_matches.extend(batch_matches)

# Create a DataFrame for the matched results
matched_df = pd.DataFrame(all_matches, columns=['school_id_a', 'school_id_b'])
matched_df.to_csv('matched_schools.csv', index=False)

# Display the first few rows of the matched results
print(matched_df.head())
