import pandas as pd
import re
from tqdm.auto import tqdm  # Auto uses best available (terminal or notebook)

# Enable tqdm pandas integration
tqdm.pandas(desc="Extracting ECG-related text")

import re

def extract_ecg_related_text(note_text):
    # Define ECG-related section headers (you can expand this)
    section_headers = [
        r'CARDIAC:?$',                 # "CARDIAC:" or "CARDIAC"
        r'CARDIAC MARKER TREND:?$',    # full cardiac marker trend section
        r'ECHO:?$',                    # "ECHO" section
        r'ECG:?$',                     # "ECG" section
        r'EKG:?$',                     # "EKG" section
        r'ELECTROCARDIOGRAM:?$',       # sometimes full word is used
    ]
    
    # Compile into regex pattern
    section_pattern = re.compile(r'^(' + '|'.join(section_headers) + r')', re.I)
    
    # Split into lines
    lines = note_text.split('\n')
    
    # Flags and storage
    in_section = False
    current_section_lines = []
    extracted_sections = []
    
    for line in lines:
        # Check if this line starts a new ECG-related section
        if section_pattern.match(line.strip()):
            # If we were already in a section, save the previous one
            if current_section_lines:
                extracted_sections.append('\n'.join(current_section_lines))
                current_section_lines = []
            
            # Start new section
            in_section = True
            current_section_lines.append(line.strip())
        elif in_section:
            # If we're inside a section:
            # If the line is empty or looks like a new unrelated section starts, stop capturing
            if line.strip() == '' or re.match(r'^[A-Z][A-Z ]{3,}$', line.strip()):
                # Save current section
                extracted_sections.append('\n'.join(current_section_lines))
                current_section_lines = []
                in_section = False
            else:
                # Continue capturing the section
                current_section_lines.append(line.strip())
    
    # If we ended while still inside a section, save it
    if current_section_lines:
        extracted_sections.append('\n'.join(current_section_lines))
    
    # Combine all extracted sections into final text
    final_ecg_text = '\n\n'.join(extracted_sections)
    
    return final_ecg_text


# Load data
df = pd.read_csv(r'C:\Users\user\Documents\Python Scripts\BRAC\ecg_chatbot_project\data\discharge.csv')

# Print total rows
print(f"Total rows to process: {len(df)}")

# Apply with progress bar
df['ecg_text'] = df['text'].progress_apply(extract_ecg_related_text)

# Save to new CSV for later indexing
df[['note_id', 'ecg_text']].to_csv(r'C:\Users\user\Documents\Python Scripts\BRAC\ecg_chatbot_project\data\discharge_ecg_text.csv', index=False)

print("ECG extraction done! Saved to data/discharge_ecg_text.csv")

