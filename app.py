import streamlit as st
import pandas as pd
import re
from typing import List, Dict, Set
import json
import io
import os
from pathlib import Path
import pyarrow.feather as feather
import configparser

def load_config(config_file):
    """Load configuration from file"""
    config = configparser.ConfigParser()
    config.read(config_file)
    
    # Load data configuration
    data_config = {}
    if 'data' in config:
        data_config = {
            'file_path': config['data'].get('file_path', 'data.feather'),
            'text_column': config['data'].get('text_column', 'content')
        }
    
    # Load output configuration
    output_config = {}
    if 'output' in config:
        output_config = {
            'filename': config['output'].get('output_filename', 'labeled_data.feather'),
            'columns': [col.strip() for col in config['output'].get('output_columns', '').split(',') if col.strip()]
        }
    
    # Load highlighting configuration
    highlighting_config = {}
    if 'highlighting' in config:
        highlighting_config = {
            'color': config['highlighting'].get('color', '#ece800'),
            'words': [word.strip() for word in config['highlighting'].get('words', '').split(',') if word.strip()]
        }
    
    # Load multiple classification tasks
    classification_tasks = {}
    if 'classification_tasks' in config:
        task_num = 1
        while f'task{task_num}_name' in config['classification_tasks']:
            task_name = config['classification_tasks'][f'task{task_num}_name']
            task_labels = [label.strip() for label in config['classification_tasks'][f'task{task_num}_labels'].split(',')]
            classification_tasks[f'task{task_num}'] = {
                'name': task_name,
                'labels': task_labels
            }
            task_num += 1
    
    # Load multiple feature tasks
    feature_tasks = {}
    if 'feature_tasks' in config:
        task_num = 1
        while f'task{task_num}_name' in config['feature_tasks']:
            task_name = config['feature_tasks'][f'task{task_num}_name']
            task_labels = [label.strip() for label in config['feature_tasks'][f'task{task_num}_labels'].split(',')]
            feature_tasks[f'task{task_num}'] = {
                'name': task_name,
                'labels': task_labels
            }
            task_num += 1
    
    # Load legacy single features section for backward compatibility
    legacy_features = {}
    if 'features' in config and 'name' in config['features']:
        legacy_features = {
            'name': config['features'].get('name', 'Features'),
            'labels': [label.strip() for label in config['features']['labels'].split(',')]
        }
    
    return data_config, highlighting_config, classification_tasks, feature_tasks, legacy_features, output_config

# Configure page
st.set_page_config(
    page_title="TV Content Labeling Tool",
    page_icon="📺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'labels' not in st.session_state:
    st.session_state.labels = {}
if 'current_index' not in st.session_state:
    st.session_state.current_index = 0
if 'data' not in st.session_state:
    st.session_state.data = None
if 'data_mode' not in st.session_state:
    st.session_state.data_mode = 'memory'  # Default to memory mode
if 'labeled_data' not in st.session_state:
    st.session_state.labeled_data = pd.DataFrame()
if 'file_path' not in st.session_state:
    st.session_state.file_path = ""
if 'total_rows' not in st.session_state:
    st.session_state.total_rows = 0
if 'chunk_size' not in st.session_state:
    st.session_state.chunk_size = 1000
if 'current_chunk' not in st.session_state:
    st.session_state.current_chunk = None
if 'chunk_start_idx' not in st.session_state:
    st.session_state.chunk_start_idx = 0
if 'selected_columns' not in st.session_state:
    st.session_state.selected_columns = []
if 'text_column' not in st.session_state:
    st.session_state.text_column = 'content'
if 'highlight_words' not in st.session_state:
    st.session_state.highlight_words = []
if 'classification_labels' not in st.session_state:
    st.session_state.classification_labels = []
if 'classification_tasks' not in st.session_state:
    st.session_state.classification_tasks = {}
if 'feature_labels' not in st.session_state:
    st.session_state.feature_labels = []
if 'feature_tasks' not in st.session_state:
    st.session_state.feature_tasks = {}
if 'labeling_locked' not in st.session_state:
    st.session_state.labeling_locked = False
if 'auto_save_counter' not in st.session_state:
    st.session_state.auto_save_counter = 0
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False

# Load default configuration on startup
if not st.session_state.classification_tasks and os.path.exists('default.cfg'):
    data_config, highlighting_config, classification_tasks, feature_tasks, legacy_features, output_config = load_config('default.cfg')
    st.session_state.classification_tasks = classification_tasks
    st.session_state.feature_tasks = feature_tasks
    if legacy_features and 'labels' in legacy_features:
        st.session_state.feature_labels = legacy_features['labels']
    if highlighting_config:
        st.session_state.highlight_words = highlighting_config.get('words', [])
    
    # Store output configuration
    if output_config:
        st.session_state.output_config = output_config
    
    # Auto-load data file if specified and exists
    if data_config and 'file_path' in data_config:
        data_file_path = data_config['file_path']
        if os.path.exists(data_file_path) and st.session_state.data is None:
            try:
                # Load the data file
                if data_file_path.endswith('.feather'):
                    st.session_state.data = pd.read_feather(data_file_path)
                elif data_file_path.endswith('.csv'):
                    st.session_state.data = pd.read_csv(data_file_path)
                elif data_file_path.endswith('.json'):
                    st.session_state.data = pd.read_json(data_file_path)
                
                if st.session_state.data is not None:
                    # Set up for large file handling
                    st.session_state.file_path = data_file_path
                    st.session_state.total_rows = len(st.session_state.data)
                    
                    # Auto-select the text column if specified
                    text_column = data_config.get('text_column', 'content')
                    if text_column in st.session_state.data.columns:
                        st.session_state.text_column = text_column
                        # Set selected columns from output config if available
                        if output_config and 'columns' in output_config and output_config['columns']:
                            # Filter output columns to only include ones that exist in the data
                            valid_columns = [col for col in output_config['columns'] if col in st.session_state.data.columns]
                            st.session_state.selected_columns = valid_columns
                        else:
                            st.session_state.selected_columns = [text_column]
                    else:
                        # If specified column doesn't exist, use the first text column
                        text_columns = [col for col in st.session_state.data.columns 
                                      if st.session_state.data[col].dtype == 'object']
                        if text_columns:
                            st.session_state.text_column = text_columns[0]
                            st.session_state.selected_columns = [text_columns[0]]
                    
                    # Initialize label columns if they don't exist
                    for task_key, task_info in st.session_state.classification_tasks.items():
                        col_name = f"{task_info['name']}_label"
                        if col_name not in st.session_state.data.columns:
                            st.session_state.data[col_name] = ""
                    
                    if st.session_state.feature_labels:
                        feature_col_name = "label_features"
                        if feature_col_name not in st.session_state.data.columns:
                            st.session_state.data[feature_col_name] = ""
                    
                    # Initialize feature task columns
                    for task_key, task_info in st.session_state.feature_tasks.items():
                        col_name = f"{task_info['name']}_features"
                        if col_name not in st.session_state.data.columns:
                            st.session_state.data[col_name] = ""
                            
            except Exception as e:
                # Don't show error on startup, just continue without auto-loading
                pass


def get_total_rows(file_path: str) -> int:
    """Get total number of rows in feather file without loading all data."""
    try:
        # For feather files, we need to load the file to get row count
        # But we can use memory-mapped reading for efficiency
        df = pd.read_feather(file_path)
        return len(df)
    except Exception as e:
        st.error(f"Error counting rows: {str(e)}")
        return 0

def load_chunk(file_path: str, start_idx: int, chunk_size: int) -> pd.DataFrame:
    """Load a specific chunk of data from feather file."""
    try:
        # Load the full feather file and slice it
        # Feather files are optimized for this kind of operation
        df = pd.read_feather(file_path)
        end_idx = min(start_idx + chunk_size, len(df))
        chunk = df.iloc[start_idx:end_idx].copy()
        return chunk
    except Exception as e:
        st.error(f"Error loading chunk: {str(e)}")
        return pd.DataFrame()

def get_current_row():
    """Get the current row data, loading new chunk if necessary."""
    if st.session_state.file_path and st.session_state.current_index < st.session_state.total_rows:
        # Check if we need to load a new chunk
        chunk_index = st.session_state.current_index - st.session_state.chunk_start_idx
        
        if (st.session_state.current_chunk is None or 
            chunk_index < 0 or 
            chunk_index >= len(st.session_state.current_chunk)):
            
            # Calculate new chunk start
            new_chunk_start = (st.session_state.current_index // st.session_state.chunk_size) * st.session_state.chunk_size
            
            # Load new chunk
            with st.spinner("Loading data chunk..."):
                st.session_state.current_chunk = load_chunk(
                    st.session_state.file_path, 
                    new_chunk_start, 
                    st.session_state.chunk_size
                )
                st.session_state.chunk_start_idx = new_chunk_start
        
        # Get row from current chunk
        chunk_index = st.session_state.current_index - st.session_state.chunk_start_idx
        if 0 <= chunk_index < len(st.session_state.current_chunk):
            return st.session_state.current_chunk.iloc[chunk_index]
    
    return None

def save_labeled_row(row_data: pd.Series, classification: str, features: List[str]):
    """Save a labeled row to the labeled_data dataframe."""
    # Create new row with selected columns plus labels
    new_row = {}
    
    # Add selected columns from original data
    for col in st.session_state.selected_columns:
        if col in row_data:
            new_row[col] = row_data[col]
    
    # Add metadata
    new_row['original_index'] = st.session_state.current_index
    new_row['label_classification'] = classification if classification else ""
    new_row['label_features'] = ', '.join(features) if features else ""
    
    # Convert to DataFrame and append
    new_row_df = pd.DataFrame([new_row])
    
    if st.session_state.labeled_data.empty:
        st.session_state.labeled_data = new_row_df
    else:
        st.session_state.labeled_data = pd.concat([st.session_state.labeled_data, new_row_df], ignore_index=True)
    
    # Increment auto-save counter and check for auto-save
    st.session_state.auto_save_counter += 1
    if st.session_state.auto_save_counter % 5 == 0:
        auto_save_to_feather()

def save_labeled_row_multi_task(row_data: pd.Series, classification_choices: Dict, feature_choices: Dict = None, features: List[str] = None):
    """Save a labeled row with multiple classification tasks and feature tasks to the labeled_data dataframe."""
    # Create new row with selected columns plus labels
    new_row = {}
    
    # Add selected columns from original data
    for col in st.session_state.selected_columns:
        if col in row_data:
            new_row[col] = row_data[col]
    
    # Add metadata
    new_row['original_index'] = st.session_state.current_index
    
    # Add each classification task as a separate column
    for task_key, choice in classification_choices.items():
        task_info = st.session_state.classification_tasks[task_key]
        task_col_name = f"{task_info['name']}_label"
        # Handle NaN and None values
        if pd.isna(choice) or choice is None or choice == "None":
            new_row[task_col_name] = ""
        else:
            new_row[task_col_name] = str(choice)
    
    # Add feature tasks
    if feature_choices:
        for task_key, selected_features in feature_choices.items():
            task_info = st.session_state.feature_tasks[task_key]
            task_col_name = f"{task_info['name']}_features"
            # Handle NaN and None values
            if pd.isna(selected_features) or selected_features is None:
                new_row[task_col_name] = ""
            elif isinstance(selected_features, (list, tuple)) and len(selected_features) > 0:
                new_row[task_col_name] = ', '.join(str(f) for f in selected_features if not pd.isna(f))
            else:
                new_row[task_col_name] = ""
    
    # Add legacy features (for backward compatibility)
    if features:
        # Handle NaN and None values
        if pd.isna(features) or features is None:
            new_row['label_features'] = ""
        elif isinstance(features, (list, tuple)) and len(features) > 0:
            new_row['label_features'] = ', '.join(str(f) for f in features if not pd.isna(f))
        else:
            new_row['label_features'] = ""
    
    # Convert to DataFrame and append
    new_row_df = pd.DataFrame([new_row])
    
    if st.session_state.labeled_data.empty:
        st.session_state.labeled_data = new_row_df
    else:
        st.session_state.labeled_data = pd.concat([st.session_state.labeled_data, new_row_df], ignore_index=True)
    
    # Increment auto-save counter and check for auto-save
    st.session_state.auto_save_counter += 1
    if st.session_state.auto_save_counter % 5 == 0:
        auto_save_to_feather()

def auto_save_to_feather():
    """Auto-save labeled data to feather file every 5 labels."""
    if not st.session_state.labeled_data.empty:
        try:
            filename = "tv_content_labeled_autosave.feather"
            st.session_state.labeled_data.to_feather(filename)
            st.toast(f"🔄 Auto-saved {len(st.session_state.labeled_data)} labels to {filename}", icon="💾")
        except Exception as e:
            st.toast(f"Auto-save failed: {str(e)}", icon="⚠️")

def is_item_fully_labeled(original_index):
    """Check if an item has all required classification tasks labeled (not None)."""
    if st.session_state.labeled_data.empty:
        return False
    
    # Find the row for this item
    item_rows = st.session_state.labeled_data[
        st.session_state.labeled_data['original_index'] == original_index
    ]
    
    if item_rows.empty:
        return False
    
    item_row = item_rows.iloc[0]
    
    # Check if all classification tasks have non-empty labels
    if st.session_state.classification_tasks:
        for task_key, task_info in st.session_state.classification_tasks.items():
            task_col_name = f"{task_info['name']}_label"
            if task_col_name not in item_row or not item_row[task_col_name] or item_row[task_col_name] == "None":
                return False
        return True
    else:
        # Backward compatibility for single classification
        if 'label_classification' in item_row:
            return item_row['label_classification'] is not None and item_row['label_classification'] != "None"
    
    return False

def save_current_selections():
    """Save current widget selections automatically (like original app)."""
    if not (st.session_state.file_path and st.session_state.current_index < st.session_state.total_rows):
        return
    
    current_id = st.session_state.current_index
    current_row = get_current_row()
    if current_row is None:
        return
    
    # Check if there are any selections to save
    has_selections = False
    classification_choices = {}
    feature_choices = {}
    
    # Check classification choices
    if st.session_state.classification_tasks:
        for task_key, task_info in st.session_state.classification_tasks.items():
            widget_key = f"classification_{current_id}_{task_key}"
            if widget_key in st.session_state:
                choice = st.session_state[widget_key]
                if choice and choice != "None":
                    classification_choices[task_key] = choice
                    has_selections = True
                else:
                    classification_choices[task_key] = None
    
    # Check feature choices
    if st.session_state.feature_tasks:
        for task_key, task_info in st.session_state.feature_tasks.items():
            selected_features = []
            for feature in task_info['labels']:
                feature_key = f"feature_{current_id}_{task_key}_{feature}"
                if feature_key in st.session_state and st.session_state[feature_key]:
                    selected_features.append(feature)
                    has_selections = True
            feature_choices[task_key] = selected_features
    
    # Save if there are any selections
    if has_selections:
        save_labeled_row_multi_task(current_row, classification_choices, feature_choices, [])

def save_current_label_if_exists():
    """Save current label before navigation if user has made selections."""
    # Check if we're on the labeling page and have classification/feature options
    if (st.session_state.data is not None and 
        st.session_state.selected_columns and
        (st.session_state.classification_labels or st.session_state.feature_labels)):
        
        current_id = st.session_state.current_index
        
        # Get current selections from session state keys
        classification = None
        features = []
        
        # Look for classification selection
        classification_key = f"classification_{current_id}"
        if classification_key in st.session_state:
            classification = st.session_state[classification_key]
        
        # Look for feature selections
        for feature in st.session_state.feature_labels:
            feature_key = f"feature_{feature}_{current_id}"
            if feature_key in st.session_state and st.session_state[feature_key]:
                features.append(feature)
        
        # Save if there are any selections
        if classification or features:
            # Lock labeling configuration on first save
            if not st.session_state.labeling_locked:
                st.session_state.labeling_locked = True
            
            # Remove existing label if it exists
            is_already_labeled = current_id in st.session_state.labeled_data['original_index'].values if not st.session_state.labeled_data.empty else False
            if is_already_labeled:
                st.session_state.labeled_data = st.session_state.labeled_data[
                    st.session_state.labeled_data['original_index'] != current_id
                ].reset_index(drop=True)
            
            # Get current row data and save
            current_row = get_current_row()
            if current_row is not None:
                save_labeled_row(current_row, classification, features)

def save_labeled_data_to_feather():
    """Save labeled data to a feather file."""
    if not st.session_state.labeled_data.empty:
        try:
            filename = "tv_content_labeled.feather"
            st.session_state.labeled_data.to_feather(filename)
            return filename
        except Exception as e:
            st.error(f"Error saving to feather: {str(e)}")
            return None
    return None

def load_data_file(file_path: str):
    """Load a data file, automatically choosing chunked or memory mode based on file size."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    if not file_path.lower().endswith('.feather'):
        raise ValueError("Only .feather files are supported")
    
    # Get file size info
    file_size = os.path.getsize(file_path)
    file_size_mb = file_size / (1024 * 1024)
    
    if file_size_mb > 200:  # Use chunked loading for large files
        # Chunked mode
        st.session_state.total_rows = get_total_rows(file_path)
        st.session_state.file_path = file_path
        st.session_state.current_index = 0
        st.session_state.chunk_start_idx = 0
        st.session_state.current_chunk = None
        st.session_state.data_mode = 'chunked'
        
        # Load first chunk to get column information
        first_chunk = load_chunk(file_path, 0, min(100, st.session_state.chunk_size))
        st.session_state.data = first_chunk  # For column selection
    else:
        # Memory mode
        full_data = pd.read_feather(file_path)
        st.session_state.data = full_data
        st.session_state.total_rows = len(full_data)
        st.session_state.file_path = file_path
        st.session_state.data_mode = 'memory'
        st.session_state.current_index = 0

def load_labeled_data_from_feather(file_path: str):
    """Load previously saved labeled data from feather file."""
    try:
        st.session_state.labeled_data = pd.read_feather(file_path)
        
        # Lock labeling if we loaded data
        if not st.session_state.labeled_data.empty:
            st.session_state.labeling_locked = True
        
        st.success(f"Loaded {len(st.session_state.labeled_data)} previously labeled items!")
    except Exception as e:
        st.error(f"Error loading feather file: {str(e)}")

def highlight_text(text: str, highlight_words: List[str], color: str = "#ff6b6b") -> str:
    """
    Highlight specified words in text with bold formatting and color.
    """
    if not highlight_words:
        return text
    
    # Create a pattern that matches any of the highlight words (case insensitive)
    pattern = '|'.join([re.escape(word) for word in highlight_words])
    if not pattern:
        return text
    
    def replace_func(match):
        return f'<span style="color: {color}; font-weight: bold;">{match.group()}</span>'
    
    highlighted = re.sub(f'({pattern})', replace_func, text, flags=re.IGNORECASE)
    return highlighted

def beautify_text(text: str) -> str:
    """
    Clean up text by replacing newlines with spaces and removing extra whitespace.
    """
    if pd.isna(text):
        return ""
    
    # Replace newlines with spaces
    cleaned = re.sub(r'\n+', ' ', str(text))
    
    # Replace multiple spaces with single spaces
    cleaned = re.sub(r'\s+', ' ', cleaned)
    
    # Strip leading/trailing whitespace
    cleaned = cleaned.strip()
    
    return cleaned

def save_labels_to_file():
    """Save current labels to a JSON file."""
    if st.session_state.labels:
        labels_json = json.dumps(st.session_state.labels, indent=2)
        return labels_json
    return None

def load_labels_from_file(uploaded_file):
    """Load labels from an uploaded JSON file."""
    try:
        labels_data = json.load(uploaded_file)
        st.session_state.labels.update(labels_data)
        st.success(f"Loaded {len(labels_data)} labels from file!")
    except Exception as e:
        st.error(f"Error loading labels: {str(e)}")

# ===================== MINIMAL SIDEBAR =====================
st.sidebar.header("⚙️ Configuration")

# Configuration file upload
uploaded_config = st.sidebar.file_uploader("Upload config file (.cfg)", type=['cfg'])
if uploaded_config is not None:
    # Save uploaded file temporarily and load it
    with open('temp_config.cfg', 'wb') as f:
        f.write(uploaded_config.getbuffer())
    
    data_config, highlighting_config, classification_tasks, feature_tasks, legacy_features, output_config = load_config('temp_config.cfg')
    st.session_state.classification_tasks = classification_tasks
    st.session_state.feature_tasks = feature_tasks
    
    # Load highlighting words from config
    if highlighting_config and 'words' in highlighting_config:
        st.session_state.highlight_words = highlighting_config['words']
    
    # Store output configuration
    if output_config:
        st.session_state.output_config = output_config
    
    # Load legacy features
    if legacy_features and 'labels' in legacy_features:
        st.session_state.feature_labels = legacy_features['labels']
    
    # Try to auto-load data file if specified in config
    if data_config and 'file_path' in data_config:
        data_file_path = data_config['file_path']
        try:
            if os.path.exists(data_file_path):
                data = load_data_file(data_file_path)
                if data is not None:
                    st.session_state.data = data
                    st.session_state.file_path = data_file_path
                    st.session_state.total_rows = len(data)
                    st.session_state.current_index = 0
                    
                    # Set text column from config
                    text_column = data_config.get('text_column', 'content')
                    if text_column in st.session_state.data.columns:
                        st.session_state.text_column = text_column
                        # Set selected columns from output config if available
                        if output_config and 'columns' in output_config and output_config['columns']:
                            # Filter output columns to only include ones that exist in the data
                            valid_columns = [col for col in output_config['columns'] if col in st.session_state.data.columns]
                            st.session_state.selected_columns = valid_columns
                        else:
                            st.session_state.selected_columns = [text_column]
                    else:
                        # Fall back to first text column
                        text_columns = [col for col in st.session_state.data.columns 
                                      if st.session_state.data[col].dtype == 'object']
                        if text_columns:
                            st.session_state.text_column = text_columns[0]
                            st.session_state.selected_columns = [text_columns[0]]
                    
                    st.sidebar.success(f"✅ Auto-loaded {len(st.session_state.data)} records!")
        except Exception as e:
            st.sidebar.error(f"Could not load data file: {str(e)}")
    
    # Remove temp file
    if os.path.exists('temp_config.cfg'):
        os.remove('temp_config.cfg')
    
    st.sidebar.success("✅ Configuration loaded successfully!")
    st.rerun()

# Display current configuration status
if st.session_state.classification_tasks:
    st.sidebar.write("**Classification Tasks:**")
    for task_key, task_info in st.session_state.classification_tasks.items():
        st.sidebar.write(f"- {task_info['name']}: {len(task_info['labels'])} labels")

if st.session_state.feature_tasks:
    st.sidebar.write("**Feature Tasks:**")
    for task_key, task_info in st.session_state.feature_tasks.items():
        st.sidebar.write(f"- {task_info['name']}: {len(task_info['labels'])} options")

# Text highlighting configuration
st.sidebar.subheader("🎨 Text Highlighting")
highlight_words_input = st.sidebar.text_area(
    "Words to highlight (comma-separated)",
    value=",".join(st.session_state.highlight_words),
    height=60,
    help="Enter words to highlight, separated by commas"
)

highlight_color = st.sidebar.color_picker(
    "Highlight color",
    value="#ece800",
    help="Choose the color for highlighted words"
)

if highlight_words_input:
    st.session_state.highlight_words = [
        word.strip() for word in highlight_words_input.split(',') 
        if word.strip()
    ]

# Main content area
st.title("📺 TV Content Labeling Tool")

# Save functionality in main window
if not st.session_state.labeled_data.empty:
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        # Auto-save on navigation
        if st.button("💾 Save Progress", type="primary", use_container_width=True):
            filename = save_labeled_data_to_feather()
            if filename:
                st.success(f"✅ Saved {len(st.session_state.labeled_data)} labels to {filename}")
    
    with col2:
        # Download button that automatically saves first
        if not st.session_state.labeled_data.empty:
            # Save to file for download
            filename = save_labeled_data_to_feather()
            if filename and os.path.exists(filename):
                with open(filename, "rb") as file:
                    st.download_button(
                        label="📥 Download File",
                        data=file.read(),
                        file_name="tv_content_labeled.feather",
                        mime="application/octet-stream",
                        use_container_width=True
                    )
            else:
                st.button("📥 Download File", disabled=True, use_container_width=True, help="Error creating file")
        else:
            st.button("📥 Download File", disabled=True, use_container_width=True, help="No labels to download")
    
    with col3:
        # Auto-save status placeholder
        pass
    
    st.markdown("---")

if st.session_state.data is None:
    st.info("� No data loaded. Please select a .feather file above or upload a config file with data path.")
    
    # Create columns for better layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### 🚀 How to use this tool:
        
        #### For Large Files (recommended for GB files):
        1. **Copy file path**: Right-click on your feather file → "Copy as path"
        2. **Paste path**: Paste the full path in the sidebar text input
        3. **Load file**: Click "📁 Load File" button
        
        #### For Small Files (< 200MB):
        1. **Upload file**: Use the file uploader in the sidebar
        
        #### Then continue with:
        5. **Select Column**: Choose the column with text content to label
        6. **Configure Highlighting**: Add words you want to highlight in the text
        7. **Set Up Labels**: Define classification and feature labeling options
        8. **Start Labeling**: Navigate through your data and apply labels
        9. **Export Results**: Download your labels when finished
        
        ### 🔄 Auto-Save Features:
        - **Auto-save on navigation**: Labels are automatically saved when you navigate (Previous/Next/Jump)
        - **Auto-save to file**: Every 5th label triggers an automatic save to `tv_content_labeled_autosave.feather`
        - **Auto-save**: Labels are automatically saved when you navigate to another item
        """)
    
    with col2:
        st.markdown("""
        ### 💡 Tips for Large Files:
        
        - **Windows**: Use `Shift + Right-click` → "Copy as path"
        - **File Explorer**: Copy from address bar
        - **Network paths**: Use UNC format `\\\\server\\share\\file.feather`
        - **Memory**: Files up to several GB should work fine
        - **Performance**: Feather files load much faster than CSV
        
        ### 📁 Supported Formats:
        - Feather files (`.feather`)
        - Apache Arrow format
        - Optimized for fast I/O
        """)
    
    # Show sample data info
    st.markdown("---")
    st.markdown("### 📊 Try with Sample Data")
    st.markdown("Want to test the app? Try loading the included `sample_data.feather` file from your project directory.")
    
    # Get current directory for sample file
    current_dir = os.getcwd()
    sample_file = os.path.join(current_dir, "sample_data.feather")
    if os.path.exists(sample_file):
        st.code(sample_file)
        st.markdown("*Copy the path above and paste it in the sidebar*")
else:
    # Check if we have the necessary configuration
    if not st.session_state.selected_columns:
        st.warning("⚠️ Please select columns to include in the labeled dataset")
        st.stop()
    
    # Get current row data
    current_row = get_current_row()
    
    if current_row is None:
        st.error("❌ Could not load current row data")
        st.stop()
    
    current_text = current_row[st.session_state.text_column] if st.session_state.text_column in current_row else ""
    
    # Check if this item is already labeled
    current_id = st.session_state.current_index
    is_already_labeled = current_id in st.session_state.labeled_data['original_index'].values if not st.session_state.labeled_data.empty else False
    
    if is_already_labeled:
        existing_label_row = st.session_state.labeled_data[st.session_state.labeled_data['original_index'] == current_id].iloc[0]
        existing_classification = existing_label_row.get('label_classification', '')
        existing_features_value = existing_label_row.get('label_features', '')
        # Handle NaN values from pandas
        if pd.isna(existing_features_value) or existing_features_value is None:
            existing_features = []
        else:
            existing_features = existing_features_value.split(', ') if existing_features_value else []
    
    # Beautify and highlight text
    beautified_text = beautify_text(current_text)
    highlighted_text = highlight_text(beautified_text, st.session_state.highlight_words, highlight_color)
    
    # Display text content
    st.subheader("📄 Content")
    
    # Show original vs beautified toggle
    show_original = st.checkbox("Show original formatting", value=False)
    
    if show_original:
        st.markdown("**Original text:**")
        st.text_area("", value=current_text, height=200, disabled=True, key="original_text")
    else:
        st.markdown("**Formatted text:**")
        bg_color = st.get_option('theme.backgroundColor')
        border_color = st.get_option('theme.borderColor')
        text_color = st.get_option('theme.textColor')        
        st.markdown(f'<div style="border: 1px solid {border_color}; padding: 15px; border-radius: 5px; background-color: {bg_color}; line-height: 1.6; color: {text_color};">{highlighted_text}</div>', unsafe_allow_html=True)
    
    # Show additional column data if selected
    if len(st.session_state.selected_columns) > 1:
        with st.expander("📋 Additional Column Data"):
            for col in st.session_state.selected_columns:
                if col != st.session_state.text_column and col in current_row:
                    st.write(f"**{col}:** {current_row[col]}")
    
    # Labeling section
    st.subheader("🏷️ Labels")
    
    # Initialize current labels
    current_classification = None
    current_features = []
    
    # Load existing labels if already labeled
    if is_already_labeled:
        current_classification = existing_classification if existing_classification else None
        current_features = [f for f in existing_features if f]
    
    # Combined labeling section (classification tasks + features in same row)
    
    # Calculate total columns needed (classification tasks + feature tasks + legacy features)
    num_classification_tasks = len(st.session_state.classification_tasks) if st.session_state.classification_tasks else 0
    num_feature_tasks = len(st.session_state.feature_tasks) if st.session_state.feature_tasks else 0
    num_legacy_features = 1 if st.session_state.feature_labels else 0
    total_cols = num_classification_tasks + num_feature_tasks + num_legacy_features
    
    if total_cols > 0:
        # Create columns for all labeling tasks
        all_cols = st.columns(total_cols)
        classification_choices = {}
        
        # Classification tasks first
        col_idx = 0
        if st.session_state.classification_tasks:
            for task_key, task_info in st.session_state.classification_tasks.items():
                with all_cols[col_idx]:
                    st.markdown(f"**{task_info['name']}**")
                    st.caption("(pick one)")
                    
                    # Get current value for this task from existing labels (simplified like original)
                    current_task_value = None
                    if is_already_labeled:
                        # Look for existing label for this task
                        existing_row = st.session_state.labeled_data[
                            st.session_state.labeled_data['original_index'] == current_id
                        ]
                        if not existing_row.empty:
                            task_col_name = f"{task_info['name']}_label"
                            if task_col_name in existing_row.columns:
                                current_task_value = existing_row.iloc[0][task_col_name]
                                # Handle NaN values
                                if pd.isna(current_task_value):
                                    current_task_value = None
                    
                    # Calculate default index directly (like original app)
                    if current_task_value and current_task_value in task_info['labels']:
                        default_index = task_info['labels'].index(current_task_value) + 1
                    else:
                        default_index = 0
                    
                    # Create radio button with simple key (exactly like original)
                    classification_choices[task_key] = st.radio(
                        f"Select {task_info['name']}",
                        options=[None] + task_info['labels'],
                        index=default_index,
                        format_func=lambda x: "None" if x is None else x,
                        key=f"classification_{current_id}_{task_key}",
                        label_visibility="collapsed"
                    )
                    col_idx += 1
        
        # Feature tasks
        feature_choices = {}
        if st.session_state.feature_tasks:
            for task_key, task_info in st.session_state.feature_tasks.items():
                with all_cols[col_idx]:
                    st.markdown(f"**{task_info['name']}**")
                    st.caption("(pick zero or more)")
                    
                    # Get current values for this feature task from existing labels
                    current_task_features = []
                    if is_already_labeled:
                        # Look for existing features for this task
                        existing_row = st.session_state.labeled_data[
                            st.session_state.labeled_data['original_index'] == current_id
                        ]
                        if not existing_row.empty:
                            task_col_name = f"{task_info['name']}_features"
                            if task_col_name in existing_row.columns:
                                existing_features_str = existing_row.iloc[0][task_col_name]
                                # Handle NaN values from pandas
                                if pd.isna(existing_features_str) or existing_features_str is None:
                                    current_task_features = []
                                elif existing_features_str:
                                    current_task_features = [f.strip() for f in existing_features_str.split(',') if f.strip()]
                                else:
                                    current_task_features = []
                    
                    # Create checkboxes for this feature task
                    selected_task_features = []
                    for feature in task_info['labels']:
                        feature_key = f"feature_{current_id}_{task_key}_{feature}"
                        
                        # Initialize session state if not exists
                        if feature_key not in st.session_state:
                            st.session_state[feature_key] = feature in current_task_features
                        
                        if st.checkbox(
                            feature,
                            key=feature_key
                        ):
                            selected_task_features.append(feature)
                    
                    feature_choices[task_key] = selected_task_features
                    col_idx += 1
    
    # Save current labels button
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("💾 Save Current Labels", type="primary", use_container_width=True):
            current_id = st.session_state.current_index
            current_row = get_current_row()
            
            if current_row is not None:
                # Collect current widget values from config-driven tasks only
                classification_choices = {}
                feature_choices = {}
                
                # Collect classification choices
                if st.session_state.classification_tasks:
                    for task_key, task_info in st.session_state.classification_tasks.items():
                        widget_key = f"classification_{current_id}_{task_key}"
                        if widget_key in st.session_state:
                            choice = st.session_state[widget_key]
                            if choice and choice != "None":
                                classification_choices[task_key] = choice
                            else:
                                classification_choices[task_key] = None
                
                # Collect feature choices
                if st.session_state.feature_tasks:
                    for task_key, task_info in st.session_state.feature_tasks.items():
                        selected_features = []
                        for feature in task_info['labels']:
                            feature_key = f"feature_{current_id}_{task_key}_{feature}"
                            if feature_key in st.session_state and st.session_state[feature_key]:
                                selected_features.append(feature)
                        feature_choices[task_key] = selected_features
                
                # Save the labels (config-driven tasks only)
                save_labeled_row_multi_task(current_row, classification_choices, feature_choices, [])
                st.success("✅ Labels saved!")
                st.rerun()

    # Navigation controls
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        if st.button("⬅️ Previous", disabled=st.session_state.current_index <= 0, key="nav_previous"):
            # Save current state before navigation (like original app)
            save_current_selections()
            st.session_state.current_index -= 1
            st.rerun()
    
    with col2:
        st.markdown(f"<div style='text-align: center;'><h3>Item {st.session_state.current_index + 1:,} of {st.session_state.total_rows:,}</h3></div>", unsafe_allow_html=True)
    
    with col3:
        if st.button("Next ➡️", disabled=st.session_state.current_index >= st.session_state.total_rows - 1, key="nav_next"):
            # Save current state before navigation (like original app)
            save_current_selections()
            st.session_state.current_index += 1
            st.rerun()

    # Progress bar
    progress = (st.session_state.current_index + 1) / st.session_state.total_rows
    
    # Quick navigation
    st.subheader("🔍 Quick Navigation")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🏠 Go to Start", key="nav_start"):
            save_current_selections()
            st.session_state.current_index = 0
            st.rerun()
    
    with col2:
        if st.button("🎯 Go to End", key="nav_end"):
            save_current_selections()
            st.session_state.current_index = st.session_state.total_rows - 1
            st.rerun()
            st.rerun()
    
    with col3:
        # Find next unlabeled item (where not all classification tasks are labeled)
        if not st.session_state.labeled_data.empty:
            # Get all items that are not fully labeled (missing classification labels)
            unlabeled_indices = [i for i in range(st.session_state.total_rows) if not is_item_fully_labeled(i)]
            
            if unlabeled_indices:
                next_unlabeled = min([i for i in unlabeled_indices if i > st.session_state.current_index], default=min(unlabeled_indices))
                if st.button(f"➡️ Next Unlabeled ({next_unlabeled + 1:,})", key="nav_next_unlabeled"):
                    save_current_selections()
                    st.session_state.current_index = next_unlabeled
                    st.rerun()
        else:
            if st.button("➡️ Next Unlabeled", key="nav_next_unlabeled_fallback"):
                save_current_selections()
                st.session_state.current_index = min(st.session_state.current_index + 1, st.session_state.total_rows - 1)
                st.rerun()
    
    # Jump to specific item
    jump_to = st.number_input(
        "Jump to item:",
        min_value=1,
        max_value=st.session_state.total_rows,
        value=st.session_state.current_index + 1
    )
    
    if st.button("🔄 Jump", key="nav_jump"):
        save_current_selections()
        st.session_state.current_index = jump_to - 1
        st.rerun()
