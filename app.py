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
import shutil
import hashlib

def update_classification_label(task_key):
    """Callback to update DataFrame when classification widget changes"""
    current_id = st.session_state.current_index
    widget_key = f"classification_{current_id}_{task_key}"
    
    if widget_key not in st.session_state:
        return
        
    new_value = st.session_state[widget_key]
    
    # Get task info
    task_info = st.session_state.classification_tasks[task_key]
    col_name = f"{task_info['name']}_label"
    
    # Ensure row exists in DataFrame - this is the ONLY place where rows are created
    if current_id not in st.session_state.labeled_data.index:
        # Create new row with current item data
        current_row = get_current_row()
        if current_row is not None:
            # Build complete row data
            new_row_data = {}
            
            # Add selected columns from original data
            for col in st.session_state.selected_columns:
                if col in current_row:
                    new_row_data[col] = current_row[col]
            
            # Initialize ALL possible label columns to empty strings
            for tk, ti in st.session_state.classification_tasks.items():
                new_row_data[f"{ti['name']}_label"] = ""
            for tk, ti in st.session_state.feature_tasks.items():
                new_row_data[f"{ti['name']}_features"] = ""
            
            # Create the complete row as a DataFrame and add it
            new_row_df = pd.DataFrame([new_row_data], index=[current_id])
            new_row_df.index.name = 'original_index'
            
            # Add the new row - no need for double-check since we already verified it doesn't exist
            st.session_state.labeled_data = pd.concat([st.session_state.labeled_data, new_row_df])
    
    # Update the specific label column
    st.session_state.labeled_data.loc[current_id, col_name] = new_value if new_value is not None else ""
    
    # Auto-save every 5 updates
    st.session_state.auto_save_counter += 1
    if st.session_state.auto_save_counter % 5 == 0:
        save_to_feather()

def update_feature_selection(task_key, feature_name):
    """Callback to update DataFrame when feature checkbox changes"""
    current_id = st.session_state.current_index
    feature_key = f"feature_{current_id}_{task_key}_{feature_name}"
    
    if feature_key not in st.session_state:
        return
    
    # Get task info
    task_info = st.session_state.feature_tasks[task_key]
    col_name = f"{task_info['name']}_features"
    
    # Ensure row exists in DataFrame
    if current_id not in st.session_state.labeled_data.index:
        # Create new row with current item data
        current_row = get_current_row()
        if current_row is not None:
            # Build complete row data
            new_row_data = {}
            
            # Add selected columns from original data
            for col in st.session_state.selected_columns:
                if col in current_row:
                    new_row_data[col] = current_row[col]
            
            # Initialize ALL possible label columns to empty strings
            for tk, ti in st.session_state.classification_tasks.items():
                new_row_data[f"{ti['name']}_label"] = ""
            for tk, ti in st.session_state.feature_tasks.items():
                new_row_data[f"{ti['name']}_features"] = ""
            
            # Create the complete row as a DataFrame and add it
            new_row_df = pd.DataFrame([new_row_data], index=[current_id])
            new_row_df.index.name = 'original_index'
            
            # Add the new row - no need for double-check since we already verified it doesn't exist
            st.session_state.labeled_data = pd.concat([st.session_state.labeled_data, new_row_df])
    
    # Collect all selected features for this task
    selected_features = []
    for feature in task_info['labels']:
        fkey = f"feature_{current_id}_{task_key}_{feature}"
        if fkey in st.session_state and st.session_state[fkey]:
            selected_features.append(feature)
    
    # Update the features column
    features_str = ', '.join(selected_features) if selected_features else ""
    st.session_state.labeled_data.loc[current_id, col_name] = features_str
    
    # Auto-save every 5 updates
    st.session_state.auto_save_counter += 1
    if st.session_state.auto_save_counter % 5 == 0:
        save_to_feather()

def reset_widgets_to_dataframe_values():
    """Reset all widget values to match the current item's values in the DataFrame"""
    current_id = st.session_state.current_index
    
    # Clear all widget keys for current item to force reset
    keys_to_clear = []
    for key in st.session_state.keys():
        if (key.startswith(f"classification_{current_id}_") or 
            key.startswith(f"feature_{current_id}_")):
            keys_to_clear.append(key)
    
    for key in keys_to_clear:
        del st.session_state[key]

def login_screen():
    """Display login screen using Streamlit's built-in authentication"""
    st.header("🔐 Content Labeling Tool")
    st.subheader("This app requires authentication.")
    st.write("Please log in with your Google account to access the labeling tool.")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.button("🚀 Log in with Google", on_click=st.login, use_container_width=True)

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
            'filename': config['output'].get('output_base_filename', 'labeled_data.feather'),
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
    page_title="Content Labeling Tool",
    page_icon="🏷️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Authentication check
if not st.user.is_logged_in:
    login_screen()
    st.stop()

# User is logged in, get user info
user_info = st.user
username = user_info.email.split('@')[0]  # Use part before @ as username
name = user_info.name or username

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
    # Initialize with proper index structure
    st.session_state.labeled_data = pd.DataFrame().set_index(pd.Index([], name='original_index'))
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
if 'auto_save_counter' not in st.session_state:
    st.session_state.auto_save_counter = 0


def get_user_filename(base_filename: str) -> str:
    """Generate user-specific filename using a hash of the user's email."""
    if st.user.is_logged_in:
        # Create a short hash of the email for the filename
        user_hash = hashlib.md5(st.user.email.encode()).hexdigest()[:8]
        username = f"user_{user_hash}"
        
        # Split filename and add username prefix
        name, ext = os.path.splitext(base_filename)
        return f"{username}_{name}{ext}"
    return base_filename

def get_output_filename() -> str:
    """Get the configured output filename, or default if not configured."""
    if hasattr(st.session_state, 'output_config') and st.session_state.output_config and st.session_state.output_config.get('filename'):
        return st.session_state.output_config['filename']
    return "content_labeled.feather"  # Default fallback

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
                    
                    # Auto-load previously saved labeled data if it exists
                    labeled_file = get_user_filename(get_output_filename())
                    if os.path.exists(labeled_file):
                        try:
                            loaded_df = pd.read_feather(labeled_file)
                            # Convert to proper indexed DataFrame
                            if 'original_index' in loaded_df.columns:
                                st.session_state.labeled_data = loaded_df.set_index('original_index')
                            else:
                                loaded_df['original_index'] = loaded_df.index
                                st.session_state.labeled_data = loaded_df.set_index('original_index')
                        except Exception:
                            # If loading fails, continue with empty labeled_data
                            pass
                            
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

# OLD FUNCTION - No longer used, replaced by callback-based updates
# def save_labeled_row_multi_task(row_data: pd.Series, classification_choices: Dict, feature_choices: Dict = None, features: List[str] = None):
#     """Save a labeled row with multiple classification tasks and feature tasks to the labeled_data dataframe."""
#     # This function has been replaced by update_classification_label and update_feature_selection callbacks
#     # that update the DataFrame immediately when widgets change

def save_to_feather():
    """Save labeled data to main feather file, and backup previous version."""
    if not st.session_state.labeled_data.empty:
        try:
            base_filename = get_output_filename()
            main_filename = get_user_filename(base_filename)
            # Create backup filename by inserting .bak before extension
            name, ext = os.path.splitext(base_filename)
            backup_base = f"{name}.bak{ext}"
            backup_filename = get_user_filename(backup_base)
            # If main file exists, copy to backup
            if os.path.exists(main_filename):
                shutil.copy2(main_filename, backup_filename)
            # Save current labeled data to main file
            # Reset index to make original_index a regular column for saving
            df_to_save = st.session_state.labeled_data.reset_index()
            df_to_save.to_feather(main_filename)
            return main_filename
        except Exception as e:
            return None
    return None

def is_item_fully_labeled(original_index):
    """Check if an item has all required classification tasks labeled (not None)."""
    if st.session_state.labeled_data.empty:
        return False
    
    # Use index-based access for better performance
    if original_index not in st.session_state.labeled_data.index:
        return False
    
    item_row = st.session_state.labeled_data.loc[original_index]
    
    # Check if all classification tasks have non-empty labels
    if st.session_state.classification_tasks:
        for task_key, task_info in st.session_state.classification_tasks.items():
            task_col_name = f"{task_info['name']}_label"
            if (task_col_name not in item_row.index or 
                pd.isna(item_row[task_col_name]) or 
                item_row[task_col_name] == "" or 
                item_row[task_col_name] == "None"):
                return False
        return True
    else:
        # Backward compatibility for single classification
        if 'label_classification' in item_row.index:
            value = item_row['label_classification']
            return not pd.isna(value) and value != "" and value != "None"
    
    return False

# OLD FUNCTION - No longer used, replaced by callback-based updates
# def save_current_selections():
#     """Save current widget selections automatically (like original app)."""
#     # This function is no longer needed since widgets now update the DataFrame directly via callbacks

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
        loaded_df = pd.read_feather(file_path)
        
        # Convert to proper indexed DataFrame
        if 'original_index' in loaded_df.columns:
            # Set original_index as the index
            st.session_state.labeled_data = loaded_df.set_index('original_index')
        else:
            # For backward compatibility, use integer index as original_index
            loaded_df['original_index'] = loaded_df.index
            st.session_state.labeled_data = loaded_df.set_index('original_index')
        
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
st.sidebar.header("👤 User Info")
st.sidebar.success(f"✅ Logged in as **{name}**")
st.sidebar.caption(f"📧 {st.user.email}")
if st.sidebar.button("🚪 Logout", use_container_width=True):
    st.logout()

# Save & Download Section
if not st.session_state.labeled_data.empty:
    #st.sidebar.markdown("---")
    st.sidebar.subheader("💾 Save & Export")
    
    # Save button
    if st.sidebar.button("💾 Save Progress", type="primary", use_container_width=True):
        filename = save_to_feather()
        if filename:
            st.sidebar.success(f"✅ Saved {len(st.session_state.labeled_data)} labels")
        else:
            st.sidebar.error("❌ Save failed.")
    
    # Download button
    filename = get_user_filename(get_output_filename())
    if os.path.exists(filename):
        with open(filename, "rb") as file:
            st.sidebar.download_button(
                label="📥 Download Results",
                data=file.read(),
                file_name=get_user_filename(get_output_filename()),
                mime="application/octet-stream",
                use_container_width=True,
                help="Download your labeled data as a feather file"
            )
    else:
        st.sidebar.button("📥 Download Results", disabled=True, use_container_width=True, help="Save progress first to enable download")
    
    # Show progress info
    st.sidebar.caption(f"📊 {len(st.session_state.labeled_data)} labels saved")


# Configuration sidebar section
#st.sidebar.markdown("---")
st.sidebar.header("⚙️ Configuration")
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
    #st.sidebar.write("**Classification Tasks:**")
    with st.sidebar.expander("Classes and Features"):
        if st.session_state.classification_tasks:
            #st.markdown("**Classification Tasks**")
            class_items = []
            for task_key, task_info in st.session_state.classification_tasks.items():
                class_items.append(f"{task_info['name']}: {len(task_info['labels'])} classes")
            #st.markdown("**Classification Tasks**<br>"+"<br>".join([f"<span style='font-size: 0.8em;'>{item}</span>" for item in class_items]), unsafe_allow_html=True)
            st.markdown(f"""<div style="line-height: 0.2;"><u>Classification Tasks</u></div><div style="font-size: 0.8em; line-height: 1.18;"><br>{"<br>".join(class_items)}</div>""", unsafe_allow_html=True)

if st.session_state.feature_tasks:
            st.write("**Feature Tasks**")
            feature_items = []
            for task_key, task_info in st.session_state.feature_tasks.items():
                feature_items.append(f"{task_info['name']}: {len(task_info['labels'])} options")
            st.markdown(f"""<div style="line-height: 0.2;"><u>Classification Tasks</u></div><div style="font-size: 0.8em; line-height: 1.18;"><br>{"<br>".join(class_items)}</div>""", unsafe_allow_html=True)


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

# App Theme Selection
st.sidebar.subheader("🎨 App Theme")
with st.sidebar.expander("Theme Settings Help"):
    st.markdown("""
    **Theme Control**: Use Streamlit's built-in theme settings:
    1. Click the **☰** hamburger menu (top right)
    2. Go to **Settings** → **Theme**
    3. Choose: **Light**, **Dark**, or **Auto**

    *Auto mode follows your system preference*
    
    💡 **Tip**: You can also add `?theme=dark` or `?theme=light` to the URL
    """)


# Main content area
# Reduce top padding with custom CSS
st.markdown("""
<style>
    .main .block-container {
        padding-top: 0.5rem;
        padding-bottom: 1rem;
    }
    .small-button {
        font-size: 0.75rem !important;
        padding: 0.2rem 0.4rem !important;
        height: 2rem !important;
    }
    .compact-section {
        margin-bottom: 0.5rem !important;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 style="margin-top: 0; margin-bottom: 1rem;">🏷️ Content Labeling Tool</h1>', unsafe_allow_html=True)

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
        - **Auto-save to file**: Every 5th label triggers an automatic save to your configured output file
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
    is_already_labeled = current_id in st.session_state.labeled_data.index if not st.session_state.labeled_data.empty else False
    
    # Beautify and highlight text
    beautified_text = beautify_text(current_text)
    highlighted_text = highlight_text(beautified_text, st.session_state.highlight_words, highlight_color)
    
    
    # Show formatted text only (removed toggle)
    st.markdown("**Formatted text:**")
    
    # Use Streamlit's container with larger height to minimize scrolling
    # Use st.text() to avoid markdown interpretation
    with st.container(height=300, border=True):
        if st.session_state.highlight_words:
            # For highlighting, we need to use markdown but escape it first
            import html
            escaped_text = html.escape(beautified_text)
            # Re-apply highlighting to escaped text
            highlighted_escaped_text = highlight_text(escaped_text, st.session_state.highlight_words, highlight_color)
            st.markdown(highlighted_escaped_text, unsafe_allow_html=True)
        else:
            # No highlighting needed, use plain text to avoid markdown interpretation
            st.text(beautified_text)
    
    # Show additional column data if selected
    if len(st.session_state.selected_columns) > 1:
        with st.expander("📋 Additional Column Data"):
            for col in st.session_state.selected_columns:
                if col != st.session_state.text_column and col in current_row:
                    st.write(f"**{col}:** {current_row[col]}")
    
    # Compact centered navigation
    _, col_nav, _ = st.columns([1, 2, 1])
    
    with col_nav:
        # All navigation in a single compact row
        nav_cols = st.columns([1, 1, 2, 1, 1, 1, 1.5, 1])
        
        with nav_cols[0]:
            if st.button("⏮️", disabled=st.session_state.current_index <= 0, key="nav_start", help="First"):
                st.session_state.current_index = 0
                reset_widgets_to_dataframe_values()
                st.rerun()
        
        with nav_cols[1]:
            if st.button("◀️", disabled=st.session_state.current_index <= 0, key="nav_previous", help="Previous"):
                st.session_state.current_index -= 1
                reset_widgets_to_dataframe_values()
                st.rerun()
        
        with nav_cols[2]:
            st.markdown(f"<div style='text-align: center; line-height: 2.3; font-weight: bold;'>{st.session_state.current_index + 1:,} / {st.session_state.total_rows:,}</div>", unsafe_allow_html=True)
        
        with nav_cols[3]:
            if st.button("▶️", disabled=st.session_state.current_index >= st.session_state.total_rows - 1, key="nav_next", help="Next"):
                st.session_state.current_index += 1
                reset_widgets_to_dataframe_values()
                st.rerun()
        
        with nav_cols[4]:
            if st.button("⏭️", disabled=st.session_state.current_index >= st.session_state.total_rows - 1, key="nav_end", help="Last"):
                st.session_state.current_index = st.session_state.total_rows - 1
                reset_widgets_to_dataframe_values()
                st.rerun()
        
        with nav_cols[5]:
            # Find next unlabeled item
            if not st.session_state.labeled_data.empty:
                try:
                    unlabeled_indices = [i for i in range(st.session_state.total_rows) if not is_item_fully_labeled(i)]
                    if unlabeled_indices:
                        next_unlabeled = min([i for i in unlabeled_indices if i > st.session_state.current_index], default=min(unlabeled_indices))
                        if st.button("🔍", key="nav_next_unlabeled", help="Next Unlabeled"):
                            st.session_state.current_index = next_unlabeled
                            reset_widgets_to_dataframe_values()
                            st.rerun()
                    else:
                        st.button("✅", disabled=True, key="nav_all_labeled", help="All Labeled")
                except Exception as e:
                    # Fallback if there's an error in checking labeled status
                    if st.button("🔍", key="nav_next_unlabeled_error", help="Next Unlabeled"):
                        st.session_state.current_index = min(st.session_state.current_index + 1, st.session_state.total_rows - 1)
                        reset_widgets_to_dataframe_values()
                        st.rerun()
            else:
                if st.button("🔍", key="nav_next_unlabeled_fallback", help="Next Unlabeled"):
                    st.session_state.current_index = min(st.session_state.current_index + 1, st.session_state.total_rows - 1)
                    reset_widgets_to_dataframe_values()
                    st.rerun()
        
        with nav_cols[6]:
            jump_to = st.number_input("Jump", min_value=1, max_value=st.session_state.total_rows, value=st.session_state.current_index + 1, key="jump_input", label_visibility="collapsed")
        
        with nav_cols[7]:
            if st.button("🎯", key="nav_jump", help="Jump to item"):
                st.session_state.current_index = jump_to - 1
                reset_widgets_to_dataframe_values()
                st.rerun()

    # Labeling section
    st.subheader("🏷️ Labels")
    
    # Initialize current labels (these are no longer used since we get values directly from DataFrame)
    current_classification = None
    current_features = []
    
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
                    
                    # Get current value for this task from existing labels using indexed access
                    current_task_value = None
                    if is_already_labeled and current_id in st.session_state.labeled_data.index:
                        task_col_name = f"{task_info['name']}_label"
                        if task_col_name in st.session_state.labeled_data.columns:
                            try:
                                current_task_value = st.session_state.labeled_data.loc[current_id, task_col_name]
                                # Ensure we have a scalar value, not a Series
                                if isinstance(current_task_value, pd.Series):
                                    current_task_value = current_task_value.iloc[0] if len(current_task_value) > 0 else None
                                # Handle NaN values - use pd.isna() on scalar values only
                                if pd.isna(current_task_value) or str(current_task_value) in ["", "nan", "None"]:
                                    current_task_value = None
                            except (KeyError, IndexError):
                                current_task_value = None
                    
                    # Calculate default index directly (like original app)
                    if current_task_value is not None and str(current_task_value) in task_info['labels']:
                        default_index = task_info['labels'].index(str(current_task_value)) + 1
                    else:
                        default_index = 0
                    
                    # Create radio button with callback
                    classification_choices[task_key] = st.radio(
                        f"Select {task_info['name']}",
                        options=[None] + task_info['labels'],
                        index=default_index,
                        format_func=lambda x: "None" if x is None else x,
                        key=f"classification_{current_id}_{task_key}",
                        label_visibility="collapsed",
                        on_change=update_classification_label,
                        args=(task_key,)
                    )
                    col_idx += 1
        
        # Feature tasks
        feature_choices = {}
        if st.session_state.feature_tasks:
            for task_key, task_info in st.session_state.feature_tasks.items():
                with all_cols[col_idx]:
                    st.markdown(f"**{task_info['name']}**")
                    st.caption("(pick zero or more)")
                    
                    # Get current values for this feature task from existing labels using indexed access
                    current_task_features = []
                    if is_already_labeled and current_id in st.session_state.labeled_data.index:
                        task_col_name = f"{task_info['name']}_features"
                        if task_col_name in st.session_state.labeled_data.columns:
                            try:
                                existing_features_str = st.session_state.labeled_data.loc[current_id, task_col_name]
                                # Ensure we have a scalar value, not a Series
                                if isinstance(existing_features_str, pd.Series):
                                    existing_features_str = existing_features_str.iloc[0] if len(existing_features_str) > 0 else None
                                # Handle NaN values from pandas
                                if pd.isna(existing_features_str) or str(existing_features_str) in ["", "nan", "None"]:
                                    current_task_features = []
                                elif existing_features_str:
                                    current_task_features = [f.strip() for f in str(existing_features_str).split(',') if f.strip()]
                                else:
                                    current_task_features = []
                            except (KeyError, IndexError):
                                current_task_features = []
                    
                    # Create checkboxes for this feature task
                    selected_task_features = []
                    for feature in task_info['labels']:
                        feature_key = f"feature_{current_id}_{task_key}_{feature}"
                        
                        # Set current value based on DataFrame data (always refresh from DataFrame)
                        current_value = feature in current_task_features
                        
                        if st.checkbox(
                            feature,
                            key=feature_key,
                            value=current_value,
                            on_change=update_feature_selection,
                            args=(task_key, feature)
                        ):
                            selected_task_features.append(feature)
                    
                    feature_choices[task_key] = selected_task_features
                    col_idx += 1
