import streamlit as st
import pandas as pd
import re
from typing import List, Dict, Set
import json
import io
import os
from pathlib import Path
import pyarrow.feather as feather

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
if 'highlight_words' not in st.session_state:
    st.session_state.highlight_words = []
if 'classification_labels' not in st.session_state:
    st.session_state.classification_labels = []
if 'feature_labels' not in st.session_state:
    st.session_state.feature_labels = []
if 'labeling_locked' not in st.session_state:
    st.session_state.labeling_locked = False
if 'auto_save_counter' not in st.session_state:
    st.session_state.auto_save_counter = 0
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False


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

def auto_save_to_feather():
    """Auto-save labeled data to feather file every 5 labels."""
    if not st.session_state.labeled_data.empty:
        try:
            filename = "tv_content_labeled_autosave.feather"
            st.session_state.labeled_data.to_feather(filename)
            st.toast(f"🔄 Auto-saved {len(st.session_state.labeled_data)} labels to {filename}", icon="💾")
        except Exception as e:
            st.toast(f"Auto-save failed: {str(e)}", icon="⚠️")

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

# Sidebar for configuration
st.sidebar.header("📊 Data Selection")

st.sidebar.divider()

# File path input for large files
st.sidebar.subheader("💽 Local File Selection")

# Initialize file path in session state
if 'selected_file_path' not in st.session_state:
    st.session_state.selected_file_path = ""

# File path input
file_path = st.sidebar.text_input(
    "Enter file path:",
    value=st.session_state.selected_file_path,
    help="Enter the full path to your feather file (e.g., C:\\Users\\username\\data\\file.feather)"
)

# Update session state when path changes
if file_path != st.session_state.selected_file_path:
    st.session_state.selected_file_path = file_path

# Browse button helper text
st.sidebar.markdown("💡 **Tip:** Copy and paste the full file path from File Explorer")

# Show example paths
with st.sidebar.expander("📋 Example file paths"):
    st.code("C:\\Users\\username\\Documents\\data.feather")
    st.code("D:\\datasets\\tv_content\\large_file.feather")
    st.code("//server/shared/data/content.feather")

# Load button for file path
col1, col2 = st.sidebar.columns([2, 1])
with col1:
    load_file = st.button("📁 Load File", use_container_width=True)
with col2:
    clear_file = st.button("🗑️ Clear", use_container_width=True)

if clear_file:
    st.session_state.data = None
    st.session_state.selected_file_path = ""
    st.session_state.labels = {}
    st.session_state.labeled_data = pd.DataFrame()
    st.session_state.current_index = 0
    st.session_state.labeling_locked = False
    st.session_state.auto_save_counter = 0
    st.rerun()

if load_file and file_path:
    try:
        if not os.path.exists(file_path):
            st.sidebar.error(f"❌ File not found: {file_path}")
        elif not file_path.lower().endswith('.feather'):
            st.sidebar.error("❌ Please select a feather file")
        else:
            # Get file size info
            file_size = os.path.getsize(file_path)
            file_size_mb = file_size / (1024 * 1024)
            
            if file_size_mb > 500:  # Show warning for very large files
                st.sidebar.warning(f"⚠️ Large file detected ({file_size_mb:.1f} MB). Using chunked loading...")
            
            with st.sidebar.status("Analyzing file...") as status:
                # Get total rows and first chunk for column info
                st.session_state.total_rows = get_total_rows(file_path)
                st.session_state.file_path = file_path
                st.session_state.current_index = 0
                st.session_state.chunk_start_idx = 0
                st.session_state.current_chunk = None
                
                # Load first chunk to get column information
                first_chunk = load_chunk(file_path, 0, min(100, st.session_state.chunk_size))
                st.session_state.data = first_chunk  # For column selection
                
                status.update(label="File loaded successfully!", state="complete")
            
            st.sidebar.success(f"✅ Ready to process {st.session_state.total_rows:,} rows")
            st.sidebar.info(f"📊 File size: {file_size_mb:.1f} MB")
            
    except FileNotFoundError:
        st.sidebar.error(f"❌ File not found: {file_path}")
    except PermissionError:
        st.sidebar.error(f"❌ Permission denied. Cannot access: {file_path}")
    except pd.errors.EmptyDataError:
        st.sidebar.error("❌ The file appears to be empty")
    except Exception as e:
        st.sidebar.error(f"❌ Error loading file: {str(e)}")

# Alternative: File upload for smaller files (loads entirely into memory)
st.sidebar.subheader("📤 Upload Small File")
st.sidebar.markdown("*For files < 200MB (loads entirely into memory)*")
uploaded_file = st.sidebar.file_uploader(
    "Upload feather file",
    type=['feather'],
    help="Upload a feather file (recommended for files < 200MB)"
)

if uploaded_file is not None:
    try:
        # Save uploaded file temporarily and load
        with open("temp_upload.feather", "wb") as f:
            f.write(uploaded_file.getvalue())
        full_data = pd.read_feather("temp_upload.feather")
        os.remove("temp_upload.feather")  # Clean up temp file
        
        st.session_state.data = full_data
        st.session_state.total_rows = len(full_data)
        st.session_state.file_path = ""  # Clear file path for uploaded files
        st.session_state.current_chunk = full_data
        st.session_state.chunk_start_idx = 0
        st.sidebar.success(f"✅ Loaded {len(full_data):,} rows into memory")
    except Exception as e:
        st.sidebar.error(f"❌ Error loading file: {str(e)}")

# Show current loaded file info
if st.session_state.data is not None:
    if st.session_state.file_path:
        st.sidebar.success(f"📊 **Large file mode:** {st.session_state.total_rows:,} total rows")
        st.sidebar.info(f"🔄 **Chunked loading:** {st.session_state.chunk_size:,} rows at a time")
    else:
        st.sidebar.success(f"📊 **Memory mode:** {st.session_state.total_rows:,} rows loaded")
    
    # Show labeled data progress
    if not st.session_state.labeled_data.empty:
        labeled_count = len(st.session_state.labeled_data)
        st.sidebar.info(f"🏷️ **Labeled so far:** {labeled_count:,} items")

# Column selection
if st.session_state.data is not None:
    st.sidebar.subheader("📋 Column Configuration")
    
    # Text column selection
    text_column = st.sidebar.selectbox(
        "Select text column to label",
        options=st.session_state.data.columns.tolist(),
        help="Choose the column containing the text content to label"
    )
    
    # Columns to include in labeled dataset
    st.sidebar.markdown("**Columns to save with labels:**")
    available_columns = st.session_state.data.columns.tolist()
    
    # Default selection includes text column
    default_selected = [text_column] if text_column in available_columns else []
    if not st.session_state.selected_columns:
        st.session_state.selected_columns = default_selected
    
    selected_columns = st.sidebar.multiselect(
        "Choose columns to include in labeled dataset",
        options=available_columns,
        default=st.session_state.selected_columns,
        help="Select which columns from the original data to save with your labels"
    )
    
    st.session_state.selected_columns = selected_columns
    
    # Show selection summary
    if selected_columns:
        st.sidebar.success(f"✅ Will save {len(selected_columns)} columns with labels")
    else:
        st.sidebar.warning("⚠️ No columns selected for saving")
    
    # Display basic info about the dataset
    st.sidebar.subheader("📈 Dataset Info")
    st.sidebar.write(f"**Total Rows:** {st.session_state.total_rows:,}")
    st.sidebar.write(f"**Available Columns:** {len(st.session_state.data.columns)}")
    
    # Chunk size configuration for large files
    if st.session_state.file_path:
        st.sidebar.subheader("⚙️ Performance Settings")
        chunk_size = st.sidebar.slider(
            "Chunk size (rows loaded at once)",
            min_value=100,
            max_value=5000,
            value=st.session_state.chunk_size,
            step=100,
            help="Larger chunks use more memory but may be faster"
        )
        if chunk_size != st.session_state.chunk_size:
            st.session_state.chunk_size = chunk_size
            st.session_state.current_chunk = None  # Force reload of chunk

# Configuration section
st.sidebar.header("⚙️ Configuration")

# Highlight words configuration
st.sidebar.subheader("🎨 Text Highlighting")
highlight_words_input = st.sidebar.text_area(
    "Words to highlight (one per line)",
    value="\n".join(st.session_state.highlight_words),
    height=100,
    help="Enter words you want to highlight in the text, one per line"
)

highlight_color = st.sidebar.color_picker(
    "Highlight color",
    value="#ff6b6b",
    help="Choose the color for highlighted words"
)

if highlight_words_input:
    st.session_state.highlight_words = [
        word.strip() for word in highlight_words_input.split('\n') 
        if word.strip()
    ]

# Classification labels configuration
st.sidebar.subheader("🏷️ Classification Labels")

# Check if labeling is locked
is_locked = st.session_state.labeling_locked and not st.session_state.labeled_data.empty

if is_locked:
    st.sidebar.info("🔒 Labels locked during labeling session")
    st.sidebar.write("**Current classification options:**")
    for label in st.session_state.classification_labels:
        st.sidebar.write(f"- {label}")
    
    # Show warning and reset option
    with st.sidebar.expander("⚠️ Change Labels (will lose progress)"):
        st.warning(f"Changing labels will abandon {len(st.session_state.labeled_data)} labeled items")
        if st.button("🗑️ Reset and Change Labels", type="secondary"):
            st.session_state.labeled_data = pd.DataFrame()
            st.session_state.labeling_locked = False
            st.session_state.auto_save_counter = 0
            st.rerun()
else:
    classification_input = st.sidebar.text_area(
        "Classification options (one per line)",
        value="\n".join(st.session_state.classification_labels),
        height=100,
        help="Enter classification options, one per line (pick one)"
    )
    
    if classification_input:
        st.session_state.classification_labels = [
            label.strip() for label in classification_input.split('\n') 
            if label.strip()
        ]

# Feature labels configuration
st.sidebar.subheader("🔖 Feature Labels")

if is_locked:
    st.sidebar.info("🔒 Labels locked during labeling session")
    st.sidebar.write("**Current feature options:**")
    for label in st.session_state.feature_labels:
        st.sidebar.write(f"- {label}")
    
    # Note: Reset option already shown in classification section
else:
    feature_input = st.sidebar.text_area(
        "Feature options (one per line)",
        value="\n".join(st.session_state.feature_labels),
        height=100,
        help="Enter feature options, one per line (pick zero or more)"
    )
    
    if feature_input:
        st.session_state.feature_labels = [
            label.strip() for label in feature_input.split('\n') 
            if label.strip()
        ]

# Labels management
st.sidebar.header("💾 Labels Management")

# Load previously saved labeled data - File path input
st.sidebar.subheader("📂 Load Previous Progress (Local File)")
progress_file_path = st.sidebar.text_input(
    "Enter path to saved progress file:",
    help="Enter the full path to your saved .feather file (e.g., C:\\Users\\username\\tv_content_labeled.feather)"
)

if progress_file_path and st.sidebar.button("� Load from Path"):
    try:
        if not os.path.exists(progress_file_path):
            st.sidebar.error(f"❌ File not found: {progress_file_path}")
        elif not progress_file_path.lower().endswith('.feather'):
            st.sidebar.error("❌ Please select a feather file")
        else:
            load_labeled_data_from_feather(progress_file_path)
    except Exception as e:
        st.sidebar.error(f"Error loading progress: {str(e)}")

# Load previously saved labeled data - File upload
st.sidebar.subheader("📤 Upload Previous Progress")
st.sidebar.markdown("*For progress files < 200MB*")
uploaded_feather = st.sidebar.file_uploader(
    "Upload saved progress file",
    type=['feather'],
    help="Upload a previously saved .feather file with labeled data"
)

if uploaded_feather is not None:
    try:
        # Save uploaded file temporarily and load
        with open("temp_labeled.feather", "wb") as f:
            f.write(uploaded_feather.getvalue())
        load_labeled_data_from_feather("temp_labeled.feather")
        os.remove("temp_labeled.feather")  # Clean up temp file
    except Exception as e:
        st.sidebar.error(f"Error loading progress: {str(e)}")

# Show current progress
if not st.session_state.labeled_data.empty:
    st.sidebar.info(f"📊 **Current progress:** {len(st.session_state.labeled_data):,} items labeled")
    st.sidebar.info("� **Labeling session active** - labels are locked")

# Main content area
st.title("📺 TV Content Labeling Tool")

# Save functionality in main window
if not st.session_state.labeled_data.empty:
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        # Manual save button
        if st.button("💾 Save Progress", type="primary", use_container_width=True):
            filename = save_labeled_data_to_feather()
            if filename:
                st.success(f"✅ Saved {len(st.session_state.labeled_data)} labels to {filename}")
    
    with col2:
        # Download button for the saved file
        if os.path.exists("tv_content_labeled.feather"):
            with open("tv_content_labeled.feather", "rb") as file:
                st.download_button(
                    label="📥 Download File",
                    data=file.read(),
                    file_name="tv_content_labeled.feather",
                    mime="application/octet-stream",
                    use_container_width=True
                )
        else:
            st.button("📥 Download File", disabled=True, use_container_width=True, help="Save progress first to enable download")
    
    with col3:
        # Show progress summary
        labeled_count = len(st.session_state.labeled_data)
        st.metric("Labels Saved", f"{labeled_count:,}")
    
    st.markdown("---")

if st.session_state.data is None:
    st.info("👆 Please select a .feather file to get started")
    
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
        - **Manual save**: Use "💾 Save Label" button for immediate saving with feedback
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
    
    # Navigation controls
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        if st.button("⬅️ Previous", disabled=st.session_state.current_index <= 0):
            save_current_label_if_exists()
            st.session_state.current_index -= 1
            st.rerun()
    
    with col2:
        st.markdown(f"<div style='text-align: center;'><h3>Item {st.session_state.current_index + 1:,} of {st.session_state.total_rows:,}</h3></div>", unsafe_allow_html=True)
    
    with col3:
        if st.button("Next ➡️", disabled=st.session_state.current_index >= st.session_state.total_rows - 1):
            save_current_label_if_exists()
            st.session_state.current_index += 1
            st.rerun()
    
    # Progress bar
    progress = (st.session_state.current_index + 1) / st.session_state.total_rows
    st.progress(progress)
    
    # Get current row data
    current_row = get_current_row()
    
    if current_row is None:
        st.error("❌ Could not load current row data")
        st.stop()
    
    current_text = current_row[text_column] if text_column in current_row else ""
    
    # Check if this item is already labeled
    current_id = st.session_state.current_index
    is_already_labeled = current_id in st.session_state.labeled_data['original_index'].values if not st.session_state.labeled_data.empty else False
    
    if is_already_labeled:
        existing_label_row = st.session_state.labeled_data[st.session_state.labeled_data['original_index'] == current_id].iloc[0]
        existing_classification = existing_label_row.get('label_classification', '')
        existing_features = existing_label_row.get('label_features', '').split(', ') if existing_label_row.get('label_features') else []
        st.info(f"✅ This item is already labeled (Classification: {existing_classification}, Features: {', '.join(existing_features)})")
    
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
                if col != text_column and col in current_row:
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
    
    col1, col2 = st.columns(2)
    
    # Classification labeling (pick one)
    with col1:
        if st.session_state.classification_labels:
            st.markdown("**Classification (pick one):**")
            
            # Create radio buttons for classification
            if current_classification and current_classification in st.session_state.classification_labels:
                default_index = st.session_state.classification_labels.index(current_classification) + 1
            else:
                default_index = 0
            
            classification_choice = st.radio(
                "Choose classification:",
                options=[None] + st.session_state.classification_labels,
                index=default_index,
                format_func=lambda x: "None" if x is None else x,
                key=f"classification_{current_id}"
            )
    
    # Feature labeling (pick zero or more)
    with col2:
        if st.session_state.feature_labels:
            st.markdown("**Features (pick zero or more):**")
            
            # Create checkboxes for features
            selected_features = []
            for feature in st.session_state.feature_labels:
                if st.checkbox(
                    feature,
                    value=feature in current_features,
                    key=f"feature_{feature}_{current_id}"
                ):
                    selected_features.append(feature)
    
    # Save label button
    if st.button("💾 Save Label", type="primary", use_container_width=True):
        # Lock labeling configuration on first save
        if not st.session_state.labeling_locked:
            st.session_state.labeling_locked = True
        
        # Remove existing label if it exists
        if is_already_labeled:
            st.session_state.labeled_data = st.session_state.labeled_data[
                st.session_state.labeled_data['original_index'] != current_id
            ].reset_index(drop=True)
        
        # Save new label
        classification = classification_choice if 'classification_choice' in locals() else None
        features = selected_features if 'selected_features' in locals() else []
        
        if classification or features:
            save_labeled_row(current_row, classification, features)
            st.success("✅ Label saved!")
            
            # Show auto-save status
            if st.session_state.auto_save_counter % 5 == 0:
                st.info("🔄 Auto-saved to file!")
            else:
                remaining = 5 - (st.session_state.auto_save_counter % 5)
                st.info(f"📊 {remaining} more labels until auto-save")
            
            st.rerun()
        else:
            st.warning("⚠️ Please select at least one classification or feature before saving")
    
    # Show current progress
    labeled_count = len(st.session_state.labeled_data)
    st.markdown(f"**Progress:** {labeled_count:,}/{st.session_state.total_rows:,} items labeled ({labeled_count/st.session_state.total_rows*100:.1f}%)")
    
    # Show auto-save status
    if hasattr(st.session_state, 'auto_save_counter'):
        remaining = 5 - (st.session_state.auto_save_counter % 5)
        if remaining == 5:
            st.sidebar.success("💾 **Auto-save:** Just saved!")
        else:
            st.sidebar.info(f"🔄 **Auto-save:** {remaining} more until next save")
    
    # Quick navigation
    st.subheader("🔍 Quick Navigation")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🏠 Go to Start"):
            save_current_label_if_exists()
            st.session_state.current_index = 0
            st.rerun()
    
    with col2:
        if st.button("🎯 Go to End"):
            save_current_label_if_exists()
            st.session_state.current_index = st.session_state.total_rows - 1
            st.rerun()
    
    with col3:
        # Find next unlabeled item
        if not st.session_state.labeled_data.empty:
            labeled_indices = set(st.session_state.labeled_data['original_index'].values)
            unlabeled_indices = [i for i in range(st.session_state.total_rows) if i not in labeled_indices]
            
            if unlabeled_indices:
                next_unlabeled = min([i for i in unlabeled_indices if i > st.session_state.current_index], default=min(unlabeled_indices))
                if st.button(f"➡️ Next Unlabeled ({next_unlabeled + 1:,})"):
                    save_current_label_if_exists()
                    st.session_state.current_index = next_unlabeled
                    st.rerun()
        else:
            if st.button("➡️ Next Unlabeled"):
                save_current_label_if_exists()
                st.session_state.current_index = min(st.session_state.current_index + 1, st.session_state.total_rows - 1)
                st.rerun()
    
    # Jump to specific item
    jump_to = st.number_input(
        "Jump to item:",
        min_value=1,
        max_value=st.session_state.total_rows,
        value=st.session_state.current_index + 1
    )
    
    if st.button("🔄 Jump"):
        save_current_label_if_exists()
        st.session_state.current_index = jump_to - 1
        st.rerun()
    
    # Export section
    if not st.session_state.labeled_data.empty:
        st.subheader("📊 Labeling Summary")
        
        # Show summary statistics
        if 'label_classification' in st.session_state.labeled_data.columns:
            classification_counts = st.session_state.labeled_data['label_classification'].value_counts()
            if not classification_counts.empty:
                st.write("**Classification distribution:**")
                for label, count in classification_counts.items():
                    if label:  # Skip empty labels
                        st.write(f"- {label}: {count}")
        
        if 'label_features' in st.session_state.labeled_data.columns:
            # Parse feature counts
            feature_counts = {}
            for features_str in st.session_state.labeled_data['label_features'].dropna():
                if features_str:
                    features = [f.strip() for f in features_str.split(',') if f.strip()]
                    for feature in features:
                        feature_counts[feature] = feature_counts.get(feature, 0) + 1
            
            if feature_counts:
                st.write("**Feature distribution:**")
                for feature, count in sorted(feature_counts.items()):
                    st.write(f"- {feature}: {count}")
