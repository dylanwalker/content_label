# TV Content Labeling Tool

A Streamlit application for labeling and annotating TV content data with advanced text processing and visualization features.

## Features

- **Flexible Data Loading**: 
  - Load large feather files (several GB) directly from local disk paths
  - Upload smaller files (< 200MB) through the web interface
  - Chunked loading for memory-efficient processing of large datasets
- **Column Selection**: Choose which columns from the original data to save with labels
- **Text Beautification**: Automatically clean text by removing newlines and extra whitespace
- **Word Highlighting**: Highlight user-defined keywords with custom colors and bold formatting
- **Dual Labeling System**:
  - **Classification**: Pick one label from multiple options (e.g., genre, sentiment)
  - **Feature Labeling**: Pick zero or more features (e.g., topics, characteristics)
- **Navigation**: Easy navigation between items with progress tracking
- **Label Management**: Save and load labeling progress as feather files
- **Export**: Download labeled data as CSV or save progress as feather files
- **Performance Optimized**: Handles large datasets with memory usage monitoring
- **Dark Mode**: Toggle between light and dark themes for comfortable viewing

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Run the Streamlit app:
```bash
streamlit run app.py
```

## Usage

### 1. Load Data

**For Large Files (recommended for GB files):**
- Copy the full path to your feather file:
  - **Windows**: Right-click file → "Copy as path" or Shift+Right-click → "Copy as path"  
  - **File Explorer**: Copy from the address bar
- Paste the path in the "Enter file path" field in the sidebar
- Click "📁 Load File" button

**For Small Files (< 200MB):**
- Click "Upload feather file" in the sidebar
- Select a feather file from the file browser

### 2. Select Columns
- Choose the column containing the text content to label from the dropdown
- Select which columns from the original data to include in the labeled dataset

### 3. Configure Highlighting
- Add words you want to highlight (one per line)
- Choose a highlight color
- Words will appear bold and colored in the text display

### 4. Set Up Labels
- **Classification Labels**: Define mutually exclusive categories (pick one)
- **Feature Labels**: Define features that can be combined (pick zero or more)

### 5. Start Labeling
- Navigate through items using Previous/Next buttons
- View beautified text with highlighted keywords
- Apply classification and feature labels
- Track progress with the progress bar

### 6. Export Results
- Download labeled data as CSV
- Save/load labeling progress as feather files
- View labeling statistics and distribution

## Sample Data

A sample feather file (`sample_data.feather`) is included to test the application. It contains TV content examples with different types of content (news, cooking, sports, weather).

## File Structure

```
tv_label/
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
├── sample_data.feather       # Sample data for testing
└── readme.md                # This file
```

## Configuration Examples

### Classification Labels (pick one):
```
News
Sports
Entertainment
Weather
Cooking
Documentary
```

### Feature Labels (pick multiple):
```
Breaking News
Live Coverage
Expert Interview
Data Visualization
Audience Participation
International Content
```

### Highlight Words:
```
breaking
live
exclusive
urgent
update
```

## Tips

- Use the "Jump to Next Unlabeled" feature to efficiently label remaining items
- Save your progress regularly using the "Save Progress" feature
- Toggle between original and formatted text view to see the difference
- Use the progress indicators to track your labeling completion

## Technical Details

- Built with Streamlit for the web interface
- Uses Pandas for data manipulation
- Implements regex-based text highlighting
- Session state management for persistent labeling
- Feather format for efficient data storage and retrieval
