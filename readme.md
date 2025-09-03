# Content Labeling Tool

A secure, multi-user Streamlit application for labeling and annotating data with Google OAuth authentication, configurable classification tasks, and user-specific data isolation.

## Features

- **🔐 Google OAuth Authentication**: Secure login with user-specific data isolation
- **👥 Multi-User Support**: Each authenticated user gets their own labeled data files
- **⚙️ Configuration-Driven**: Flexible setup through `.cfg` files
- **📊 Multi-Task Classification**: Support for multiple classification and feature labeling tasks
- **🎨 Text Highlighting**: Configurable keyword highlighting with custom colors
- **💾 Auto-Save**: Automatic progress saving every 5 labels with backup files
- **📁 Large File Support**: Handle multi-GB feather files with chunked loading
- **🔍 Smart Navigation**: Jump to unlabeled items, progress tracking
- **📤 Export Options**: Download results as feather files with configurable columns

## Quick Setup Guide

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/content_label.git
cd content_label
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Set Up Google OAuth Authentication

#### Create Google OAuth Project:
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing one
3. Enable the **Google+ API** or **Google Identity API**
4. Go to **Credentials** → **Create Credentials** → **OAuth 2.0 Client ID**
5. Set application type to **Web application**
6. Add authorized redirect URIs:
   - For local development: `http://localhost:8501`
   - For Streamlit Cloud: `https://your-app-name.streamlit.app`

#### Configure Authentication:
Create a `secrets.toml` file in the project root:
```toml
[oauth.google]
client_id = "your-google-client-id.googleusercontent.com"
client_secret = "your-google-client-secret"
redirect_uri = "http://localhost:8501"  # or your deployed URL

[oauth]
cookie_secret = "your-random-string-for-cookie-encryption"
```

#### Add Approved Users:
- In Google Cloud Console, go to **OAuth consent screen**
- Add test users (their email addresses) who should have access
- For production, submit for verification or keep in testing mode

### 4. Prepare Your Data
- Convert your data to `.feather` format (Apache Arrow)
- Place the file in the project directory or note its full path
- Ensure your data has the columns you want to label

### 5. Configure the Application

Edit `default.cfg` to match your labeling needs:

```ini
[data]
# Path to your data file
file_path = your_data.feather
# Column containing text to be labeled
text_column = content

[output]
# Base filename for user-specific labeled data
output_base_filename = labeled_data.feather
# Columns to include in output (comma-separated)
output_columns = datetime,text,embedding,metadata

[highlighting]
# Highlight color (hex code)
color = #ece800
# Words to highlight (comma-separated)
words = keyword1,keyword2,important,urgent

[classification_tasks]
# Define multiple classification tasks
task1_name = Content Type
task1_labels = News, Sports, Entertainment, Weather

task2_name = Urgency Level  
task2_labels = Low, Medium, High, Critical

[feature_tasks]
# Define feature labeling tasks (multi-select)
task1_name = Content Features
task1_labels = Breaking News, Live Coverage, Expert Interview, Analysis
```

### 6. Run the Application

**Local Development:**
```bash
streamlit run app.py
```

**Deploy to Streamlit Cloud:**
1. Push your repository to GitHub
2. Connect to [Streamlit Cloud](https://streamlit.io/cloud)
3. Add your `secrets.toml` content to the Streamlit Cloud secrets
4. Deploy your app

## Configuration Reference

### `[data]` Section
- `file_path`: Path to your feather data file
- `text_column`: Column name containing text content to label

### `[output]` Section  
- `output_base_filename`: Base name for output files (users get `user_hash_filename`)
- `output_columns`: Columns from input data to include in labeled output

### `[highlighting]` Section
- `color`: Hex color code for highlighted text (#rrggbb)
- `words`: Comma-separated list of words to highlight in text

### `[classification_tasks]` Section
Multiple classification tasks where users pick **one** option:
```ini
task1_name = Task Display Name
task1_labels = Option1, Option2, Option3

task2_name = Another Task
task2_labels = Choice A, Choice B, Choice C
```

### `[feature_tasks]` Section  
Multiple feature tasks where users can pick **zero or more** options:
```ini
task1_name = Feature Set Name
task1_labels = Feature1, Feature2, Feature3

task2_name = Another Feature Set
task2_labels = Attribute A, Attribute B, Attribute C
```

## User Workflow

### For Labelers:
1. **Login**: Use approved Google account to access the tool
2. **Auto-Load**: Data and configuration load automatically from `default.cfg`
3. **Navigate**: Use Previous/Next buttons or jump to specific items
4. **Label**: Select classifications and features for each item
5. **Auto-Save**: Progress saves automatically every 5 labels
6. **Download**: Export your labeled data when complete

### For Administrators:
1. **Setup**: Configure `default.cfg` with your labeling tasks
2. **Deploy**: Host on Streamlit Cloud with proper authentication
3. **Manage**: Add/remove users through Google OAuth console
4. **Monitor**: Each user's data is saved separately as `user_hash_filename.feather`

## File Structure

```
content_label/
├── app.py                           # Main application
├── default.cfg                      # Configuration file
├── secrets.toml                     # OAuth secrets (local only)
├── requirements.txt                 # Python dependencies
├── sample_data.feather             # Sample data
├── user_hash1_labeled_data.feather # User 1's labeled data
├── user_hash2_labeled_data.feather # User 2's labeled data
└── readme.md                       # This documentation
```

## Security Features

- **OAuth Authentication**: Only approved Google accounts can access
- **User Isolation**: Each user's labeled data is completely separate
- **Auto-Backup**: Backup files (`.bak.feather`) created on each save
- **Session Management**: Secure session handling with encrypted cookies

## Technical Notes

- **Performance**: Handles multi-GB files through chunked loading
- **Format**: Uses Apache Arrow feather format for fast I/O
- **State Management**: Streamlit session state preserves work across navigation
- **Responsive**: Works on desktop and tablet devices
- **Multi-User**: Supports unlimited concurrent users with data isolation

## Troubleshooting

**Authentication Issues:**
- Verify `secrets.toml` has correct Google OAuth credentials
- Ensure user email is added to OAuth consent screen
- Check redirect URIs match your deployment URL

**Data Loading Issues:**  
- Confirm file path in `default.cfg` is correct and accessible
- Ensure data file is in feather format
- Check that specified `text_column` exists in your data

**Configuration Issues:**
- Validate `default.cfg` syntax (no spaces in section headers)
- Ensure task numbers are sequential (task1, task2, task3...)
- Check that specified `output_columns` exist in your data

## Example Use Cases

- **Content Moderation**: Label social media posts for policy violations
- **Research Annotation**: Classify academic papers or articles  
- **Media Analysis**: Tag TV/radio content for themes and topics
- **Customer Feedback**: Categorize support tickets or reviews
- **Document Processing**: Label legal documents or contracts
