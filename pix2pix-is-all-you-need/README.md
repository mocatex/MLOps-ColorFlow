# General Information

We use [GCS](https://cloud.google.com/) for storage and [DVC](https://dvc.org/) for data version control.

# DVC

## Setup DVC and link to GCS

```bash
cd mlops-coco

# Initialize DVC in a subdirectory
dvc init --subdir 

# Track one data directory
dvc add images

# Authenticate for GCS access
gcloud auth login

# Allow local tools (for example DVC) to use your Google credentials
gcloud auth application-default login

# Set the default project for application-default credentials
gcloud auth application-default set-quota-project mlops-colorflow

# Add your GCS bucket as default DVC remote
dvc remote add -d storage gs://mlops-coco

# Pin the GCP project for this remote
dvc remote modify storage projectname mlops-colorflow

# Commit DVC metadata to Git
git add .
git commit -m "track images with DVC"
git push

# Upload tracked data to GCS
dvc push
```

## Download data with DVC

```bash
# Authenticate for GCS access (needed for private bucket)
gcloud auth login
# Allow local tools (for example DVC) to use your Google credentials
gcloud auth application-default login
# Set the default project for application-default credentials
gcloud auth application-default set-quota-project mlops-colorflow
# Pull data from DVC remote (GCS)
dvc pull
```

# Google Cloud Storage

## Install Google Cloud CLI

Follow [install-sdk](https://docs.cloud.google.com/sdk/docs/install-sdk) or the following steps to install the Google Cloud CLI:

Run these install commands outside this repository folder to avoid committing a local SDK directory by mistake.

```bash
# 1. Determine the platform
uname -m

# 2. Download the package for your platform
# macOS 64-bit (x86_64)
FILE_NAME='google-cloud-cli-darwin-x86_64.tar.gz'
# macOS 64-bit (ARM64, Apple silicon)
FILE_NAME='google-cloud-cli-darwin-arm.tar.gz'
# Download the appropriate file for your platform and then run the installer
curl -O https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/${FILE_NAME}

# 3. Extract the archive
tar -xf ${FILE_NAME}

# 4. Run the installation script 
./google-cloud-sdk/install.sh

# 5. Reopen the terminal or run the following command to update your PATH
source ~/.bashrc  # or source ~/.zshrc if you use zsh

# 6. Verify the installation
gcloud --version
```

## Create a GCP Bucket

```bash
# list projects to find your project ID
gcloud projects list

# Create a bucket named <YOUR_BUCKET_NAME> in the project <YOUR_PROJECT_ID>
gcloud storage buckets create gs://<YOUR_BUCKET_NAME> --location=europe-west1 --project=<YOUR_PROJECT_ID>
# example
gcloud storage buckets create gs://mlops-coco --location=europe-west1 --project=mlops-colorflow 

# or alternatively, set the default project and then create the bucket
gcloud config set project <YOUR_PROJECT_ID>
# Create a bucket named coco (if it doesn't exist)
gcloud storage buckets create gs://<YOUR_BUCKET_NAME> --location=europe-west1

# Note that YOUR_BUCKET_NAME must be globally unique across all GCP users.
```

## Grant access to GCS bucket

Common roles:
- `roles/storage.objectViewer`: read objects
- `roles/storage.objectCreator`: upload only
- `roles/storage.objectAdmin`: read/write/delete objects

```bash
# Grant read access to one user
gcloud storage buckets add-iam-policy-binding gs://mlops-coco \
    --member="user:someone@example.com" \
    --role="roles/storage.objectAdmin"
```

## Upload to GCS

> Not needed because we use DVC, just kept for reference.

```bash
# First login with your Google account
gcloud auth login

# Set the default project (if not set already)
gcloud config set project mlops-colorflow

# Allow local tools (for example DVC) to use your Google credentials
gcloud auth application-default login

# Set the default project for application-default credentials (if not set already)
gcloud auth application-default set-quota-project mlops-colorflow

# Upload the contents of the mlops-coco directory to the bucket mlops-coco
cd mlops-coco
find . -name '.DS_Store' -delete # optionally before uploading
gcloud storage rsync --recursive --exclude="(^|/)\\.DS_Store$" . gs://mlops-coco
```

## Download from GCS

> Not needed because we use DVC, just kept for reference.

```bash
# First login with your Google account
gcloud auth login 

# Optionally, list the contents of the bucket to verify access
gcloud storage ls gs://mlops-coco

# Download the bucket mlops-coco to the local directory
mkdir -p ./mlops-coco
gcloud storage rsync --recursive gs://mlops-coco ./mlops-coco
```

# Oxen

We don't use Oxen for this project, but I kept it for reference. Oxen includes storage and version control in one tool.

## Install Oxen CLI

```bash
brew install oxen
oxen --version
```

## Download from Oxen

```bash
oxen clone https://hub.oxen.ai/domizai/mlops-coco
```

## Push to Oxen

```bash
oxen add .
oxen commit -m "Commit message"
oxen push origin main
```

## Setup a new Oxen repository

```bash
cd dir/to/dataset
oxen init
oxen config --name "Your Name" --email "you@example.com"
oxen config --set-remote origin https://hub.oxen.ai/<namespace>/<repo>
oxen config --auth hub.oxen.ai <API-Key>
```