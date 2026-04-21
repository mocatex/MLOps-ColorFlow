# General Informations

We use [GCS](https://cloud.google.com/) for storage and [Oxen](https://www.oxen.ai/) for version control. Note that you can skip the GCS part.

# Oxen

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

# Google Cloud Storage

## Install Google Cloud CLI

Follow [install-sdk](https://docs.cloud.google.com/sdk/docs/install-sdk) or the following steps to install the Google Cloud CLI:

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

## Create a GCP Bucker

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

## Upload to GCS

```bash
# First login with your Google account
gcloud auth login

# Upload the contents of the coco directory to the bucket coco
gcloud storage cp --recursive coco/train_sample/. gs://mlops-coco
```

## Download from GCS

```bash
# First login with your Google account
gcloud auth login 

# Download the bucket coco to a local directory named coco (current directory)
gcloud storage cp --recursive gs://mlops-coco coco/train_sample
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
