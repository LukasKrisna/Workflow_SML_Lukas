import os
import json
import mlflow
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from pathlib import Path

def get_credentials():
    creds_json = os.environ.get('GDRIVE_CREDENTIALS')
    if not creds_json:
        raise ValueError("GDRIVE_CREDENTIALS not found in environment variables")
    
    creds_info = json.loads(creds_json)
    creds = Credentials.from_authorized_user_info(creds_info)
    return creds

def get_latest_run():
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name("Diabetes_Classification_Basic")
    
    if experiment is None:
        experiments = client.search_experiments()
        if not experiments:
            raise ValueError("No experiments found")
        experiment = experiments[0]
    
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=1
    )
    
    if not runs:
        raise ValueError("No runs found in experiment")
    
    return runs[0]

def upload_file(service, file_path, folder_id, file_name=None):
    if file_name is None:
        file_name = Path(file_path).name
    
    file_metadata = {
        'name': file_name,
        'parents': [folder_id]
    }
    
    media = MediaFileUpload(file_path, resumable=True)
    file = service.files().create(
        body=file_metadata,
        media_body=media,
        fields='id, name'
    ).execute()
    
    print(f"Uploaded: {file.get('name')} (ID: {file.get('id')})")
    return file

def upload_artifacts():
    creds = get_credentials()
    service = build('drive', 'v3', credentials=creds)
    
    folder_id = os.environ.get('GDRIVE_FOLDER_ID')
    if not folder_id:
        raise ValueError("GDRIVE_FOLDER_ID not found in environment variables")
    
    run = get_latest_run()
    run_id = run.info.run_id
    
    print(f"Uploading artifacts from run: {run_id}")
    
    artifact_path = Path(run.info.artifact_uri.replace('file://', ''))
    
    if not artifact_path.exists():
        print(f"Warning: Artifact path does not exist: {artifact_path}")
        return
    
    uploaded_files = []
    
    for file_path in artifact_path.rglob('*'):
        if file_path.is_file():
            try:
                relative_path = file_path.relative_to(artifact_path)
                upload_name = f"{run_id}_{relative_path}"
                upload_file(service, str(file_path), folder_id, str(upload_name))
                uploaded_files.append(str(upload_name))
            except Exception as e:
                print(f"Error uploading {file_path}: {e}")
    
    metrics_file = Path('metrics.json')
    with open(metrics_file, 'w') as f:
        json.dump(run.data.metrics, f, indent=2)
    
    upload_file(service, str(metrics_file), folder_id, f"{run_id}_metrics.json")
    uploaded_files.append(f"{run_id}_metrics.json")
    
    print(f"\nTotal files uploaded: {len(uploaded_files)}")
    print("Upload to Google Drive completed successfully")

if __name__ == "__main__":
    upload_artifacts()

