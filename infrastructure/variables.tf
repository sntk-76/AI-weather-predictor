variable "project_id" {
    description = "The project id "
    default = "eastern-bedrock-464312-h1"
}

variable "service_account" {
    description = "The service account json file directory for the gcp connection"
    default = "/home/sina.tvk.1997/AI-weather-predictor/authentication/service_account.json"
}

variable "region" {
    description = "The region of the project"
    default = "europe-west8"
}

variable "bucket_name" {
    description = "The name of the bucket forthe project that contain three folders"
    default = "eastern-bedrock-464312-h1_bucket"
}

variable "bigquery_name_1" {
    description = "The name of the data set for the clean data after transformation"
    default = "cleaned_data"
}