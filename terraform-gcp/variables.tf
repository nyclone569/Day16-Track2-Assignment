variable "project_id" {
  description = "GCP Project ID"
  type        = string
}

variable "region" {
  description = "GCP Region"
  type        = string
  default     = "us-west1"
}

variable "zone" {
  description = "GCP Zone"
  type        = string
  default     = "us-west1-b"
}

variable "hf_token" {
  description = "Hugging Face Token for gated models (like Gemma)"
  type        = string
  sensitive   = true
  default     = ""
}

variable "model_id" {
  description = "Hugging Face Model ID to serve"
  type        = string
  default     = "google/gemma-4-E2B-it"
}

variable "machine_type" {
  description = "GCE Machine Type for the CPU node"
  type        = string
  default     = "n2-highmem-8"  # 8 vCPU, 64 GB RAM - tương đương r5.2xlarge của AWS
}

# GPU variables - không dùng nữa nhưng giữ lại để tránh lỗi
variable "gpu_type" {
  description = "GPU accelerator type (not used in CPU mode)"
  type        = string
  default     = ""
}

variable "gpu_count" {
  description = "Number of GPUs to attach (not used in CPU mode)"
  type        = number
  default     = 0
}
