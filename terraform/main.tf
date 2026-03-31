terraform {
  required_version = ">= 1.5"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

module "networking" {
  source       = "./modules/networking"
  project_id   = var.project_id
  region       = var.region
  cluster_name = var.cluster_name
}

module "gke" {
  source           = "./modules/gke"
  project_id       = var.project_id
  region           = var.region
  cluster_name     = var.cluster_name
  network          = module.networking.network_name
  subnetwork       = module.networking.subnetwork_name
  cpu_node_count   = 2
  cpu_machine_type = "e2-standard-4"
}
