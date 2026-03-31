resource "google_container_cluster" "primary" {
  name     = var.cluster_name
  location = var.region
  project  = var.project_id

  network    = var.network
  subnetwork = var.subnetwork

  # Autopilot disabled — we manage node pools explicitly
  enable_autopilot = false

  # Remove default node pool (we create our own)
  remove_default_node_pool = true
  initial_node_count       = 1

  ip_allocation_policy {
    cluster_secondary_range_name  = "pods"
    services_secondary_range_name = "services"
  }
}

resource "google_container_node_pool" "cpu_pool" {
  name       = "${var.cluster_name}-cpu-pool"
  location   = var.region
  cluster    = google_container_cluster.primary.name
  node_count = var.cpu_node_count
  project    = var.project_id

  node_config {
    machine_type = var.cpu_machine_type
    disk_size_gb = 50
    disk_type    = "pd-standard"

    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform",
    ]
  }
}
