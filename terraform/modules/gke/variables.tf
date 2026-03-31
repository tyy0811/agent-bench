variable "project_id" {
  type = string
}

variable "region" {
  type = string
}

variable "cluster_name" {
  type = string
}

variable "network" {
  type = string
}

variable "subnetwork" {
  type = string
}

variable "cpu_node_count" {
  type    = number
  default = 2
}

variable "cpu_machine_type" {
  type    = string
  default = "e2-standard-4"
}
