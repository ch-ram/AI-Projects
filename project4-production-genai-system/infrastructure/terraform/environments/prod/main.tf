terraform {
  required_version = ">= 1.8.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.70"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.6"
    }
  }

  backend "s3" {
    bucket         = "genai-terraform-state"
    key            = "production/terraform.tfstate"
    region         = "us-east-1"
    dynamodb_table = "terraform-locks"
    encrypt        = true
  }
}

provider "aws" {
  region = var.aws_region
  default_tags {
    tags = {
      Project     = "genai-platform"
      Environment = "production"
      ManagedBy   = "terraform"
      Owner       = "psr-it-solutions"
    }
  }
}

variable "aws_region" {
  default = "us-east-1"
}

# ── VPC ─────────────────────────────────────────────────────

module "vpc" {
  source      = "../../modules/vpc"
  environment = "production"
  vpc_cidr    = "10.0.0.0/16"
}

# ── EKS ─────────────────────────────────────────────────────

module "eks" {
  source              = "../../modules/eks"
  cluster_name        = "genai-prod"
  cluster_version     = "1.30"
  environment         = "production"
  vpc_id              = module.vpc.vpc_id
  subnet_ids          = module.vpc.private_subnet_ids
  node_instance_types = ["t3.xlarge"]
  node_desired_size   = 3
  node_min_size       = 2
  node_max_size       = 10
}

# ── RDS (PostgreSQL + pgvector) ─────────────────────────────

module "rds" {
  source            = "../../modules/rds"
  identifier        = "genai-prod"
  environment       = "production"
  vpc_id            = module.vpc.vpc_id
  subnet_ids        = module.vpc.private_subnet_ids
  instance_class    = "db.r6g.large"
  allocated_storage = 100
}

# ── S3 ──────────────────────────────────────────────────────

module "s3" {
  source      = "../../modules/s3"
  bucket_name = "genai-documents-prod"
  environment = "production"
}

# ── Outputs ─────────────────────────────────────────────────

output "eks_cluster_name" { value = module.eks.cluster_name }
output "eks_endpoint" { value = module.eks.cluster_endpoint }
output "rds_endpoint" { value = module.rds.endpoint }
output "s3_bucket" { value = module.s3.bucket_name }
