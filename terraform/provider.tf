terraform {
    required_providers {
        aws = {
            source  = "hashicorp/aws"
            version = "~> 6.0"
        }
    }
}

provider "aws" {}

terraform {
    backend "s3" {
        bucket = "tf-remote-backend-ehb-2660"
        key    = "path/to/my/key"
        region = "eu-west-1"
    }
}