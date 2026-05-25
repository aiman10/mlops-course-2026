resource "aws_s3_bucket" "this" {
  bucket = "mlops-course-dev-863745572000"

  tags = {
    Environment = "Dev"
  }
}