#!/bin/bash
echo "worker 1"
kubectl logs -f epoch-100-non-salr-job-1-worker-worker-0 | grep Test-Accuracy
echo "worker 2"
kubectl logs -f epoch-100-non-salr-job-2-worker-worker-0 | grep Test-Accuracy
echo "worker 3"
kubectl logs -f epoch-100-non-salr-job-3-worker-worker-0 | grep Test-Accuracy
echo "worker 4"
kubectl logs -f epoch-100-non-salr-job-4-worker-worker-0 | grep Test-Accuracy
echo "worker 5"
kubectl logs -f epoch-100-non-salr-job-5-worker-worker-0 | grep Test-Accuracy
echo "worker 6"
kubectl logs -f epoch-100-non-salr-job-6-worker-worker-0 | grep Test-Accuracy
