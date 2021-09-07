#!/bin/bash
echo "job 1"
kubectl logs -f job1-salr-worker-0 | grep Test-Accuracy
echo "job 2"
kubectl logs -f job2-salr-worker-0 | grep Test-Accuracy
echo "job 3"
kubectl logs -f job3-salr-worker-0 | grep Test-Accuracy
echo "job 4"
kubectl logs -f job4-salr-worker-0 | grep Test-Accuracy