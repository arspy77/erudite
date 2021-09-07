#!/bin/bash
echo "job 1"
kubectl logs -f job1-queue-non-salr-worker-0 | grep Test-Accuracy
echo "job 2"
kubectl logs -f job2-queue-non-salr-worker-0 | grep Test-Accuracy
echo "job 3"
kubectl logs -f job3-queue-non-salr-worker-0 | grep Test-Accuracy
echo "job 4"
kubectl logs -f job4-queue-non-salr-worker-0 | grep Test-Accuracy
echo "job 5"
kubectl logs -f job5-queue-non-salr-worker-0 | grep Test-Accuracy
echo "job 6"
kubectl logs -f job6-queue-non-salr-worker-0 | grep Test-Accuracy
echo "job 7"
kubectl logs -f job7-queue-non-salr-worker-0 | grep Test-Accuracy
echo "job 8"
kubectl logs -f job8-queue-non-salr-worker-0 | grep Test-Accuracy
echo "job 9"
kubectl logs -f job9-queue-non-salr-worker-0 | grep Test-Accuracy
echo "job 10"
kubectl logs -f job10-queue-non-salr-worker-0 | grep Test-Accuracy
echo "job 11"
kubectl logs -f job11-queue-non-salr-worker-0 | grep Test-Accuracy
echo "job 12"
kubectl logs -f job12-queue-non-salr-worker-0 | grep Test-Accuracy