#!/bin/bash
kubectl create -f examples/epoch-100-non-salr-job-1-worker.yaml
kubectl create -f examples/epoch-100-non-salr-job-2-worker.yaml
kubectl create -f examples/epoch-100-non-salr-job-3-worker.yaml
kubectl create -f examples/epoch-100-non-salr-job-4-worker.yaml
kubectl create -f examples/epoch-100-non-salr-job-5-worker.yaml
kubectl create -f examples/epoch-100-non-salr-job-6-worker.yaml
kubectl get pods -w