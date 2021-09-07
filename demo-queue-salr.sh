#!/bin/bash
kubectl create -f examples/job1-queue-salr.yaml
kubectl create -f examples/job2-queue-salr.yaml
kubectl create -f examples/job3-queue-salr.yaml
kubectl create -f examples/job4-queue-salr.yaml
kubectl create -f examples/job5-queue-salr.yaml
kubectl create -f examples/job6-queue-salr.yaml
kubectl create -f examples/job7-queue-salr.yaml
kubectl create -f examples/job8-queue-salr.yaml
kubectl create -f examples/job9-queue-salr.yaml
kubectl create -f examples/job10-queue-salr.yaml
kubectl create -f examples/job11-queue-salr.yaml
kubectl create -f examples/job12-queue-salr.yaml
kubectl get pods -w