package com.example.cnnlib.activate;

public abstract class ActivationFunc {
    public abstract double active(double in);

    public abstract double diffActive(double in);

    public abstract ActivateType getType();

    public enum ActivateType {
        RELU,
        Sigmoid,
        Tanh
    }
}
