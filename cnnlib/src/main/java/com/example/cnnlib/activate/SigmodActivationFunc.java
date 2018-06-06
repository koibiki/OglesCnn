package com.example.cnnlib.activate;

public class SigmodActivationFunc extends ActivationFunc {

    @Override
    public double active(double in) {
        return 1.0 / (1.0 + Math.exp(-in));
    }

    @Override
    public double diffActive(double in) {
        return (Math.exp(-in)) / ((1 + Math.exp(-in)) * (1 + Math.exp(-in)));
    }

    @Override
    public ActivateType getType() {
        return ActivateType.Sigmoid;
    }
}
