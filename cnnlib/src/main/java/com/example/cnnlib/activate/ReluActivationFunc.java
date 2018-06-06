package com.example.cnnlib.activate;

public class ReluActivationFunc extends ActivationFunc {

    @Override
    public double active(double in) {
        return in > 0 ? in : 0;
    }

    @Override
    public double diffActive(double in) {
        return in <= 0 ? 0.1 : 1.0;
    }

    @Override
    public ActivateType getType() {
        return ActivateType.RELU;
    }

}
