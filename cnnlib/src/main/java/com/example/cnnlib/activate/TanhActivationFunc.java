package com.example.cnnlib.activate;

public class TanhActivationFunc extends ActivationFunc {

    private double tanh(double in) {
        double ef = Math.exp(in);
        double efx = Math.exp(-in);
        return (ef - efx) / (ef + efx);
    }

    @Override
    public double active(double in) {
        return tanh(in);
    }

    @Override
    public double diffActive(double in) {
        return (1 - tanh(in) * tanh(in));
    }

    @Override
    public ActivateType getType() {
        return ActivateType.Tanh;
    }

}
