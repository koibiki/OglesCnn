package com.example.cnnlib.model;

public class Kennel {

    // 每个 Kennel 容量(size)不能超过1000
    public final int[] shape = new int[3];
    public final float[] data;

    public Kennel(int width, int height, int channel, float[] data) {
        this.shape[0] = width;
        this.shape[1] = height;
        this.shape[2] = channel;
        this.data = data;
    }

    public int getArea() {
        return shape[0] * shape[1];
    }

    public int getSize() {
        return shape[0] * shape[1] * shape[2];
    }

}
