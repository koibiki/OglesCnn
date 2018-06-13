package com.example.cnnlib.model;

public class Kennel {

    // 每个 Kennel 容量(size)不能超过900
    public final int[] shape = new int[3];
    public final float[] data;
    public int area;
    public int size;

    public Kennel(int width, int height, int channel, float[] data) {
        this.shape[0] = width;
        this.shape[1] = height;
        this.shape[2] = channel;
        this.data = data;
        this.area = width * height;
        this.size = width * height * channel;
    }

    public int getArea() {
        return area;
    }

    public int getSize() {
        return size;
    }

}
