package com.example.cnnlib.model;

public class LayerParams {

    public final int[] inputShape = new int[3];     // w,h,c
    public final int[] outputShape = new int[3];    // w,h,c

    public enum TYPE {
        FLAT, SAME
    }

    public LayerParams(int width, int height, int channel, TYPE type) {
        this.inputShape[0] = width;
        this.inputShape[1] = height;
        this.inputShape[2] = channel;
        if (type == TYPE.FLAT) {
            this.outputShape[0] = width * height;
            this.outputShape[1] = channel;
            this.outputShape[2] = 1;
        } else if (type == TYPE.SAME) {
            this.outputShape[0] = width;
            this.outputShape[1] = height;
            this.outputShape[2] = channel;
        }
    }

    // 用于卷积和池化层
    public LayerParams(int inputWidth, int inputHeight, int inputChannel, int outputWidth, int outputHeight, int outputChannel) {
        this.inputShape[0] = inputWidth;
        this.inputShape[1] = inputHeight;
        this.inputShape[2] = inputChannel;
        this.outputShape[0] = outputWidth;
        this.outputShape[1] = outputHeight;
        this.outputShape[2] = outputChannel;
    }

}